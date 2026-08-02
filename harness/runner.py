"""Run a whole task suite, persist the results and compare runs."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from harness.agent import run_task
from harness.config import Settings
from harness.db import session_scope
from harness.models import Run, Task, TaskResult
from harness.scoring import score_task

SUITE_DIR = Path(__file__).resolve().parent.parent / "data"


def load_suite(suite: str) -> list[Task]:
    """Read a suite file from data/ and return its tasks as unsaved Task rows."""
    path = SUITE_DIR / f"{suite}.json"
    if not path.exists():
        raise FileNotFoundError(f"suite {suite!r} not found at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[Task] = []
    for item in raw:
        tasks.append(
            Task(
                id=item["id"],
                suite=suite,
                name=item["name"],
                prompt=item["prompt"],
                expected_tool=item.get("expected_tool"),
                expected_args=json.dumps(item.get("expected_args") or {}),
                must_contain=item.get("must_contain"),
            )
        )
    return tasks


def _persist_tasks(suite: str, tasks: list[Task]) -> list[Task]:
    """Insert the suite's tasks once and return the stored rows with their ids."""
    with session_scope() as session:
        existing = {
            row.id: row
            for row in session.scalars(select(Task).where(Task.suite == suite)).all()
        }
        stored: list[Task] = []
        for task in tasks:
            row = existing.get(task.id)
            if row is None:
                session.add(task)
                session.flush()
                row = task
            else:
                row.name = task.name
                row.prompt = task.prompt
                row.expected_tool = task.expected_tool
                row.expected_args = task.expected_args
                row.must_contain = task.must_contain
            stored.append(row)
        session.flush()
        for row in stored:
            session.expunge(row)
        return stored


async def _run_one(task: Task, settings: Settings) -> tuple[Task, bool, float, int, str]:
    transcript = await run_task(task, settings)
    passed, score, reason = await score_task(transcript, task, settings)
    return task, passed, score, transcript.steps, reason


async def run_suite(suite: str, settings: Settings, run_id: int | None = None) -> Run:
    """Run every task in a suite, store the results and return the finished Run row.

    When run_id is given the row already exists (the API queues it) and is updated in
    place; otherwise a new row is created here.
    """
    tasks = _persist_tasks(suite, load_suite(suite))

    if run_id is None:
        run = Run(
            suite=suite,
            model=settings.model,
            prompt_version=settings.prompt_version,
            status="running",
            started_at=datetime.now(timezone.utc),
        )
        with session_scope() as session:
            session.add(run)
            session.flush()
            run_id = run.id
    else:
        with session_scope() as session:
            queued = session.get(Run, run_id)
            if queued is None:
                raise RuntimeError(f"run {run_id} not found")
            queued.status = "running"
            queued.started_at = datetime.now(timezone.utc)

    try:
        outcomes = await asyncio.gather(*(_run_one(task, settings) for task in tasks))
    except Exception:
        with session_scope() as session:
            failed = session.get(Run, run_id)
            if failed is not None:
                failed.status = "failed"
                failed.finished_at = datetime.now(timezone.utc)
        raise

    passed_count = 0
    with session_scope() as session:
        for task, passed, score, steps, reason in outcomes:
            passed_count += int(passed)
            session.add(
                TaskResult(
                    run_id=run_id,
                    task_id=task.id,
                    passed=passed,
                    score=score,
                    steps=steps,
                    judge_reason=reason,
                )
            )
        stored = session.get(Run, run_id)
        if stored is None:
            raise RuntimeError(f"run {run_id} disappeared while it was running")
        stored.status = "finished"
        stored.finished_at = datetime.now(timezone.utc)
        stored.pass_rate = round(passed_count / len(outcomes), 4) if outcomes else 0.0
        session.flush()
        session.expunge(stored)
        return stored


def compare_runs(
    baseline_id: int,
    candidate_id: int,
    session: Session | None = None,
) -> list[dict]:
    """Compare two runs task by task and return the rows that changed."""
    if session is None:
        with session_scope() as own_session:
            return compare_runs(baseline_id, candidate_id, session=own_session)

    baseline = {
        row.task_id: row
        for row in session.scalars(
            select(TaskResult).where(TaskResult.run_id == baseline_id)
        ).all()
    }
    candidate = {
        row.task_id: row
        for row in session.scalars(
            select(TaskResult).where(TaskResult.run_id == candidate_id)
        ).all()
    }
    names = {
        row.id: row.name
        for row in session.scalars(
            select(Task).where(Task.id.in_(set(baseline) | set(candidate)))
        ).all()
    }

    rows: list[dict] = []
    for task_id in sorted(set(baseline) | set(candidate)):
        before = baseline.get(task_id)
        after = candidate.get(task_id)
        before_passed = bool(before and before.passed)
        after_passed = bool(after and after.passed)
        rows.append(
            {
                "task_id": task_id,
                "name": names.get(task_id, f"task-{task_id}"),
                "baseline_passed": before_passed,
                "candidate_passed": after_passed,
                "delta": int(after_passed) - int(before_passed),
            }
        )
    return rows
