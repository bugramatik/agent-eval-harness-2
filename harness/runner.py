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

