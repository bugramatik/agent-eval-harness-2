"""FastAPI app: start a run, read a run, compare two runs, health check."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from harness import __version__
from harness.config import Settings, get_settings
from harness.db import init_db, session_scope
from harness.models import Run, TaskResult
from harness.runner import compare_runs, run_suite

logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Eval Harness", version=__version__)


class RunRequest(BaseModel):
    suite: str = Field(default="default", min_length=1, max_length=64)
    prompt_version: str = Field(default="v1", min_length=1, max_length=64)


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    suite: str
    model: str
    prompt_version: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    pass_rate: float | None


class ResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    passed: bool
    score: float
    steps: int
    judge_reason: str
    latency_ms: int


class RunDetail(RunOut):
    results: list[ResultOut] = Field(default_factory=list)


class RegressionOut(BaseModel):
    task_id: str
    baseline_passed: bool
    candidate_passed: bool
    delta: float


def get_session() -> Session:
    with session_scope() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@app.on_event("startup")
def on_startup() -> None:
    init_db()

