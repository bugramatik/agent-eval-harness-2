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

