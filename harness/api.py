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

