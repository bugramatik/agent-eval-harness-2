"""Deterministic checks and the LLM judge that turn a transcript into a score."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from harness.agent import Transcript
from harness.config import Settings
from harness.models import Task
