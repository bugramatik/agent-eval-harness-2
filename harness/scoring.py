"""Deterministic checks and the LLM judge that turn a transcript into a score."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from harness.agent import Transcript
from harness.config import Settings
from harness.models import Task

logger = logging.getLogger(__name__)

JUDGE_PROMPT = (
    "You grade one run of a tool-calling assistant. You get the user task, the tool "
    "calls the assistant made and its final answer. Reply with a single JSON object "
    "with the keys score (a number from 0 to 1) and reason (one short sentence). "
    "Score 1 when the answer is correct and grounded in the tool output, 0.5 when it "
    "is partly right, 0 when it is wrong or made up."
)


def _parse_expected_args(expected_args: str | None) -> dict[str, Any] | None:
    if not expected_args:
        return None
    try:
        parsed = json.loads(expected_args)
    except json.JSONDecodeError:
        logger.warning("expected_args is not valid JSON: %r", expected_args)
        return None
    return parsed if isinstance(parsed, dict) else None


def _args_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Every expected key must be present with the same value; extra keys are allowed."""
    for key, value in expected.items():
        if key not in actual:
            return False
        if isinstance(value, str) and isinstance(actual[key], str):
            if value.strip().lower() != actual[key].strip().lower():
                return False
        elif actual[key] != value:
            return False
    return True

