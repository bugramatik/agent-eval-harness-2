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


def check_tool(
    transcript: Transcript,
    expected_tool: str | None,
    expected_args: str | None,
) -> tuple[bool, str]:
    """Check that the expected tool was called, with the expected arguments."""
    if not expected_tool:
        if transcript.tool_calls:
            seen = ", ".join(sorted({call.name for call in transcript.tool_calls}))
            return False, f"no tool expected, called: {seen}"
        return True, "no tool expected and none called"

    calls = [call for call in transcript.tool_calls if call.name == expected_tool]
    if not calls:
        seen = ", ".join(sorted({call.name for call in transcript.tool_calls})) or "none"
        return False, f"expected tool {expected_tool!r}, called: {seen}"

    expected = _parse_expected_args(expected_args)
    if expected is None:
        return True, f"called {expected_tool}"

    for call in calls:
        if _args_match(call.arguments, expected):
            return True, f"called {expected_tool} with matching arguments"

    first = calls[0].arguments
    return False, f"called {expected_tool} but arguments {json.dumps(first, sort_keys=True)} did not match"


def check_contains(transcript: Transcript, must_contain: str | None) -> tuple[bool, str]:
    """Check that the final answer contains the required text, case-insensitively."""
    if not must_contain:
        return True, "no required text"

    answer = transcript.final_answer or ""
    if must_contain.strip().lower() in answer.lower():
        return True, f"answer contains {must_contain!r}"
    return False, f"answer does not contain {must_contain!r}"


def _transcript_for_judge(transcript: Transcript) -> str:
    calls = [
        {"tool": call.name, "arguments": call.arguments} for call in transcript.tool_calls
    ]
    return json.dumps(
        {"tool_calls": calls, "final_answer": transcript.final_answer, "steps": transcript.steps},
        indent=2,
        sort_keys=True,
    )


async def judge(transcript: Transcript, task: Task, settings: Settings) -> tuple[float, str]:
    """Ask the judge model to grade the transcript. Returns (score, reason)."""
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        timeout=settings.request_timeout,
    )
    user_message = (
        f"Task: {task.prompt}\n\n"
        f"Run:\n{_transcript_for_judge(transcript)}"
    )
    try:
        response = await client.chat.completions.create(
            model=settings.judge_model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)
        score = float(payload.get("score", 0.0))
        reason = str(payload.get("reason", "")).strip() or "no reason given"
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("judge returned an unreadable answer: %s", exc)
        return 0.0, f"judge answer could not be read: {exc}"
    except Exception as exc:  # network, auth, rate limit
        logger.warning("judge call failed: %s", exc)
        return 0.0, f"judge call failed: {exc}"

    score = min(max(score, 0.0), 1.0)
    return score, reason


async def score_task(
    transcript: Transcript,
    task: Task,
    settings: Settings,
) -> tuple[bool, float, str]:
    """Combine the deterministic checks and the judge into (passed, score, reason).

    The deterministic checks are gates: a run that calls the wrong tool or misses the
    required text fails no matter what the judge says. When the gates pass, the score
    is the mean of the check score and the judge score.
    """
    tool_ok, tool_reason = check_tool(transcript, task.expected_tool, task.expected_args)
    text_ok, text_reason = check_contains(transcript, task.must_contain)

    if not tool_ok or not text_ok:
        reasons = [reason for ok, reason in ((tool_ok, tool_reason), (text_ok, text_reason)) if not ok]
        return False, 0.0, "; ".join(reasons)

    judge_score, judge_reason = await judge(transcript, task, settings)
    check_score = 1.0
    score = round((check_score + judge_score) / 2, 4)
    passed = score >= settings.pass_threshold
    reason = f"checks passed; judge {judge_score:.2f}: {judge_reason}"
    return passed, score, reason
