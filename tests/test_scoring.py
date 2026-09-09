"""Tests for the deterministic checks and the score aggregation in harness.scoring."""

import pytest

from harness.agent import ToolCall, Transcript
from harness.config import Settings
from harness.models import Task
from harness.scoring import check_contains, check_tool, score_task


def make_task(**overrides) -> Task:
    values = {
        "id": "orders-lookup",
        "suite": "default",
        "name": "Look up an order by id",
        "prompt": "Find order A-1042 and tell me its status.",
        "expected_tool": "get_order",
        "expected_args": {"order_id": "A-1042"},
        "must_contain": "shipped",
    }
    values.update(overrides)
    return Task(**values)


def make_transcript(answer: str = "Order A-1042 is shipped.", calls=None) -> Transcript:
    return Transcript(
        final_answer=answer,
        tool_calls=calls if calls is not None else [ToolCall(name="get_order", arguments={"order_id": "A-1042"})],
        steps=2,
        latency_ms=410,
    )


def test_check_tool_pass():
    ok, reason = check_tool(make_transcript(), "get_order", {"order_id": "A-1042"})
    assert ok is True
    assert "get_order" in reason


def test_check_tool_wrong_args():
    transcript = make_transcript(calls=[ToolCall(name="get_order", arguments={"order_id": "A-1"})])
    ok, reason = check_tool(transcript, "get_order", {"order_id": "A-1042"})
    assert ok is False
    assert "order_id" in reason


def test_check_tool_expected_none_but_called():
    ok, reason = check_tool(make_transcript(), None, {})
    assert ok is False
    assert "no tool" in reason.lower()


def test_check_contains_missing():
    ok, reason = check_contains(make_transcript(answer="Order A-1042 is pending."), "shipped")
    assert ok is False
    assert "shipped" in reason


def test_check_contains_is_case_insensitive():
    ok, _ = check_contains(make_transcript(answer="It SHIPPED yesterday."), "shipped")
    assert ok is True


@pytest.mark.asyncio
async def test_score_task_combines_checks(monkeypatch):
    async def fake_judge(transcript, task, settings):
        return 0.8, "answer is grounded in the tool result"

    monkeypatch.setattr("harness.scoring.judge", fake_judge)
    transcript = make_transcript()
    passed, score, reason = await score_task(transcript, make_task(), Settings(openai_api_key="test"))
    assert passed is True
    assert score == pytest.approx(0.9)
    assert "grounded" in reason


@pytest.mark.asyncio
async def test_score_task_fails_when_a_check_fails(monkeypatch):
    async def fake_judge(transcript, task, settings):
        return 1.0, "looks right"

    monkeypatch.setattr("harness.scoring.judge", fake_judge)
    transcript = make_transcript(answer="I could not find it.")
    passed, score, reason = await score_task(transcript, make_task(), Settings(openai_api_key="test"))
    assert passed is False
    assert score < 1.0
    assert "shipped" in reason
