"""Run a single task against an OpenAI-compatible model with tool calling."""

from __future__ import annotations

import json
import time
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from harness.config import Settings
from harness.models import Task


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by its id and return its status and total.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order id, e.g. ORD-1042."}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refund_order",
            "description": "Issue a refund for an order. Only call after the order is looked up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number", "description": "Amount in the order currency."},
                    "reason": {"type": "string"},
                },
                "required": ["order_id", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "Search the support FAQ for a short query and return matching entries.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a support assistant. Use the provided tools when they help answer the "
    "request. Call a tool only when it is needed, pass exact ids, and finish with a short "
    "answer that states what you did."
)


def _fake_tool_result(name: str, arguments: dict[str, Any]) -> str:
    """Return a deterministic stand-in result so runs are reproducible without backends."""
    if name == "lookup_order":
        return json.dumps(
            {
                "order_id": arguments.get("order_id"),
                "status": "delivered",
                "total": 84.5,
                "currency": "EUR",
            }
        )
    if name == "refund_order":
        return json.dumps(
            {
                "order_id": arguments.get("order_id"),
                "refunded": True,
                "amount": arguments.get("amount"),
            }
        )
    if name == "search_faq":
        return json.dumps(
            {
                "query": arguments.get("query"),
                "results": [
                    {"title": "Return window", "body": "Items can be returned within 30 days."},
                    {"title": "Refund timing", "body": "Refunds settle in 3 to 5 business days."},
                ],
            }
        )
    return json.dumps({"error": f"unknown tool {name}"})


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Transcript(BaseModel):
    final_answer: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    steps: int = 0
    latency_ms: int = 0
    error: str | None = None


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def run_task(task: Task, settings: Settings) -> Transcript:
    """Run one task through the model, executing tool calls until it answers or runs out of steps."""
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        timeout=settings.request_timeout,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    transcript = Transcript()
    started = time.perf_counter()

    for step in range(1, settings.max_steps + 1):
        transcript.steps = step
        try:
            response = await client.chat.completions.create(
                model=settings.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )
        except Exception as exc:  # network, auth or provider errors end the run for this task
            transcript.error = f"{type(exc).__name__}: {exc}"
            break

        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            transcript.final_answer = (message.content or "").strip()
            break

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments or "{}",
                        },
                    }
                    for call in calls
                ],
            }
        )

        for call in calls:
            arguments = _parse_arguments(call.function.arguments)
            transcript.tool_calls.append(ToolCall(name=call.function.name, arguments=arguments))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": _fake_tool_result(call.function.name, arguments),
                }
            )
    else:
        transcript.error = f"no final answer within {settings.max_steps} steps"

    transcript.latency_ms = int((time.perf_counter() - started) * 1000)
    return transcript
