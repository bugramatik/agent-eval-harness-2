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

