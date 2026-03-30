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
