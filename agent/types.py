"""Shared data types for the simple agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class LLMResult:
    content: str | None
    tool_calls: list[ToolCall]


@dataclass(frozen=True)
class ToolCallSignature:
    """Stable signature for a tool call plus its observed result."""

    name: str
    args_hash: str
    result_hash: str | None = None
