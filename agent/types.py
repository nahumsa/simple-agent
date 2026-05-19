"""Shared data types for the simple agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

JsonObject: TypeAlias = dict[str, object]
ToolSpec: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: JsonObject
    raw_arguments: str | None = None
    extra_content: JsonObject | None = None


@dataclass(frozen=True)
class LLMResult:
    content: str | None
    tool_calls: list[ToolCall]


@dataclass(frozen=True)
class RawLLMToolCall:
    id: str
    name: str
    arguments: str
    extra_content: JsonObject | None = None


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: list[RawLLMToolCall]


@dataclass(frozen=True)
class ToolCallSignature:
    """Stable signature for a tool call plus its observed result."""

    name: str
    args_hash: str
    result_hash: str | None = None
