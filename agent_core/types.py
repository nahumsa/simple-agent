"""Shared data types for the simple agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

JsonObject: TypeAlias = dict[str, object]
ToolSpec: TypeAlias = dict[str, object]


class AgentState(Enum):
    """Runtime lifecycle states for an agent session."""

    READY = "ready"
    RUNNING_TURN = "running_turn"
    CANCELLING = "cancelling"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


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


@dataclass(frozen=True)
class ChatTurnResult:
    """Framework-neutral result for one chat turn."""

    content: str
    framework: str
    iterations: int | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: object | None = None
