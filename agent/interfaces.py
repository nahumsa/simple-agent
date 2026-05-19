"""Protocol interfaces expected by SimpleAgentLoop."""

from __future__ import annotations

from typing import Protocol

from agent.config import AgentConfig
from agent.types import JsonObject, LLMResponse, LLMResult, ToolCall, ToolSpec


class Context(Protocol):
    @property
    def needs_compaction(self) -> bool: ...
    def messages(self) -> list[JsonObject]: ...
    def add_user_message(self, content: str) -> None: ...
    def add_assistant_message(self, content: str) -> None: ...
    def add_assistant_tool_calls(self, result: LLMResult) -> None: ...
    def add_tool_result(
        self,
        tool_call_id: str,
        name: str,
        content: str,
        *,
        success: bool,
    ) -> None: ...
    def undo_last_turn(self) -> None: ...
    async def compact(self) -> None: ...


class Tools(Protocol):
    def specs(self) -> list[ToolSpec]: ...
    async def call(
        self,
        name: str,
        args: JsonObject,
        *,
        session: object,
        tool_call_id: str,
    ) -> tuple[str, bool]: ...
    async def cancel_running(self) -> None: ...


class LLM(Protocol):
    async def complete(
        self,
        *,
        messages: list[JsonObject],
        tools: list[ToolSpec],
    ) -> LLMResponse: ...


class AgentSession(Protocol):
    config: AgentConfig
    context: Context
    tools: Tools
    llm: LLM
    running: bool
    cancelled: bool
    pending_approval: list[ToolCall] | None

    async def emit(self, event: str, payload: JsonObject) -> None: ...
    async def save(self) -> None: ...
