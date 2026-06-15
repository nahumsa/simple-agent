"""Protocol interfaces expected by SimpleAgentLoop."""

from __future__ import annotations

from typing import Protocol

from agent_core.config import AgentConfig
from agent_core.types import (
    AgentState,
    ChatTurnResult,
    JsonObject,
    LLMResponse,
    LLMResult,
    ToolSpec,
)


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


class EventSink(Protocol):
    async def handle(self, event: str, payload: JsonObject) -> None: ...


class AgentSession(Protocol):
    config: AgentConfig
    context: Context
    tools: Tools
    llm: LLM
    state: AgentState
    running: bool
    cancelled: bool

    async def emit(self, event: str, payload: JsonObject) -> None: ...
    async def save(self) -> None: ...


class ChatAgent(Protocol):
    async def run_turn(self, user_text: str) -> ChatTurnResult: ...
