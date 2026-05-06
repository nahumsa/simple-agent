"""Protocol interfaces expected by SimpleAgentLoop."""

from __future__ import annotations

from typing import Any, Protocol

from agent.types import LLMResult


class Context(Protocol):
    @property
    def needs_compaction(self) -> bool: ...
    def messages(self) -> list[dict[str, Any]]: ...
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
    def specs(self) -> list[dict[str, Any]]: ...
    async def call(
        self,
        name: str,
        args: dict[str, Any],
        *,
        session: Any,
        tool_call_id: str,
    ) -> tuple[str, bool]: ...
    async def cancel_running(self) -> None: ...


class LLM(Protocol):
    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any: ...
