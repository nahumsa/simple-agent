"""In-memory conversation context implementation."""

from __future__ import annotations

from typing import Any

from agent.types import LLMResult


class InMemoryContext:
    """Small in-memory conversation store for the CLI."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self._messages: list[dict[str, Any]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    @property
    def needs_compaction(self) -> bool:
        return False

    def messages(self) -> list[dict[str, Any]]:
        return self._messages.copy()

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def add_assistant_tool_calls(self, result: LLMResult) -> None:
        self._messages.append(
            {
                "role": "assistant",
                "content": result.content,
                "tool_calls": [
                    {
                        "id": tool.id,
                        "name": tool.name,
                        "arguments": tool.args,
                    }
                    for tool in result.tool_calls
                ],
            }
        )

    def add_tool_result(
        self,
        tool_call_id: str,
        name: str,
        content: str,
        *,
        success: bool,
    ) -> None:
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content,
                "success": success,
            }
        )

    def undo_last_turn(self) -> None:
        while self._messages and self._messages[-1].get("role") not in {"system", "user"}:
            self._messages.pop()
        if self._messages and self._messages[-1].get("role") == "user":
            self._messages.pop()

    async def compact(self) -> None:
        return None
