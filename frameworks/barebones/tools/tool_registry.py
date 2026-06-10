"""Command-style tool registry for barebones agent tools."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agent_core.types import JsonObject, ToolSpec


class ToolCommand(Protocol):
    """A single executable tool command."""

    name: str

    def spec(self) -> ToolSpec: ...

    async def execute(
        self,
        args: JsonObject,
        *,
        session: object,
        tool_call_id: str,
    ) -> tuple[str, bool]: ...


class ToolRegistry:
    """Registry that exposes tool specs and dispatches calls by tool name."""

    def __init__(self, tools: Sequence[ToolCommand]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    async def call(
        self,
        name: str,
        args: JsonObject,
        *,
        session: object,
        tool_call_id: str,
    ) -> tuple[str, bool]:
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}", False

        return await tool.execute(args, session=session, tool_call_id=tool_call_id)
