"""Empty tool registry for plain chat mode."""

from __future__ import annotations

from agent_core.types import JsonObject, ToolSpec


class NoTools:
    """Empty tool registry for plain chat mode."""

    def specs(self) -> list[ToolSpec]:
        return []

    async def call(
        self,
        name: str,
        args: JsonObject,
        *,
        session: object,
        tool_call_id: str,
    ) -> tuple[str, bool]:
        return f"Unknown tool: {name}", False

    async def cancel_running(self) -> None:
        return None
