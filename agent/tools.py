"""Tool registries for the simple agent."""

from __future__ import annotations

from typing import Any


class NoTools:
    """Empty tool registry for plain chat mode."""

    def specs(self) -> list[dict[str, Any]]:
        return []

    async def call(
        self,
        name: str,
        args: dict[str, Any],
        *,
        session: Any,
        tool_call_id: str,
    ) -> tuple[str, bool]:
        return f"Unknown tool: {name}", False

    async def cancel_running(self) -> None:
        return None
