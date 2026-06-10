"""Composable decorators for tool registries."""

from __future__ import annotations

import asyncio
import logging

from agent_core.interfaces import Tools
from agent_core.types import JsonObject, ToolSpec

logger = logging.getLogger(__name__)


class LoggingTools:
    """Decorator that logs tool calls without changing behavior."""

    def __init__(self, inner: Tools) -> None:
        self.inner = inner

    def specs(self) -> list[ToolSpec]:
        return self.inner.specs()

    async def call(
        self,
        name: str,
        args: JsonObject,
        *,
        session: object,
        tool_call_id: str,
    ) -> tuple[str, bool]:
        logger.debug("Calling tool %s (%s)", name, tool_call_id)
        output, success = await self.inner.call(
            name,
            args,
            session=session,
            tool_call_id=tool_call_id,
        )
        logger.debug("Tool %s (%s) success=%s", name, tool_call_id, success)
        return output, success

    async def cancel_running(self) -> None:
        await self.inner.cancel_running()


class TimeoutTools:
    """Decorator that applies an async timeout around tool calls."""

    def __init__(self, inner: Tools, *, timeout_seconds: int) -> None:
        self.inner = inner
        self.timeout_seconds = timeout_seconds

    def specs(self) -> list[ToolSpec]:
        return self.inner.specs()

    async def call(
        self,
        name: str,
        args: JsonObject,
        *,
        session: object,
        tool_call_id: str,
    ) -> tuple[str, bool]:
        return await asyncio.wait_for(
            self.inner.call(
                name,
                args,
                session=session,
                tool_call_id=tool_call_id,
            ),
            timeout=self.timeout_seconds,
        )

    async def cancel_running(self) -> None:
        await self.inner.cancel_running()
