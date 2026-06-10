"""Composable decorators for LLM cross-cutting behavior."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from agent_core.interfaces import LLM
from agent_core.types import JsonObject, LLMResponse, ToolSpec

logger = logging.getLogger(__name__)


class LoggingLLM:
    """Decorator that logs LLM calls without changing behavior."""

    def __init__(self, inner: LLM) -> None:
        self.inner = inner

    async def complete(
        self,
        *,
        messages: list[JsonObject],
        tools: list[ToolSpec],
    ) -> LLMResponse:
        logger.debug("Calling LLM with %d messages and %d tools", len(messages), len(tools))
        response = await self.inner.complete(messages=messages, tools=tools)
        logger.debug(
            "LLM returned content=%s tool_calls=%d",
            response.content is not None,
            len(response.tool_calls),
        )
        return response


class TimeoutLLM:
    """Decorator that applies an async timeout around LLM calls."""

    def __init__(self, inner: LLM, *, timeout_seconds: int) -> None:
        self.inner = inner
        self.timeout_seconds = timeout_seconds

    async def complete(
        self,
        *,
        messages: list[JsonObject],
        tools: list[ToolSpec],
    ) -> LLMResponse:
        return await asyncio.wait_for(
            self.inner.complete(messages=messages, tools=tools),
            timeout=self.timeout_seconds,
        )


class RetryingLLM:
    """Decorator that retries transient LLM failures."""

    def __init__(
        self,
        inner: LLM,
        *,
        attempts: int = 2,
        retry_exceptions: Sequence[type[BaseException]] = (OSError, TimeoutError),
    ) -> None:
        self.inner = inner
        self.attempts = max(1, attempts)
        self.retry_exceptions = tuple(retry_exceptions)

    async def complete(
        self,
        *,
        messages: list[JsonObject],
        tools: list[ToolSpec],
    ) -> LLMResponse:
        last_error: BaseException | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                return await self.inner.complete(messages=messages, tools=tools)
            except self.retry_exceptions as exc:
                last_error = exc
                if attempt == self.attempts:
                    break
                logger.warning("LLM call failed on attempt %d/%d: %s", attempt, self.attempts, exc)

        if last_error is not None:
            raise last_error
        raise RuntimeError("RetryingLLM exhausted without recording an error")
