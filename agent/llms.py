from __future__ import annotations

import asyncio
import json
import urllib.request
from types import SimpleNamespace
from typing import Any

from agent.config import LLMConfig


class EchoLLM:
    """Dependency-free fallback LLM used when no OpenAI API key is provided."""

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        last_user = next(
            (
                message["content"]
                for message in reversed(messages)
                if message["role"] == "user"
            ),
            "",
        )
        return SimpleNamespace(content=f"Echo: {last_user}", tool_calls=[])


class OpenAIChatLLM:
    """Minimal OpenAI-compatible chat client using only the standard library."""

    def __init__(
        self, *, api_key: str, model: str, base_url: str, timeout_seconds: int
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        return await asyncio.to_thread(self._complete_sync, messages)

    def _complete_sync(self, messages: list[dict[str, Any]]) -> Any:
        payload = json.dumps({"model": self.model, "messages": messages}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode())

        message = data["choices"][0]["message"]
        return SimpleNamespace(content=message.get("content"), tool_calls=[])


def build_llm(config: LLMConfig) -> Any:
    provider = config.provider

    if provider == "echo":
        return EchoLLM()

    if provider == "ollama":
        return OpenAIChatLLM(
            api_key=config.api_key or "ollama",
            model=config.model,
            base_url=config.base_url,
            timeout_seconds=config.request_timeout_seconds,
        )

    api_key = config.resolved_api_key
    if not api_key:
        return EchoLLM()

    return OpenAIChatLLM(
        api_key=api_key,
        model=config.model,
        base_url=config.base_url,
        timeout_seconds=config.request_timeout_seconds,
    )
