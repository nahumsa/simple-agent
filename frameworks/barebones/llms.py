from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping

from agent_core.config import LLMConfig, Provider
from agent_core.interfaces import LLM
from agent_core.types import JsonObject, LLMResponse, RawLLMToolCall, ToolSpec
from frameworks.barebones.llm_adapters import (
    ChatPayloadAdapter,
    OllamaOpenAICompatibleAdapter,
    OpenAIChatCompletionsAdapter,
    to_openai_message,
    to_tool_call,
)
from frameworks.barebones.llm_decorators import LoggingLLM, RetryingLLM, TimeoutLLM

logger = logging.getLogger(__name__)


class OpenAIChatLLM:
    """Minimal OpenAI-compatible chat client using only the standard library."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: int,
        adapter: ChatPayloadAdapter | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.adapter = adapter or OpenAIChatCompletionsAdapter()

    async def complete(
        self,
        *,
        messages: list[JsonObject],
        tools: list[ToolSpec],
    ) -> LLMResponse:
        return await asyncio.to_thread(self._complete_sync, messages, tools)

    def _complete_sync(
        self, messages: list[JsonObject], tools: list[ToolSpec]
    ) -> LLMResponse:
        payload_dict = self.adapter.to_provider_payload(
            model=self.model,
            messages=messages,
            tools=tools,
        )

        _debug_json("LLM request payload", payload_dict)
        payload = json.dumps(payload_dict).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode(errors="replace")
            logger.error("LLM HTTP error %s %s: %s", exc.code, exc.reason, error_body)
            raise

        _debug_json("LLM response payload", data)
        result = self.adapter.from_provider_response(data)
        _debug_json(
            "LLM parsed response",
            {
                "content": result.content,
                "tool_calls": [call.__dict__ for call in result.tool_calls],
            },
        )
        return result


def _debug_json(label: str, data: object) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("%s:\n%s", label, json.dumps(data, indent=2, ensure_ascii=False, default=str))


# Backward-compatible helper names used by tests and callers.
def _to_openai_message(message: JsonObject) -> JsonObject:
    return to_openai_message(message)


def _to_tool_call(call: Mapping[str, object]) -> RawLLMToolCall:
    return to_tool_call(call)


def _openai_llm(config: LLMConfig) -> LLM:
    api_key = config.resolved_api_key
    if not api_key:
        raise RuntimeError(
            "OpenAI provider requires an API key. Set LLM_API_KEY or OPENAI_API_KEY."
        )
    return OpenAIChatLLM(
        api_key=api_key,
        model=config.model,
        base_url=config.base_url,
        timeout_seconds=config.request_timeout_seconds,
        adapter=OpenAIChatCompletionsAdapter(),
    )


def _ollama_llm(config: LLMConfig) -> LLM:
    return OpenAIChatLLM(
        api_key=config.api_key or "ollama",
        model=config.model,
        base_url=config.base_url,
        timeout_seconds=config.request_timeout_seconds,
        adapter=OllamaOpenAICompatibleAdapter(),
    )


LLM_PROVIDER_REGISTRY: dict[Provider, Callable[[LLMConfig], LLM]] = {
    "ollama": _ollama_llm,
    "openai": _openai_llm,
}


def register_llm_provider(provider: Provider, builder: Callable[[LLMConfig], LLM]) -> None:
    """Register or replace an LLM provider builder."""
    LLM_PROVIDER_REGISTRY[provider] = builder


def build_llm(config: LLMConfig) -> LLM:
    builder = LLM_PROVIDER_REGISTRY.get(config.provider)
    if builder is None:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

    llm = builder(config)
    return LoggingLLM(
        TimeoutLLM(
            RetryingLLM(llm, attempts=2),
            timeout_seconds=config.request_timeout_seconds,
        )
    )
