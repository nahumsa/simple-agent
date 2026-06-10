from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import cast

from agent_core.config import LLMConfig
from agent_core.interfaces import LLM
from agent_core.types import JsonObject, LLMResponse, RawLLMToolCall, ToolSpec

logger = logging.getLogger(__name__)


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
        messages: list[JsonObject],
        tools: list[ToolSpec],
    ) -> LLMResponse:
        return await asyncio.to_thread(self._complete_sync, messages, tools)

    def _complete_sync(
        self, messages: list[JsonObject], tools: list[ToolSpec]
    ) -> LLMResponse:
        payload_dict: JsonObject = {
            "model": self.model,
            "messages": [_to_openai_message(message) for message in messages],
        }
        if tools:
            payload_dict["tools"] = tools
            payload_dict["tool_choice"] = "auto"

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
        message = data["choices"][0]["message"]
        result = LLMResponse(
            content=message.get("content"),
            tool_calls=[_to_tool_call(call) for call in message.get("tool_calls", [])],
        )
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


def _to_openai_message(message: JsonObject) -> JsonObject:
    role = message["role"]
    if role == "assistant" and "tool_calls" in message:
        tool_calls = message["tool_calls"]
        return {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": [
                _to_openai_tool_call(call)
                for call in tool_calls
                if isinstance(call, dict)
            ]
            if isinstance(tool_calls, list)
            else [],
        }

    if role == "tool":
        openai_message = {
            "role": "tool",
            "tool_call_id": message["tool_call_id"],
            "content": message["content"],
        }
        if message.get("name"):
            openai_message["name"] = message["name"]
        return openai_message

    return {"role": role, "content": message.get("content") or ""}


def _to_openai_tool_call(call: JsonObject) -> JsonObject:
    openai_call: JsonObject = {
        "id": call["id"],
        "type": "function",
        "function": {
            "name": call["name"],
            "arguments": call.get("raw_arguments") or json.dumps(call["arguments"]),
        },
    }
    if call.get("extra_content") is not None:
        openai_call["extra_content"] = call["extra_content"]
    return openai_call


def _to_tool_call(call: Mapping[str, object]) -> RawLLMToolCall:
    function = call.get("function", {})
    function_data = function if isinstance(function, dict) else {}
    return RawLLMToolCall(
        id=str(call.get("id", "")),
        name=str(function_data.get("name", "")),
        arguments=str(function_data.get("arguments", "{}")),
        extra_content=cast(JsonObject | None, call.get("extra_content")),
    )


def build_llm(config: LLMConfig) -> LLM:
    provider = config.provider

    if provider == "ollama":
        return OpenAIChatLLM(
            api_key=config.api_key or "ollama",
            model=config.model,
            base_url=config.base_url,
            timeout_seconds=config.request_timeout_seconds,
        )

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
    )
