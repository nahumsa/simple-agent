"""Provider payload adapters for OpenAI-compatible chat APIs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol, cast

from agent_core.types import JsonObject, LLMResponse, RawLLMToolCall, ToolSpec


class ChatPayloadAdapter(Protocol):
    """Serialize chat requests and parse provider responses."""

    def to_provider_payload(
        self,
        *,
        model: str,
        messages: list[JsonObject],
        tools: list[ToolSpec],
    ) -> JsonObject: ...

    def from_provider_response(self, data: JsonObject) -> LLMResponse: ...


class OpenAIChatCompletionsAdapter:
    """Adapter for OpenAI Chat Completions compatible APIs."""

    def to_provider_payload(
        self,
        *,
        model: str,
        messages: list[JsonObject],
        tools: list[ToolSpec],
    ) -> JsonObject:
        payload: JsonObject = {
            "model": model,
            "messages": [to_openai_message(message) for message in messages],
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def from_provider_response(self, data: JsonObject) -> LLMResponse:
        choices = data["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM response did not include choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("LLM response choice was not an object")

        message = first_choice["message"]
        if not isinstance(message, dict):
            raise ValueError("LLM response message was not an object")

        raw_tool_calls = message.get("tool_calls", [])
        tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []
        content = message.get("content")
        return LLMResponse(
            content=content if isinstance(content, str) else None,
            tool_calls=[to_tool_call(call) for call in tool_calls if isinstance(call, dict)],
        )


class OllamaOpenAICompatibleAdapter(OpenAIChatCompletionsAdapter):
    """Adapter for Ollama's OpenAI-compatible chat endpoint."""


class GeminiOpenAICompatibleAdapter(OpenAIChatCompletionsAdapter):
    """Adapter for Gemini OpenAI-compatible responses.

    Gemini may include provider-specific `extra_content` such as thought signatures.
    The base OpenAI-compatible serializer preserves that field on replayed tool calls.
    """


def to_openai_message(message: JsonObject) -> JsonObject:
    role = message["role"]
    if role == "assistant" and "tool_calls" in message:
        tool_calls = message["tool_calls"]
        return {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": [
                to_openai_tool_call(call)
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


def to_openai_tool_call(call: JsonObject) -> JsonObject:
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


def to_tool_call(call: Mapping[str, object]) -> RawLLMToolCall:
    id_val = call.get("id")
    function = call.get("function", {})
    function_data = function if isinstance(function, Mapping) else {}
    name_val = function_data.get("name")
    arguments_val = function_data.get("arguments")
    return RawLLMToolCall(
        id="" if id_val is None else str(id_val),
        name="" if name_val is None else str(name_val),
        arguments="{}" if arguments_val is None else str(arguments_val),
        extra_content=cast(JsonObject | None, call.get("extra_content")),
    )
