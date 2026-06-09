"""Doom-loop detection for repeated tool-call and chat patterns.
This was based on https://github.com/huggingface/ml-intern/tree/main"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

from agent_core.types import JsonObject, ToolCallSignature

DoomLoopDetector = Callable[[list[JsonObject]], str | None]


@dataclass(frozen=True)
class ChatMessageSignature:
    """Stable signature for a conversational message."""

    role: str
    content_hash: str


def normalize_jsonish(value: str) -> str:
    """Return a stable JSON representation when possible.

    This makes these hash identically:
        {"b": 2, "a": 1}
        {"a":1,"b":2}
    """
    if not value:
        return ""

    try:
        return json.dumps(json.loads(value), sort_keys=True, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return value


def short_hash(value: str) -> str:
    return hashlib.md5(normalize_jsonish(value).encode()).hexdigest()[:12]


def find_tool_result_hash(
    following_messages: list[JsonObject],
    tool_call_id: str | None,
) -> str | None:
    """Find the matching tool result before the next user/assistant message."""
    for message in following_messages:
        role = message.get("role")

        if role == "tool" and message.get("tool_call_id") == tool_call_id:
            return short_hash(str(message.get("content") or ""))

        if role in {"user", "assistant"}:
            break

    return None


def extract_recent_tool_signatures(
    messages: list[JsonObject],
    lookback: int = 30,
) -> list[ToolCallSignature]:
    """Extract recent assistant tool calls as comparable signatures."""
    signatures: list[ToolCallSignature] = []
    recent = messages[-lookback:]

    for index, message in enumerate(recent):
        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = tool_call.get("id")
            name = tool_call.get("name", "")
            args = tool_call.get("arguments", {})
            args_str = json.dumps(args, sort_keys=True)
            result_hash = find_tool_result_hash(
                recent[index + 1 :],
                str(tool_call_id) if tool_call_id is not None else None,
            )

            signatures.append(
                ToolCallSignature(
                    name=str(name),
                    args_hash=short_hash(args_str),
                    result_hash=result_hash,
                )
            )

    return signatures


def detect_identical_consecutive(
    signatures: list[ToolCallSignature],
    threshold: int = 3,
) -> str | None:
    """Detect A, A, A where A is the same tool, args, and result."""
    if len(signatures) < threshold:
        return None

    count = 1
    for current, previous in zip(signatures[1:], signatures[:-1]):
        if current == previous:
            count += 1
            if count >= threshold:
                return current.name
        else:
            count = 1

    return None


def extract_recent_chat_signatures(
    messages: list[JsonObject],
    *,
    role: str,
    lookback: int = 30,
) -> list[ChatMessageSignature]:
    """Extract recent non-tool-call chat messages for repetition checks."""
    signatures: list[ChatMessageSignature] = []

    for message in messages[-lookback:]:
        if message.get("role") != role:
            continue
        if message.get("tool_calls"):
            continue

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        signatures.append(
            ChatMessageSignature(role=role, content_hash=short_hash(content.strip()))
        )

    return signatures


def detect_repeated_chat_message(
    messages: list[JsonObject],
    *,
    role: str,
    threshold: int = 3,
) -> bool:
    """Detect the same user or assistant chat message repeated recently."""
    signatures = extract_recent_chat_signatures(messages, role=role)
    if len(signatures) < threshold:
        return False

    count = 1
    for current, previous in zip(signatures[1:], signatures[:-1]):
        if current == previous:
            count += 1
            if count >= threshold:
                return True
        else:
            count = 1

    return False


def detect_repeating_sequence(
    signatures: list[ToolCallSignature],
    min_sequence_len: int = 2,
    max_sequence_len: int = 5,
    min_repetitions: int = 2,
) -> list[ToolCallSignature] | None:
    """Detect repeated tails such as A, B, A, B or A, B, C, A, B, C."""
    n = len(signatures)

    for sequence_len in range(min_sequence_len, max_sequence_len + 1):
        required = sequence_len * min_repetitions
        if n < required:
            continue

        tail = signatures[-required:]
        pattern = tail[:sequence_len]

        if all(
            tail[offset : offset + sequence_len] == pattern
            for offset in range(sequence_len, required, sequence_len)
        ):
            return pattern

    return None


def check_repeated_user_messages(messages: list[JsonObject]) -> str | None:
    """Return a corrective prompt when the user repeats the same message."""
    if not detect_repeated_chat_message(messages, role="user"):
        return None

    return (
        "[SYSTEM: REPETITION GUARD] The user has repeated the same message "
        "multiple times. Do not repeat the same answer. Acknowledge the repetition, "
        "take a different approach, and ask a clarifying question if needed."
    )


def check_repeated_assistant_messages(messages: list[JsonObject]) -> str | None:
    """Return a corrective prompt when the assistant repeats the same response."""
    if not detect_repeated_chat_message(messages, role="assistant"):
        return None

    return (
        "[SYSTEM: REPETITION GUARD] You have repeated the same response multiple "
        "times. Stop repeating yourself. Provide a materially different answer, "
        "summarize what is blocking progress, or ask the user for clarification."
    )


def check_repeated_tool_calls(messages: list[JsonObject]) -> str | None:
    """Return a corrective prompt for identical repeated tool calls."""
    repeated_tool = detect_identical_consecutive(
        extract_recent_tool_signatures(messages)
    )
    if not repeated_tool:
        return None

    return (
        f"[SYSTEM: REPETITION GUARD] You have called '{repeated_tool}' "
        "with the same arguments multiple times and received the same result. "
        "Stop repeating this approach. Try a different tool, different arguments, "
        "or explain what you are stuck on and ask the user for guidance."
    )


def check_repeating_tool_sequence(messages: list[JsonObject]) -> str | None:
    """Return a corrective prompt for repeating tool-call sequences."""
    repeating_pattern = detect_repeating_sequence(
        extract_recent_tool_signatures(messages)
    )
    if not repeating_pattern:
        return None

    pattern = " -> ".join(signature.name for signature in repeating_pattern)
    return (
        "[SYSTEM: REPETITION GUARD] You are stuck in a repeating cycle of "
        f"tool calls: [{pattern}]. Stop this cycle and try a fundamentally "
        "different approach."
    )


DOOM_LOOP_DETECTORS: tuple[DoomLoopDetector, ...] = (
    check_repeated_user_messages,
    check_repeated_assistant_messages,
    check_repeated_tool_calls,
    check_repeating_tool_sequence,
)


def check_for_doom_loop(messages: list[JsonObject]) -> str | None:
    """Return a corrective prompt if recent tool use or chat looks stuck."""
    for detector in DOOM_LOOP_DETECTORS:
        prompt = detector(messages)
        if prompt:
            return prompt

    return None
