"""Doom-loop detection for repeated tool-call patterns.
This was based on https://github.com/huggingface/ml-intern/tree/main"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agent.types import ToolCallSignature


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
    following_messages: list[dict[str, Any]],
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
    messages: list[dict[str, Any]],
    lookback: int = 30,
) -> list[ToolCallSignature]:
    """Extract recent assistant tool calls as comparable signatures."""
    signatures: list[ToolCallSignature] = []
    recent = messages[-lookback:]

    for index, message in enumerate(recent):
        if message.get("role") != "assistant":
            continue

        for tool_call in message.get("tool_calls") or []:
            tool_call_id = tool_call.get("id")
            name = tool_call.get("name", "")
            args = tool_call.get("arguments", {})
            args_str = json.dumps(args, sort_keys=True)
            result_hash = find_tool_result_hash(recent[index + 1 :], tool_call_id)

            signatures.append(
                ToolCallSignature(
                    name=name,
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


def check_for_doom_loop(messages: list[dict[str, Any]]) -> str | None:
    """Return a corrective prompt if recent tool use looks stuck."""
    signatures = extract_recent_tool_signatures(messages)
    if len(signatures) < 3:
        return None

    repeated_tool = detect_identical_consecutive(signatures)
    if repeated_tool:
        return (
            f"[SYSTEM: REPETITION GUARD] You have called '{repeated_tool}' "
            "with the same arguments multiple times and received the same result. "
            "Stop repeating this approach. Try a different tool, different arguments, "
            "or explain what you are stuck on and ask the user for guidance."
        )

    repeating_pattern = detect_repeating_sequence(signatures)
    if repeating_pattern:
        pattern = " -> ".join(signature.name for signature in repeating_pattern)
        return (
            "[SYSTEM: REPETITION GUARD] You are stuck in a repeating cycle of "
            f"tool calls: [{pattern}]. Stop this cycle and try a fundamentally "
            "different approach."
        )

    return None
