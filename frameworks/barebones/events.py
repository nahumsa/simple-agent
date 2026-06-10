"""Event sinks for observing agent session events."""

from __future__ import annotations

from agent_core.types import JsonObject


class ConsoleEventSink:
    """Render selected session events to the terminal."""

    def __init__(self, *, emit_assistant_messages: bool = True) -> None:
        self.emit_assistant_messages = emit_assistant_messages

    async def handle(self, event: str, payload: JsonObject) -> None:
        if event == "assistant_message" and self.emit_assistant_messages:
            print(f"assistant: {payload.get('content', '<no content>')}")
        elif event in {"error", "tool_output", "approval_required", "interrupted"}:
            print(f"[{event}] {payload}")
