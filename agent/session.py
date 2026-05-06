"""CLI session wiring for SimpleAgentLoop."""

from __future__ import annotations

from typing import Any

from agent.config import AgentConfig
from agent.context import InMemoryContext
from agent.tools import NoTools
from agent.types import ToolCall


class CliSession:
    """Minimal session object expected by SimpleAgentLoop."""

    def __init__(self, llm: Any, config: AgentConfig | None = None) -> None:
        self.context = InMemoryContext()
        self.tools = NoTools()
        self.llm = llm
        self.config = config or AgentConfig()
        self.running = True
        self.cancelled = False
        self.pending_approval: list[ToolCall] | None = None

    async def emit(self, event: str, payload: dict[str, Any]) -> None:
        if event == "assistant_message":
            print(f"assistant: {payload['content']}")
        elif event in {"error", "tool_output", "approval_required", "interrupted"}:
            print(f"[{event}] {payload}")

    async def save(self) -> None:
        return None
