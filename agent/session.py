"""CLI session wiring for SimpleAgentLoop."""

from __future__ import annotations

from agent.config import AgentConfig
from agent.context import InMemoryContext
from typing import cast

from agent.interfaces import Context, LLM, Tools
from agent.tools import NoTools
from agent.types import JsonObject, ToolCall


class CliSession:
    """Minimal session object expected by SimpleAgentLoop."""

    def __init__(self, llm: object, config: AgentConfig | None = None, tools: Tools | None = None) -> None:
        self.config = config or AgentConfig()
        self.tools: Tools = tools or NoTools()
        self.context: Context = InMemoryContext(system_prompt=self.config.system_prompt)
        self.llm: LLM = cast(LLM, llm)
        self.running = True
        self.cancelled = False
        self.pending_approval: list[ToolCall] | None = None

    async def emit(self, event: str, payload: JsonObject) -> None:
        if event == "assistant_message":
            print(f"assistant: {payload['content']}")
        elif event in {"error", "tool_output", "approval_required", "interrupted"}:
            print(f"[{event}] {payload}")

    async def save(self) -> None:
        return None
