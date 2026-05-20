"""CLI session wiring for SimpleAgentLoop."""

from __future__ import annotations

from typing import cast

from agent_core.config import AgentConfig
from agent_core.interfaces import Context, LLM, Tools
from agent_core.types import JsonObject, ToolCall
from frameworks.barebones.context import InMemoryContext
from frameworks.barebones.tools import NoTools


class CliSession:
    """Minimal session object expected by SimpleAgentLoop."""

    def __init__(
        self,
        llm: object,
        config: AgentConfig | None = None,
        tools: Tools | None = None,
        *,
        emit_messages: bool = True,
    ) -> None:
        self.config = config or AgentConfig()
        self.tools: Tools = tools or NoTools()
        self.context: Context = InMemoryContext(system_prompt=self.config.system_prompt)
        self.llm: LLM = cast(LLM, llm)
        self.emit_messages = emit_messages
        self.running = True
        self.cancelled = False
        self.pending_approval: list[ToolCall] | None = None

    async def emit(self, event: str, payload: JsonObject) -> None:
        if event == "assistant_message" and self.emit_messages:
            print(f"assistant: {payload['content']}")
        elif event in {"error", "tool_output", "approval_required", "interrupted"}:
            print(f"[{event}] {payload}")

    async def save(self) -> None:
        return None
