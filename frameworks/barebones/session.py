"""CLI session wiring for SimpleAgentLoop."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from agent_core.config import AgentConfig
from agent_core.interfaces import Context, EventSink, LLM, Tools
from agent_core.types import AgentState, JsonObject
from frameworks.barebones.context import InMemoryContext
from frameworks.barebones.events import ConsoleEventSink
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
        event_sinks: Sequence[EventSink] | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.tools: Tools = tools or NoTools()
        self.context: Context = InMemoryContext(system_prompt=self.config.system_prompt)
        self.llm: LLM = cast(LLM, llm)
        self.event_sinks: list[EventSink] = list(event_sinks) if event_sinks is not None else [
            ConsoleEventSink(emit_assistant_messages=emit_messages)
        ]
        self.state = AgentState.READY
        self._cancelled = False

    @property
    def running(self) -> bool:
        return self.state not in {AgentState.SHUTTING_DOWN, AgentState.STOPPED}

    @running.setter
    def running(self, value: bool) -> None:
        self.state = AgentState.READY if value else AgentState.STOPPED

    @property
    def cancelled(self) -> bool:
        return self._cancelled or self.state == AgentState.CANCELLING

    @cancelled.setter
    def cancelled(self, value: bool) -> None:
        self._cancelled = value
        if value:
            self.state = AgentState.CANCELLING
        elif self.state == AgentState.CANCELLING:
            self.state = AgentState.READY

    async def emit(self, event: str, payload: JsonObject) -> None:
        for sink in self.event_sinks:
            await sink.handle(event, payload)

    async def save(self) -> None:
        return None
