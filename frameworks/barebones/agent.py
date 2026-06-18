"""Barebones chat agent adapter and composition root."""

from __future__ import annotations

from dataclasses import dataclass

from agent_core.config import AppConfig
from agent_core.types import ChatTurnResult
from frameworks.barebones.llms import build_llm
from frameworks.barebones.loop import SimpleAgentLoop
from frameworks.barebones.session import CliSession
from frameworks.barebones.tools import ChallengeDataTools, LoggingTools


@dataclass(frozen=True)
class BarebonesComponents:
    """Objects composed to run the barebones agent."""

    tools: ChallengeDataTools
    session: CliSession
    loop: SimpleAgentLoop


class BarebonesChatFramework:
    """Adapter exposing the hand-written loop through the shared agent contract."""

    def __init__(
        self,
        config: AppConfig,
        components: BarebonesComponents | None = None,
    ) -> None:
        self.config = config
        self.components = components or build_barebones_components(config)
        self.tools = self.components.tools
        self.loop = self.components.loop

    async def run_turn(self, user_text: str) -> ChatTurnResult:
        content = await self.loop.run_turn(user_text)
        return ChatTurnResult(content=content or "")


def build_barebones_framework(config: AppConfig) -> BarebonesChatFramework:
    """Build the barebones chat agent from explicit components."""
    return BarebonesChatFramework(config, build_barebones_components(config))


def build_barebones_components(config: AppConfig) -> BarebonesComponents:
    """Composition root for the barebones agent."""
    tools = ChallengeDataTools()
    session = CliSession(
        build_llm(config.llm),
        config.agent,
        tools=LoggingTools(tools),
        emit_messages=False,
    )
    loop = SimpleAgentLoop(session)
    return BarebonesComponents(tools=tools, session=session, loop=loop)
