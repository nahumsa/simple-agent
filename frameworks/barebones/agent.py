"""Barebones chat framework adapter."""

from __future__ import annotations

from agent_core.config import AgentConfig, AppConfig
from agent_core.types import ChatTurnResult
from frameworks.barebones.llms import build_llm
from frameworks.barebones.loop import SimpleAgentLoop
from frameworks.barebones.session import CliSession
from frameworks.barebones.tools import ChallengeDataTools


class BarebonesChatFramework:
    """Adapter exposing the hand-written loop through the shared framework contract."""

    framework = "barebones"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.tools = ChallengeDataTools()
        agent_config = _agent_config_with_challenge_context(config.agent, self.tools)
        self.loop = SimpleAgentLoop(
            CliSession(
                build_llm(config.llm),
                agent_config,
                tools=self.tools,
                emit_messages=False,
            )
        )

    async def run_turn(self, user_text: str) -> ChatTurnResult:
        content = await self.loop.run_turn(user_text)
        return ChatTurnResult(content=content or "", framework=self.framework)


def _agent_config_with_challenge_context(
    config: AgentConfig,
    tools: ChallengeDataTools,
) -> AgentConfig:
    """Append the extracted challenge index to the system prompt when available."""
    parts = [part for part in [config.system_prompt, tools.initial_context()] if part]
    if not parts:
        return config
    return AgentConfig(
        max_iterations=config.max_iterations,
        system_prompt="\n\n".join(parts),
    )
