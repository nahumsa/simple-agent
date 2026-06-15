"""Factory for the barebones chat agent."""

from __future__ import annotations

from agent_core.config import AppConfig
from agent_core.interfaces import ChatAgent


def build_chat_agent(config: AppConfig) -> ChatAgent:
    """Build the barebones chat agent."""
    from frameworks.barebones.agent import build_barebones_framework

    return build_barebones_framework(config)
