"""Factory for framework-specific chat adapters."""

from __future__ import annotations

from agent_core.config import AppConfig, Framework
from agent_core.interfaces import ChatFramework


def build_chat_framework(framework: Framework, config: AppConfig) -> ChatFramework:
    """Build the selected chat framework using lazy imports."""
    if framework == "barebones":
        from frameworks.barebones.agent import BarebonesChatFramework

        return BarebonesChatFramework(config)

    if framework == "langchain":
        raise RuntimeError(
            "LangChain adapter is not implemented yet.\n\n"
            "When dependencies are added, install them with:\n\n"
            "  uv sync --group langchain"
        )

    if framework == "pydantic-ai":
        raise RuntimeError(
            "Pydantic AI adapter is not implemented yet.\n\n"
            "When dependencies are added, install them with:\n\n"
            "  uv sync --group pydantic-ai"
        )

    raise ValueError(f"Unsupported framework: {framework}")
