"""Registry-backed factory for framework-specific chat adapters."""

from __future__ import annotations

from collections.abc import Callable

from agent_core.config import AppConfig, Framework
from agent_core.interfaces import ChatFramework

FrameworkBuilder = Callable[[AppConfig], ChatFramework]


def _build_barebones(config: AppConfig) -> ChatFramework:
    from frameworks.barebones.agent import build_barebones_framework

    return build_barebones_framework(config)


def _build_unimplemented_framework(name: str, dependency_group: str) -> FrameworkBuilder:
    def builder(config: AppConfig) -> ChatFramework:
        raise RuntimeError(
            f"{name} adapter is not implemented yet.\n\n"
            "When dependencies are added, install them with:\n\n"
            f"  uv sync --group {dependency_group}"
        )

    return builder


FRAMEWORK_REGISTRY: dict[Framework, FrameworkBuilder] = {
    "barebones": _build_barebones,
    "langchain": _build_unimplemented_framework("LangChain", "langchain"),
    "pydantic-ai": _build_unimplemented_framework("Pydantic AI", "pydantic-ai"),
}


def register_framework(framework: Framework, builder: FrameworkBuilder) -> None:
    """Register or replace a chat framework builder."""
    FRAMEWORK_REGISTRY[framework] = builder


def build_chat_framework(framework: Framework, config: AppConfig) -> ChatFramework:
    """Build the selected chat framework using lazy imports."""
    builder = FRAMEWORK_REGISTRY.get(framework)
    if builder is None:
        raise ValueError(f"Unsupported framework: {framework}")
    return builder(config)
