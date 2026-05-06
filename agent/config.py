"""Central configuration for the simple agent application."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Mapping, cast

Provider = Literal["ollama", "openai", "echo"]

PROVIDER_CHOICES: tuple[Provider, ...] = ("ollama", "openai", "echo")
DEFAULT_PROVIDER: Provider = "ollama"
DEFAULT_MODEL = "gemma4:latest"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
LLM_PROVIDER_ENV = "LLM_PROVIDER"
LLM_MODEL_ENV = "LLM_MODEL"
LLM_BASE_URL_ENV = "LLM_BASE_URL"
LLM_API_KEY_ENV = "LLM_API_KEY"
LLM_REQUEST_TIMEOUT_SECONDS_ENV = "LLM_REQUEST_TIMEOUT_SECONDS"
DEFAULT_MAX_ITERATIONS = 8
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class LLMConfig:
    """Configuration needed to construct an LLM client."""

    provider: Provider = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    request_timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "LLMConfig":
        """Build LLM configuration from environment variables."""
        env = environ or os.environ
        provider = _env_provider(env)

        return cls(
            provider=provider,
            model=env.get(LLM_MODEL_ENV, DEFAULT_MODEL),
            base_url=env.get(LLM_BASE_URL_ENV, DEFAULT_BASE_URL),
            api_key=env.get(LLM_API_KEY_ENV) or env.get(OPENAI_API_KEY_ENV),
            request_timeout_seconds=_env_int(
                env,
                LLM_REQUEST_TIMEOUT_SECONDS_ENV,
                DEFAULT_REQUEST_TIMEOUT_SECONDS,
            ),
        )

    @property
    def resolved_api_key(self) -> str | None:
        """Return the configured API key, falling back to supported API-key env vars."""
        if self.api_key:
            return self.api_key
        return os.getenv(LLM_API_KEY_ENV) or os.getenv(OPENAI_API_KEY_ENV)


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for agent-loop behavior."""

    max_iterations: int = DEFAULT_MAX_ITERATIONS


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    llm: LLMConfig
    agent: AgentConfig = field(default_factory=AgentConfig)


_APP_PROVIDERS = set(PROVIDER_CHOICES)


def _env_provider(env: Mapping[str, str]) -> Provider:
    provider = env.get(LLM_PROVIDER_ENV, DEFAULT_PROVIDER)
    if provider not in _APP_PROVIDERS:
        choices = ", ".join(PROVIDER_CHOICES)
        raise ValueError(f"Invalid {LLM_PROVIDER_ENV}={provider!r}. Expected one of: {choices}.")
    return cast(Provider, provider)


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={value!r}. Expected an integer.") from exc
