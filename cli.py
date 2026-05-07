"""Command-line interface for chatting with the simple agent."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from agent.config import (
    AgentConfig,
    AppConfig,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_SYSTEM_PROMPT_FILE,
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    LLM_PROVIDER_ENV,
    LLM_REQUEST_TIMEOUT_SECONDS_ENV,
    LLMConfig,
    OPENAI_API_KEY_ENV,
    PROVIDER_CHOICES,
    SYSTEM_PROMPT_FILE_ENV,
)
from agent.llms import build_llm
from agent.loop import SimpleAgentLoop
from agent.session import CliSession


def read_system_prompt(path: str | None) -> str | None:
    """Read a system prompt from a markdown file, if one is configured."""
    configured_path = path or os.getenv(SYSTEM_PROMPT_FILE_ENV)
    if not configured_path:
        default_path = Path(DEFAULT_SYSTEM_PROMPT_FILE)
        if not default_path.exists():
            return None
        configured_path = DEFAULT_SYSTEM_PROMPT_FILE

    prompt_path = Path(configured_path)
    try:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"System prompt file not found: {prompt_path}") from exc

    return prompt or None


def config_from_args(args: argparse.Namespace) -> AppConfig:
    env_llm = LLMConfig.from_environment()
    return AppConfig(
        llm=LLMConfig(
            provider=args.provider or env_llm.provider,
            api_key=args.api_key or env_llm.api_key,
            model=args.model or env_llm.model,
            base_url=args.base_url or env_llm.base_url,
            request_timeout_seconds=args.request_timeout_seconds or env_llm.request_timeout_seconds,
        ),
        agent=AgentConfig(
            max_iterations=args.max_iterations,
            system_prompt=read_system_prompt(args.system_prompt_file),
        ),
    )


async def chat(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    agent = SimpleAgentLoop(CliSession(build_llm(config.llm), config.agent))
    print(f"Chat started with {config.llm.provider}:{config.llm.model}. Type /exit or /quit to stop.")

    while True:
        try:
            user_text = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_text in {"/exit", "/quit"}:
            break
        if not user_text:
            continue

        await agent.run_turn(user_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the simple agent.")
    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default=None,
        help=f"LLM provider to use. Defaults to ${LLM_PROVIDER_ENV} or {DEFAULT_PROVIDER}.",
    )
    parser.add_argument(
        "--api-key",
        help=f"OpenAI-compatible API key. Defaults to ${LLM_API_KEY_ENV} or ${OPENAI_API_KEY_ENV}.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Chat model name. Defaults to ${LLM_MODEL_ENV} or {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            f"OpenAI-compatible base URL. Defaults to ${LLM_BASE_URL_ENV} or "
            f"{DEFAULT_BASE_URL}. Use https://api.openai.com/v1 for OpenAI."
        ),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=None,
        help=(
            "HTTP request timeout for OpenAI-compatible providers. "
            f"Defaults to ${LLM_REQUEST_TIMEOUT_SECONDS_ENV}."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Maximum LLM/tool iterations per turn.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help=(
            "Markdown file containing the system prompt. Defaults to "
            f"${SYSTEM_PROMPT_FILE_ENV} or {DEFAULT_SYSTEM_PROMPT_FILE} if it exists."
        ),
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(chat(parse_args()))
