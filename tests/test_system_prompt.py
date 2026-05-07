import argparse

from agent.config import AgentConfig
from agent.context import InMemoryContext
from agent.session import CliSession
from cli import config_from_args


class DummyLLM:
    pass


def _args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "provider": None,
        "api_key": None,
        "model": None,
        "base_url": None,
        "request_timeout_seconds": None,
        "max_iterations": 8,
        "system_prompt_file": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_context_starts_with_system_prompt() -> None:
    context = InMemoryContext(system_prompt="You are concise.")

    assert context.messages() == [{"role": "system", "content": "You are concise."}]


def test_undo_preserves_system_prompt() -> None:
    context = InMemoryContext(system_prompt="You are concise.")
    context.add_user_message("hello")
    context.add_assistant_message("hi")

    context.undo_last_turn()

    assert context.messages() == [{"role": "system", "content": "You are concise."}]


def test_cli_session_uses_configured_system_prompt() -> None:
    session = CliSession(DummyLLM(), AgentConfig(system_prompt="You are concise."))

    assert session.context.messages() == [{"role": "system", "content": "You are concise."}]


def test_config_from_args_reads_system_prompt_file_env(monkeypatch, tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("You are concise.\n", encoding="utf-8")
    monkeypatch.setenv("SYSTEM_PROMPT_FILE", str(prompt_file))

    config = config_from_args(_args())

    assert config.agent.system_prompt == "You are concise."


def test_config_from_args_prefers_system_prompt_file_arg(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / "env.md"
    arg_file = tmp_path / "arg.md"
    env_file.write_text("env prompt", encoding="utf-8")
    arg_file.write_text("arg prompt", encoding="utf-8")
    monkeypatch.setenv("SYSTEM_PROMPT_FILE", str(env_file))

    config = config_from_args(_args(system_prompt_file=str(arg_file)))

    assert config.agent.system_prompt == "arg prompt"


def test_config_from_args_reads_default_project_system_prompt(monkeypatch, tmp_path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompt_file = prompts_dir / "system_prompt.md"
    prompt_file.write_text("default prompt", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SYSTEM_PROMPT_FILE", raising=False)

    config = config_from_args(_args())

    assert config.agent.system_prompt == "default prompt"
