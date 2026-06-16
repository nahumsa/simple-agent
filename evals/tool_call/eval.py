"""Evaluate whether the agent chooses expected tools for user prompts.

Dataset format:
[
  {
    "id": "search-json-parser",
    "user_prompt": "Find the JSON parser challenge. Use tools before answering.",
    "expected_tool_calls": ["search_challenges"],
    "expected_final_contains": ["JSON"]
  }
]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

ROOT_DIR = Path(__file__).resolve().parents[2]
TOOL_CALL_DATASETS_DIR = Path("evals/tool_call/datasets")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent_core.config import (  # noqa: E402
    AgentConfig,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_SYSTEM_PROMPT_FILE,
    LLMConfig,
    PROVIDER_CHOICES,
)
from agent_core.interfaces import EventSink  # noqa: E402
from agent_core.types import JsonObject  # noqa: E402
from cli import read_system_prompt  # noqa: E402
from evals.core.cli import (  # noqa: E402
    add_common_output_args,
    emit_report,
    output_config_from_args,
)
from evals.core.datasets import (  # noqa: E402
    latest_versioned_dataset,
    load_json_list,
    optional_string,
    required_string,
    string_list_field,
)
from evals.core.output import (  # noqa: E402
    dataclass_report_to_json,
    default_csv_results_path as core_default_csv_results_path,
    safe_filename_part,
    write_csv_rows,
)
from evals.core.progress import ProgressReporter  # noqa: E402
from frameworks.barebones.events import ConsoleEventSink  # noqa: E402
from frameworks.barebones.llms import build_llm  # noqa: E402
from frameworks.barebones.loop import SimpleAgentLoop  # noqa: E402
from frameworks.barebones.session import CliSession  # noqa: E402
from frameworks.barebones.tools import ChallengeDataTools, LoggingTools  # noqa: E402


@dataclass(frozen=True)
class AgentToolCallEvalCase:
    """One prompt with expected agent tool calls."""

    id: str
    user_prompt: str
    expected_tool_calls: list[str]
    expected_final_contains: list[str]


@dataclass(frozen=True)
class AgentToolCallEvalResult:
    """Evaluation result for one agent prompt."""

    id: str
    user_prompt: str
    expected_tool_calls: list[str]
    actual_tool_calls: list[str]
    missing_tool_calls: list[str]
    expected_final_contains: list[str]
    missing_final_substrings: list[str]
    passed: bool
    final_response_preview: str


@dataclass(frozen=True)
class AgentToolCallEvalReport:
    """Aggregate agent tool-call evaluation report."""

    dataset: str
    provider: str
    model: str
    case_count: int
    pass_rate: float
    results: list[AgentToolCallEvalResult]


class RecordingEventSink:
    """Capture agent events emitted during eval turns."""

    def __init__(self) -> None:
        self.events: list[tuple[str, JsonObject]] = []

    async def handle(self, event: str, payload: JsonObject) -> None:
        self.events.append((event, payload))

    def tool_call_names(self) -> list[str]:
        names: list[str] = []
        for event, payload in self.events:
            tool = payload.get("tool")
            if event == "tool_call" and isinstance(tool, str):
                names.append(tool)
        return names


def load_dataset(path: Path) -> list[AgentToolCallEvalCase]:
    """Load and validate an agent tool-call eval dataset."""
    raw_cases = load_json_list(path)
    return [
        _parse_case(index, raw_case)
        for index, raw_case in enumerate(raw_cases, start=1)
    ]


def _parse_case(index: int, raw_case: object) -> AgentToolCallEvalCase:
    """Validate and convert one raw JSON object into an eval case."""
    if not isinstance(raw_case, dict):
        raise ValueError(f"Case {index} must be a JSON object.")

    return AgentToolCallEvalCase(
        id=_optional_string(raw_case, "id", default=f"case-{index}", case_index=index),
        user_prompt=_required_string(raw_case, "user_prompt", case_index=index),
        expected_tool_calls=_string_list_field(
            raw_case,
            "expected_tool_calls",
            default=[],
            case_index=index,
        ),
        expected_final_contains=_string_list_field(
            raw_case,
            "expected_final_contains",
            default=[],
            case_index=index,
        ),
    )


def _required_string(
    raw_case: dict[object, object], field: str, *, case_index: int
) -> str:
    return required_string(raw_case, field, case_index=case_index)


def _optional_string(
    raw_case: dict[object, object],
    field: str,
    *,
    default: str,
    case_index: int,
) -> str:
    return optional_string(raw_case, field, default=default, case_index=case_index)


def _string_list_field(
    raw_case: dict[object, object],
    field: str,
    *,
    default: list[str],
    case_index: int,
) -> list[str]:
    return string_list_field(
        raw_case,
        field,
        default=default,
        case_index=case_index,
    )


def _agent_config_with_challenge_context(
    config: AgentConfig,
    tools: ChallengeDataTools,
) -> AgentConfig:
    parts = [part for part in [config.system_prompt, tools.initial_context()] if part]
    if not parts:
        return config
    return AgentConfig(
        max_iterations=config.max_iterations,
        system_prompt="\n\n".join(parts),
    )


def build_eval_loop(
    *,
    llm_config: LLMConfig,
    agent_config: AgentConfig,
    emit_messages: bool,
) -> tuple[SimpleAgentLoop, RecordingEventSink]:
    """Build a real agent loop plus event recorder for one eval case."""
    tools = ChallengeDataTools()
    recorder = RecordingEventSink()
    sinks: list[EventSink] = [recorder]
    if emit_messages:
        sinks.append(ConsoleEventSink(emit_assistant_messages=True))

    session = CliSession(
        build_llm(llm_config),
        _agent_config_with_challenge_context(agent_config, tools),
        tools=LoggingTools(tools),
        emit_messages=False,
        event_sinks=sinks,
    )
    return SimpleAgentLoop(session), recorder


def _missing_expected_tool_calls(
    expected_tool_calls: list[str],
    actual_tool_calls: list[str],
) -> list[str]:
    """Return expected tool names not found as an ordered subsequence."""
    missing: list[str] = []
    search_from = 0
    for expected in expected_tool_calls:
        try:
            matched_at = actual_tool_calls.index(expected, search_from)
        except ValueError:
            missing.append(expected)
        else:
            search_from = matched_at + 1
    return missing


def evaluate_case(
    case: AgentToolCallEvalCase,
    actual_tool_calls: list[str],
    final_response: str,
) -> AgentToolCallEvalResult:
    """Score one agent turn."""
    missing_tool_calls = _missing_expected_tool_calls(
        case.expected_tool_calls,
        actual_tool_calls,
    )
    final_response_lower = final_response.casefold()
    missing_final_substrings = [
        substring
        for substring in case.expected_final_contains
        if substring.casefold() not in final_response_lower
    ]
    passed = not missing_tool_calls and not missing_final_substrings
    return AgentToolCallEvalResult(
        id=case.id,
        user_prompt=case.user_prompt,
        expected_tool_calls=case.expected_tool_calls,
        actual_tool_calls=actual_tool_calls,
        missing_tool_calls=missing_tool_calls,
        expected_final_contains=case.expected_final_contains,
        missing_final_substrings=missing_final_substrings,
        passed=passed,
        final_response_preview=final_response[:500].replace("\n", " "),
    )


async def run_eval_async(
    *,
    dataset_path: Path,
    llm_config: LLMConfig,
    agent_config: AgentConfig,
    emit_messages: bool,
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> AgentToolCallEvalReport:
    """Run all agent tool-call eval cases and compute aggregate metrics."""
    cases = load_dataset(dataset_path)
    results: list[AgentToolCallEvalResult] = []
    total_cases = len(cases)
    progress_reporter = ProgressReporter(enabled=progress, stream=progress_stream)

    for index, case in enumerate(cases, start=1):
        progress_reporter.case_started(index, total_cases, case.id)
        loop, recorder = build_eval_loop(
            llm_config=llm_config,
            agent_config=agent_config,
            emit_messages=emit_messages,
        )
        final_response = await loop.run_turn(case.user_prompt)
        result = evaluate_case(case, recorder.tool_call_names(), final_response or "")
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        actual_tools = ",".join(result.actual_tool_calls) or "none"
        progress_reporter.case_finished(
            index,
            total_cases,
            status,
            case.id,
            f"tools={actual_tools}",
        )

    case_count = len(results)
    pass_rate = (
        sum(result.passed for result in results) / case_count if case_count else 0.0
    )
    return AgentToolCallEvalReport(
        dataset=str(dataset_path),
        provider=llm_config.provider,
        model=llm_config.model,
        case_count=case_count,
        pass_rate=pass_rate,
        results=results,
    )


def run_eval(
    *,
    dataset_path: Path,
    llm_config: LLMConfig,
    agent_config: AgentConfig,
    emit_messages: bool,
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> AgentToolCallEvalReport:
    """Synchronous wrapper for the async agent eval."""
    return asyncio.run(
        run_eval_async(
            dataset_path=dataset_path,
            llm_config=llm_config,
            agent_config=agent_config,
            emit_messages=emit_messages,
            progress=progress,
            progress_stream=progress_stream,
        )
    )


def print_text_report(report: AgentToolCallEvalReport) -> None:
    """Print a concise human-readable report."""
    print(f"Dataset: {report.dataset}")
    print(f"Model: {report.provider}:{report.model}")
    print(f"Cases: {report.case_count}")
    print(f"Pass rate: {report.pass_rate:.3f}")
    print()

    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.id}")
        print(f"  prompt: {result.user_prompt}")
        print(f"  expected tools: {', '.join(result.expected_tool_calls) or '(none)'}")
        print(f"  actual tools: {', '.join(result.actual_tool_calls) or '(none)'}")
        if result.missing_tool_calls:
            print(f"  missing tools: {', '.join(result.missing_tool_calls)}")
        if result.missing_final_substrings:
            print(f"  missing final text: {', '.join(result.missing_final_substrings)}")
        print(f"  final: {result.final_response_preview}")


def report_to_json(report: AgentToolCallEvalReport) -> dict[str, Any]:
    """Convert report dataclasses to JSON-serializable dictionaries."""
    return dataclass_report_to_json(report)


def write_csv_report(
    report: AgentToolCallEvalReport, path: Path | None = None
) -> None:
    """Write one CSV row per eval case."""
    fieldnames = [
        "id",
        "user_prompt",
        "expected_tool_calls",
        "actual_tool_calls",
        "missing_tool_calls",
        "expected_final_contains",
        "missing_final_substrings",
        "passed",
        "final_response_preview",
        "dataset",
        "provider",
        "model",
    ]
    rows = (
        {
            "id": result.id,
            "user_prompt": result.user_prompt,
            "expected_tool_calls": ";".join(result.expected_tool_calls),
            "actual_tool_calls": ";".join(result.actual_tool_calls),
            "missing_tool_calls": ";".join(result.missing_tool_calls),
            "expected_final_contains": ";".join(result.expected_final_contains),
            "missing_final_substrings": ";".join(
                result.missing_final_substrings
            ),
            "passed": result.passed,
            "final_response_preview": result.final_response_preview,
            "dataset": report.dataset,
            "provider": report.provider,
            "model": report.model,
        }
        for result in report.results
    )
    write_csv_rows(fieldnames=fieldnames, rows=rows, path=path)


def default_csv_results_path(
    report: AgentToolCallEvalReport, results_dir: Path
) -> Path:
    """Build a timestamped CSV path for an eval report."""
    return core_default_csv_results_path(
        prefix="tool_call",
        model=report.model,
        results_dir=results_dir,
    )


def _safe_filename_part(value: str) -> str:
    """Return a filesystem-friendly filename segment."""
    return safe_filename_part(value)


def latest_tool_call_dataset() -> Path:
    """Return the newest versioned tool-call dataset path."""
    return latest_versioned_dataset(
        datasets_dir=TOOL_CALL_DATASETS_DIR,
        versioned_prefix="tool_call_v",
        fallback_name="tool_call.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate agent tool selection.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=latest_tool_call_dataset(),
        help=(
            "JSON dataset of user prompts and expected agent tool calls. "
            "Defaults to the newest evals/tool_call/datasets/tool_call_v*.json file."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default=None,
        help=f"LLM provider. Defaults to $LLM_PROVIDER or {DEFAULT_PROVIDER}.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Chat model name. Defaults to $LLM_MODEL or {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"OpenAI-compatible base URL. Defaults to $LLM_BASE_URL or {DEFAULT_BASE_URL}.",
    )
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key.")
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=None,
        help="HTTP request timeout for OpenAI-compatible providers.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Maximum LLM/tool iterations per eval case.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help=(
            "Markdown file containing the system prompt. Defaults to "
            f"$SYSTEM_PROMPT_FILE or {DEFAULT_SYSTEM_PROMPT_FILE} if it exists."
        ),
    )
    parser.add_argument(
        "--emit-messages",
        action="store_true",
        help="Print live assistant/tool messages while running evals.",
    )
    add_common_output_args(parser, default_results_dir=Path("evals/tool_call/results"))
    return parser.parse_args()


def _llm_config_from_args(args: argparse.Namespace) -> LLMConfig:
    env_llm = LLMConfig.from_environment()
    return LLMConfig(
        provider=args.provider or env_llm.provider,
        model=args.model or env_llm.model,
        base_url=args.base_url or env_llm.base_url,
        api_key=args.api_key or env_llm.api_key or os.getenv("OPENAI_API_KEY"),
        request_timeout_seconds=(
            args.request_timeout_seconds or env_llm.request_timeout_seconds
        ),
    )


def main() -> None:
    args = parse_args()
    report = run_eval(
        dataset_path=args.dataset,
        llm_config=_llm_config_from_args(args),
        agent_config=AgentConfig(
            max_iterations=args.max_iterations,
            system_prompt=read_system_prompt(args.system_prompt_file),
        ),
        emit_messages=args.emit_messages,
        progress=not args.no_progress,
    )
    emit_report(
        report,
        output_config_from_args(args),
        output_prefix="tool_call",
        model_name=report.model,
        print_text_report=print_text_report,
        report_to_json=report_to_json,
        write_csv_report=write_csv_report,
    )


if __name__ == "__main__":
    main()
