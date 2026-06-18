"""Shared CLI helpers for eval scripts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from evals.core.output import (
    default_csv_results_path,
    json_summary_path_for_csv_results,
    write_json_summary,
)

ReportT = TypeVar("ReportT")


@dataclass(frozen=True)
class CommonOutputConfig:
    """Common output and result-saving options for eval scripts."""

    json: bool
    csv: bool
    csv_output: Path | None
    results_dir: Path
    no_save_results: bool
    no_progress: bool


def add_common_output_args(
    parser: argparse.ArgumentParser, *, default_results_dir: Path
) -> None:
    """Add output/result CLI arguments shared by eval scripts."""
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON.",
    )
    output_group.add_argument(
        "--csv",
        action="store_true",
        help="Print one CSV row per eval case.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help=(
            "Write CSV output to this file instead of the default results path. "
            "A compact JSON summary is written next to it."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        help="Directory for timestamped CSV and compact JSON summary result files.",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Do not save timestamped CSV or compact JSON summary result files.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Do not print per-case progress to stderr.",
    )


def output_config_from_args(args: argparse.Namespace) -> CommonOutputConfig:
    """Build common output config from parsed argparse args."""
    return CommonOutputConfig(
        json=args.json,
        csv=args.csv,
        csv_output=args.csv_output,
        results_dir=args.results_dir,
        no_save_results=args.no_save_results,
        no_progress=args.no_progress,
    )


def emit_report(
    report: ReportT,
    config: CommonOutputConfig,
    *,
    output_prefix: str,
    model_name: str,
    print_text_report: Callable[[ReportT], None],
    report_to_json: Callable[[ReportT], dict[str, Any]],
    write_csv_report: Callable[[ReportT, Path | None], None],
    summary_to_json: Callable[[ReportT], dict[str, Any]],
) -> Path | None:
    """Save optional CSV and compact JSON summary results, then print output."""
    saved_results_path = None
    saved_summary_path = None
    if config.csv_output:
        config.csv_output.parent.mkdir(parents=True, exist_ok=True)
        write_csv_report(report, config.csv_output)
        saved_results_path = config.csv_output
    elif not config.no_save_results:
        saved_results_path = default_csv_results_path(
            prefix=output_prefix,
            model=model_name,
            results_dir=config.results_dir,
        )
        saved_results_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv_report(report, saved_results_path)

    if saved_results_path:
        saved_summary_path = json_summary_path_for_csv_results(saved_results_path)
        write_json_summary(summary_to_json(report), saved_summary_path)

    if config.csv:
        write_csv_report(report, None)
    elif config.json:
        print(json.dumps(report_to_json(report), indent=2))
    else:
        print_text_report(report)
        if saved_results_path:
            print(f"\nSaved CSV results: {saved_results_path}")
        if saved_summary_path:
            print(f"Saved JSON summary: {saved_summary_path}")

    return saved_results_path
