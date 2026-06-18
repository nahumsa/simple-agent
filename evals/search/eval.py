"""Evaluate markdown search quality against a labeled dataset.

Dataset format:
[
  {
    "id": "wc",
    "query": "word count command line tool",
    "expected_paths": ["001-challenge-wc.md"]
  }
]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

ROOT_DIR = Path(__file__).resolve().parents[2]
SEARCH_DATASETS_DIR = Path("evals/search/datasets")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evals.core.cli import (  # noqa: E402
    add_common_output_args,
    emit_report,
    output_config_from_args,
)
from evals.core.datasets import (  # noqa: E402
    latest_versioned_dataset,
    load_json_list,
)
from evals.core.output import (  # noqa: E402
    dataclass_report_to_json,
    default_csv_results_path as core_default_csv_results_path,
    write_csv_rows,
)
from evals.core.progress import ProgressReporter  # noqa: E402
from frameworks.barebones.tools import DuckDBFTSMarkdownSearch, SearchResult  # noqa: E402


@dataclass(frozen=True)
class SearchEvalCase:
    """One labeled search query."""

    id: str
    query: str
    expected_paths: list[str]


@dataclass(frozen=True)
class SearchEvalResult:
    """Evaluation result for one search query."""

    id: str
    query: str
    expected_paths: list[str]
    returned_paths: list[str]
    hit: bool
    reciprocal_rank: float
    recall: float


@dataclass(frozen=True)
class SearchEvalReport:
    """Aggregate search evaluation report."""

    dataset: str
    data_dir: str
    model: str
    limit: int
    case_count: int
    hit_rate: float
    mean_reciprocal_rank: float
    mean_recall: float
    results: list[SearchEvalResult]


def load_dataset(path: Path) -> list[SearchEvalCase]:
    """Load and validate a search eval dataset."""
    raw_cases = load_json_list(path)

    cases: list[SearchEvalCase] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Case {index} must be a JSON object.")

        case_id = raw_case.get("id") or f"case-{index}"
        query = raw_case.get("query")
        expected_paths = raw_case.get("expected_paths", raw_case.get("expected_path"))

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Case {index} has an invalid id.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Case {index} must include a non-empty query string.")
        if isinstance(expected_paths, str):
            expected_paths = [expected_paths]
        if not isinstance(expected_paths, list) or not expected_paths:
            raise ValueError(
                f"Case {index} must include expected_paths as a non-empty list."
            )
        if not all(isinstance(path, str) and path.strip() for path in expected_paths):
            raise ValueError(f"Case {index} expected_paths must contain strings only.")

        cases.append(
            SearchEvalCase(
                id=case_id,
                query=query,
                expected_paths=expected_paths,
            )
        )

    return cases


def evaluate_case(
    case: SearchEvalCase,
    results: list[SearchResult],
) -> SearchEvalResult:
    """Score one query result list."""
    returned_paths = [result.path for result in results]
    expected = set(case.expected_paths)
    matched_positions = [
        position
        for position, returned_path in enumerate(returned_paths, start=1)
        if returned_path in expected
    ]

    hit = bool(matched_positions)
    reciprocal_rank = 1 / matched_positions[0] if matched_positions else 0.0
    recall = len(expected.intersection(returned_paths)) / len(expected)

    return SearchEvalResult(
        id=case.id,
        query=case.query,
        expected_paths=case.expected_paths,
        returned_paths=returned_paths,
        hit=hit,
        reciprocal_rank=reciprocal_rank,
        recall=recall,
    )


def run_eval(
    *,
    dataset_path: Path,
    data_dir: Path,
    db_path: Path,
    limit: int,
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> SearchEvalReport:
    """Run all search eval cases and compute aggregate metrics."""
    cases = load_dataset(dataset_path)
    search = DuckDBFTSMarkdownSearch(data_dir=data_dir, db_path=db_path)
    results: list[SearchEvalResult] = []
    total_cases = len(cases)
    progress_reporter = ProgressReporter(enabled=progress, stream=progress_stream)

    for index, case in enumerate(cases, start=1):
        progress_reporter.case_started(index, total_cases, f"{case.id}: {case.query}")
        result = evaluate_case(case, search.search(case.query, limit=limit))
        results.append(result)
        status = "PASS" if result.hit else "FAIL"
        progress_reporter.case_finished(index, total_cases, status, case.id)

    case_count = len(results)
    hit_rate = sum(result.hit for result in results) / case_count if case_count else 0.0
    mean_reciprocal_rank = (
        sum(result.reciprocal_rank for result in results) / case_count
        if case_count
        else 0.0
    )
    mean_recall = (
        sum(result.recall for result in results) / case_count if case_count else 0.0
    )

    return SearchEvalReport(
        dataset=str(dataset_path),
        data_dir=str(data_dir),
        model="duckdb-fts",
        limit=limit,
        case_count=case_count,
        hit_rate=hit_rate,
        mean_reciprocal_rank=mean_reciprocal_rank,
        mean_recall=mean_recall,
        results=results,
    )


def print_text_report(report: SearchEvalReport) -> None:
    """Print a concise human-readable report."""
    print(f"Dataset: {report.dataset}")
    print(f"Data dir: {report.data_dir}")
    print(f"Model: {report.model}")
    print(f"Limit: {report.limit}")
    print(f"Cases: {report.case_count}")
    print(f"Hit rate@{report.limit}: {report.hit_rate:.3f}")
    print(f"MRR@{report.limit}: {report.mean_reciprocal_rank:.3f}")
    print(f"Mean recall@{report.limit}: {report.mean_recall:.3f}")
    print()

    for result in report.results:
        status = "PASS" if result.hit else "FAIL"
        print(f"[{status}] {result.id}: {result.query}")
        print(f"  expected: {', '.join(result.expected_paths)}")
        print(f"  returned: {', '.join(result.returned_paths) or '(none)'}")
        print(f"  rr={result.reciprocal_rank:.3f} recall={result.recall:.3f}")


def report_to_json(report: SearchEvalReport) -> dict[str, Any]:
    """Convert report dataclasses to JSON-serializable dictionaries."""
    return dataclass_report_to_json(report)


def report_to_summary_json(report: SearchEvalReport) -> dict[str, Any]:
    """Build the compact JSON summary saved for every search eval run."""
    hits = sum(result.hit for result in report.results)
    return {
        "eval": "search",
        "dataset": report.dataset,
        "data_dir": report.data_dir,
        "model": report.model,
        "limit": report.limit,
        "case_count": report.case_count,
        "passed": hits,
        "failed": report.case_count - hits,
        "hit_rate": report.hit_rate,
        "mean_reciprocal_rank": report.mean_reciprocal_rank,
        "mean_recall": report.mean_recall,
    }


def write_csv_report(report: SearchEvalReport, path: Path | None = None) -> None:
    """Write one CSV row per eval case."""
    fieldnames = [
        "id",
        "query",
        "expected_paths",
        "returned_paths",
        "hit",
        "reciprocal_rank",
        "recall",
        "limit",
        "dataset",
        "data_dir",
        "model",
    ]
    rows = (
        {
            "id": result.id,
            "query": result.query,
            "expected_paths": ";".join(result.expected_paths),
            "returned_paths": ";".join(result.returned_paths),
            "hit": result.hit,
            "reciprocal_rank": f"{result.reciprocal_rank:.6f}",
            "recall": f"{result.recall:.6f}",
            "limit": report.limit,
            "dataset": report.dataset,
            "data_dir": report.data_dir,
            "model": report.model,
        }
        for result in report.results
    )
    write_csv_rows(fieldnames=fieldnames, rows=rows, path=path)


def default_csv_results_path(report: SearchEvalReport, results_dir: Path) -> Path:
    """Build a timestamped CSV path for an eval report."""
    return core_default_csv_results_path(
        prefix="search",
        model=report.model,
        results_dir=results_dir,
    )



def latest_search_dataset() -> Path:
    """Return the newest versioned synthetic search dataset path."""
    return latest_versioned_dataset(
        datasets_dir=SEARCH_DATASETS_DIR,
        versioned_prefix="search_synthetic_v",
        fallback_name="search_synthetic_v1.json",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate challenge markdown search.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=latest_search_dataset(),
        help=(
            "JSON dataset of search queries and expected markdown paths. "
            "Defaults to the newest evals/search/datasets/search_synthetic_v*.json file."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/extracted_data"),
        help="Directory containing extracted markdown files.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/challenge_search_eval.duckdb"),
        help="DuckDB path used for the eval search index.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to score per query.",
    )
    add_common_output_args(parser, default_results_dir=Path("evals/search/results"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_eval(
        dataset_path=args.dataset,
        data_dir=args.data_dir,
        db_path=args.db_path,
        limit=max(1, args.limit),
        progress=not args.no_progress,
    )
    emit_report(
        report,
        output_config_from_args(args),
        output_prefix="search",
        model_name=report.model,
        print_text_report=print_text_report,
        report_to_json=report_to_json,
        write_csv_report=write_csv_report,
        summary_to_json=report_to_summary_json,
    )


if __name__ == "__main__":
    main()
