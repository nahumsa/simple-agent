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
import csv
import json
import re
import sys
from datetime import UTC, datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise ValueError("Dataset must be a JSON list of cases.")

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
) -> SearchEvalReport:
    """Run all search eval cases and compute aggregate metrics."""
    cases = load_dataset(dataset_path)
    search = DuckDBFTSMarkdownSearch(data_dir=data_dir, db_path=db_path)
    results = [
        evaluate_case(case, search.search(case.query, limit=limit)) for case in cases
    ]

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
    return asdict(report)


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
    output = path.open("w", newline="", encoding="utf-8") if path else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in report.results:
            writer.writerow(
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
            )
    finally:
        if path:
            output.close()


def default_csv_results_path(report: SearchEvalReport, results_dir: Path) -> Path:
    """Build a timestamped CSV path for an eval report."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model = _safe_filename_part(report.model)
    return results_dir / f"search_{timestamp}_{model}.csv"


def _safe_filename_part(value: str) -> str:
    """Return a filesystem-friendly filename segment."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return cleaned or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate challenge markdown search.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/search/datasets/search_smoke.json"),
        help="JSON dataset of search queries and expected markdown paths.",
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
        help="Write CSV output to this file instead of the default results path.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("evals/search/results"),
        help="Directory for timestamped CSV result files.",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Do not save a timestamped CSV result file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(
        dataset_path=args.dataset,
        data_dir=args.data_dir,
        db_path=args.db_path,
        limit=max(1, args.limit),
    )
    saved_results_path = None
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        write_csv_report(report, args.csv_output)
        saved_results_path = args.csv_output
    elif not args.no_save_results:
        saved_results_path = default_csv_results_path(report, args.results_dir)
        saved_results_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv_report(report, saved_results_path)

    if args.csv:
        write_csv_report(report)
    elif args.json:
        print(json.dumps(report_to_json(report), indent=2))
    else:
        print_text_report(report)
        if saved_results_path:
            print(f"\nSaved CSV results: {saved_results_path}")


if __name__ == "__main__":
    main()
