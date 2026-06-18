import csv
import re
from pathlib import Path

from evals.core.cli import CommonOutputConfig, emit_report
from evals.core.datasets import latest_versioned_dataset, load_json_list
from evals.core.output import (
    json_summary_path_for_csv_results,
    safe_filename_part,
    write_csv_rows,
)
from evals.search.eval import (
    SearchEvalReport,
    SearchEvalResult,
    load_dataset as load_search_dataset,
    report_to_summary_json as search_report_to_summary_json,
    write_csv_report as write_search_csv_report,
)
from evals.tool_call.eval import load_dataset as load_tool_call_dataset


def test_latest_versioned_dataset_returns_highest_version(tmp_path) -> None:
    (tmp_path / "sample_v1.json").write_text("[]", encoding="utf-8")
    (tmp_path / "sample_v10.json").write_text("[]", encoding="utf-8")
    (tmp_path / "sample_v2.json").write_text("[]", encoding="utf-8")
    (tmp_path / "sample_vx.json").write_text("[]", encoding="utf-8")

    latest = latest_versioned_dataset(
        datasets_dir=tmp_path,
        versioned_prefix="sample_v",
        fallback_name="sample.json",
    )

    assert latest == tmp_path / "sample_v10.json"


def test_latest_versioned_dataset_falls_back_when_no_versioned_files(tmp_path) -> None:
    latest = latest_versioned_dataset(
        datasets_dir=tmp_path,
        versioned_prefix="sample_v",
        fallback_name="sample.json",
    )

    assert latest == tmp_path / "sample.json"


def test_load_json_list_rejects_non_list_dataset(tmp_path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"id": "not-a-list"}', encoding="utf-8")

    try:
        load_json_list(dataset)
    except ValueError as exc:
        assert str(exc) == "Dataset must be a JSON list of cases."
    else:
        raise AssertionError("Expected ValueError")


def test_safe_filename_part_matches_existing_eval_behavior() -> None:
    assert safe_filename_part(" openai/gpt-oss:120b free ") == "openai-gpt-oss-120b-free"
    assert safe_filename_part("///") == "unknown"


def test_write_csv_rows_writes_headers_and_rows(tmp_path) -> None:
    output_path = tmp_path / "results.csv"

    write_csv_rows(
        fieldnames=["id", "items"],
        rows=[{"id": "case-1", "items": "a;b"}],
        path=output_path,
    )

    with output_path.open(newline="", encoding="utf-8") as output:
        rows = list(csv.DictReader(output))

    assert rows == [{"id": "case-1", "items": "a;b"}]


def test_json_summary_path_sits_next_to_csv_result() -> None:
    path = json_summary_path_for_csv_results(Path("results/search_123_model.csv"))

    assert path == Path("results/search_123_model_summary.json")


def test_search_dataset_accepts_single_expected_path(tmp_path) -> None:
    dataset = tmp_path / "search.json"
    dataset.write_text(
        '[{"id": "wc", "query": "word count", "expected_path": "001.md"}]',
        encoding="utf-8",
    )

    cases = load_search_dataset(dataset)

    assert len(cases) == 1
    assert cases[0].expected_paths == ["001.md"]


def test_tool_call_dataset_accepts_string_shorthand_fields(tmp_path) -> None:
    dataset = tmp_path / "tool_call.json"
    dataset.write_text(
        "["
        '{"id": "case", '
        '"user_prompt": "Find JSON", '
        '"expected_tool_calls": "search_challenges", '
        '"expected_final_contains": "JSON"}'
        "]",
        encoding="utf-8",
    )

    cases = load_tool_call_dataset(dataset)

    assert len(cases) == 1
    assert cases[0].expected_tool_calls == ["search_challenges"]
    assert cases[0].expected_final_contains == ["JSON"]


def test_default_search_result_filename_shape() -> None:
    from evals.search.eval import default_csv_results_path

    report = SearchEvalReport(
        dataset="dataset.json",
        data_dir="data",
        model="duckdb/fts",
        limit=5,
        case_count=0,
        hit_rate=0.0,
        mean_reciprocal_rank=0.0,
        mean_recall=0.0,
        results=[],
    )

    path = default_csv_results_path(report, results_dir=Path("results"))

    assert re.fullmatch(
        r"results/search_\d{8}T\d{6}Z_duckdb-fts\.csv",
        path.as_posix(),
    )


def test_search_report_to_summary_json_is_compact() -> None:
    report = SearchEvalReport(
        dataset="dataset.json",
        data_dir="data",
        model="duckdb-fts",
        limit=5,
        case_count=2,
        hit_rate=0.5,
        mean_reciprocal_rank=0.5,
        mean_recall=0.25,
        results=[
            SearchEvalResult(
                id="hit",
                query="wc",
                expected_paths=["001.md"],
                returned_paths=["001.md"],
                hit=True,
                reciprocal_rank=1.0,
                recall=1.0,
            ),
            SearchEvalResult(
                id="miss",
                query="json",
                expected_paths=["002.md"],
                returned_paths=[],
                hit=False,
                reciprocal_rank=0.0,
                recall=0.0,
            ),
        ],
    )

    assert search_report_to_summary_json(report) == {
        "eval": "search",
        "dataset": "dataset.json",
        "data_dir": "data",
        "model": "duckdb-fts",
        "limit": 5,
        "case_count": 2,
        "passed": 1,
        "failed": 1,
        "hit_rate": 0.5,
        "mean_reciprocal_rank": 0.5,
        "mean_recall": 0.25,
    }


def test_emit_report_saves_default_compact_json_summary(tmp_path, capsys) -> None:
    report = SearchEvalReport(
        dataset="dataset.json",
        data_dir="data",
        model="duckdb-fts",
        limit=5,
        case_count=0,
        hit_rate=0.0,
        mean_reciprocal_rank=0.0,
        mean_recall=0.0,
        results=[],
    )
    config = CommonOutputConfig(
        json=False,
        csv=False,
        csv_output=None,
        results_dir=tmp_path,
        no_save_results=False,
        no_progress=False,
    )

    saved_csv = emit_report(
        report,
        config,
        output_prefix="search",
        model_name=report.model,
        print_text_report=lambda _: None,
        report_to_json=lambda _: {},
        write_csv_report=write_search_csv_report,
        summary_to_json=search_report_to_summary_json,
    )

    assert saved_csv is not None
    summary_path = json_summary_path_for_csv_results(saved_csv)
    assert saved_csv.exists()
    assert summary_path.exists()
    assert '"eval": "search"' in summary_path.read_text(encoding="utf-8")
    assert "Saved JSON summary:" in capsys.readouterr().out
