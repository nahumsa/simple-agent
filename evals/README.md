# Evals

Each eval type has its own folder:

- `evals/search/` evaluates markdown search quality.
- `evals/tool_call/` evaluates built-in tool execution and validation.

Both evals support text, JSON, and CSV output. By default, each eval run prints per-case progress to stderr and saves a timestamped CSV result file plus a compact JSON summary file under that eval's `results/` folder. Filenames include the eval name, run datetime, and model/search backend name; summary files add `_summary.json`.

Shared mechanics live under `evals/core/` so new evals do not need to copy/paste dataset loading, latest-version discovery, progress reporting, CSV writing, result-path generation, or common output CLI flags.

---

## Search eval

Search evals measure whether challenge markdown search returns the expected files for labeled queries.

Run the latest included synthetic regression dataset (`evals/search/datasets/search_synthetic_v1.json`):

```bash
uv run python evals/search/eval.py
# or
uv run python -m evals.run search
```

This dataset expands the original smoke coverage with realistic failure-oriented queries across:

- wording style: exact names, capability descriptions, colloquial/noisy requests;
- retrieval risk: unique targets, confusable neighbors, multi-result families;
- topic clusters: CLI/text tools, networking/protocols, data formats, web/apps, AI/devtools, and support docs.

When more `search_synthetic_v*.json` datasets are added, the eval automatically uses the highest version number by default.

Run a custom dataset:

```bash
uv run python evals/search/eval.py --dataset path/to/search_dataset.json --limit 10
```

Print machine-readable output:

```bash
uv run python evals/search/eval.py --json
```

Print CSV output:

```bash
uv run python evals/search/eval.py --csv
```

Write CSV output to a specific file. A compact JSON summary is also written next to it with `_summary.json` appended to the CSV stem:

```bash
uv run python evals/search/eval.py --csv-output evals/search/results/search_smoke.csv
```

Disable progress output:

```bash
uv run python evals/search/eval.py --no-progress
```

Default saved result filenames look like:

```text
evals/search/results/search_20260615T143012Z_duckdb-fts.csv
evals/search/results/search_20260615T143012Z_duckdb-fts_summary.json
```

Search dataset format:

```json
[
  {
    "id": "wc",
    "query": "word count command line tool",
    "expected_paths": ["001-challenge-wc.md"]
  }
]
```

You can also use `expected_path` for a single expected file.

Reported metrics:

- `hit_rate`: fraction of queries with at least one expected file in the top results.
- `mean_reciprocal_rank`: rewards expected files appearing higher in the result list.
- `mean_recall`: fraction of expected files returned per query.

The compact JSON summary includes: `eval`, `dataset`, `data_dir`, `model`, `limit`, `case_count`, `passed`, `failed`, `hit_rate`, `mean_reciprocal_rank`, and `mean_recall`.

---

## Tool-call eval

Tool-call evals run the real agent with an LLM and check whether the agent chooses the expected tools for each user prompt. These evals require your configured LLM provider to be running/reachable.

Run the latest included tool-call dataset (`evals/tool_call/datasets/tool_call_v2.json`):

```bash
uv run python evals/tool_call/eval.py
# or
uv run python -m evals.run tool-call
```

The original v1 dataset is kept at `evals/tool_call/datasets/tool_call.json` for comparison.

Run a custom dataset:

```bash
uv run python evals/tool_call/eval.py --dataset path/to/tool_call_dataset.json
```

Print CSV output:

```bash
uv run python evals/tool_call/eval.py --csv
```

Write CSV output to a specific file. A compact JSON summary is also written next to it with `_summary.json` appended to the CSV stem:

```bash
uv run python evals/tool_call/eval.py --csv-output evals/tool_call/results/tool_call_smoke.csv
```

Disable progress output:

```bash
uv run python evals/tool_call/eval.py --no-progress
```

Default saved result filenames look like:

```text
evals/tool_call/results/tool_call_20260615T143012Z_gemma4-latest.csv
evals/tool_call/results/tool_call_20260615T143012Z_gemma4-latest_summary.json
```

Tool-call dataset format:

```json
[
  {
    "id": "search-json-parser",
    "user_prompt": "Find the JSON parser challenge. Use tools before answering.",
    "expected_tool_calls": ["search_challenges"],
    "expected_final_contains": ["JSON"]
  }
]
```

Reported metric:

- `pass_rate`: fraction of agent cases where the expected tool calls appear and the final answer contains the expected text.

The compact JSON summary includes: `eval`, `dataset`, `provider`, `model`, `case_count`, `passed`, `failed`, and `pass_rate`.

---

## Adding a new eval

Create a folder like:

```text
evals/my_eval/
  eval.py
  datasets/my_eval_v1.json
  results/
```

In `eval.py`, keep only eval-specific pieces local:

- case/result/report dataclasses;
- dataset field validation using helpers from `evals.core.datasets`;
- case execution and scoring;
- aggregate metric calculation;
- text report formatting and CSV row shape.

Use the shared helpers for common behavior:

- `evals.core.datasets.load_json_list()`
- `evals.core.datasets.latest_versioned_dataset()`
- `evals.core.progress.ProgressReporter`
- `evals.core.output.write_csv_rows()`
- `evals.core.cli.add_common_output_args()`
- `evals.core.cli.emit_report()`

Keep current per-eval entrypoints for backwards compatibility, and prefer extracting another core helper only after the same pattern appears in multiple evals.
