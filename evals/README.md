# Evals

Each eval type has its own folder:

- `evals/search/` evaluates markdown search quality.
- `evals/tool_call/` evaluates built-in tool execution and validation.

Both evals support text, JSON, and CSV output. By default, each eval run prints per-case progress to stderr and saves a timestamped CSV result file under that eval's `results/` folder. The filename includes the eval name, run datetime, and model/search backend name.

---

## Search eval

Search evals measure whether challenge markdown search returns the expected files for labeled queries.

Run the included smoke dataset:

```bash
uv run python evals/search/eval.py
```

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

Write CSV output to a specific file:

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

---

## Tool-call eval

Tool-call evals run the real agent with an LLM and check whether the agent chooses the expected tools for each user prompt. These evals require your configured LLM provider to be running/reachable.

Run the latest included tool-call dataset (`evals/tool_call/datasets/tool_call_v2.json`):

```bash
uv run python evals/tool_call/eval.py
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

Write CSV output to a specific file:

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
