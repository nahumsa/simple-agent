.PHONY: install ruff mypy lint split-challenge-data test evals eval-search eval-search-smoke eval-tool-call eval-tool-call-help ci

install:
	uv sync --dev

ruff:
	uv run ruff check .

mypy:
	uv run mypy .

lint: ruff mypy

split-challenge-data:
	uv run python scripts/split_challenge_data.py

test:
	uv run pytest

evals: eval-search eval-tool-call

eval-search:
	uv run python evals/search/eval.py

eval-search-smoke:
	uv run python evals/search/eval.py --dataset evals/search/datasets/search_smoke.json --no-save-results --no-progress

eval-tool-call:
	uv run python evals/tool_call/eval.py

eval-tool-call-help:
	uv run python evals/tool_call/eval.py --help

ci: lint test eval-search-smoke
