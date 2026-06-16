.PHONY: install ruff mypy lint split-challenge-data test eval-search eval-tool-call ci

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

eval-search:
	uv run python evals/search/eval.py

eval-tool-call:
	uv run python evals/tool_call/eval.py

ci: lint test
