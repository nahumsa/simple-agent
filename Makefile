.PHONY: install ruff mypy lint split-challenge-data test ci

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

ci: lint test
