.PHONY: install ruff mypy lint test ci

install:
	uv sync --dev

ruff:
	uv run ruff check .

mypy:
	uv run mypy .

lint: ruff mypy

test:
	uv run pytest

ci: lint test
