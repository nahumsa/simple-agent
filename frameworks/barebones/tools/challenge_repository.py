"""Repository for Coding Challenges markdown data."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from frameworks.barebones.tools.markdown_search import (
    DuckDBFTSMarkdownSearch,
    SearchResult,
)


class MarkdownSearch(Protocol):
    """Search backend used by the challenge repository."""

    def search(self, query: str, *, limit: int) -> list[SearchResult]: ...


class ChallengeRepository:
    """Filesystem and search access for extracted Coding Challenges markdown files."""

    def __init__(
        self,
        data_dir: Path | str = "data/extracted_data",
        search_db_path: Path | str = "data/challenge_search.duckdb",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.index_file = self.data_dir / "index.md"
        self.search_index: MarkdownSearch = DuckDBFTSMarkdownSearch(
            self.data_dir,
            Path(search_db_path),
        )

    def read_index_for_context(self) -> str | None:
        """Return the challenge index as system-prompt context, if available."""
        try:
            index = self.index_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None

        return (
            "The following Coding Challenges index is available. "
            "Use the read_file tool to inspect any referenced markdown file in detail.\n\n"
            f"{index}"
        )

    def read_index(self) -> tuple[str, bool]:
        """Read the challenge index markdown file."""
        return self._read_file(self.index_file)

    def read_markdown(self, requested_path: str) -> tuple[str, bool]:
        """Safely read a markdown file relative to the challenge data directory."""
        relative_path = Path(requested_path)
        if relative_path.is_absolute():
            return "Path must be relative to data/extracted_data", False
        if relative_path.suffix != ".md":
            return "Only markdown files can be read", False

        path = self.data_dir / relative_path
        try:
            data_root = self.data_dir.resolve(strict=True)
            resolved_path = path.resolve(strict=True)
        except FileNotFoundError:
            return f"File not found: {requested_path}", False

        if data_root not in resolved_path.parents and resolved_path != data_root:
            return "Path escapes data/extracted_data", False

        return self._read_file(resolved_path)

    def search(self, query: str, *, limit: int) -> list[SearchResult]:
        """Search challenge markdown files."""
        return self.search_index.search(query, limit=limit)

    @staticmethod
    def _read_file(path: Path) -> tuple[str, bool]:
        try:
            return path.read_text(encoding="utf-8"), True
        except FileNotFoundError:
            return f"File not found: {path}", False
        except OSError as exc:
            return f"Could not read {path}: {exc}", False
