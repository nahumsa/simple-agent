"""Tool-facing adapter for Coding Challenges markdown data."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from agent_core.types import JsonObject, ToolSpec
from frameworks.barebones.tools.markdown_search import (
    DuckDBFTSMarkdownSearch,
    MarkdownSearchError,
)
from frameworks.barebones.tools.url_fetch import FETCH_URL_SPEC, call_fetch_url


class ChallengeDataTools:
    """Tools for browsing and full-text searching extracted Coding Challenges markdown files."""

    def __init__(
        self,
        data_dir: Path | str = "data/extracted_data",
        search_db_path: Path | str = "data/challenge_search.duckdb",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.index_file = self.data_dir / "index.md"
        self.search_index = DuckDBFTSMarkdownSearch(
            self.data_dir,
            Path(search_db_path),
        )

    def initial_context(self) -> str | None:
        """Return the challenge index so it can be added to the agent context."""
        try:
            index = self.index_file.read_text(encoding="utf-8").strip()

        except FileNotFoundError:
            return None

        return (
            "The following Coding Challenges index is available. "
            "Use the read_file tool to inspect any referenced markdown file in detail.\n\n"
            f"{index}"
        )

    def specs(self) -> list[ToolSpec]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_challenge_index",
                    "description": "Read the Coding Challenges markdown index.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Read a markdown file from data/extracted_data. "
                        "Use paths from the challenge index, for example "
                        "001-challenge-wc.md."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative markdown path to read from data/extracted_data.",
                            }
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_challenges",
                    "description": (
                        "Full-text search the Coding Challenges markdown files with DuckDB FTS. "
                        "Use this before reading individual files when the relevant challenge is unknown."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query, for example: parsing command line flags.",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of matches to return. Defaults to 5.",
                                "minimum": 1,
                                "maximum": 20,
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            FETCH_URL_SPEC,
        ]

    async def call(
        self,
        name: str,
        args: JsonObject,
        *,
        session: object,
        tool_call_id: str,
    ) -> tuple[str, bool]:
        if name == "read_challenge_index":
            return self._read_file(self.index_file)

        if name == "read_file":
            requested_path = args.get("path")
            if not isinstance(requested_path, str) or not requested_path.strip():
                return "Missing required string argument: path", False
            return self._read_relative_markdown(requested_path)

        if name == "search_challenges":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return "Missing required string argument: query", False
            limit_arg = args.get("limit", 5)
            if not isinstance(limit_arg, int):
                return "limit must be an integer", False
            limit = max(1, min(limit_arg, 20))
            try:
                results = self.search_index.search(query, limit=limit)
            except MarkdownSearchError as exc:
                return str(exc), False
            return json.dumps(
                {"query": query, "results": [asdict(result) for result in results]},
                indent=2,
            ), True

        if name == "fetch_url":
            return call_fetch_url(args)

        return f"Unknown tool: {name}", False

    async def cancel_running(self) -> None:
        return None

    def _read_relative_markdown(self, requested_path: str) -> tuple[str, bool]:
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

    @staticmethod
    def _read_file(path: Path) -> tuple[str, bool]:
        try:
            return path.read_text(encoding="utf-8"), True
        except FileNotFoundError:
            return f"File not found: {path}", False
        except OSError as exc:
            return f"Could not read {path}: {exc}", False
