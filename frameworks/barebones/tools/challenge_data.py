"""Tool-facing adapter for Coding Challenges markdown data."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from agent_core.types import JsonObject, ToolSpec
import frameworks.barebones.tools.url_fetch as url_fetch
from frameworks.barebones.tools.challenge_repository import ChallengeRepository, MarkdownSearch
from frameworks.barebones.tools.markdown_search import MarkdownSearchError
from frameworks.barebones.tools.tool_registry import ToolRegistry


class ReadMarkdownFileTool:
    """Command that validates and reads an extracted markdown file."""

    name = "read_file"

    def __init__(self, repository: ChallengeRepository) -> None:
        self.repository = repository

    def spec(self) -> ToolSpec:
        return {
            "type": "function",
            "function": {
                "name": self.name,
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
        }

    async def execute(
        self,
        args: JsonObject,
        *,
        session: object,
        tool_call_id: str,
    ) -> tuple[str, bool]:
        requested_path = args.get("path")
        if not isinstance(requested_path, str) or not requested_path.strip():
            return "Missing required string argument: path", False
        return self.repository.read_markdown(requested_path)


class SearchChallengesTool:
    """Command that searches extracted challenge markdown files."""

    name = "search_challenges"

    def __init__(self, repository: ChallengeRepository) -> None:
        self.repository = repository

    def spec(self) -> ToolSpec:
        return {
            "type": "function",
            "function": {
                "name": self.name,
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
        }

    async def execute(
        self,
        args: JsonObject,
        *,
        session: object,
        tool_call_id: str,
    ) -> tuple[str, bool]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return "Missing required string argument: query", False

        limit_arg = args.get("limit", 5)
        if not isinstance(limit_arg, int):
            return "limit must be an integer", False

        limit = max(1, min(limit_arg, 20))
        try:
            results = self.repository.search(query, limit=limit)
        except MarkdownSearchError as exc:
            return str(exc), False

        return json.dumps(
            {"query": query, "results": [asdict(result) for result in results]},
            indent=2,
        ), True


class FetchURLTool:
    """Command that fetches public HTTP(S) URLs."""

    name = "fetch_url"

    def spec(self) -> ToolSpec:
        return url_fetch.FETCH_URL_SPEC

    async def execute(
        self,
        args: JsonObject,
        *,
        session: object,
        tool_call_id: str,
    ) -> tuple[str, bool]:
        return url_fetch.call_fetch_url(args)


class ChallengeDataTools:
    """Tools for searching and reading extracted Coding Challenges markdown files."""

    def __init__(
        self,
        data_dir: Path | str = "data/extracted_data",
        search_db_path: Path | str = "data/challenge_search.duckdb",
    ) -> None:
        self.repository = ChallengeRepository(data_dir, search_db_path)
        self.registry = ToolRegistry(
            [
                ReadMarkdownFileTool(self.repository),
                SearchChallengesTool(self.repository),
                FetchURLTool(),
            ]
        )

    @property
    def data_dir(self) -> Path:
        return self.repository.data_dir

    @property
    def index_file(self) -> Path:
        return self.repository.index_file

    @property
    def search_index(self) -> MarkdownSearch:
        return self.repository.search_index

    @search_index.setter
    def search_index(self, search_index: MarkdownSearch) -> None:
        self.repository.search_index = search_index

    def initial_context(self) -> str | None:
        """Return the challenge index so it can be added to the agent context."""
        return self.repository.read_index_for_context()

    def specs(self) -> list[ToolSpec]:
        return self.registry.specs()

    async def call(
        self,
        name: str,
        args: JsonObject,
        *,
        session: object,
        tool_call_id: str,
    ) -> tuple[str, bool]:
        return await self.registry.call(
            name,
            args,
            session=session,
            tool_call_id=tool_call_id,
        )

    async def cancel_running(self) -> None:
        return None

    def _read_relative_markdown(self, requested_path: str) -> tuple[str, bool]:
        return self.repository.read_markdown(requested_path)

    @staticmethod
    def _read_file(path: Path) -> tuple[str, bool]:
        return ChallengeRepository._read_file(path)
