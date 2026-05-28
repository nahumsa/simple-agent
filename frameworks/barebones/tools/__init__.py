"""Tool registries for the simple agent."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from agent_core.types import JsonObject, ToolSpec


class NoTools:
    """Empty tool registry for plain chat mode."""

    def specs(self) -> list[ToolSpec]:
        return []

    async def call(
        self,
        name: str,
        args: JsonObject,
        *,
        session: object,
        tool_call_id: str,
    ) -> tuple[str, bool]:
        return f"Unknown tool: {name}", False

    async def cancel_running(self) -> None:
        return None


class DuckDBFTSMarkdownSearch:
    """Persistent DuckDB full-text search index for markdown files."""

    def __init__(
        self,
        data_dir: Path,
        db_path: Path,
        *,
        rebuild_interval: timedelta = timedelta(days=1),
    ) -> None:
        self.data_dir = data_dir
        self.db_path = db_path
        self.rebuild_interval = rebuild_interval

    def search(self, query: str, *, limit: int) -> tuple[str, bool]:
        """Search markdown documents, rebuilding stale FTS indexes when file count changes."""
        try:
            import duckdb
        except ImportError:
            return "DuckDB is not installed. Run `uv sync` to install project dependencies.", False

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(str(self.db_path)) as conn:
                self._install_and_load_fts(conn)
                documents = self._markdown_documents()
                self._refresh_index_if_needed(conn, documents)
                rows = conn.execute(
                    """
                    select
                        path,
                        title,
                        fts_main_documents.match_bm25(id, ?) as score,
                        left(regexp_replace(content, '\\s+', ' ', 'g'), 500) as snippet
                    from documents
                    where score is not null
                    order by score desc
                    limit ?
                    """,
                    [query, limit],
                ).fetchall()
        except Exception as exc:  # DuckDB extension errors are surfaced as different classes.
            return f"DuckDB FTS search failed: {exc}", False

        results = [
            {
                "path": path,
                "title": title,
                "score": score,
                "snippet": snippet,
            }
            for path, title, score, snippet in rows
        ]
        return json.dumps({"query": query, "results": results}, indent=2), True

    @staticmethod
    def _install_and_load_fts(conn: Any) -> None:
        conn.execute("INSTALL fts")
        conn.execute("LOAD fts")

    def _markdown_documents(self) -> list[tuple[int, str, str, str]]:
        documents = []
        markdown_paths = [path for path in sorted(self.data_dir.glob("*.md")) if path.name != "index.md"]
        for index, path in enumerate(markdown_paths, start=1):
            content = path.read_text(encoding="utf-8")
            title = self._extract_title(content) or path.stem
            documents.append((index, path.name, title, content))
        return documents

    @staticmethod
    def _extract_title(content: str) -> str | None:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or None
        return None

    def _refresh_index_if_needed(
        self,
        conn: Any,
        documents: list[tuple[int, str, str, str]],
    ) -> None:
        conn.execute("create table if not exists search_metadata (key varchar primary key, value varchar)")
        stored_count = self._metadata_value(conn, "document_count")
        current_count = len(documents)
        documents_table_exists = self._table_exists(conn, "documents")
        if documents_table_exists and stored_count == str(current_count):
            return
        if documents_table_exists and not self._rebuild_interval_elapsed(conn):
            return

        now = datetime.now(UTC).isoformat()
        conn.execute("drop table if exists documents")
        conn.execute(
            """
            create table documents (
                id integer primary key,
                path varchar not null,
                title varchar not null,
                content varchar not null
            )
            """
        )
        conn.executemany(
            "insert into documents (id, path, title, content) values (?, ?, ?, ?)",
            documents,
        )
        conn.execute(
            """
            pragma create_fts_index(
                'documents',
                'id',
                'title',
                'content',
                stemmer = 'porter',
                stopwords = 'english',
                ignore = '(\\.|[^a-z])+',
                strip_accents = 1,
                lower = 1,
                overwrite = 1
            )
            """
        )
        conn.executemany(
            "insert or replace into search_metadata (key, value) values (?, ?)",
            [("document_count", str(current_count)), ("last_built_at", now)],
        )

    @staticmethod
    def _metadata_value(conn: Any, key: str) -> str | None:
        row = conn.execute("select value from search_metadata where key = ?", [key]).fetchone()
        if row is None:
            return None
        return row[0]

    @staticmethod
    def _table_exists(conn: Any, table_name: str) -> bool:
        row = conn.execute(
            "select count(*) from information_schema.tables where table_name = ?",
            [table_name],
        ).fetchone()
        return bool(row and row[0])

    def _rebuild_interval_elapsed(self, conn: Any) -> bool:
        last_built_at = self._metadata_value(conn, "last_built_at")
        if last_built_at is None:
            return True
        try:
            last_built = datetime.fromisoformat(last_built_at)
        except ValueError:
            return True
        if last_built.tzinfo is None:
            last_built = last_built.replace(tzinfo=UTC)
        return datetime.now(UTC) - last_built >= self.rebuild_interval


class ChallengeDataTools:
    """Tools for browsing and full-text searching extracted Coding Challenges markdown files."""

    def __init__(
        self,
        data_dir: Path | str = "data/extracted_data",
        search_db_path: Path | str = "data/challenge_search.duckdb",
        *,
        search_index_rebuild_interval: timedelta = timedelta(days=1),
    ) -> None:
        self.data_dir = Path(data_dir)
        self.index_file = self.data_dir / "index.md"
        self.search_index = DuckDBFTSMarkdownSearch(
            self.data_dir,
            Path(search_db_path),
            rebuild_interval=search_index_rebuild_interval,
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
            return self.search_index.search(query, limit=max(1, min(limit_arg, 20)))

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
