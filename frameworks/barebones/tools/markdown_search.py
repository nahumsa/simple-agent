"""DuckDB-backed full-text search over markdown files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SearchResult:
    path: str
    title: str
    score: float
    snippet: str


class MarkdownSearchError(RuntimeError):
    """Raised when markdown search cannot be completed."""


@dataclass(frozen=True)
class _MarkdownDocument:
    id: int
    path: str
    title: str
    content: str
    sha256: str

    def index_record(self) -> tuple[int, str, str, str]:
        return (self.id, self.path, self.title, self.content)

    def manifest_entry(self) -> dict[str, str]:
        return {
            "path": self.path,
            "sha256": self.sha256,
        }


class DuckDBFTSMarkdownSearch:
    """Persistent DuckDB full-text search index for markdown files."""

    def __init__(
        self,
        data_dir: Path,
        db_path: Path,
    ) -> None:
        self.data_dir = data_dir
        self.db_path = db_path

    def search(self, query: str, *, limit: int) -> list[SearchResult]:
        """Search markdown documents, atomically rebuilding when the corpus manifest changes."""
        try:
            import duckdb
        except ImportError as exc:
            raise MarkdownSearchError(
                "DuckDB is not installed. Run `uv sync` to install project dependencies."
            ) from exc

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(str(self.db_path)) as conn:
                self._install_and_load_fts(conn)
                documents = self._markdown_corpus()
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
        except (
            Exception
        ) as exc:  # DuckDB extension errors are surfaced as different classes.
            raise MarkdownSearchError(f"DuckDB FTS search failed: {exc}") from exc

        return [
            SearchResult(
                path=path,
                title=title,
                score=score,
                snippet=snippet,
            )
            for path, title, score, snippet in rows
        ]

    @staticmethod
    def _install_and_load_fts(conn: Any) -> None:
        conn.execute("INSTALL fts")
        conn.execute("LOAD fts")

    def _markdown_documents(self) -> list[tuple[int, str, str, str]]:
        return [document.index_record() for document in self._markdown_corpus()]

    def _markdown_corpus(self) -> list[_MarkdownDocument]:
        documents: list[_MarkdownDocument] = []
        markdown_paths = [
            path
            for path in sorted(self.data_dir.glob("*.md"))
            if path.name != "index.md"
        ]
        for index, path in enumerate(markdown_paths, start=1):
            data = path.read_bytes()
            content = data.decode("utf-8")
            title = self._extract_title(content) or path.stem
            documents.append(
                _MarkdownDocument(
                    id=index,
                    path=path.name,
                    title=title,
                    content=content,
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            )
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
        documents: list[_MarkdownDocument],
    ) -> None:
        conn.execute(
            "create table if not exists search_metadata (key varchar primary key, value varchar)"
        )
        manifest = [document.manifest_entry() for document in documents]
        manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
        documents_table_exists = self._table_exists(conn, "documents")
        if (
            documents_table_exists
            and self._metadata_value(conn, "corpus_manifest_hash") == manifest_hash
            and self._metadata_value(conn, "corpus_manifest") == manifest_json
        ):
            return

        conn.execute("begin transaction")
        try:
            conn.execute("drop table if exists documents")
            conn.execute("""
                create table documents (
                    id integer primary key,
                    path varchar not null,
                    title varchar not null,
                    content varchar not null
                )
                """)
            conn.executemany(
                "insert into documents (id, path, title, content) values (?, ?, ?, ?)",
                [document.index_record() for document in documents],
            )
            conn.execute("""
                pragma create_fts_index(
                    'documents',
                    'id',
                    'title',
                    'content',
                    stemmer = 'porter',
                    stopwords = 'english',
                    ignore = '(\\\\.|[^a-z])+',
                    strip_accents = 1,
                    lower = 1,
                    overwrite = 1
                )
                """)
            conn.executemany(
                "insert or replace into search_metadata (key, value) values (?, ?)",
                [
                    ("corpus_manifest", manifest_json),
                    ("corpus_manifest_hash", manifest_hash),
                ],
            )
            conn.execute("commit")
        except Exception:
            conn.execute("rollback")
            raise

    @staticmethod
    def _metadata_value(conn: Any, key: str) -> str | None:
        row = conn.execute(
            "select value from search_metadata where key = ?", [key]
        ).fetchone()
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
