import asyncio
import json

from frameworks.barebones.tools import ChallengeDataTools, DuckDBFTSMarkdownSearch, SearchResult


def test_markdown_documents_ignore_index_and_extract_titles(tmp_path) -> None:
    (tmp_path / "index.md").write_text("# Challenge Index\n", encoding="utf-8")
    (tmp_path / "001-parser.md").write_text(
        "Intro before title\n\n## Build Your Own Parser\nParse command line flags.",
        encoding="utf-8",
    )
    (tmp_path / "002-no-heading.md").write_text(
        "No markdown heading here.",
        encoding="utf-8",
    )

    search = DuckDBFTSMarkdownSearch(tmp_path, tmp_path / "search.duckdb")

    assert search._markdown_documents() == [
        (
            1,
            "001-parser.md",
            "Build Your Own Parser",
            "Intro before title\n\n## Build Your Own Parser\nParse command line flags.",
        ),
        (2, "002-no-heading.md", "002-no-heading", "No markdown heading here."),
    ]


def test_search_returns_matching_markdown_documents(tmp_path) -> None:
    data_dir = tmp_path / "docs"
    db_path = tmp_path / "db" / "challenge_search.duckdb"
    data_dir.mkdir()
    (data_dir / "index.md").write_text("# Index\n", encoding="utf-8")
    (data_dir / "001-shell.md").write_text(
        "# Build Your Own Shell\n\nImplement pipes and command execution.",
        encoding="utf-8",
    )
    (data_dir / "002-editor.md").write_text(
        "# Build Your Own Text Editor\n\nHandle cursor movement and syntax highlighting.",
        encoding="utf-8",
    )

    results = DuckDBFTSMarkdownSearch(data_dir, db_path).search("pipes", limit=5)

    assert [result.path for result in results] == ["001-shell.md"]
    assert results[0].title == "Build Your Own Shell"
    assert "Implement pipes and command execution." in results[0].snippet
    assert db_path.exists()


def test_search_refreshes_index_when_markdown_file_count_changes(tmp_path) -> None:
    (tmp_path / "001-shell.md").write_text(
        "# Shell\n\nImplement command execution.",
        encoding="utf-8",
    )
    search = DuckDBFTSMarkdownSearch(tmp_path, tmp_path / "search.duckdb")

    assert search.search("zebra", limit=5) == []

    (tmp_path / "002-cache.md").write_text(
        "# Cache\n\nStore zebra tokens for later retrieval.",
        encoding="utf-8",
    )

    refreshed_results = search.search("zebra", limit=5)

    assert [result.path for result in refreshed_results] == ["002-cache.md"]


def test_search_refreshes_index_when_markdown_content_changes(tmp_path) -> None:
    markdown_path = tmp_path / "001-shell.md"
    markdown_path.write_text(
        "# Shell\n\nImplement command execution.",
        encoding="utf-8",
    )
    search = DuckDBFTSMarkdownSearch(tmp_path, tmp_path / "search.duckdb")

    assert search.search("zebra", limit=5) == []

    markdown_path.write_text(
        "# Shell\n\nImplement command execution with zebra tokens.",
        encoding="utf-8",
    )

    refreshed_results = search.search("zebra", limit=5)

    assert [result.path for result in refreshed_results] == ["001-shell.md"]


def test_challenge_data_tool_validates_and_clamps_search_arguments(tmp_path) -> None:
    class SearchSpy:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def search(self, query: str, *, limit: int) -> list[SearchResult]:
            self.calls.append((query, limit))
            return [
                SearchResult(
                    path="001-parser.md",
                    title="Parser",
                    score=1.5,
                    snippet="Parse command line flags.",
                )
            ]

    tools = ChallengeDataTools(data_dir=tmp_path, search_db_path=tmp_path / "search.duckdb")
    spy = SearchSpy()
    tools.search_index = spy  # type: ignore[assignment]

    missing_query = asyncio.run(
        tools.call("search_challenges", {}, session=object(), tool_call_id="test")
    )
    invalid_limit = asyncio.run(
        tools.call(
            "search_challenges",
            {"query": "parser", "limit": "5"},
            session=object(),
            tool_call_id="test",
        )
    )
    clamped_low = asyncio.run(
        tools.call(
            "search_challenges",
            {"query": "parser", "limit": -10},
            session=object(),
            tool_call_id="test",
        )
    )
    clamped_high = asyncio.run(
        tools.call(
            "search_challenges",
            {"query": "editor", "limit": 999},
            session=object(),
            tool_call_id="test",
        )
    )

    assert missing_query == ("Missing required string argument: query", False)
    assert invalid_limit == ("limit must be an integer", False)
    expected_results = [
        {
            "path": "001-parser.md",
            "title": "Parser",
            "score": 1.5,
            "snippet": "Parse command line flags.",
        }
    ]
    assert clamped_low == (
        json.dumps({"query": "parser", "results": expected_results}, indent=2),
        True,
    )
    assert clamped_high == (
        json.dumps({"query": "editor", "results": expected_results}, indent=2),
        True,
    )
    assert spy.calls == [("parser", 1), ("editor", 20)]
