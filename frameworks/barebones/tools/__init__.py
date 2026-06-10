"""Tool registries for the simple agent."""

from __future__ import annotations

from frameworks.barebones.tools.challenge_data import ChallengeDataTools
from frameworks.barebones.tools.challenge_repository import ChallengeRepository
from frameworks.barebones.tools.decorators import LoggingTools, TimeoutTools
from frameworks.barebones.tools.markdown_search import (
    DuckDBFTSMarkdownSearch,
    MarkdownSearchError,
    SearchResult,
)
from frameworks.barebones.tools.no_tools import NoTools
from frameworks.barebones.tools.url_fetch import FetchedURL, URLFetchError, fetch_url

__all__ = [
    "ChallengeDataTools",
    "ChallengeRepository",
    "LoggingTools",
    "TimeoutTools",
    "DuckDBFTSMarkdownSearch",
    "MarkdownSearchError",
    "FetchedURL",
    "NoTools",
    "SearchResult",
    "URLFetchError",
    "fetch_url",
]
