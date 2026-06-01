"""Tool registries for the simple agent."""

from __future__ import annotations

from frameworks.barebones.tools.challenge_data import ChallengeDataTools
from frameworks.barebones.tools.markdown_search import (
    DuckDBFTSMarkdownSearch,
    MarkdownSearchError,
    SearchResult,
)
from frameworks.barebones.tools.no_tools import NoTools

__all__ = [
    "ChallengeDataTools",
    "DuckDBFTSMarkdownSearch",
    "MarkdownSearchError",
    "NoTools",
    "SearchResult",
]
