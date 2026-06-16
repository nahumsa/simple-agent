"""Shared progress reporting for eval scripts."""

from __future__ import annotations

import sys
from typing import TextIO


class ProgressReporter:
    """Print per-case eval progress when enabled."""

    def __init__(self, *, enabled: bool, stream: TextIO | None = None) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stderr

    def case_started(self, index: int, total: int, label: str) -> None:
        """Report that a case started."""
        if self.enabled:
            print(f"[{index}/{total}] RUN {label}", file=self.stream, flush=True)

    def case_finished(
        self,
        index: int,
        total: int,
        status: str,
        label: str,
        details: str = "",
    ) -> None:
        """Report that a case finished."""
        if not self.enabled:
            return
        suffix = f" {details}" if details else ""
        print(
            f"[{index}/{total}] {status} {label}{suffix}",
            file=self.stream,
            flush=True,
        )
