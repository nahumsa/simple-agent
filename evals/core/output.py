"""Shared output helpers for eval scripts."""

from __future__ import annotations

import csv
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


def safe_filename_part(value: str) -> str:
    """Return a filesystem-friendly filename segment."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return cleaned or "unknown"


def default_csv_results_path(*, prefix: str, model: str, results_dir: Path) -> Path:
    """Build a timestamped CSV path for an eval report."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return results_dir / f"{prefix}_{timestamp}_{safe_filename_part(model)}.csv"


def dataclass_report_to_json(report: object) -> dict[str, Any]:
    """Convert a dataclass report to a JSON-serializable dictionary."""
    if not is_dataclass(report):
        raise TypeError("report must be a dataclass instance")
    value = asdict(cast(Any, report))
    if not isinstance(value, dict):
        raise TypeError("dataclass report must convert to a dictionary")
    return value


def write_csv_rows(
    *,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, object]],
    path: Path | None = None,
) -> None:
    """Write CSV rows to a path or stdout."""
    output = path.open("w", newline="", encoding="utf-8") if path else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if path:
            output.close()
