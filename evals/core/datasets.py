"""Shared dataset loading and validation helpers for eval scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


def load_json_list(path: Path, *, description: str = "Dataset") -> list[object]:
    """Load a JSON file and ensure the top-level value is a list."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{description} must be a JSON list of cases.")
    return value


def latest_versioned_dataset(
    *,
    datasets_dir: Path,
    versioned_prefix: str,
    fallback_name: str,
) -> Path:
    """Return the newest dataset matching '<versioned_prefix><number>.json'."""
    versioned_datasets: list[tuple[int, Path]] = []
    pattern = re.compile(rf"{re.escape(versioned_prefix)}(\d+)\.json")
    for path in datasets_dir.glob(f"{versioned_prefix}*.json"):
        match = pattern.fullmatch(path.name)
        if match:
            versioned_datasets.append((int(match.group(1)), path))

    if versioned_datasets:
        return max(versioned_datasets)[1]

    return datasets_dir / fallback_name


def required_string(
    raw_case: Mapping[object, object], field: str, *, case_index: int
) -> str:
    """Return a required non-empty string field from a raw eval case."""
    value = raw_case.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Case {case_index} must include a non-empty {field}.")
    return value


def optional_string(
    raw_case: Mapping[object, object],
    field: str,
    *,
    default: str,
    case_index: int,
) -> str:
    """Return an optional non-empty string field from a raw eval case."""
    value = raw_case.get(field, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Case {case_index} has an invalid {field}.")
    return value


def string_list_field(
    raw_case: Mapping[object, object],
    field: str,
    *,
    default: list[str] | None = None,
    case_index: int,
) -> list[str]:
    """Return a string-list field, accepting a single string as shorthand."""
    value: Any = raw_case.get(field, [] if default is None else default)
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"Case {case_index} {field} must be a list.")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"Case {case_index} {field} must contain strings only.")
    return value
