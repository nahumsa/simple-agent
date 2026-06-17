"""Unified dispatcher for eval entrypoints.

Examples:
    python -m evals.run search --limit 10
    python -m evals.run tool-call --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from typing import Callable

_COMMAND_MODULES = {
    "search": "evals.search.eval",
    "tool-call": "evals.tool_call.eval",
    "tool_call": "evals.tool_call.eval",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the unified eval dispatcher argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m evals.run",
        description="Run an eval suite.",
    )
    parser.add_argument(
        "eval_name",
        choices=sorted(_COMMAND_MODULES),
        metavar="eval",
        help="Eval suite to run.",
    )
    parser.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the selected eval suite.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch to a specific eval's existing main function."""
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        return

    args = parser.parse_args(argv)
    module_name = _COMMAND_MODULES[args.eval_name]

    module = import_module(module_name)
    eval_main = getattr(module, "main")
    if not callable(eval_main):
        raise TypeError(f"{module_name}.main is not callable")

    sys.argv = [f"python -m evals.run {args.eval_name}", *args.eval_args]
    cast_main: Callable[[], None] = eval_main
    cast_main()


if __name__ == "__main__":
    main()
