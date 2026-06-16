"""Unified dispatcher for eval entrypoints.

Examples:
    python -m evals.run search --limit 10
    python -m evals.run tool-call --model gpt-4o-mini
"""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Callable

_COMMAND_MODULES = {
    "search": "evals.search.eval",
    "tool-call": "evals.tool_call.eval",
    "tool_call": "evals.tool_call.eval",
}


def _usage() -> str:
    commands = ", ".join(sorted(_COMMAND_MODULES))
    return f"Usage: python -m evals.run <eval> [args...]\n\nAvailable evals: {commands}"


def main() -> None:
    """Dispatch to a specific eval's existing main function."""
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return

    command = sys.argv[1]
    module_name = _COMMAND_MODULES.get(command)
    if module_name is None:
        print(f"Unknown eval: {command}\n\n{_usage()}", file=sys.stderr)
        raise SystemExit(2)

    module = import_module(module_name)
    eval_main = getattr(module, "main")
    if not callable(eval_main):
        raise TypeError(f"{module_name}.main is not callable")

    sys.argv = [f"python -m evals.run {command}", *sys.argv[2:]]
    cast_main: Callable[[], None] = eval_main
    cast_main()


if __name__ == "__main__":
    main()
