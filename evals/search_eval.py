"""Backward-compatible entrypoint for search evals."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evals.search.eval import main  # noqa: E402

if __name__ == "__main__":
    main()
