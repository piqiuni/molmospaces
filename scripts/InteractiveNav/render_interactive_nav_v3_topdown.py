"""CLI entry point for an InteractiveNav V3 evaluation top-down report."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav.evaluation.episode_topdown import main


if __name__ == "__main__":
    raise SystemExit(main())
