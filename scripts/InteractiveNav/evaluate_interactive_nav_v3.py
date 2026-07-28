"""Canonical CLI entry point for the standalone InteractiveNav V3 evaluator.

This command intentionally routes to
:mod:`scripts.InteractiveNav.evaluation.benchmark_runner`, rather than the
legacy compatibility runner or the upstream ``molmo_spaces.evaluation``
package.  The benchmark runner owns the frozen V3 protocol, interaction action
validation, reproducibility manifest, and spawn-safe multi-process execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav.evaluation.benchmark_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
