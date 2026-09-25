#!/usr/bin/env python3
"""Show compact progress for one pool or the current three-task campaign."""

import argparse
from datetime import datetime
from pathlib import Path
import time

from progress import format_pool


VARIANTS = ("no_interaction_graph", "no_task_decision", "no_outcome_update")


def pools_at(path):
    path = path.resolve()
    if (path / "pool_manifest.json").exists():
        return [path]
    return [path / variant / "run/pool" for variant in VARIANTS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval-s", type=float, default=10.0)
    args = parser.parse_args()
    if args.interval_s <= 0:
        parser.error("interval must be positive")
    pools = pools_at(args.path)
    while True:
        print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}]", flush=True)
        for pool in pools:
            print(format_pool(pool), flush=True)
        if not args.follow or all((pool / "pool_result.json").exists() for pool in pools):
            return
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
