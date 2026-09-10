#!/usr/bin/env python3
"""Check that two branches have identical shared simulator contract files."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCOPE = Path(__file__).with_name("simulator_scope.txt")


def load_scope(path: Path) -> tuple[str, ...]:
    entries = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not entries:
        raise ValueError(f"Simulator scope is empty: {path}")
    if len(entries) != len(set(entries)):
        raise ValueError(f"Simulator scope contains duplicate paths: {path}")
    return entries


def blob_oid(ref: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def compare_refs(
    simulator_ref: str, full_ref: str, paths: tuple[str, ...]
) -> list[tuple[str, str | None, str | None]]:
    differences = []
    for path in paths:
        simulator_oid = blob_oid(simulator_ref, path)
        full_oid = blob_oid(full_ref, path)
        if simulator_oid is None or full_oid is None or simulator_oid != full_oid:
            differences.append((path, simulator_oid, full_oid))
    return differences


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-ref", default="interactive-nav/sim")
    parser.add_argument("--full-ref", default="codex/exp-setting")
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    args = parser.parse_args(argv)

    paths = load_scope(args.scope.resolve())
    differences = compare_refs(args.sim_ref, args.full_ref, paths)
    if not differences:
        print(
            f"Simulator scope is identical: {args.sim_ref} == {args.full_ref} "
            f"({len(paths)} files)"
        )
        return 0

    print(
        f"Simulator scope differs: {args.sim_ref} != {args.full_ref} "
        f"({len(differences)}/{len(paths)} files)"
    )
    for path, simulator_oid, full_oid in differences:
        simulator_status = simulator_oid or "MISSING"
        full_status = full_oid or "MISSING"
        print(f"- {path}: sim={simulator_status} full={full_status}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
