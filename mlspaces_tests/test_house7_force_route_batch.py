from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "InteractiveNav"
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.InteractiveNav.run_house7_force_route_batch import (
    load_route_ids,
    split_round_robin,
    validate_worker_count,
)


def test_worker_limit_is_hard_capped_at_two() -> None:
    assert validate_worker_count(1) == 1
    assert validate_worker_count(2) == 2
    with pytest.raises(ValueError, match="cannot exceed 2"):
        validate_worker_count(3)


def test_routes_are_sharded_without_duplication() -> None:
    shards = split_round_robin(["r1", "r2", "r3", "r4", "r5"], 2)
    assert shards == [["r1", "r3", "r5"], ["r2", "r4"]]


def test_route_selection_preserves_frozen_config_order(tmp_path: Path) -> None:
    path = tmp_path / "routes.yaml"
    path.write_text(
        yaml.safe_dump(
            {"routes": [{"route_id": "r1"}, {"route_id": "r2"}, {"route_id": "r3"}]}
        ),
        encoding="utf-8",
    )
    assert load_route_ids(path, ["r3", "r1"]) == ["r1", "r3"]
    with pytest.raises(ValueError, match="Unknown route ids"):
        load_route_ids(path, ["missing"])
