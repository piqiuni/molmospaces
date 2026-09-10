from pathlib import Path

from scripts.InteractiveNav.check_simulator_scope_parity import (
    compare_refs,
    load_scope,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPE_PATH = (
    REPO_ROOT / "scripts" / "InteractiveNav" / "simulator_scope.txt"
)


def test_simulator_scope_manifest_is_unique_and_present_in_current_head() -> None:
    paths = load_scope(SCOPE_PATH)

    assert len(paths) == len(set(paths))
    assert compare_refs("HEAD", "HEAD", paths) == []
