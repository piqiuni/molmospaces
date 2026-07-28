from __future__ import annotations

import json
import sys
from pathlib import Path

from std_msgs.msg import String


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from semantic_candidate_node import (
    SemanticCandidateNode,
    resolve_initial_scan_complete,
)


def test_initial_scan_completion_prefers_explicit_compatibility_field() -> None:
    assert resolve_initial_scan_complete(
        {"initial_scan_complete": False, "initial_spin": {"done": True}}
    ) is False
    assert resolve_initial_scan_complete(
        {"initial_scan_complete": True, "initial_spin": {"done": False}}
    ) is True


def test_initial_scan_completion_falls_back_to_nested_spin_state() -> None:
    assert resolve_initial_scan_complete({"initial_spin": {"done": False}}) is False
    assert resolve_initial_scan_complete({"initial_spin": {"done": True}}) is True


def test_initial_scan_completion_is_safe_when_status_is_missing() -> None:
    assert resolve_initial_scan_complete({}) is False


def test_target_update_republishes_immediately() -> None:
    node = object.__new__(SemanticCandidateNode)
    node.target_context = {"episode_active": True}
    published = []
    node._publish = lambda _event: published.append(dict(node.target_context))

    node._target_callback(String(data=json.dumps({"episode_active": False})))

    assert published == [{"episode_active": False}]


def test_graph_callback_immediately_publishes_refreshed_candidates() -> None:
    node = object.__new__(SemanticCandidateNode)
    node.graph = {"graph_revision": 2}
    published_revisions = []
    node._publish = lambda _event: published_revisions.append(
        node.graph["graph_revision"]
    )

    node._graph_callback(String(data=json.dumps({"graph_revision": 3, "nodes": []})))

    assert node.graph["graph_revision"] == 3
    assert published_revisions == [3]


def test_graph_callback_does_not_republish_unchanged_revision() -> None:
    node = object.__new__(SemanticCandidateNode)
    node.graph = {"episode_id": "episode_1", "graph_revision": 3}
    published_revisions = []
    node._publish = lambda _event: published_revisions.append(
        node.graph["graph_revision"]
    )

    node._graph_callback(
        String(
            data=json.dumps(
                {"episode_id": "episode_1", "graph_revision": 3, "nodes": []}
            )
        )
    )

    assert published_revisions == []
