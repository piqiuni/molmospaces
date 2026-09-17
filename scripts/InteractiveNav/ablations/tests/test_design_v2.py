from copy import deepcopy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from ablations import VARIANTS
from ablations.flat_memory import FlatObjectCandidateGenerator, flat_objects, candidate_node_class
from ablations.launch import render_artifacts
from ablations.perception import REFRESH_PROFILES, inference_node_class
from ablations.policies import without_outcome_continuations
from ablations.sensory_candidates import PerceptionOnlyCandidateGenerator
from semantic_decision_py_pkg.behavior_candidates import CandidateGeneratorConfig
from semantic_mapping_py_pkg.portal_state_consensus import PortalStateConsensus

REPO = Path(__file__).resolve().parents[4]


def graph(state="closed"):
    return {"episode_id": "test", "nodes": [{
        "id": "door", "type": "portal", "label": "door", "name": "door",
        "centroid": [0., 0., 1.], "aabb_center": [0., 0., 1.],
        "aabb_extent": [.1, 1., 2.], "confidence": .95,
        "is_currently_visible": True,
        "attributes": {"visible_pixels": 2048, "visible_fraction": .9,
                       "consecutive_observations": 4, "source_object_name": "door",
                       "connected_room_ids": ["a", "b"], "drawer_scan_completed": True},
        "interaction": {"state": state, "is_interactable": True, "traversable": state == "open",
                        "state_confidence": .95, "operation_history": [{"success": True}]},
    }, {"id": "a", "type": "room"}], "edges": [{"relation": "connects"}]}


def test_flat_candidate_generation_is_independent_of_interaction_state_and_edges():
    generator = FlatObjectCandidateGenerator(CandidateGeneratorConfig())
    closed, opened = graph(), graph("open")
    before = deepcopy(closed)
    left = generator.generate({}, closed, (-1., 0.))
    right = generator.generate({}, opened, (-1., 0.))
    assert left and [c.to_dict() for c in left] == [c.to_dict() for c in right]
    assert closed == before
    interactions = [c for c in left if c.behavior_type == "INTERACT"]
    assert interactions and all(c.metadata["observation_required"] for c in interactions)
    assert all("state" not in c.metadata and "state_age_ratio" not in c.features for c in left)
    memory = flat_objects(closed)
    assert memory["edges"] == [] and len(memory["nodes"]) == 1
    assert "interaction" not in memory["nodes"][0]
    assert "drawer_scan_completed" not in memory["nodes"][0]["attributes"]
    assert "connected_room_ids" not in memory["nodes"][0]["attributes"]


def test_flat_node_does_not_retain_incoming_state_or_relations():
    class Node:
        def __init__(self):
            self.graph = {}
            self.generator = SimpleNamespace(config=CandidateGeneratorConfig())
            self.startup_scan_lifecycle = SimpleNamespace(observe_episode=lambda _: None)

    node = candidate_node_class(Node)()
    for state in ("closed", "open"):
        node._graph_callback(SimpleNamespace(data=json.dumps(graph(state))))
        assert node.graph == flat_objects(graph(state))
        assert "interaction" not in node.graph["nodes"][0]


def test_perception_only_traversal_needs_visual_and_occupancy_confirmation():
    generator = PerceptionOnlyCandidateGenerator(CandidateGeneratorConfig())
    source = graph("open")
    assert generator._portal_traversal_candidates(source, (-1., 0.)) == []
    attrs = source["nodes"][0]["attributes"]
    attrs.update(portal_state_consensus_accepted=True, connectivity_status="connected",
                 portal_state_gate={"accepted": True, "observation_capture_step": 80},
                 portal_state_consensus={"accepted": True}, observed_connected_room_ids=["a", "b"])
    before = deepcopy(source)
    proposals = generator._portal_traversal_candidates(source, (-1., 0.))
    assert proposals and source == before
    assert proposals[0].metadata["perception_open_occ_confirmed"]
    assert proposals[0].metadata["state"] == "open"
    assert without_outcome_continuations({"candidates": [c.to_dict() for c in proposals]})["candidate_count"] == len(proposals)
    attrs["portal_state_consensus_accepted"] = False
    assert generator._portal_traversal_candidates(source, (-1., 0.)) == []


@pytest.mark.parametrize("variant", VARIANTS)
def test_continuous_profile_applies_identically_to_all_four_groups(tmp_path, variant):
    artifacts = render_artifacts(REPO, tmp_path, variant, "/usr/bin/python3", "continuous")
    mapping = ET.fromstring(artifacts[tmp_path / "semantic_mapping_py.launch"])
    inference = next(n for n in mapping.iter("node") if n.get("type") == "interaction_attribute_inference_node.py")
    params = {p.get("name"): p.get("value") for p in inference.findall("param")}
    for name, value in REFRESH_PROFILES["continuous"].items():
        assert float(params[name]) == float(value)
    assert "runner.sh" in {p.name for p in artifacts}
    assert bool(inference.get("launch-prefix")) == (variant == "no_outcome_update")


def test_m1_result_invalidation_does_not_import_backend_state():
    class Base:
        pass

    node = inference_node_class(Base)()
    node.lock = threading.Lock()
    node.aliases = {"door_alias": "door"}
    node.generations = {"door": 2}
    for key in ("completed", "last_request", "pending", "target_visual_history", "portal_observation_streaks"):
        setattr(node, key, {"door": {"old": True}})
    node.request_queue = {"door"}
    node.portal_state_consensus = PortalStateConsensus(cooldown_steps=60)
    node.portal_state_consensus.record_authoritative("door", "closed", capture_step=10)
    node._interaction_result_callback(SimpleNamespace(data=json.dumps({
        "object_id": "door_alias", "success": True, "post_state": "open", "capture_step": 20,
    })))
    assert node.generations["door"] == 3 and not node.pending and not node.completed
    assert not node.request_queue
    assert node.portal_state_consensus.cached_stable_result("door", capture_step=20) is None
    assert node.portal_state_consensus.can_request("door", capture_step=20, observation_pose_xyyaw=[0, 0, 0])[0]


def test_known_m1_state_refreshes_periodically_and_pending_requests_are_deduplicated(monkeypatch):
    pytest.importorskip("rospy")
    import interaction_attribute_inference_node as inference

    node = inference.InteractionAttributeInferenceNode.__new__(inference.InteractionAttributeInferenceNode)
    node.lock = threading.Lock()
    node.success_refresh_interval_s = 30.
    node.min_interval_s = 5.
    node.pending, node.generations, node.last_request = {}, {}, {"cabinet": 100.}
    node.completed = {"cabinet": {"signature": "same-view", "completed_at": 100., "refresh_interval_s": 30.}}
    node.request_sequence = 0
    node.current_episode_id = "test"
    monkeypatch.setattr(inference.time, "monotonic", lambda: 129.)
    assert node._try_reserve("cabinet", "same-view") is None
    monkeypatch.setattr(inference.time, "monotonic", lambda: 131.)
    assert node._try_reserve("cabinet", "same-view") is not None
    assert node._try_reserve("cabinet", "same-view") is None


def test_continuous_m1_preserves_distinct_view_confirmation_for_state_changes():
    consensus = PortalStateConsensus(cooldown_steps=60)
    consensus.record_authoritative("door", "closed", capture_step=10)
    assert not consensus.can_request("door", capture_step=69, observation_pose_xyyaw=[0, 0, 0])[0]
    assert consensus.can_request("door", capture_step=70, observation_pose_xyyaw=[0, 0, 0])[0]
    first = consensus.observe("door", "open", capture_step=70, observation_pose_xyyaw=[0, 0, 0])
    assert not first["accepted"]
    assert not consensus.can_request("door", capture_step=71, observation_pose_xyyaw=[0, 0, 0])[0]
    second = consensus.observe("door", "open", capture_step=72, observation_pose_xyyaw=[.5, 0, 0])
    third = consensus.observe("door", "open", capture_step=73, observation_pose_xyyaw=[1., 0, 0])
    assert not second["accepted"] and third["accepted"]
