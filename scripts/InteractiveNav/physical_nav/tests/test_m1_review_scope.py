"""Physical M1 discovery is broader than permission to execute interactions."""

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml

from semantic_mapping_py_pkg.attribute_filter import (
    DEFAULT_EXCLUDE_KEYWORDS,
    DEFAULT_INTERACTION_KEYWORDS,
    is_interaction_attribute_candidate,
)
from semantic_mapping_py_pkg.graph_rules import OPENABLE_CONTAINER_LABELS, PORTAL_LABELS


ADDON = Path(__file__).resolve().parents[1]


def _review_scope():
    config = yaml.safe_load((ADDON / "config/physical_nav.yaml").read_text())
    return tuple(config["include_keywords"]), tuple(config["exclude_keywords"])


@pytest.mark.parametrize("label", sorted(
    PORTAL_LABELS | OPENABLE_CONTAINER_LABELS | {"locker", "safe", "freezer"}
))
@pytest.mark.parametrize("label_field", ["semantic_name", "semantic_class", "raw_class"])
def test_all_portals_and_openable_containers_are_reviewable(label, label_field):
    include, exclude = _review_scope()
    assert is_interaction_attribute_candidate({label_field: label}, include, exclude)


@pytest.mark.parametrize("label", ["bedroom_door", "bedside_cabinet", "toilet_door"])
def test_room_context_in_a_name_does_not_exclude_a_portal_or_container(label):
    include, exclude = _review_scope()
    assert is_interaction_attribute_candidate({"semantic_class": label}, include, exclude)


@pytest.mark.parametrize("label", ["box", "storage_bin", "chair", "table", "sofa"])
def test_ordinary_objects_are_not_unconditionally_reviewed(label):
    include, exclude = _review_scope()
    assert not is_interaction_attribute_candidate({"semantic_class": label}, include, exclude)


@pytest.mark.parametrize("label", ["box", "storage_bin", "unknown"])
def test_explicit_articulated_receptacle_remains_reviewable(label):
    include, exclude = _review_scope()
    assert is_interaction_attribute_candidate({
        "semantic_class": label, "is_receptacle": True, "is_articulable": True,
    }, include, exclude)


def test_review_does_not_expand_interaction_goal_whitelist():
    include, exclude = _review_scope()
    goals = yaml.safe_load((ADDON / "config/semantic_shadow_override.yaml").read_text())
    assert goals["interaction_goals"]["allowed_semantic_types"] == ["door", "fridge"]
    assert is_interaction_attribute_candidate({"semantic_class": "microwave"}, include, exclude)
    assert "microwave" not in goals["interaction_goals"]["allowed_semantic_types"]


def test_launch_does_not_override_configured_m1_review_scope():
    launch = ET.parse(ADDON / "launch/physical_nav_readonly.launch").getroot()
    node = launch.find("node[@name='interaction_attribute_inference']")
    assert node is not None
    assert node.find("rosparam[@file='$(arg config_file)']") is not None
    for name in ("include_keywords", "exclude_keywords"):
        assert node.find(f"rosparam[@param='{name}']") is None
        assert node.find(f"param[@name='{name}']") is None


def test_physical_review_expansion_does_not_change_generic_defaults():
    assert not is_interaction_attribute_candidate(
        {"semantic_name": "Safe"}, DEFAULT_INTERACTION_KEYWORDS, DEFAULT_EXCLUDE_KEYWORDS,
    )
    assert "microwave" not in DEFAULT_INTERACTION_KEYWORDS
