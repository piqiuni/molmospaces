from __future__ import annotations

import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCRIPTS = PACKAGE_ROOT / "scripts"
if str(PACKAGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SCRIPTS))
MLLM_SCRIPTS = PACKAGE_SCRIPTS.parents[1] / "semantic_mllm_py_pkg" / "scripts"
if str(MLLM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MLLM_SCRIPTS))

def _include_arg(group: ET.Element, name: str) -> str | None:
    include = group.find("include")
    assert include is not None
    arg = include.find(f"arg[@name='{name}']")
    return None if arg is None else arg.attrib.get("value")


def _semantic_mapping_node_module():
    pytest.importorskip("rospy")
    import semantic_mapping_node as semantic_mapping_module
    from semantic_mapping_node import SemanticMappingNode

    return semantic_mapping_module, SemanticMappingNode


def test_scene_callback_warns_and_does_not_mutate_scene_without_cloud_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic_mapping_module, semantic_mapping_node = _semantic_mapping_node_module()
    warnings: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        semantic_mapping_module.rospy,
        "logwarn_throttle",
        lambda *args: warnings.append(args),
        raising=False,
    )
    node = object.__new__(semantic_mapping_node)
    node.enable_scene_mapping = True
    node.subscribe_pointcloud = False
    node.latest_scene = None

    semantic_mapping_node.scene_callback(
        node,
        SimpleNamespace(data='{"scene_attribute":"bedroom"}'),
    )

    assert node.latest_scene is None
    assert len(warnings) == 1
    assert warnings[0][0] == 5.0
    assert "subscribe_pointcloud=false" in str(warnings[0][1])


def test_pointcloud_callback_remains_available_for_enabled_legacy_scene_mapping() -> None:
    _semantic_mapping_module, semantic_mapping_node = _semantic_mapping_node_module()
    node = object.__new__(semantic_mapping_node)
    node.lock = threading.RLock()
    node.latest_cloud = None
    cloud = object()

    semantic_mapping_node.pointcloud_callback(node, cloud)

    assert node.latest_cloud is cloud


def test_launch_wiring_keeps_legacy_modes_enabled_and_disables_realtime_gt_only() -> None:
    semantic_launch = ET.parse(
        PACKAGE_ROOT / "launch" / "semantic_mapping_py.launch"
    ).getroot()
    launch_arg = semantic_launch.find("arg[@name='subscribe_pointcloud']")
    assert launch_arg is not None
    assert launch_arg.attrib["default"] == "true"
    semantic_node = semantic_launch.find(".//node[@name='semantic_mapping_py']")
    assert semantic_node is not None
    param = semantic_node.find("param[@name='semantic_map/subscribe_pointcloud']")
    assert param is not None
    assert param.attrib == {
        "name": "semantic_map/subscribe_pointcloud",
        "value": "$(arg subscribe_pointcloud)",
        "type": "bool",
    }

    nav_launch = ET.parse(
        PACKAGE_ROOT.parent / "nav_pkg" / "launch" / "molmospaces_nav_system.launch"
    ).getroot()
    source_groups = {
        group.attrib.get("if", ""): group for group in nav_launch.findall("group")
    }
    detector_group = next(
        group
        for condition, group in source_groups.items()
        if "semantic_source') == 'detector'" in condition
    )
    realtime_gt_group = next(
        group
        for condition, group in source_groups.items()
        if "semantic_source') == 'realtime_gt'" in condition
    )
    offline_gt_group = next(
        group
        for condition, group in source_groups.items()
        if "semantic_source') == 'offline_gt'" in condition
    )

    assert _include_arg(detector_group, "subscribe_pointcloud") == "true"
    assert _include_arg(realtime_gt_group, "subscribe_pointcloud") == "false"
    assert _include_arg(offline_gt_group, "subscribe_pointcloud") == "true"
