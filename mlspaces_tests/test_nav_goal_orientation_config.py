import math
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_global_planner_interpolates_to_goal_orientation() -> None:
    launch_path = REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "nav_pkg" / "launch" / "nav.launch"
    root = ET.parse(launch_path).getroot()
    params = {
        element.attrib.get("name"): element.attrib.get("value")
        for element in root.iter("param")
    }
    assert params["GlobalPlanner/orientation_mode"] == "3"
    assert int(params["GlobalPlanner/orientation_window_size"]) >= 1
    assert params["OrientedGlobalPlanner/orient_path_tangents"] == "true"
    base_global_planner = next(
        element
        for element in root.iter("param")
        if element.attrib.get("name") == "base_global_planner"
    )
    assert base_global_planner.attrib["value"] == "$(arg base_global_planner)"
    launch_args = {element.attrib.get("name"): element.attrib for element in root.iter("arg")}
    assert launch_args["base_global_planner"]["default"] == "nav_pkg/OrientedGlobalPlanner"


def test_dwa_requires_terminal_goal_yaw() -> None:
    config_path = (
        REPO_ROOT
        / "Interactive-Nav-SG-nav"
        / "src"
        / "nav_pkg"
        / "configs"
        / "controller"
        / "dwa_controller_params.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["DWAPlannerROS"]
    assert float(config["yaw_goal_tolerance"]) <= 0.25
    assert config["latch_xy_goal_tolerance"] is True


def test_v3_prerotation_uses_navigation_speed_cap_and_forward_sector() -> None:
    semantic_config_path = (
        REPO_ROOT
        / "scripts"
        / "InteractiveNav"
        / "configs"
        / "semantic_decision"
        / "object_goal_v3_full_mllm.yaml"
    )
    dwa_config_path = (
        REPO_ROOT
        / "Interactive-Nav-SG-nav"
        / "src"
        / "nav_pkg"
        / "configs"
        / "controller"
        / "dwa_controller_params.yaml"
    )
    executor = yaml.safe_load(semantic_config_path.read_text(encoding="utf-8"))["executor"]
    dwa = yaml.safe_load(dwa_config_path.read_text(encoding="utf-8"))["DWAPlannerROS"]

    assert math.isclose(float(executor["rear_goal_enter_angle_rad"]), math.pi / 6.0)
    assert float(executor["rear_goal_exit_angle_rad"]) == float(
        executor["rear_goal_enter_angle_rad"]
    )
    assert float(executor["rear_goal_rotate_speed_rad_s"]) <= float(dwa["max_vel_theta"])
    assert executor["rear_goal_prerotate_step_sync_enabled"] is True
    assert math.isclose(float(executor["rear_goal_prerotate_control_dt_s"]), 0.2)
    assert int(executor["rear_goal_prerotate_max_control_steps"]) == 12


def test_v3_dwa_biases_leave_room_for_local_obstacle_avoidance() -> None:
    config_path = (
        REPO_ROOT
        / "scripts"
        / "InteractiveNav"
        / "configs"
        / "semantic_decision"
        / "semantic_interaction_nav.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["DWAPlannerROS"]

    # These are supported by stock dwa_local_planner/DWAPlannerROS.  Keep the
    # local trajectory from being over-tethered to a narrow global route while
    # restoring the native high-speed footprint expansion.
    assert float(config["path_distance_bias"]) == 18.0
    assert float(config["goal_distance_bias"]) == 20.0
    assert float(config["occdist_scale"]) == 0.10
    assert float(config["max_scaling_factor"]) == 0.20


def test_v3_global_costmap_bounds_post_open_occ_propagation_at_10_hz() -> None:
    config_path = (
        REPO_ROOT
        / "scripts"
        / "InteractiveNav"
        / "configs"
        / "semantic_decision"
        / "semantic_interaction_nav.yaml"
    )
    costmap = yaml.safe_load(config_path.read_text(encoding="utf-8"))["global_costmap"]

    assert float(costmap["update_frequency"]) == 10.0
    assert float(costmap["publish_frequency"]) == 10.0


def test_v3_post_open_uses_a_bounded_causal_costmap_refresh() -> None:
    config_path = (
        REPO_ROOT
        / "scripts"
        / "InteractiveNav"
        / "configs"
        / "semantic_decision"
        / "object_goal_v3_full_mllm.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    policy = config["policy"]
    executor = config["executor"]

    assert policy["post_interaction_refresh_enabled"] is False
    # The executor waits for raw OCC -> planning OCC -> global costmap, so
    # permit one bounded multi-stage refresh rather than a graph/M1 wait or
    # the former 30-second replan loop.
    assert float(executor["post_interaction_costmap_fresh_timeout_s"]) <= 2.0
    assert float(executor["post_interaction_costmap_fresh_poll_interval_s"]) <= 0.05
    assert float(executor["post_interaction_traversal_make_plan_retry_window_s"]) <= 0.35
    assert float(executor["post_interaction_traversal_make_plan_retry_interval_s"]) == 0.10
    assert (
        config["topics"]["global_costmap_updates"]
        == "/move_base/global_costmap/costmap_updates"
    )
    assert config["topics"]["raw_occupancy_grid"] == "/struct_mapping/occ_map"
    assert (
        config["topics"]["planning_occupancy_grid"]
        == "/semantic_mapping/planning_occ_map"
    )


def test_default_executor_subscribes_to_the_global_costmap_used_by_make_plan() -> None:
    config_path = (
        REPO_ROOT
        / "Interactive-Nav-SG-nav"
        / "src"
        / "semantic_decision_py_pkg"
        / "config"
        / "default.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["topics"]["global_costmap"] == "/move_base/global_costmap/costmap"
    assert (
        config["topics"]["global_costmap_updates"]
        == "/move_base/global_costmap/costmap_updates"
    )
    assert config["topics"]["raw_occupancy_grid"] == "/struct_mapping/occ_map"
    assert (
        config["topics"]["planning_occupancy_grid"]
        == "/semantic_mapping/planning_occ_map"
    )
    assert float(config["executor"]["post_interaction_costmap_fresh_timeout_s"]) <= 2.0


def test_v3_inherits_native_interaction_standoff_values() -> None:
    default_config_path = (
        REPO_ROOT
        / "Interactive-Nav-SG-nav"
        / "src"
        / "semantic_decision_py_pkg"
        / "config"
        / "default.yaml"
    )
    native_runtime_path = (
        REPO_ROOT
        / "scripts"
        / "InteractiveNav"
        / "configs"
        / "semantic_decision"
        / "object_goal_runtime.yaml"
    )
    v3_runtime_path = (
        REPO_ROOT
        / "scripts"
        / "InteractiveNav"
        / "configs"
        / "semantic_decision"
        / "object_goal_v3_full_mllm.yaml"
    )

    default_candidate = yaml.safe_load(
        default_config_path.read_text(encoding="utf-8")
    )["candidate"]
    assert float(default_candidate["portal_standoff_m"]) == 0.85
    assert float(default_candidate["container_standoff_m"]) == 0.50
    assert float(default_candidate["drawer_standoff_m"]) == 0.50
    assert float(default_candidate["interaction_safety_margin_m"]) == 0.25

    # Both object-goal overlays intentionally inherit these four placement
    # parameters rather than silently changing Native NavToObj approach range.
    for runtime_path in (native_runtime_path, v3_runtime_path):
        candidate = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))["candidate"]
        for key in (
            "portal_standoff_m",
            "container_standoff_m",
            "drawer_standoff_m",
            "interaction_safety_margin_m",
        ):
            assert key not in candidate


def test_oriented_global_planner_plugin_is_exported() -> None:
    package_path = REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "nav_pkg" / "package.xml"
    package_root = ET.parse(package_path).getroot()
    export = next(element for element in package_root if element.tag == "export")
    plugin_path = export.find("nav_core").attrib["plugin"]
    assert plugin_path.endswith("oriented_global_planner_plugin.xml")
    plugin_root = ET.parse(package_path.parent / plugin_path.rsplit("/", 1)[-1]).getroot()
    classes = {element.attrib["name"] for element in plugin_root.iter("class")}
    assert "nav_pkg/OrientedGlobalPlanner" in classes


def test_explorer_publishes_the_frontier_viewpoint_yaw() -> None:
    source_path = (
        REPO_ROOT
        / "Interactive-Nav-SG-nav"
        / "src"
        / "explore_py_pkg"
        / "scripts"
        / "explore_py_node.py"
    )
    source = source_path.read_text(encoding="utf-8")
    publish_method = source.split("    def _publish_active_goal(self):", 1)[1].split(
        "    def _publish_status(self):", 1
    )[0]
    assert "_quaternion_z_w_from_yaw(goal.yaw)" in publish_method
    assert "msg.pose.orientation.z = qz" in publish_method
    assert "msg.pose.orientation.w = qw" in publish_method


def test_house7_force_route_allows_short_ros_pose_lag() -> None:
    config_path = (
        REPO_ROOT
        / "scripts"
        / "InteractiveNav"
        / "configs"
        / "semantic_decision"
        / "house7_force_route_nav.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert float(config["global_costmap"]["transform_tolerance"]) == 3.0
    assert float(config["local_costmap"]["transform_tolerance"]) == 3.0
