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


def test_v3_prerotation_matches_ordinary_full_mllm_navigation() -> None:
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
    ordinary_executor = yaml.safe_load(
        (
            REPO_ROOT
            / "scripts"
            / "InteractiveNav"
            / "configs"
            / "semantic_decision"
            / "full_mllm_interactive_exploration.yaml"
        ).read_text(encoding="utf-8")
    )["executor"]
    dwa = yaml.safe_load(dwa_config_path.read_text(encoding="utf-8"))["DWAPlannerROS"]

    assert float(executor["rear_goal_enter_angle_rad"]) == float(
        ordinary_executor["rear_goal_enter_angle_rad"]
    )
    assert float(executor["rear_goal_exit_angle_rad"]) == 0.20
    assert float(executor["rear_goal_rotate_speed_rad_s"]) <= float(dwa["max_vel_theta"])
    assert executor["rear_goal_prerotate_step_sync_enabled"] is True
    assert math.isclose(float(executor["rear_goal_prerotate_control_dt_s"]), 0.2)
    assert int(executor["rear_goal_prerotate_max_control_steps"]) == int(
        ordinary_executor["rear_goal_prerotate_max_control_steps"]
    )


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
    # The executor's paired inner container point has a 0.25 m public pose
    # gate.  DWA must not declare success farther away than that contract.
    assert float(config["xy_goal_tolerance"]) <= 0.25


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


def test_v3_force_interaction_timeout_overrides_only_the_evaluator_profile() -> None:
    default_config_path = (
        REPO_ROOT
        / "Interactive-Nav-SG-nav"
        / "src"
        / "semantic_decision_py_pkg"
        / "config"
        / "default.yaml"
    )
    v3_config_path = (
        REPO_ROOT
        / "scripts"
        / "InteractiveNav"
        / "configs"
        / "semantic_decision"
        / "object_goal_v3_full_mllm.yaml"
    )

    default_executor = yaml.safe_load(default_config_path.read_text(encoding="utf-8"))["executor"]
    v3_executor = yaml.safe_load(v3_config_path.read_text(encoding="utf-8"))["executor"]

    # The V3 overlay is loaded into semantic_behavior_executor, while normal
    # navigation continues to inherit the native thirty-second default.
    assert float(default_executor["interaction_timeout_s"]) == 30.0
    assert float(v3_executor["interaction_timeout_s"]) == 90.0


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


def test_v3_channel_and_mllm_contract_matches_ordinary_profile() -> None:
    ordinary_runtime_path = (
        REPO_ROOT
        / "scripts"
        / "InteractiveNav"
        / "configs"
        / "semantic_decision"
        / "full_mllm_interactive_exploration.yaml"
    )
    v3_runtime_path = (
        REPO_ROOT
        / "scripts"
        / "InteractiveNav"
        / "configs"
        / "semantic_decision"
        / "object_goal_v3_full_mllm.yaml"
    )

    ordinary = yaml.safe_load(ordinary_runtime_path.read_text(encoding="utf-8"))
    v3 = yaml.safe_load(v3_runtime_path.read_text(encoding="utf-8"))

    for key in (
        "portal_standoff_m",
        "interaction_safety_margin_m",
        "interaction_ready_distance_m",
        "interaction_ready_yaw_tolerance_rad",
        "container_require_same_room",
        "container_allow_connected_room",
        "container_action_standoff_m",
        "drawer_action_standoff_m",
        "fridge_action_standoff_m",
        "container_m1_capture_standoff_m",
        "drawer_m1_capture_standoff_m",
        "fridge_m1_capture_standoff_m",
        "container_navigation_anchor_outer_offset_m",
        "container_navigation_anchor_ring_count",
        "container_navigation_anchor_tangent_offset_m",
        "drawer_navigation_anchor_aabb_fan_enabled",
        "drawer_navigation_anchor_fan_clearances_m",
        "drawer_navigation_anchor_fan_angles_deg",
        "container_two_stage_observation_max_attempts",
        "container_m1_same_pose_samples_per_view",
        "container_m1_max_viewpoints",
        "container_m1_max_total_requests",
        "container_m1_front_axis_from_capture",
        "container_safe_staging_arrival_tolerance_m",
        "container_interaction_ready_distance_m",
        "container_action_lateral_offset_m",
    ):
        assert v3["candidate"][key] == ordinary["candidate"][key]
    for key in (
        "timeout_s",
        "max_tokens",
        "consecutive_timeout_limit",
        "timeout_cooldown_s",
        "skill_timeout_s",
        "skill_max_output_tokens",
        "verification_timeout_s",
        "verification_max_output_tokens",
    ):
        assert v3["model"][key] == ordinary["model"][key]
    for key in (
        "interaction_observation_timeout_s",
        "interaction_observation_same_pose_samples_per_view",
        "interaction_observation_max_viewpoints",
        "interaction_observation_max_total_requests",
        "container_m1_capture_yaw_tolerance_rad",
        "container_m1_distinct_view_arrival_tolerance_m",
        "container_m1_distinct_view_arrival_yaw_tolerance_rad",
        "container_m1_capture_dwa_profile_enabled",
        "container_m1_capture_dwa_reconfigure_server",
        "container_m1_capture_dwa_reconfigure_timeout_s",
        "container_m1_capture_dwa_terminal_yaw_settle_max_task_steps",
        "container_m1_capture_successor_quiescence_timeout_s",
        "move_base_successor_quiescence_timeout_s",
        "container_outer_staging_abort_replan_global_costmap_wait_s",
        "container_outer_staging_abort_replan_global_costmap_poll_interval_s",
        "container_outer_staging_abort_replan_plan_retry_window_s",
        "container_outer_staging_abort_replan_plan_retry_interval_s",
        "container_outer_staging_abort_replan_plan_health_confirmations",
        "container_outer_staging_abort_replan_plan_health_interval_s",
        "container_two_stage_fallback_max_attempts",
        "container_inner_corridor_enabled",
        "container_inner_corridor_segment_m",
        "container_inner_corridor_arrival_tolerance_m",
        "container_inner_corridor_max_segments",
        "container_pre_action_confirmation_count",
        "rear_goal_enter_angle_rad",
        "rear_goal_prerotate_max_control_steps",
        "navigation_failure_recovery_enabled",
        "navigation_failure_recovery_max_attempts",
        "navigation_stagnation_timeout_task_steps",
        "navigation_max_task_steps",
        "interaction_navigation_max_task_steps",
        "navigation_step_sync_stall_timeout_s",
        "navigation_stagnation_post_rotation_grace_s",
        "interaction_final_align_enabled",
        "interaction_final_align_max_distance_m",
        "interaction_final_align_yaw_tolerance_rad",
        "interaction_dwa_terminal_yaw_settle_max_task_steps",
    ):
        assert v3["executor"][key] == ordinary["executor"][key]


def test_v3_effective_interaction_profile_has_only_evaluator_boundary_differences() -> None:
    """Prevent silent algorithm drift outside the sealed V3 evaluator seam."""

    def deep_merge(base: dict, overlay: dict) -> dict:
        result = dict(base)
        for key, value in overlay.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    default = yaml.safe_load(
        (
            REPO_ROOT
            / "Interactive-Nav-SG-nav"
            / "src"
            / "semantic_decision_py_pkg"
            / "config"
            / "default.yaml"
        ).read_text(encoding="utf-8")
    )
    config_root = (
        REPO_ROOT / "scripts" / "InteractiveNav" / "configs" / "semantic_decision"
    )
    ordinary = deep_merge(
        default,
        yaml.safe_load(
            (config_root / "full_mllm_interactive_exploration.yaml").read_text(
                encoding="utf-8"
            )
        ),
    )
    v3 = deep_merge(
        default,
        yaml.safe_load(
            (config_root / "object_goal_v3_full_mllm.yaml").read_text(encoding="utf-8")
        ),
    )
    allowed = {
        "candidate": {
            # Restricted public M1 evidence is fail-closed, and object-goal
            # candidates may traverse a connected room.
            "portal_allow_unknown_state",
            "portal_require_attribute_ready",
            "target_allow_connected_room",
        },
        "policy": {
            # The evaluator owns one causal post-open barrier and therefore
            # disables the ordinary asynchronous duplicate refresh path.
            "direct_atomic_outcome_belief_enabled",
            "post_interaction_refresh_enabled",
        },
        "model": {
            # V3 resolves the deployed model only from its secure env file.
            "model",
        },
        "completion": set(),
        "executor": {
            # Public commands cross the opaque evaluator seam; a loaded batch
            # gets a longer result-delivery timeout without changing force.
            "evaluator_opaque_open_only",
            "interaction_timeout_s",
        },
    }
    for section, allowed_keys in allowed.items():
        ordinary_section = ordinary.get(section, {})
        v3_section = v3.get(section, {})
        actual = {
            key
            for key in set(ordinary_section) | set(v3_section)
            if ordinary_section.get(key, object()) != v3_section.get(key, object())
        }
        assert actual == allowed_keys


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
