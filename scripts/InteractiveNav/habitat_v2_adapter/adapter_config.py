"""Strict configuration profile for the Habitat ObjectNav-v2 bridge.

The profile intentionally describes only the adapter boundary.  It never
modifies Habitat's official task YAML and it never configures the original ROS
interactive-execution stack.  The optional ObjectGoal Module-3 verifier has a
closed STOP/CONTINUE/RESCAN vocabulary and remains separate from interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml


class AdapterProfileError(ValueError):
    """Raised when a profile would violate the public navigation-only boundary."""


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdapterProfileError(f"{path} must be a mapping")
    return dict(value)


def _keys(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise AdapterProfileError(f"{path} has unsupported key(s): {', '.join(unexpected)}")


def _required(value: dict[str, Any], names: set[str], path: str) -> None:
    missing = sorted(names - set(value))
    if missing:
        raise AdapterProfileError(f"{path} is missing required key(s): {', '.join(missing)}")


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise AdapterProfileError(f"{path} must be boolean")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterProfileError(f"{path} must be numeric")
    return float(value)


def _positive(value: Any, path: str, *, allow_zero: bool = False) -> float:
    number = _number(value, path)
    if number < 0.0 or (number == 0.0 and not allow_zero):
        raise AdapterProfileError(f"{path} must be {'non-negative' if allow_zero else 'positive'}")
    return number


@dataclass(frozen=True)
class HabitatV2AdapterProfile:
    """Validated profile plus enough provenance to reproduce one evaluation."""

    profile: str
    source_path: Path
    sha256: str
    raw: dict[str, Any]

    @property
    def module1(self) -> dict[str, Any]:
        return dict(self.raw["modules"]["module1"])

    @property
    def module2(self) -> dict[str, Any]:
        return dict(self.raw["modules"]["module2"])

    @property
    def module3(self) -> dict[str, Any]:
        return dict(self.raw["modules"]["module3"])

    def policy_overrides(self) -> dict[str, Any]:
        """Map profile fields to the adapter runtime dataclass only.

        Keeping this mapping here prevents the benchmark runner from accidentally
        treating a local approach tolerance as Habitat's official success radius.
        """

        module1 = self.module1
        sidecar = dict(module1["sidecar"])
        module2 = self.module2
        approach = dict(self.raw["approach"])
        stop_proxy = dict(self.raw["stop_proxy"])
        return {
            "mllm_endpoint": str(module2["endpoint"]),
            "mllm_model": str(module2["model"]),
            "mllm_timeout_s": float(module2["timeout_s"]),
            "target_goal_lock_enabled": bool(module2.get("target_goal_lock_enabled", False)),
            "target_goal_pre_score": float(module2.get("target_goal_pre_score", 1.0)),
            "module1_detector_enabled": bool(module1["enabled"] and module1["mode"] == "detector_only_sidecar"),
            "module1_detector_endpoint": str(sidecar["endpoint"]),
            "module1_detector_timeout_s": float(sidecar["timeout_s"]),
            "module1_detector_interval_steps": int(sidecar["interval_steps"]),
            "module1_detector_min_confidence": float(sidecar["min_confidence"]),
            "module1_detector_include_depth": bool(sidecar["include_depth"]),
            "module1_original_scripts": str(sidecar["original_module1_scripts"]),
            "module1_detector_require_healthy": bool(sidecar["require_healthy"]),
            "full_ros_stack_enabled": bool(module1["enabled"] and module1["mode"] == "full_ros_navigation_stack"),
            "full_ros_stack_endpoint": str(sidecar["endpoint"]),
            "full_ros_stack_timeout_s": float(sidecar["timeout_s"]),
            "full_ros_stack_interval_steps": int(sidecar["interval_steps"]),
            "full_ros_stack_require_healthy": bool(sidecar["require_healthy"]),
            "original_ros_navigation_enabled": bool(
                module1["enabled"] and module1["mode"] == "full_ros_navigation_stack"
            ),
            "mllm_visual_enabled": bool(self.raw["modules"]["module2"].get("vision_localization_enabled", False)),
            "module3_enabled": bool(self.module3["enabled"]),
            "module3_fail_closed": bool(self.module3["fail_closed"]),
            "module3_mode": str(self.module3["mode"]),
            "module3_stop_trigger_distance_m": float(self.module3.get("stop_trigger_distance_m", 0.25)),
            "module3_detector_confirmations": int(self.module3.get("detector_confirmations", 1)),
            "module3_min_confidence": float(self.module3.get("min_confidence", 0.80)),
            "module3_max_bbox_center_distance_m": float(
                self.module3.get("max_bbox_center_distance_m", 0.85)
            ),
            "module3_missing_target_max_steps": int(self.module3.get("missing_target_max_steps", 18)),
            "module3_max_verification_steps": int(self.module3.get("max_verification_steps", 48)),
            "module3_semantic_rejection_limit": int(self.module3.get("semantic_rejection_limit", 3)),
            "allowed_behavior_types": tuple(str(item) for item in self.module3["allowed_behavior_types"]),
            "vision_min_depth_m": float(approach["min_depth_m"]),
            "vision_navigate_max_depth_m": float(approach["max_target_depth_m"]),
            "vision_waypoint_standoff_m": float(approach["waypoint_standoff_m"]),
            "vision_waypoint_reached_distance_m": float(approach["waypoint_reached_distance_m"]),
            "visual_goal_max_contiguous_steps": int(approach["max_contiguous_steps"]),
            "target_track_standoff_m": float(approach["target_track_standoff_m"]),
            "target_track_standoff_tolerance_m": float(
                approach.get("target_track_standoff_tolerance_m", 0.10)
            ),
            "goal_reached_distance_m": float(approach["route_goal_reached_distance_m"]),
            "ros_goal_terminal_grace_steps": int(approach.get("ros_goal_terminal_grace_steps", 3)),
            "target_standoff_failure_radius_m": float(
                approach.get("target_standoff_failure_radius_m", 0.35)
            ),
            "target_standoff_failure_cooldown_steps": int(
                approach.get("target_standoff_failure_cooldown_steps", 500)
            ),
            "target_standoff_retry_angles_deg": tuple(
                float(value)
                for value in approach.get("target_standoff_retry_angles_deg", [35.0, -35.0, 70.0, -70.0])
            ),
            "vision_stop_enabled": bool(stop_proxy["enabled"]),
            "vision_stop_depth_m": float(stop_proxy["depth_m"]),
            "vision_stop_center_tolerance": float(stop_proxy["center_tolerance"]),
            "vision_stop_confirmations": int(stop_proxy["confirmations"]),
            "persistent_target_tracking": bool(
                module1["enabled"]
                and module1["mode"] in {"detector_only_sidecar", "full_ros_navigation_stack"}
            ),
        }

    def public_summary(self) -> dict[str, Any]:
        """Return configuration provenance without embedding source-code paths."""

        return {
            "profile": self.profile,
            "schema_version": int(self.raw["schema_version"]),
            "source_path": str(self.source_path),
            "sha256": self.sha256,
            "module1": {
                "enabled": bool(self.module1["enabled"]),
                "mode": str(self.module1["mode"]),
                "transport": str(self.module1["sidecar"]["transport"]),
            },
            "module2": {
                "enabled": bool(self.module2["enabled"]),
                "mode": str(self.module2["mode"]),
                "target_goal_lock_enabled": bool(self.module2.get("target_goal_lock_enabled", False)),
                "target_goal_pre_score": float(self.module2.get("target_goal_pre_score", 1.0)),
            },
            "module3": {
                "enabled": bool(self.module3["enabled"]),
                "fail_closed": bool(self.module3["fail_closed"]),
                "max_bbox_center_distance_m": float(
                    self.module3.get("max_bbox_center_distance_m", 0.85)
                ),
                "allowed_behavior_types": list(self.module3["allowed_behavior_types"]),
                "forbidden_behavior_types": list(self.module3["forbidden_behavior_types"]),
                "forbidden_habitat_actions": list(self.module3["forbidden_habitat_actions"]),
            },
            "approach": {
                "target_track_standoff_m": float(self.raw["approach"]["target_track_standoff_m"]),
                "target_track_standoff_tolerance_m": float(
                    self.raw["approach"].get("target_track_standoff_tolerance_m", 0.10)
                ),
                "route_goal_reached_distance_m": float(
                    self.raw["approach"]["route_goal_reached_distance_m"]
                ),
            },
            "official_task_assertions": dict(self.raw["official_task_assertions"]),
            "stop_proxy": dict(self.raw["stop_proxy"]),
        }


def load_adapter_profile(path: Path) -> HabitatV2AdapterProfile:
    """Load a strict, detector-only Habitat-v2 integration profile."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise AdapterProfileError(f"adapter profile does not exist: {source_path}")
    text = source_path.read_text(encoding="utf-8")
    try:
        raw = _mapping(yaml.safe_load(text), "profile")
    except yaml.YAMLError as exc:
        raise AdapterProfileError(f"invalid YAML in {source_path}: {exc}") from exc
    _keys(
        raw,
        {
            "schema_version",
            "profile",
            "official_task_assertions",
            "modules",
            "approach",
            "stop_proxy",
            "logging",
        },
        "profile",
    )
    _required(
        raw,
        {"schema_version", "profile", "official_task_assertions", "modules", "approach", "stop_proxy", "logging"},
        "profile",
    )
    if raw["schema_version"] != 1:
        raise AdapterProfileError("profile.schema_version must equal 1")
    if not isinstance(raw["profile"], str) or not raw["profile"].strip():
        raise AdapterProfileError("profile.profile must be a non-empty string")

    official = _mapping(raw["official_task_assertions"], "official_task_assertions")
    _keys(official, {"habitat_config", "distance_to", "success_distance_m", "requires_explicit_stop"}, "official_task_assertions")
    _required(official, {"habitat_config", "distance_to", "success_distance_m", "requires_explicit_stop"}, "official_task_assertions")
    if official["habitat_config"] != "benchmark/nav/objectnav/objectnav_v2_hm3d_stretch.yaml":
        raise AdapterProfileError("official_task_assertions.habitat_config must be the official v2 Stretch config")
    if official["distance_to"] != "VIEW_POINTS":
        raise AdapterProfileError("official_task_assertions.distance_to must be VIEW_POINTS")
    if abs(_number(official["success_distance_m"], "official_task_assertions.success_distance_m") - 0.1) > 1e-8:
        raise AdapterProfileError("official_task_assertions.success_distance_m must remain 0.1")
    if not _bool(official["requires_explicit_stop"], "official_task_assertions.requires_explicit_stop"):
        raise AdapterProfileError("official_task_assertions.requires_explicit_stop must remain true")

    modules = _mapping(raw["modules"], "modules")
    _keys(modules, {"module1", "module2", "module3"}, "modules")
    _required(modules, {"module1", "module2", "module3"}, "modules")
    module1 = _mapping(modules["module1"], "modules.module1")
    _keys(module1, {"enabled", "mode", "sidecar"}, "modules.module1")
    _required(module1, {"enabled", "mode", "sidecar"}, "modules.module1")
    if not _bool(module1["enabled"], "modules.module1.enabled"):
        raise AdapterProfileError("modules.module1.enabled must be true for this integration profile")
    if module1["mode"] not in {"detector_only_sidecar", "full_ros_navigation_stack"}:
        raise AdapterProfileError(
            "modules.module1.mode must be detector_only_sidecar or full_ros_navigation_stack"
        )
    sidecar = _mapping(module1["sidecar"], "modules.module1.sidecar")
    _keys(
        sidecar,
        {
            "transport",
            "endpoint",
            "timeout_s",
            "interval_steps",
            "min_confidence",
            "include_depth",
            "original_module1_scripts",
            "require_healthy",
        },
        "modules.module1.sidecar",
    )
    _required(
        sidecar,
        {
            "transport",
            "endpoint",
            "timeout_s",
            "interval_steps",
            "min_confidence",
            "include_depth",
            "original_module1_scripts",
            "require_healthy",
        },
        "modules.module1.sidecar",
    )
    if sidecar["transport"] != "http_sidecar":
        raise AdapterProfileError("modules.module1.sidecar.transport must be http_sidecar on this ROS-less host")
    if not isinstance(sidecar["endpoint"], str) or not sidecar["endpoint"].startswith(("http://", "https://")):
        raise AdapterProfileError("modules.module1.sidecar.endpoint must be an HTTP URL")
    _positive(sidecar["timeout_s"], "modules.module1.sidecar.timeout_s")
    interval = _positive(sidecar["interval_steps"], "modules.module1.sidecar.interval_steps")
    if int(interval) != interval:
        raise AdapterProfileError("modules.module1.sidecar.interval_steps must be an integer")
    confidence = _number(sidecar["min_confidence"], "modules.module1.sidecar.min_confidence")
    if not 0.0 <= confidence <= 1.0:
        raise AdapterProfileError("modules.module1.sidecar.min_confidence must be in [0, 1]")
    _bool(sidecar["include_depth"], "modules.module1.sidecar.include_depth")
    _bool(sidecar["require_healthy"], "modules.module1.sidecar.require_healthy")
    scripts_path = Path(str(sidecar["original_module1_scripts"])).expanduser()
    if not scripts_path.is_dir():
        raise AdapterProfileError("modules.module1.sidecar.original_module1_scripts must be an existing directory")

    module2 = _mapping(modules["module2"], "modules.module2")
    _keys(
        module2,
        {
            "enabled",
            "mode",
            "endpoint",
            "model",
            "timeout_s",
            "vision_localization_enabled",
            "target_goal_lock_enabled",
            "target_goal_pre_score",
        },
        "modules.module2",
    )
    _required(module2, {"enabled", "mode", "endpoint", "model", "timeout_s", "vision_localization_enabled"}, "modules.module2")
    if not _bool(module2["enabled"], "modules.module2.enabled"):
        raise AdapterProfileError("modules.module2.enabled must be true")
    if module2["mode"] != "original_model_policy":
        raise AdapterProfileError("modules.module2.mode must be original_model_policy")
    if not isinstance(module2["endpoint"], str) or not module2["endpoint"].startswith(("http://", "https://")):
        raise AdapterProfileError("modules.module2.endpoint must be an HTTP URL")
    if not isinstance(module2["model"], str) or not module2["model"].strip():
        raise AdapterProfileError("modules.module2.model must be a non-empty string")
    _positive(module2["timeout_s"], "modules.module2.timeout_s")
    _bool(module2["vision_localization_enabled"], "modules.module2.vision_localization_enabled")
    if "target_goal_lock_enabled" in module2:
        _bool(module2["target_goal_lock_enabled"], "modules.module2.target_goal_lock_enabled")
    if "target_goal_pre_score" in module2:
        score = float(module2["target_goal_pre_score"])
        if not 0.0 <= score <= 1.0:
            raise AdapterProfileError("modules.module2.target_goal_pre_score must be in [0, 1]")

    module3 = _mapping(modules["module3"], "modules.module3")
    _keys(module3, {"enabled", "mode", "fail_closed", "allowed_behavior_types", "forbidden_behavior_types", "forbidden_habitat_actions", "allowed_outputs", "stop_trigger_distance_m", "detector_confirmations", "min_confidence", "max_bbox_center_distance_m", "missing_target_max_steps", "max_verification_steps", "semantic_rejection_limit"}, "modules.module3")
    _required(module3, {"enabled", "mode", "fail_closed", "allowed_behavior_types", "forbidden_behavior_types", "forbidden_habitat_actions"}, "modules.module3")
    module3_enabled = _bool(module3["enabled"], "modules.module3.enabled")
    expected_mode = "objectgoal_stop_verified" if module3_enabled else "disabled"
    if module3["mode"] != expected_mode:
        raise AdapterProfileError(f"modules.module3.mode must be {expected_mode}")
    if not _bool(module3["fail_closed"], "modules.module3.fail_closed"):
        raise AdapterProfileError("modules.module3.fail_closed must be true")
    allowed_behaviors = module3["allowed_behavior_types"]
    if not isinstance(allowed_behaviors, list) or set(allowed_behaviors) != {"EXPLORE", "NAVIGATE"}:
        raise AdapterProfileError("modules.module3.allowed_behavior_types must be exactly [EXPLORE, NAVIGATE]")
    forbidden_behaviors = module3["forbidden_behavior_types"]
    forbidden_actions = module3["forbidden_habitat_actions"]
    if not isinstance(forbidden_behaviors, list) or "INTERACT" not in forbidden_behaviors:
        raise AdapterProfileError("modules.module3.forbidden_behavior_types must include INTERACT")
    if not isinstance(forbidden_actions, list) or "INTERACT" not in forbidden_actions:
        raise AdapterProfileError("modules.module3.forbidden_habitat_actions must include INTERACT")
    if module3_enabled:
        if set(module3.get("allowed_outputs") or []) != {"STOP", "CONTINUE", "RESCAN"}:
            raise AdapterProfileError("enabled ObjectGoal Module-3 outputs must be STOP/CONTINUE/RESCAN")
        _positive(module3.get("stop_trigger_distance_m"), "modules.module3.stop_trigger_distance_m")
        confirmations = _positive(module3.get("detector_confirmations"), "modules.module3.detector_confirmations")
        if int(confirmations) != confirmations:
            raise AdapterProfileError("modules.module3.detector_confirmations must be an integer")
        confidence = _number(module3.get("min_confidence"), "modules.module3.min_confidence")
        if not 0.0 <= confidence <= 1.0:
            raise AdapterProfileError("modules.module3.min_confidence must be in [0,1]")
        _positive(
            module3.get("max_bbox_center_distance_m", 0.85),
            "modules.module3.max_bbox_center_distance_m",
        )
        for key in (
            "missing_target_max_steps",
            "max_verification_steps",
            "semantic_rejection_limit",
        ):
            value = _positive(module3.get(key), f"modules.module3.{key}")
            if int(value) != value:
                raise AdapterProfileError(f"modules.module3.{key} must be an integer")

    approach = _mapping(raw["approach"], "approach")
    _keys(
        approach,
        {
            "min_depth_m",
            "max_target_depth_m",
            "waypoint_standoff_m",
            "waypoint_reached_distance_m",
            "max_contiguous_steps",
            "target_track_standoff_m",
            "target_track_standoff_tolerance_m",
            "route_goal_reached_distance_m",
            "ros_goal_terminal_grace_steps",
            "target_standoff_failure_radius_m",
            "target_standoff_failure_cooldown_steps",
            "target_standoff_retry_angles_deg",
        },
        "approach",
    )
    _required(
        approach,
        {
            "min_depth_m",
            "max_target_depth_m",
            "waypoint_standoff_m",
            "waypoint_reached_distance_m",
            "max_contiguous_steps",
            "target_track_standoff_m",
            "route_goal_reached_distance_m",
        },
        "approach",
    )
    for name in ("min_depth_m", "max_target_depth_m", "waypoint_standoff_m", "waypoint_reached_distance_m", "target_track_standoff_m", "route_goal_reached_distance_m"):
        _positive(approach[name], f"approach.{name}", allow_zero=name == "min_depth_m")
    if "target_track_standoff_tolerance_m" in approach:
        _positive(
            approach["target_track_standoff_tolerance_m"],
            "approach.target_track_standoff_tolerance_m",
        )
    if float(approach["max_target_depth_m"]) < float(approach["min_depth_m"]):
        raise AdapterProfileError("approach.max_target_depth_m must be >= approach.min_depth_m")
    steps = _positive(approach["max_contiguous_steps"], "approach.max_contiguous_steps")
    if int(steps) != steps:
        raise AdapterProfileError("approach.max_contiguous_steps must be an integer")
    if "ros_goal_terminal_grace_steps" in approach:
        value = _positive(
            approach["ros_goal_terminal_grace_steps"],
            "approach.ros_goal_terminal_grace_steps",
            allow_zero=True,
        )
        if int(value) != value:
            raise AdapterProfileError("approach.ros_goal_terminal_grace_steps must be an integer")
    if "target_standoff_failure_radius_m" in approach:
        _positive(
            approach["target_standoff_failure_radius_m"],
            "approach.target_standoff_failure_radius_m",
        )
    if "target_standoff_failure_cooldown_steps" in approach:
        value = _positive(
            approach["target_standoff_failure_cooldown_steps"],
            "approach.target_standoff_failure_cooldown_steps",
        )
        if int(value) != value:
            raise AdapterProfileError(
                "approach.target_standoff_failure_cooldown_steps must be an integer"
            )
    if "target_standoff_retry_angles_deg" in approach:
        values = approach["target_standoff_retry_angles_deg"]
        if not isinstance(values, list) or not values:
            raise AdapterProfileError("approach.target_standoff_retry_angles_deg must be a non-empty list")
        for index, value in enumerate(values):
            _number(value, f"approach.target_standoff_retry_angles_deg[{index}]")

    stop_proxy = _mapping(raw["stop_proxy"], "stop_proxy")
    _keys(stop_proxy, {"enabled", "habitat_action", "depth_m", "center_tolerance", "confirmations"}, "stop_proxy")
    _required(stop_proxy, {"enabled", "habitat_action", "depth_m", "center_tolerance", "confirmations"}, "stop_proxy")
    _bool(stop_proxy["enabled"], "stop_proxy.enabled")
    if stop_proxy["habitat_action"] != "velocity_stop":
        raise AdapterProfileError("stop_proxy.habitat_action must be velocity_stop")
    _positive(stop_proxy["depth_m"], "stop_proxy.depth_m")
    tolerance = _number(stop_proxy["center_tolerance"], "stop_proxy.center_tolerance")
    if not 0.0 <= tolerance <= 0.5:
        raise AdapterProfileError("stop_proxy.center_tolerance must be in [0, 0.5]")
    confirmations = _positive(stop_proxy["confirmations"], "stop_proxy.confirmations")
    if int(confirmations) != confirmations:
        raise AdapterProfileError("stop_proxy.confirmations must be an integer")

    logging = _mapping(raw["logging"], "logging")
    _keys(logging, {"public_trace"}, "logging")
    _required(logging, {"public_trace"}, "logging")
    _bool(logging["public_trace"], "logging.public_trace")

    return HabitatV2AdapterProfile(
        profile=str(raw["profile"]),
        source_path=source_path,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        raw=raw,
    )


def validate_habitat_v2_invariants(config: Any, profile: HabitatV2AdapterProfile) -> None:
    """Assert the composed official task was not weakened by the adapter profile."""

    expected = profile.raw["official_task_assertions"]
    try:
        task = config.habitat.task
        task_type = str(task.type)
        distance_to = str(task.measurements.distance_to_goal.distance_to)
        success_distance = float(task.measurements.success.success_distance)
        end_on_success = bool(task.end_on_success)
    except (AttributeError, TypeError, ValueError) as exc:
        raise AdapterProfileError("could not inspect composed Habitat-v2 task config") from exc
    if task_type != "ObjectNav-v2":
        raise AdapterProfileError(f"expected ObjectNav-v2 task, got {task_type!r}")
    if distance_to != expected["distance_to"]:
        raise AdapterProfileError(f"expected distance_to={expected['distance_to']}, got {distance_to}")
    if abs(success_distance - float(expected["success_distance_m"])) > 1e-8:
        raise AdapterProfileError(
            f"expected success_distance={expected['success_distance_m']}, got {success_distance}"
        )
    if bool(expected["requires_explicit_stop"]) and not end_on_success:
        raise AdapterProfileError("the official task must keep end_on_success enabled")
