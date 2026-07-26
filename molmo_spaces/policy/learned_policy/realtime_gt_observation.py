from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


def gather_joint_info(*args, **kwargs):
    from molmo_spaces.utils.articulation_utils import gather_joint_info as implementation

    return implementation(*args, **kwargs)


def body_aabb(*args, **kwargs):
    from molmo_spaces.utils.mj_model_and_data_utils import body_aabb as implementation

    return implementation(*args, **kwargs)


def _joint_type_name(joint_type: Any) -> str:
    text = str(joint_type).lower()
    if "hinge" in text:
        return "hinge"
    if "slide" in text:
        return "slide"
    try:
        numeric_type = int(np.asarray(joint_type).reshape(-1)[0])
    except (IndexError, TypeError, ValueError):
        return "none"
    if numeric_type == int(mujoco.mjtJoint.mjJNT_HINGE):
        return "hinge"
    if numeric_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
        return "slide"
    return "none"


def _safe_body_aabb(model, data, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        return body_aabb(model, data, body_id, visual_only=True)
    except Exception:
        return data.xpos[body_id].copy(), np.zeros(3, dtype=np.float64)


def _quat_xyzw(rotation: np.ndarray) -> list[float]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    return [float(qx), float(qy), float(qz), float(qw)]


def _joint_closed_open_values(joint_range: list[float]) -> tuple[float, float]:
    lower, upper = float(joint_range[0]), float(joint_range[1])
    closed = 0.0 if lower <= 0.0 <= upper else min((lower, upper), key=abs)
    opened = lower if abs(lower - closed) >= abs(upper - closed) else upper
    return closed, opened


def _joint_axis_world(model, data, joint_id: int) -> np.ndarray | None:
    if hasattr(data, "xaxis"):
        axis = np.asarray(data.xaxis[joint_id], dtype=np.float64)
    else:
        body_id = int(model.jnt_bodyid[joint_id])
        rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        axis = rotation @ np.asarray(model.jnt_axis[joint_id], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        return None
    return axis / norm


def _aligned_mean_axis(axes: list[np.ndarray]) -> list[float] | None:
    if not axes:
        return None
    reference = axes[0]
    aligned = [axis if float(np.dot(axis, reference)) >= 0.0 else -axis for axis in axes]
    mean_axis = np.mean(aligned, axis=0)
    norm = float(np.linalg.norm(mean_axis))
    if norm <= 1e-8:
        return [float(reference[0]), float(reference[1])]
    mean_axis /= norm
    return [float(mean_axis[0]), float(mean_axis[1])]


def _interaction_approach_axis_xy(model, data, joint_infos: list[dict[str, Any]]) -> list[float] | None:
    slide_axes: list[np.ndarray] = []
    hinge_axes: list[np.ndarray] = []
    for joint_info in joint_infos:
        joint_name = str(joint_info.get("joint_name") or "")
        joint_type = str(joint_info.get("joint_type") or "none")
        joint_range = list(joint_info.get("joint_range") or [0.0, 0.0])
        if not joint_name or joint_type not in {"hinge", "slide"} or len(joint_range) < 2:
            continue
        try:
            joint_id = int(model.joint(joint_name).id)
        except (KeyError, TypeError, ValueError):
            continue
        axis_world = _joint_axis_world(model, data, joint_id)
        if axis_world is None:
            continue
        closed_value, open_value = _joint_closed_open_values(joint_range)
        opening_delta = float(open_value - closed_value)
        if abs(opening_delta) <= 1e-8:
            continue
        opening_sign = 1.0 if opening_delta > 0.0 else -1.0
        if joint_type == "slide":
            axis_xy = opening_sign * axis_world[:2]
            norm = float(np.linalg.norm(axis_xy))
            if norm > 1e-6:
                slide_axes.append(axis_xy / norm)
            continue
        if not hasattr(data, "xanchor"):
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        body_center, _ = _safe_body_aabb(model, data, body_id)
        radial = np.asarray(body_center, dtype=np.float64) - np.asarray(
            data.xanchor[joint_id], dtype=np.float64
        )
        current_value = float(joint_info.get("joint_value", closed_value) or 0.0)
        angle_to_closed = -(current_value - closed_value)
        cosine = float(np.cos(angle_to_closed))
        sine = float(np.sin(angle_to_closed))
        closed_radial = (
            radial * cosine
            + np.cross(axis_world, radial) * sine
            + axis_world * float(np.dot(axis_world, radial)) * (1.0 - cosine)
        )
        tangent_xy = np.cross(opening_sign * axis_world, closed_radial)[:2]
        norm = float(np.linalg.norm(tangent_xy))
        if norm > 1e-6:
            hinge_axes.append(tangent_xy / norm)
    return _aligned_mean_axis(slide_axes or hinge_axes)


def _bbox_area(bbox: list[float] | list[int]) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0]) + 1.0) * max(
        0.0, float(bbox[3]) - float(bbox[1]) + 1.0
    )


def _project_aabb_bbox(
    camera_position: np.ndarray,
    camera_forward: np.ndarray,
    camera_up: np.ndarray,
    fov_deg: float,
    image_size: list[int],
    center: np.ndarray,
    size: np.ndarray,
) -> list[float] | None:
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 1 or height <= 1:
        return None
    forward = np.asarray(camera_forward, dtype=np.float64)
    up = np.asarray(camera_up, dtype=np.float64)
    forward_norm = float(np.linalg.norm(forward))
    up_norm = float(np.linalg.norm(up))
    if forward_norm <= 1e-8 or up_norm <= 1e-8:
        return None
    forward /= forward_norm
    up /= up_norm
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-8:
        return None
    right /= right_norm
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1e-8)
    half = 0.5 * np.abs(np.asarray(size, dtype=np.float64))
    offsets = np.asarray(
        [
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    relative = np.asarray(center, dtype=np.float64)[None, :] + offsets - np.asarray(
        camera_position, dtype=np.float64
    )[None, :]
    depth = relative @ forward
    valid = depth > 1e-3
    if int(np.count_nonzero(valid)) < 2:
        return None
    relative = relative[valid]
    depth = depth[valid]
    fov_rad = np.deg2rad(min(179.0, max(1.0, float(fov_deg))))
    focal = 0.5 * float(height) / max(np.tan(0.5 * fov_rad), 1e-6)
    xs = 0.5 * float(width - 1) + (relative @ right) / depth * focal
    ys = 0.5 * float(height - 1) - (relative @ up) / depth * focal
    min_x = max(0.0, float(np.min(xs)))
    min_y = max(0.0, float(np.min(ys)))
    max_x = min(float(width - 1), float(np.max(xs)))
    max_y = min(float(height - 1), float(np.max(ys)))
    if max_x < min_x or max_y < min_y:
        return None
    return [min_x, min_y, max_x, max_y]


def _visible_fraction(
    bbox_2d: list[int],
    camera_position: np.ndarray,
    camera_forward: np.ndarray,
    camera_up: np.ndarray,
    fov_deg: float,
    image_size: list[int],
    center: np.ndarray,
    size: np.ndarray,
) -> tuple[float, list[float] | None]:
    projected_bbox = _project_aabb_bbox(
        camera_position,
        camera_forward,
        camera_up,
        fov_deg,
        image_size,
        center,
        size,
    )
    projected_area = _bbox_area(projected_bbox or [])
    if projected_area <= 1e-6:
        return 0.0, projected_bbox
    return min(1.0, _bbox_area(bbox_2d) / projected_area), projected_bbox


@dataclass
class _ObjectSpec:
    source_name: str
    metadata: dict[str, Any]
    body_id: int
    joint_names: tuple[str, ...]
    is_door: bool
    is_receptacle: bool
    is_articulable: bool
    is_pickup_candidate: bool
    parent_source_name: str = ""


class RealtimeGTObservationPublisher:
    def __init__(
        self,
        rospy_module,
        string_message_type,
        topic: str = "/semantic_mapping/gt_observations",
        camera_name: str = "head_camera",
        min_visible_pixels: int = 16,
        min_visible_fraction: float = 0.2,
        required_consecutive_observations: int = 2,
        max_distance_m: float = 6.0,
        step_interval: int = 3,
        queue_size: int = 1,
        async_processing: bool = True,
    ) -> None:
        self._rospy = rospy_module
        self._String = string_message_type
        self.topic = str(topic)
        self.camera_name = str(camera_name)
        self.min_visible_pixels = max(1, int(min_visible_pixels))
        self.min_visible_fraction = min(1.0, max(0.0, float(min_visible_fraction)))
        self.required_consecutive_observations = max(
            1, int(required_consecutive_observations)
        )
        self.max_distance_m = max(0.0, float(max_distance_m))
        self.step_interval = max(1, int(step_interval))
        self.publisher = self._rospy.Publisher(self.topic, self._String, queue_size=queue_size)
        self.episode_index = 0
        self.episode_id = ""
        self.frame_index = 0
        self.next_instance_index = 1
        self.instance_ids: dict[str, str] = {}
        self._qualified_streaks: dict[str, int] = {}
        self._episode_reset_pending = True
        self._cache_model_identity: int | None = None
        self._specs: list[_ObjectSpec] = []
        self._geom_to_spec = np.empty(0, dtype=np.int32)
        self._async_processing = bool(async_processing)
        self._publish_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker = None
        self.dropped_payload_count = 0
        if self._async_processing:
            self._worker = threading.Thread(target=self._publish_worker, name="realtime-gt-publisher", daemon=True)
            self._worker.start()

    def reset(self) -> None:
        self.episode_index += 1
        self.episode_id = f"episode_{self.episode_index:06d}"
        self.frame_index = 0
        self.next_instance_index = 1
        self.instance_ids.clear()
        self._qualified_streaks.clear()
        self._episode_reset_pending = True
        self._cache_model_identity = None
        self._specs = []
        self._geom_to_spec = np.empty(0, dtype=np.int32)
        self._clear_queue()

    def close(self) -> None:
        if self._worker is None:
            return
        self._stop_event.set()
        self._clear_queue()
        try:
            self._publish_queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=2.0)
        self._worker = None

    def publish(self, task, stamp=None, step_index: int | None = None, force: bool = False) -> dict[str, Any] | None:
        if task is None or getattr(task, "env", None) is None:
            return None
        capture_step = int(self.frame_index if step_index is None else step_index)
        if not force and not self._episode_reset_pending and capture_step % self.step_interval != 0:
            return None
        env = task.env
        if self.camera_name not in env.camera_manager.registry:
            self._rospy.logwarn_throttle(2.0, "RealtimeGTObservationPublisher: camera %s not found", self.camera_name)
            return None
        self._ensure_cache(env)
        render_t0 = time.perf_counter()
        try:
            segmentation = np.asarray(env.render_segmentation_frame(self.camera_name))[..., :2]
        except Exception as exc:
            self._rospy.logwarn_throttle(2.0, "Realtime GT segmentation failed: %s", exc)
            return None
        render_ms = (time.perf_counter() - render_t0) * 1000.0

        snapshot_t0 = time.perf_counter()
        visible = self._visible_instances(segmentation)
        model = env.current_model
        data = env.current_data
        camera = env.camera_manager.registry[self.camera_name]
        camera_position = np.asarray(camera.pos, dtype=np.float64).copy()
        camera_forward = np.asarray(camera.forward, dtype=np.float64).copy()
        camera_up = np.asarray(camera.up, dtype=np.float64).copy()
        image_size = [int(segmentation.shape[1]), int(segmentation.shape[0])]
        observations = []
        qualified_sources = set()
        for spec_index, visible_pixels, bbox_2d in visible:
            spec = self._specs[spec_index]
            position = np.asarray(data.xpos[spec.body_id], dtype=np.float64).copy()
            distance_m = float(np.linalg.norm(position - camera_position))
            if self.max_distance_m > 0.0 and distance_m > self.max_distance_m:
                continue
            center, size = _safe_body_aabb(model, data, spec.body_id)
            visible_fraction, projected_bbox_2d = _visible_fraction(
                bbox_2d,
                camera_position,
                camera_forward,
                camera_up,
                float(camera.fov),
                image_size,
                center,
                size,
            )
            if visible_fraction <= self.min_visible_fraction:
                continue
            qualified_sources.add(spec.source_name)
            streak = self._qualified_streaks.get(spec.source_name, 0) + 1
            self._qualified_streaks[spec.source_name] = streak
            if streak < self.required_consecutive_observations:
                continue
            observations.append(
                self._build_observation(
                    env,
                    spec,
                    visible_pixels,
                    bbox_2d,
                    image_size,
                    distance_m,
                    center,
                    size,
                    visible_fraction,
                    projected_bbox_2d,
                    streak,
                )
            )
        for source_name in list(self._qualified_streaks):
            if source_name not in qualified_sources:
                self._qualified_streaks.pop(source_name, None)
        capture_stamp_sec = (
            float(stamp.to_sec()) if stamp is not None and hasattr(stamp, "to_sec") else time.time()
        )
        payload = {
            "episode_id": self.episode_id,
            "episode_reset": bool(self._episode_reset_pending),
            "frame_index": int(self.frame_index),
            "capture_step": capture_step,
            "camera_name": self.camera_name,
            "camera_pose_world": {
                "position": camera_position.tolist(),
                "forward": camera_forward.tolist(),
                "up": camera_up.tolist(),
                "fov_deg": float(camera.fov),
            },
            "stamp_sec": capture_stamp_sec,
            "capture_stamp_sec": capture_stamp_sec,
            "source_mode": "realtime_gt_observation",
            "observation_performed": True,
            "image_size": image_size,
            "observations": observations,
            "performance": {
                "gt_render_ms": render_ms,
                "gt_snapshot_ms": (time.perf_counter() - snapshot_t0) * 1000.0,
                "visible_instance_count": len(visible),
                "qualified_instance_count": len(qualified_sources),
                "published_object_count": len(observations),
                "dropped_payload_count": int(self.dropped_payload_count),
            },
        }
        self._submit_payload(payload)
        self._episode_reset_pending = False
        self.frame_index += 1
        return payload

    def _ensure_cache(self, env) -> None:
        model = env.current_model
        if self._cache_model_identity == id(model):
            return
        object_manager = env.object_managers[env.current_batch_index]
        objects_meta = dict((env.current_scene_metadata or {}).get("objects", {}) or {})
        entries: list[tuple[str, dict[str, Any], bool]] = []
        seen = set()
        for source_name, metadata in objects_meta.items():
            try:
                model.body(source_name)
            except KeyError:
                continue
            entries.append((str(source_name), dict(metadata or {}), False))
            seen.add(str(source_name))
        try:
            door_names = object_manager.find_door_names()
        except Exception:
            door_names = []
        for door_name in door_names:
            if str(door_name) not in seen:
                entries.append((str(door_name), {"category": "Door", "object_id": door_name}, True))

        specs = []
        body_to_spec = {}
        for source_name, metadata, force_door in entries:
            body_id = int(model.body(source_name).id)
            category = "Door" if force_door else metadata.get("category") or source_name
            joint_names = self._joint_names(model, body_id, metadata)
            is_door = bool(force_door or "door" in str(category).lower() or "door" in source_name.lower())
            try:
                is_receptacle = bool(object_manager.has_receptacle_site(source_name))
            except Exception:
                is_receptacle = False
            try:
                is_pickup_candidate = bool(object_manager.has_free_joint(source_name))
            except Exception:
                is_pickup_candidate = False
            try:
                is_articulable = bool(object_manager.is_object_articulable(source_name))
            except Exception:
                is_articulable = bool(joint_names)
            body_to_spec[body_id] = len(specs)
            specs.append(
                _ObjectSpec(
                    source_name=source_name,
                    metadata=metadata,
                    body_id=body_id,
                    joint_names=joint_names,
                    is_door=is_door,
                    is_receptacle=is_receptacle,
                    is_articulable=is_articulable,
                    is_pickup_candidate=is_pickup_candidate,
                )
            )

        geom_to_spec = np.full(int(model.ngeom), -1, dtype=np.int32)
        canonical_door_specs = self._canonical_door_root_specs(model, specs)
        door_body_to_spec = {}
        for door_name in door_names:
            try:
                door_body_id = int(model.body(door_name).id)
            except KeyError:
                continue
            root_id = int(model.body_rootid[door_body_id])
            spec_index = canonical_door_specs.get(root_id)
            if spec_index is None:
                spec_index = body_to_spec.get(door_body_id)
            if spec_index is not None:
                door_body_to_spec[door_body_id] = int(spec_index)
        for geom_id in range(int(model.ngeom)):
            body_id = int(model.geom_bodyid[geom_id])
            door_spec = self._door_spec_for_body(
                model, body_id, door_body_to_spec
            )
            if door_spec is not None:
                geom_to_spec[geom_id] = door_spec
                continue
            while body_id >= 0:
                spec_index = body_to_spec.get(body_id)
                if spec_index is not None:
                    geom_to_spec[geom_id] = spec_index
                    break
                parent_id = int(model.body_parentid[body_id])
                if parent_id == body_id:
                    break
                body_id = parent_id
        self._specs = specs
        body_to_source = {spec.body_id: spec.source_name for spec in specs}
        for spec in self._specs:
            parent_body_id = int(model.body_parentid[spec.body_id])
            visited = {spec.body_id}
            while parent_body_id >= 0 and parent_body_id not in visited:
                visited.add(parent_body_id)
                parent_source_name = body_to_source.get(parent_body_id)
                if parent_source_name:
                    spec.parent_source_name = parent_source_name
                    break
                next_parent_body_id = int(model.body_parentid[parent_body_id])
                if next_parent_body_id == parent_body_id:
                    break
                parent_body_id = next_parent_body_id
        self._geom_to_spec = geom_to_spec
        self._cache_model_identity = id(model)

    @staticmethod
    def _canonical_door_root_specs(model, specs: list[_ObjectSpec]) -> dict[int, int]:
        """Map an articulated doorway root to its single root-level GT spec."""
        result = {}
        for spec_index, spec in enumerate(specs):
            root_id = int(model.body_rootid[spec.body_id])
            if (
                spec.is_door
                and spec.is_articulable
                and int(spec.body_id) == root_id
            ):
                result[root_id] = int(spec_index)
        return result

    @staticmethod
    def _door_spec_for_body(model, body_id: int, door_body_to_spec: dict[int, int]):
        """Return the door spec only for an explicit door body hierarchy.

        A MuJoCo articulated root can contain non-door siblings.  Mapping every
        geom sharing that root to the door inflated door masks across rooms.
        """
        current = int(body_id)
        visited = set()
        while current >= 0 and current not in visited:
            if current in door_body_to_spec:
                return int(door_body_to_spec[current])
            visited.add(current)
            parent = int(model.body_parentid[current])
            if parent == current:
                break
            current = parent
        return None

    def _visible_instances(self, segmentation: np.ndarray) -> list[tuple[int, int, list[int]]]:
        if not self._specs or self._geom_to_spec.size == 0:
            return []
        geom_mask = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        ys, xs = np.nonzero(geom_mask)
        if ys.size == 0:
            return []
        geom_ids = segmentation[..., 0][geom_mask].astype(np.int64, copy=False)
        valid_geom = (geom_ids >= 0) & (geom_ids < self._geom_to_spec.size)
        ys = ys[valid_geom]
        xs = xs[valid_geom]
        spec_indices = self._geom_to_spec[geom_ids[valid_geom]]
        valid_spec = spec_indices >= 0
        ys = ys[valid_spec]
        xs = xs[valid_spec]
        spec_indices = spec_indices[valid_spec]
        if spec_indices.size == 0:
            return []
        counts = np.bincount(spec_indices, minlength=len(self._specs))
        min_x = np.full(len(self._specs), segmentation.shape[1], dtype=np.int32)
        min_y = np.full(len(self._specs), segmentation.shape[0], dtype=np.int32)
        max_x = np.full(len(self._specs), -1, dtype=np.int32)
        max_y = np.full(len(self._specs), -1, dtype=np.int32)
        np.minimum.at(min_x, spec_indices, xs)
        np.minimum.at(min_y, spec_indices, ys)
        np.maximum.at(max_x, spec_indices, xs)
        np.maximum.at(max_y, spec_indices, ys)
        result = []
        for spec_index in np.flatnonzero(counts >= self.min_visible_pixels):
            result.append(
                (
                    int(spec_index),
                    int(counts[spec_index]),
                    [int(min_x[spec_index]), int(min_y[spec_index]), int(max_x[spec_index]), int(max_y[spec_index])],
                )
            )
        return result

    def _build_observation(
        self,
        env,
        spec: _ObjectSpec,
        visible_pixels: int,
        bbox_2d: list[int],
        image_size: list[int],
        distance_m: float,
        center: np.ndarray,
        size: np.ndarray,
        visible_fraction: float,
        projected_bbox_2d: list[float] | None,
        consecutive_observations: int,
    ) -> dict[str, Any]:
        model = env.current_model
        data = env.current_data
        joint_infos = self._joint_infos(model, data, spec.joint_names)
        primary_joint = max(
            joint_infos,
            key=lambda item: abs(float(item["joint_range"][1]) - float(item["joint_range"][0])),
            default={"joint_name": "", "joint_type": "none", "joint_range": [0.0, 0.0], "joint_value": None},
        )
        instance_id = self.instance_ids.get(spec.source_name)
        if instance_id is None:
            instance_id = f"gt_{self.next_instance_index:06d}"
            self.next_instance_index += 1
            self.instance_ids[spec.source_name] = instance_id
        metadata = spec.metadata
        category = "Door" if spec.is_door else metadata.get("category") or spec.source_name
        interaction_approach_axis_xy = None
        if spec.is_receptacle and spec.is_articulable:
            interaction_approach_axis_xy = _interaction_approach_axis_xy(
                model, data, joint_infos
            )
        return {
            "observation_id": f"{self.episode_id}_frame_{self.frame_index:06d}_{instance_id}",
            "instance_id": instance_id,
            "source_object_name": spec.source_name,
            "name": spec.source_name,
            "object_id": metadata.get("object_id"),
            "asset_id": metadata.get("asset_id"),
            "semantic_name": str(category),
            "category": str(category),
            "parent": spec.parent_source_name or None,
            "confidence": 1.0,
            "position": [float(value) for value in data.xpos[spec.body_id]],
            "orientation": _quat_xyzw(data.xmat[spec.body_id]),
            "interaction_approach_axis_xy": interaction_approach_axis_xy,
            "aabb_center": [float(value) for value in center],
            "aabb_size": [float(value) for value in size],
            "distance_m": float(distance_m),
            "is_door": spec.is_door,
            "is_movable_door": bool(spec.is_door and spec.is_articulable),
            "is_receptacle": spec.is_receptacle,
            "is_articulable": spec.is_articulable,
            "is_pickup_candidate": spec.is_pickup_candidate,
            "joint_infos": joint_infos,
            "primary_joint_name": primary_joint.get("joint_name", ""),
            "joint_type": primary_joint.get("joint_type", "none"),
            "joint_range": list(primary_joint.get("joint_range") or [0.0, 0.0]),
            "joint_value": primary_joint.get("joint_value"),
            "visible_pixels": int(visible_pixels),
            "visible_fraction": float(visible_fraction),
            "projected_bbox_2d": list(projected_bbox_2d or []),
            "consecutive_observations": int(consecutive_observations),
            "bbox_2d": list(bbox_2d),
            "image_size": list(image_size),
            "camera_name": self.camera_name,
            "frame_index": int(self.frame_index),
            "episode_id": self.episode_id,
            "source": "realtime_gt",
        }

    @staticmethod
    def _joint_names(model, body_id: int, metadata: dict[str, Any]) -> tuple[str, ...]:
        names = set((metadata.get("name_map", {}).get("joints", {}) or {}).keys())
        root_id = int(model.body_rootid[body_id])
        for joint_id in range(int(model.njnt)):
            joint_body_id = int(model.jnt_bodyid[joint_id])
            if int(model.body_rootid[joint_body_id]) == root_id:
                joint_name = model.joint(joint_id).name
                if joint_name:
                    names.add(str(joint_name))
        return tuple(sorted(names))

    @staticmethod
    def _joint_infos(model, data, joint_names: tuple[str, ...]) -> list[dict[str, Any]]:
        result = []
        for joint_name in joint_names:
            try:
                info = gather_joint_info(model, data, joint_name)
            except Exception:
                continue
            joint_type = _joint_type_name(info.get("joint_type"))
            if joint_type not in {"hinge", "slide"}:
                continue
            result.append(
                {
                    "joint_name": joint_name,
                    "joint_type": joint_type,
                    "joint_range": [float(value) for value in info.get("joint_range", [0.0, 0.0])],
                    "joint_value": float(info.get("joint_pos", 0.0)),
                }
            )
        return result

    def _submit_payload(self, payload: dict[str, Any]) -> None:
        if not self._async_processing:
            self._publish_payload(payload)
            return
        try:
            self._publish_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._publish_queue.get_nowait()
                self.dropped_payload_count += 1
            except queue.Empty:
                pass
            self._publish_queue.put_nowait(payload)

    def _publish_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._publish_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if payload is None:
                break
            self._publish_payload(payload)

    def _publish_payload(self, payload: dict[str, Any]) -> None:
        publish_t0 = time.perf_counter()
        payload["publish_stamp_sec"] = time.time()
        payload["processing_latency_ms"] = max(
            0.0, (payload["publish_stamp_sec"] - float(payload["capture_stamp_sec"])) * 1000.0
        )
        payload["performance"]["gt_publish_ms"] = (time.perf_counter() - publish_t0) * 1000.0
        self.publisher.publish(self._String(data=json.dumps(payload, separators=(",", ":"))))

    def _clear_queue(self) -> None:
        while True:
            try:
                self._publish_queue.get_nowait()
            except queue.Empty:
                return
