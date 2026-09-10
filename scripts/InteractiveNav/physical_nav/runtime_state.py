#!/usr/bin/env python3
"""Thread-safe state shared by the physical gateway and web renderer."""

from __future__ import annotations

import copy
import base64
import math
import threading
import time
from typing import Any


def telemetry_yaw(telemetry: Any, default: float | None = None) -> float | None:
    """Return the live body yaw from a telemetry snapshot.

    Unitree packets normally carry ``yaw`` directly.  Older bridges (and a
    partially populated ROS odometry mirror) may omit that scalar while still
    providing the body IMU ``rpy`` or quaternion.  Keep the fallback here so
    every presentation path uses the same convention instead of silently
    drawing a zero/frozen heading.  Unitree's list quaternion is ``wxyz``;
    dictionary/ROS quaternions use the usual ``xyzw`` names.
    """

    if not isinstance(telemetry, dict):
        return default

    def finite(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    direct = finite(telemetry.get("yaw"))
    if direct is not None:
        return direct
    imu = telemetry.get("imu")
    if isinstance(imu, dict):
        rpy = imu.get("rpy")
        if isinstance(rpy, (list, tuple)) and len(rpy) >= 3:
            value = finite(rpy[2])
            if value is not None:
                return value
        raw = imu.get("quaternion") or imu.get("orientation")
        if isinstance(raw, dict):
            x, y, z, w = (finite(raw.get(key)) for key in ("x", "y", "z", "w"))
        elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
            # The Unitree SDK exposes imu_state.quaternion as [w, x, y, z].
            w, x, y, z = (finite(value) for value in raw[:4])
        else:
            x = y = z = w = None
        if None not in (x, y, z, w):
            return math.atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
    return default


class RuntimeState:
    # ``summary_snapshot`` feeds the browser's low-rate status endpoint.  A
    # detector receipt also contains sparse masks and packed point-cloud
    # fields, which are needed by the ROS projection but never by the status
    # cards.  Keep this allow-list beside the snapshot method so a new heavy
    # detector field cannot accidentally turn a one-Hz dashboard poll into a
    # multi-megabyte deepcopy/GIL stall.
    _SUMMARY_DETECTION_KEYS = frozenset({
        "semantic_class", "semantic_class_raw", "raw_class", "class",
        "confidence", "bbox", "bbox_2d", "mask_area", "visible_fraction",
        "depth_median_m", "distance_m", "category", "name", "label",
        "instance_id", "object_id", "target_id", "capture_seq",
        "visualization_track_id", "source_frame", "geometry_skipped",
        "camera_position", "camera_box3d_center", "camera_box3d_size",
        "position", "world_position", "world_box3d_center",
        "world_box3d_size", "world_box3d_orientation", "world_box3d_yaw",
        "world_box3d_marker_size", "yaw", "aabb_center", "aabb_size",
        "map_transform_status",
    })

    # Navigation plans are intentionally absent from the low-rate browser
    # summary.  A global/local NavPath can contain tens of thousands of pose
    # dictionaries and is only needed by the visualization snapshot.  Keep
    # the status cards' state fields, while bounding the few nested structures
    # that are useful for the candidate/timeline cards.
    _SUMMARY_NAVIGATION_KEYS = (
        "explore_status", "current_subgoal", "candidates", "selection",
        "execution_state", "behavior_feedback", "interaction_result",
        "decision_trace",
    )
    _SUMMARY_DECISION_TRACE_KEYS = (
        "timestamp", "active_candidate_id", "model_selected_candidate_id",
        "executed_candidate_id", "model_reason", "selection_override_reason",
        "model_result_source", "model_error", "phase", "event",
    )

    @classmethod
    def _summary_detections(cls, value: Any) -> list[dict[str, Any]]:
        """Project detector receipts to the fields consumed by web cards.

        Do not mutate the live receipt and do not deep-copy fields that the
        browser cannot use (masks, polygons and packed segment points can be
        hundreds of thousands of Python objects per frame).  The full
        ``snapshot()``/``/api/ros-state`` path remains unchanged for ROS and
        offline diagnostics.
        """
        if not isinstance(value, list):
            return []
        compact: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            compact.append({
                key: copy.deepcopy(item[key])
                for key in cls._SUMMARY_DETECTION_KEYS
                if key in item
            })
        return compact

    @classmethod
    def _bounded_summary_value(
        cls,
        value: Any,
        *,
        depth: int = 0,
        max_list: int = 32,
        max_dict: int = 64,
        max_string: int = 2000,
    ) -> Any:
        """Copy a small status value without traversing unbounded payloads.

        Navigation status is produced by several older ROS publishers, so an
        allow-list for every nested field would be brittle.  This bounded
        copier keeps the useful shape for the dashboard while preventing a
        debug trace or an accidental point list from reintroducing a large
        deepcopy under ``RuntimeState``'s lock.
        """
        if depth >= 4:
            if isinstance(value, (dict, list, tuple)):
                return "<truncated>"
            if isinstance(value, str):
                return value[:max_string]
            return copy.deepcopy(value)
        if isinstance(value, dict):
            return {
                key: cls._bounded_summary_value(
                    item,
                    depth=depth + 1,
                    max_list=max_list,
                    max_dict=max_dict,
                    max_string=max_string,
                )
                for key, item in list(value.items())[:max_dict]
            }
        if isinstance(value, (list, tuple)):
            return [
                cls._bounded_summary_value(
                    item,
                    depth=depth + 1,
                    max_list=max_list,
                    max_dict=max_dict,
                    max_string=max_string,
                )
                for item in list(value)[:max_list]
            ]
        if isinstance(value, str):
            return value[:max_string]
        return copy.deepcopy(value)

    @classmethod
    def _summary_navigation(cls, value: Any) -> dict[str, Any]:
        """Return the bounded navigation view used by ``/api/state-summary``.

        In particular, never copy ``global_plan``/``local_plan`` here.  The
        full plans remain available through ``snapshot()`` and the revisioned
        visualization endpoint; the one-Hz status endpoint only renders the
        active candidate and execution state.
        """
        navigation = value if isinstance(value, dict) else {}
        compact: dict[str, Any] = {}
        for key in cls._SUMMARY_NAVIGATION_KEYS:
            if key not in navigation:
                continue
            item = navigation.get(key)
            if key == "candidates" and isinstance(item, dict):
                candidate_items = item.get("candidates")
                candidate_copy = {
                    key_name: cls._bounded_summary_value(value_item)
                    for key_name, value_item in item.items()
                    if key_name != "candidates"
                }
                if isinstance(candidate_items, list):
                    candidate_copy["candidates"] = [
                        cls._bounded_summary_value(candidate)
                        for candidate in candidate_items[:20]
                    ]
                else:
                    candidate_copy["candidates"] = []
                compact[key] = candidate_copy
            elif key == "decision_trace" and isinstance(item, dict):
                compact[key] = {
                    trace_key: copy.deepcopy(item[trace_key])
                    for trace_key in cls._SUMMARY_DECISION_TRACE_KEYS
                    if trace_key in item
                }
            else:
                compact[key] = cls._bounded_summary_value(item)
        return compact

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.rgb = None
        self.depth = None
        self.rgb_b64 = ""
        self.depth_b64 = ""
        self.depth_scale = 0.001
        self.intrinsics: dict[str, Any] = {}
        self.rgb_intrinsics: dict[str, Any] = {}
        self.depth_intrinsics: dict[str, Any] = {}
        self.depth_to_color_extrinsics: dict[str, Any] = {}
        self.camera_frame = ""
        self.depth_frame = ""
        self.camera_imu: dict[str, Any] = {}
        self.capture_timing: dict[str, Any] = {}
        self.calibration: dict[str, Any] = {}
        self.frame_seq = -1
        # Local 5 Hz navigation/viewer step. Unlike the Go2 sensor sequence,
        # this starts from zero whenever the policy-side navigation restarts.
        self.navigation_step = -1
        self.frame_stamp = 0.0
        self.sync_ms = None
        self.telemetry: dict[str, Any] = {}
        # Pose arrives on two asynchronous lanes: the capture packet (10 Hz)
        # and the lower-rate telemetry mirror.  Keep the newest source stamp
        # so a delayed mirror packet cannot move the dashboard arrow back to
        # an older heading/position.
        self._telemetry_pose_stamp = float("-inf")
        # RGB-D receipts have a local monotonically increasing sequence even
        # when the Go2 and policy-host wall clocks are not synchronized.  A
        # newer capture sequence is authoritative for pose, so a temporary
        # NTP/clock rollback cannot freeze the dashboard heading until the
        # source timestamp catches up again.
        self._telemetry_capture_seq = -1
        # ``/physical_nav/odom`` mirrors the same RGB-D receipt, but its ROS
        # header stamp has no capture sequence.  Remember the latest RGB-D
        # stamp so an odom echo cannot overwrite the authoritative capture
        # yaw with a stale second-lane value.
        self._telemetry_capture_frame_stamp = float("-inf")
        # Some legacy telemetry mirrors omit ``received_at``.  Allow the
        # first such pose for backwards compatibility, but remember it so a
        # later un-stamped mirror cannot rewind a pose that arrived with a
        # capture/source timestamp.
        self._telemetry_unstamped_pose_seen = False
        # The web gateway may outlive the direct sensor ROS bridge.  A bridge
        # reconnect can reset both its local capture sequence and the Go2 wall
        # clock, so keep the transport generation alongside the pose ordering
        # keys and reset those keys when a new generation arrives.
        self._telemetry_transport_session: str | None = None
        self._telemetry_transport_connection = -1
        self._telemetry_retired_sessions: set[str] = set()
        self.detections: list[dict[str, Any]] = []
        self.detection_meta: dict[str, Any] = {}
        self.mapped_detections: list[dict[str, Any]] = []
        self.mapped_detection_meta: dict[str, Any] = {}
        self.graph: dict[str, Any] = {}
        self.occupancy = None
        # Additional raw grids mirror the recorder's map stages.  They remain
        # in-memory for the live renderer; /api/state exposes metadata only.
        self.room_grid = None
        self.global_costmap = None
        self.local_costmap = None
        self.consistency: dict[str, Any] = {}
        self.qwen: dict[str, Any] = {"requests": [], "results": []}
        # Native M1/M2 model traces and M3 interaction-result evaluations are
        # retained as a small bounded history for the live dashboard.
        self.mllm_events: list[dict[str, Any]] = []
        self.m1_input_images: dict[str, bytes] = {}
        self.navigation: dict[str, Any] = {
            "explore_status": {}, "current_subgoal": {}, "candidates": {},
            "selection": {}, "execution_state": {}, "behavior_feedback": {},
            "interaction_result": {}, "decision_trace": {},
        }
        self.link: dict[str, Any] = {
            "connected": False,
            "connections": 0,
            "last_hello": {},
            "last_packet_at": 0.0,
            "last_disconnect_at": 0.0,
        }
        self.counters = {"frames": 0, "detections": 0, "graph_updates": 0, "dropped": 0}
        # Monotonic revisions let renderers reuse expensive map/graph products
        # when only the camera frame changed.  Keeping these counters here is
        # cheaper than hashing megabyte-sized OccupancyGrid payloads on every
        # six-panel tick.
        self.revision = 0
        self.map_revision = 0
        # ``map_revision`` covers all map-stage receipts (OCC, room grid and
        # costmaps).  Keep a narrower counter for the raw OCC endpoint so a
        # room-grid heartbeat does not make the browser download the large
        # occupancy payload again.
        self.occupancy_revision = 0
        self.graph_revision = 0
        # Mapped detector positions are a high-rate perception overlay, not
        # semantic-graph topology.  Keep their revision separate so a fresh
        # YOLO frame cannot invalidate/download the (often multi-megabyte)
        # OCC/graph visualization payload.
        self.detection_revision = 0
        # Consistency is a diagnostic heartbeat, not semantic graph topology.
        # Keep it separate so its 1-Hz evaluator cannot invalidate the raw OCC
        # visualization payload.
        self.consistency_revision = 0
        self.navigation_revision = 0
        # Pose/heading changes are independent of camera receipts and map
        # topology.  The six-panel renderer uses this revision to refresh the
        # robot arrow without treating every RGB frame as a heavy map update.
        self.telemetry_revision = 0
        self.last_error = ""

    def update_frame(self, **kwargs: Any) -> None:
        with self._lock:
            # RGB-D packets carry a capture-time telemetry subset (position,
            # yaw and velocity) so the dashboard can update the robot arrow
            # even when the independent low-rate telemetry mirror is delayed.
            # Merge that subset instead of replacing the full telemetry
            # snapshot; otherwise battery/range fields disappear on every
            # camera frame.  Keep the update atomic with the frame receipt so
            # a renderer cannot observe a new image paired with an old yaw.
            frame_telemetry = kwargs.pop("telemetry", None)
            frame_seq = kwargs.get("frame_seq")
            frame_stamp = kwargs.get("frame_stamp")
            telemetry_transport_session = kwargs.pop(
                "telemetry_transport_session", None
            )
            telemetry_transport_connection = kwargs.pop(
                "telemetry_transport_connection", None
            )
            for key, value in kwargs.items():
                setattr(self, key, value)
            if isinstance(frame_telemetry, dict) and frame_telemetry:
                changed = self._merge_telemetry_patch(
                    frame_telemetry,
                    fallback_stamp=frame_stamp,
                    capture_seq=frame_seq,
                    capture_stamp=frame_stamp,
                    transport_session=telemetry_transport_session,
                    transport_connection=telemetry_transport_connection,
                )
                if changed:
                    self.telemetry_revision += 1
            self.counters["frames"] += 1
            self.revision += 1

    def advance_navigation_step(self) -> int:
        with self._lock:
            self.navigation_step += 1
            return self.navigation_step

    def set_calibration(self, **value: Any) -> None:
        with self._lock:
            self.calibration = copy.deepcopy(value)

    def update_topic(self, name: str, value: Any) -> None:
        if name == "mllm_events":
            if isinstance(value, list):
                for event in value:
                    self.add_mllm_event(event)
            else:
                self.add_mllm_event(value)
            return
        with self._lock:
            changed = True
            if name == "telemetry" and isinstance(value, dict):
                # Direct /api/telemetry packets may be either a complete
                # snapshot or a lower-rate partial update.  Treat both as
                # patches so a partial update cannot erase a fresh yaw (or a
                # camera-frame packet cannot erase battery metadata).
                changed = self._merge_telemetry_patch(
                    value,
                    transport_session=value.get("_transport_session"),
                    transport_connection=value.get("_transport_connection"),
                )
                # ``value`` is assigned below for backwards compatibility;
                # use the merged state produced by the helper rather than
                # replacing it with a potentially stale complete snapshot.
                value = copy.deepcopy(self.telemetry)
            if name in {"detections", "mapped_detections"} and isinstance(value, dict):
                meta_attr = "detection_meta" if name == "detections" else "mapped_detection_meta"
                setattr(self, meta_attr, {k: v for k, v in value.items() if k not in {"detections", "objects"}})
                value = value.get("detections", value.get("objects", []))
            setattr(self, name, value)
            self.revision += 1
            if name in {"occupancy", "room_grid", "global_costmap", "local_costmap"}:
                self.map_revision += 1
                if name == "occupancy":
                    self.occupancy_revision += 1
            elif name == "mapped_detections":
                self.detection_revision += 1
            elif name == "graph":
                self.graph_revision += 1
            elif name == "consistency":
                self.consistency_revision += 1
            elif name == "navigation":
                self.navigation_revision += 1
            elif name == "telemetry":
                if changed:
                    self.telemetry_revision += 1
            if name == "detections":
                self.counters["detections"] = len(value) if isinstance(value, list) else 0
            elif name == "graph":
                self.counters["graph_updates"] += 1

    def update_navigation(self, name: str, value: Any) -> None:
        """Atomically publish one navigation product and its revision.

        The HTTP ROS-state endpoint receives a freshly decoded JSON object,
        so transferring ownership is cheaper than deep-copying a potentially
        very large NavPath.  Readers still observe the assignment and the
        revision update under the same lock; full snapshots are responsible
        for copying data when they need an immutable view.
        """
        with self._lock:
            self.navigation[str(name)] = value
            self.revision += 1
            self.navigation_revision += 1

    def _merge_telemetry_patch(
        self,
        patch: dict[str, Any],
        *,
        fallback_stamp: Any = None,
        capture_seq: Any = None,
        capture_stamp: Any = None,
        transport_session: Any = None,
        transport_connection: Any = None,
    ) -> bool:
        """Merge telemetry while rejecting stale pose fields.

        RGB-D receipts and the independent telemetry mirror can cross in
        flight.  ``received_at`` is generated by the Go2 state subscriber and
        is therefore a better ordering key for pose than host arrival time.
        Diagnostics such as battery data are still merged from an older
        packet; only pose-bearing fields are protected by the freshness gate.
        """
        if not isinstance(patch, dict) or not patch:
            return False
        # A persistent web gateway can receive frames from a newly restarted
        # direct sensor bridge.  Both the bridge-local sequence and the Go2
        # source clock may reset in that case.  Reset pose freshness keys at
        # the transport-generation boundary before comparing this patch.
        session = transport_session
        if session in (None, ""):
            session = patch.get("_transport_session")
        session = str(session) if session not in (None, "") else None
        raw_connection = transport_connection
        if raw_connection in (None, ""):
            raw_connection = patch.get("_transport_connection", -1)
        try:
            connection = int(raw_connection)
        except (TypeError, ValueError):
            connection = -1
        generation_is_stale = False
        # Once the direct sensor bridge has announced a transport generation,
        # an untagged packet can only be a delayed legacy/compatibility lane.
        # It may still carry useful diagnostics (battery, error codes, etc.),
        # but its pose must never replace the tagged capture-time yaw/position
        # because there is no way to tell which source clock it belongs to.
        # The direct RGB-D and odom paths attach the session metadata, so this
        # guard does not discard an authoritative pose stream.
        untagged_pose_blocked = (
            session is None and self._telemetry_transport_session is not None
        )
        if session is not None:
            current_session = self._telemetry_transport_session
            current_connection = self._telemetry_transport_connection
            if session in self._telemetry_retired_sessions:
                generation_is_stale = True
                new_generation = False
            else:
                new_generation = (
                    current_session is None
                    or session != current_session
                    or (
                        connection >= 0
                        and current_connection >= 0
                        and connection > current_connection
                    )
                )
            if new_generation:
                if current_session is not None and current_session != session:
                    self._telemetry_retired_sessions.add(current_session)
                    # A bounded set is enough to reject delayed HTTP packets
                    # without retaining one token per reconnect forever.
                    if len(self._telemetry_retired_sessions) > 32:
                        self._telemetry_retired_sessions = set(
                            list(self._telemetry_retired_sessions)[-16:]
                        )
                self._telemetry_transport_session = session
                self._telemetry_transport_connection = connection
                self._telemetry_pose_stamp = float("-inf")
                self._telemetry_capture_seq = -1
                self._telemetry_capture_frame_stamp = float("-inf")
                self._telemetry_unstamped_pose_seen = False
            elif (
                connection >= 0
                and current_connection >= 0
                and connection < current_connection
            ):
                generation_is_stale = True
        elif self._telemetry_transport_session is not None:
            # Once the direct bridge has advertised a transport generation,
            # an unmarked packet can only come from the deprecated legacy
            # socket (or from a delayed pre-restart HTTP request).  Keep its
            # diagnostics/battery fields mergeable, but never let its pose
            # fields overwrite the tagged RGB-D/telemetry stream.  The direct
            # bridge and current sensor-frame path always carry the marker;
            # this preserves legacy compatibility only before activation.
            generation_is_stale = True
        # These markers are ordering metadata only. Never expose them as part
        # of the public telemetry snapshot consumed by the browser/ROS APIs.
        if "_transport_session" in patch or "_transport_connection" in patch:
            patch = {
                key: value
                for key, value in patch.items()
                if key not in {"_transport_session", "_transport_connection"}
            }
        if not patch:
            return False
        # Some ROS/legacy telemetry mirrors omit the scalar yaw but retain the
        # Unitree body IMU.  Materialize the derived value before the freshness
        # merge so all downstream consumers (renderer, raw visualization API,
        # and browser partial updates) observe one canonical heading field.
        normalized_patch = patch
        derived_yaw = telemetry_yaw(patch)
        try:
            direct_yaw_valid = math.isfinite(float(patch.get("yaw")))
        except (TypeError, ValueError):
            direct_yaw_valid = False
        if derived_yaw is not None and ("yaw" not in patch or not direct_yaw_valid):
            normalized_patch = dict(patch)
            normalized_patch["yaw"] = derived_yaw
        patch = normalized_patch
        pose_keys = {
            "position", "velocity", "yaw", "yaw_speed", "imu", "camera_imu",
            "mode", "progress", "gait_type", "body_height", "received_at",
        }
        try:
            source_stamp = float(patch.get("received_at"))
        except (TypeError, ValueError):
            source_stamp = float("nan")
        if not math.isfinite(source_stamp):
            try:
                source_stamp = float(fallback_stamp)
            except (TypeError, ValueError):
                source_stamp = float("nan")
        has_pose = any(key in patch for key in pose_keys)
        try:
            capture_seq_int = int(capture_seq)
        except (TypeError, ValueError):
            capture_seq_int = None
        try:
            capture_stamp_float = float(capture_stamp)
        except (TypeError, ValueError):
            capture_stamp_float = float("nan")
        # The sequence belongs only to the RGB-D capture lane; telemetry
        # mirror/odom patches do not pass it.  Prefer this ordering key over
        # wall-clock stamps for a newer frame so a cross-machine clock step
        # cannot pin the rendered arrow at the previous yaw.
        capture_is_fresh = (
            has_pose
            and capture_seq_int is not None
            and capture_seq_int > self._telemetry_capture_seq
        )
        pose_is_fresh = (
            not generation_is_stale
            and not untagged_pose_blocked
            and (
            not has_pose
            or (
                math.isfinite(source_stamp)
                and source_stamp >= self._telemetry_pose_stamp - 1e-6
            )
            or (
                not math.isfinite(source_stamp)
                and not math.isfinite(self._telemetry_pose_stamp)
                and not self._telemetry_unstamped_pose_seen
            )
            or capture_is_fresh
            )
        )
        # The ROS gateway mirrors ``/physical_nav/odom`` into this same web
        # state.  Odom is generated from the RGB-D receipt, but it has no
        # capture sequence; accepting an echo after the authoritative frame
        # can overwrite a live Go2 yaw with a stale/identity quaternion and
        # make the browser arrow look frozen.  Keep odom as a fallback when
        # no capture pose exists, while rejecting an echo at the latest RGB-D
        # stamp.  A newer odom sample (strictly beyond the capture stamp) is
        # still allowed for legacy streams that do not carry frame telemetry.
        odom_echo = str(patch.get("pose_source") or "") == "physical_nav_odom"
        if (
            odom_echo
            and capture_seq_int is None
            and math.isfinite(source_stamp)
            and math.isfinite(self._telemetry_capture_frame_stamp)
            and source_stamp <= self._telemetry_capture_frame_stamp + 1e-3
        ):
            pose_is_fresh = False
        merged = copy.deepcopy(self.telemetry)
        for key, value in patch.items():
            if key in pose_keys and not pose_is_fresh:
                continue
            # Battery/IMU diagnostics are sometimes sent as partial nested
            # dictionaries.  Merge those leaves instead of erasing fields
            # that arrived on the other telemetry lane.
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                nested = copy.deepcopy(merged[key])
                nested.update(copy.deepcopy(value))
                merged[key] = nested
            else:
                merged[key] = copy.deepcopy(value)
        changed = merged != self.telemetry
        self.telemetry = merged
        if has_pose and pose_is_fresh and math.isfinite(source_stamp):
            self._telemetry_pose_stamp = max(self._telemetry_pose_stamp, source_stamp)
        elif has_pose and pose_is_fresh and not math.isfinite(source_stamp):
            self._telemetry_unstamped_pose_seen = True
        if has_pose and pose_is_fresh and capture_seq_int is not None:
            self._telemetry_capture_seq = max(self._telemetry_capture_seq, capture_seq_int)
            if math.isfinite(capture_stamp_float):
                self._telemetry_capture_frame_stamp = max(
                    self._telemetry_capture_frame_stamp,
                    capture_stamp_float,
                )
        return changed

    def link_connected(self, hello: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.link["connections"] = int(self.link.get("connections", 0)) + 1
            self.link["connected"] = True
            if hello:
                self.link["last_hello"] = copy.deepcopy(hello)
            self.link["last_packet_at"] = time.time()

    def link_packet(self, packet_type: str) -> None:
        with self._lock:
            self.link["last_packet_at"] = time.time()
            self.link["last_packet_type"] = str(packet_type)

    def sensor_link_active(self) -> None:
        """A received RGB-D frame proves the sensor link is alive.

        The direct 12335/mirror transport carries sensor data independently of
        the legacy 12334 WebSocket, so a fresh frame stream is the authoritative
        "connected" signal for the watchdog and the dashboard.
        """
        with self._lock:
            self.link["connected"] = True
            self.link["last_packet_at"] = time.time()
            self.link["last_packet_type"] = "sensor_frame"

    def link_disconnected(self) -> None:
        with self._lock:
            self.link["connections"] = max(0, int(self.link.get("connections", 1)) - 1)
            self.link["connected"] = bool(self.link["connections"])
            self.link["last_disconnect_at"] = time.time()

    def add_qwen(self, request: dict[str, Any], result: dict[str, Any] | None = None) -> None:
        with self._lock:
            if request:
                self.qwen.setdefault("requests", []).append(request)
                self.qwen["requests"] = self.qwen["requests"][-100:]
            if result is not None:
                self.qwen.setdefault("results", []).append(result)
                self.qwen["results"] = self.qwen["results"][-100:]

    def add_mllm_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            if not isinstance(event, dict):
                return
            stored = copy.deepcopy(event)
            data_url = str(stored.pop("m1_input_image_data_url", "") or "")
            if data_url.startswith("data:image/") and "," in data_url:
                key = "{}:{}:{}".format(
                    stored.get("episode_id", "default"),
                    stored.get("request_sequence", len(self.mllm_events)),
                    int(float(stored.get("timestamp", time.time())) * 1000),
                )
                try:
                    self.m1_input_images[key] = base64.b64decode(
                        data_url.split(",", 1)[1], validate=True
                    )
                    stored["m1_input_image_key"] = key
                except (ValueError, TypeError):
                    pass
            self.mllm_events.append(stored)
            self.mllm_events = self.mllm_events[-100:]
            active_keys = {
                str(item.get("m1_input_image_key"))
                for item in self.mllm_events
                if item.get("m1_input_image_key")
            }
            self.m1_input_images = {
                key: value
                for key, value in self.m1_input_images.items()
                if key in active_keys
            }

    def m1_input_image(self, key: str) -> bytes | None:
        with self._lock:
            value = self.m1_input_images.get(str(key))
            return bytes(value) if value is not None else None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            graph = copy.deepcopy(self.graph)
            return {
                "frame_seq": self.frame_seq,
                "navigation_step": self.navigation_step,
                "frame_stamp": self.frame_stamp,
                "camera_frame": self.camera_frame,
                "depth_frame": self.depth_frame or self.camera_frame,
                "calibration": copy.deepcopy(self.calibration),
                "sync_ms": self.sync_ms,
                "depth_scale": self.depth_scale,
                "intrinsics": copy.deepcopy(self.intrinsics),
                "rgb_intrinsics": copy.deepcopy(self.rgb_intrinsics or self.intrinsics),
                "depth_intrinsics": copy.deepcopy(self.depth_intrinsics or self.intrinsics),
                "depth_to_color_extrinsics": copy.deepcopy(self.depth_to_color_extrinsics),
                "camera_imu": copy.deepcopy(self.camera_imu),
                "capture_timing": copy.deepcopy(self.capture_timing),
                "raw_frame_available": bool(self.rgb_b64 and self.depth_b64),
                "telemetry": copy.deepcopy(self.telemetry),
                "detections": copy.deepcopy(self.detections),
                "detection_meta": copy.deepcopy(self.detection_meta),
                "mapped_detections": copy.deepcopy(self.mapped_detections),
                "mapped_detection_meta": copy.deepcopy(self.mapped_detection_meta),
                "graph": graph,
                "occupancy": ({k: self.occupancy.get(k) for k in ("width", "height", "resolution", "origin", "frame_id")} if isinstance(self.occupancy, dict) else None),
                "room_grid": ({k: self.room_grid.get(k) for k in ("width", "height", "resolution", "origin", "frame_id")} if isinstance(self.room_grid, dict) else None),
                "global_costmap": ({k: self.global_costmap.get(k) for k in ("width", "height", "resolution", "origin", "frame_id")} if isinstance(self.global_costmap, dict) else None),
                "local_costmap": ({k: self.local_costmap.get(k) for k in ("width", "height", "resolution", "origin", "frame_id")} if isinstance(self.local_costmap, dict) else None),
                "consistency": copy.deepcopy(self.consistency),
                "qwen": copy.deepcopy(self.qwen),
                "mllm_events": copy.deepcopy(self.mllm_events),
                "navigation": copy.deepcopy(self.navigation),
                "link": copy.deepcopy(self.link),
                "counters": dict(self.counters),
                "last_error": self.last_error,
                "revision": self.revision,
                "map_revision": self.map_revision,
                "occupancy_revision": self.occupancy_revision,
                "graph_revision": self.graph_revision,
                "detection_revision": self.detection_revision,
                "consistency_revision": self.consistency_revision,
                "navigation_revision": self.navigation_revision,
                "telemetry_revision": self.telemetry_revision,
                "read_only": True,
                "status": "READ_ONLY_BLOCKED",
                "generated_at": time.time(),
            }

    def summary_snapshot(self) -> dict[str, Any]:
        """Return only fields consumed by the live dashboard.

        The full snapshot deep-copies the accumulated semantic graph and
        MLLM trace.  Doing that once per browser poll caused visible pauses as
        the graph grew.  The summary endpoint only needs graph counts and
        compact recent events, so avoid copying map payloads and graph nodes.
        """
        with self._lock:
            graph = self.graph if isinstance(self.graph, dict) else {}
            graph_summary = {
                "scene_id": graph.get("scene_id"),
                "graph_revision": graph.get("graph_revision"),
                "capture_step": graph.get("capture_step"),
                "node_count": len(graph.get("nodes") or []),
                "edge_count": len(graph.get("edges") or []),
            }
            return {
                "frame_seq": self.frame_seq,
                "navigation_step": self.navigation_step,
                "frame_stamp": self.frame_stamp,
                "telemetry": copy.deepcopy(self.telemetry),
                # Keep the status endpoint independent from raw segmentation
                # payload size.  ``snapshot()`` still exposes full receipts
                # to the ROS/debug path; this endpoint only needs compact box
                # and label metadata.
                "detections": self._summary_detections(self.detections),
                "detection_meta": dict(self.detection_meta),
                "mapped_detections": self._summary_detections(self.mapped_detections),
                "graph": graph_summary,
                "consistency": copy.deepcopy(self.consistency),
                "mllm_events": copy.deepcopy(self.mllm_events[-40:]),
                "navigation": self._summary_navigation(self.navigation),
                "link": dict(self.link),
                "counters": dict(self.counters),
                "last_error": self.last_error,
                "read_only": True,
                "status": "READ_ONLY_BLOCKED",
                "generated_at": time.time(),
            }

    def health_snapshot(self) -> dict[str, Any]:
        """Return a constant-size liveness view without copying maps/traces."""
        with self._lock:
            perception_seq = self.detection_meta.get("seq", -1)
            perception_stamp = self.detection_meta.get("stamp", 0.0)
            now = time.time()
            try:
                frame_stamp = float(self.frame_stamp or 0.0)
            except (TypeError, ValueError):
                frame_stamp = 0.0
            try:
                perception_stamp = float(perception_stamp or 0.0)
            except (TypeError, ValueError):
                perception_stamp = 0.0
            frame_age = max(0.0, now - frame_stamp) if frame_stamp > 0.0 else None
            perception_age = (
                max(0.0, now - perception_stamp)
                if perception_stamp > 0.0
                else None
            )
            # These fixed informational thresholds mirror the watchdog
            # defaults.  ``health_errors`` remains the authoritative checker
            # because it can use operator-specific limits; exposing the age
            # explicitly prevents ``ok=true`` (which only means “a frame has
            # existed”) from being mistaken for a live stream in dashboards.
            frame_stale = frame_age is None or frame_age > 15.0
            perception_stale = perception_age is None or perception_age > 15.0
            return {
                "ok": bool(self.frame_seq >= 0),
                "read_only": True,
                "frame_seq": self.frame_seq,
                "frame_stamp": self.frame_stamp,
                "frame_age_s": frame_age,
                "frame_stale": frame_stale,
                "perception_seq": perception_seq,
                "perception_stamp": perception_stamp,
                "perception_age_s": perception_age,
                "perception_stale": perception_stale,
                "navigation_step": self.navigation_step,
                "link_connected": bool(self.link.get("connected")),
                "last_packet_at": self.link.get("last_packet_at", 0.0),
                "generated_at": now,
                "last_error": self.last_error,
            }

    def visualization_snapshot(self) -> dict[str, Any]:
        """Return raw map/graph receipts for presentation-only redrawers.

        Keep this endpoint limited to data consumed by the presentation
        redrawers.  Room/global/local costmaps duplicate OCC at hundreds of
        kilobytes per poll and are available through their dedicated ROS/UI
        paths; including them here made the browser download megabyte-sized
        payloads and serialized the runtime lock for too long.
        """
        with self._lock:
            graph = copy.deepcopy(self.graph)
            return {
                "frame_seq": self.frame_seq,
                "navigation_step": self.navigation_step,
                "telemetry": copy.deepcopy(self.telemetry),
                "occupancy": copy.deepcopy(self.occupancy),
                "graph": graph,
                "mapped_detections": copy.deepcopy(self.mapped_detections),
                "navigation": copy.deepcopy(self.navigation),
                "map_revision": self.map_revision,
                "occupancy_revision": self.occupancy_revision,
                "graph_revision": self.graph_revision,
                "detection_revision": self.detection_revision,
                "telemetry_revision": self.telemetry_revision,
                "navigation_revision": self.navigation_revision,
            }

    def visualization_revisions(self) -> dict[str, int]:
        """Return constant-size revision counters for conditional web polls."""
        with self._lock:
            return {
                "map_revision": int(self.map_revision),
                "occupancy_revision": int(self.occupancy_revision),
                "graph_revision": int(self.graph_revision),
                "detection_revision": int(self.detection_revision),
                "telemetry_revision": int(self.telemetry_revision),
                "navigation_revision": int(self.navigation_revision),
            }

    def visualization_delta(self, fields: set[str] | list[str] | tuple[str, ...]) -> dict[str, Any]:
        """Return only high-rate visualization fields that changed.

        OCC and the semantic graph are intentionally absent from this path.
        A capture-time pose or mapped-detection update can therefore refresh
        the browser arrow/overlay without copying and serializing the raw
        occupancy grid.  ``partial`` tells clients to merge the response
        into their last full visualization snapshot.
        """
        requested = {str(field) for field in fields}
        with self._lock:
            payload: dict[str, Any] = {
                "partial": sorted(requested),
                "frame_seq": self.frame_seq,
                "navigation_step": self.navigation_step,
            }
            if "telemetry" in requested:
                payload["telemetry"] = copy.deepcopy(self.telemetry)
            if "mapped_detections" in requested:
                payload["mapped_detections"] = copy.deepcopy(self.mapped_detections)
            if "graph" in requested:
                payload["graph"] = copy.deepcopy(self.graph)
            if "navigation" in requested:
                payload["navigation"] = copy.deepcopy(self.navigation)
            return payload

    def raw_frame(self) -> dict[str, Any]:
        """Return the latest encoded frame for the system-Python ROS bridge."""
        with self._lock:
            return {
                "seq": self.frame_seq,
                "stamp": self.frame_stamp,
                "rgb": self.rgb_b64,
                "depth": self.depth_b64,
                "width": int(self.intrinsics.get("width", self.rgb.shape[1] if self.rgb is not None else 0)),
                "height": int(self.intrinsics.get("height", self.rgb.shape[0] if self.rgb is not None else 0)),
                "camera_frame": self.camera_frame,
                "depth_frame": self.depth_frame or self.camera_frame,
                "depth_scale": self.depth_scale,
                "intrinsics": copy.deepcopy(self.intrinsics),
                "rgb_intrinsics": copy.deepcopy(self.rgb_intrinsics or self.intrinsics),
                "depth_intrinsics": copy.deepcopy(self.depth_intrinsics or self.intrinsics),
                "depth_to_color_extrinsics": copy.deepcopy(self.depth_to_color_extrinsics),
                "camera_imu": copy.deepcopy(self.camera_imu),
                "intrinsics": copy.deepcopy(self.intrinsics),
                "color_depth_sync_ms": self.sync_ms,
                "telemetry": copy.deepcopy(self.telemetry),
            }

    def occupancy_snapshot(self) -> dict[str, Any]:
        """Return only the latest occupancy grid for the web map panel."""
        with self._lock:
            value = self.occupancy
            return copy.deepcopy(value) if isinstance(value, dict) else {}
