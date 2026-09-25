#!/usr/bin/env python3
"""ROS-side perception adapter and optional dashboard mirror.

The physical launch consumes authoritative ROS sensor/YOLO messages. Legacy
HTTP sensor/state readers are opt-in replay compatibility, not live inputs.
Dashboard delivery is latest-only and never owns capture or algorithm state.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import struct
import sys
import threading
import time
import urllib.request
from typing import Any, Callable

import numpy as np
import rospy
from geometry_msgs.msg import Point, TransformStamped, PointStamped
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, String
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from PIL import Image as PILImage


# Small state receipts (telemetry, plans, and decision status) are latest-only
# and should reach the optional dashboard promptly.  Keep the large
# OccupancyGrid lane on its independent, slower wake-up below; changing this
# value therefore does not add map serialization or ROS callback work.
_STATE_POST_WAIT_S = 0.05

# These topics are presentation-only heartbeats.  The ROS publishers and the
# semantic/navigation consumers keep their original rates; only the optional
# HTTP mirror is coalesced so a persistent web gateway cannot spend a full CPU
# core parsing and serializing an unchanged/near-identical graph at 10 Hz.
# Event-like results are intentionally absent and remain edge-triggered.
_STATE_MIRROR_MIN_PERIODS_S = {
    "graph": 1.0,
    "consistency": 1.0,
    "explore_status": 0.5,
    "decision_trace": 0.5,
    "candidates": 0.5,
    "selection": 0.2,
    "execution_state": 0.2,
    "behavior_feedback": 0.2,
}


def _state_post_wait_seconds() -> float:
    """Return the bounded wait used by the small web-state worker.

    Keeping the policy as a pure helper makes the latency contract explicit
    and lets tests verify it without importing/starting a ROS node.
    """

    return _STATE_POST_WAIT_S


def _state_mirror_min_period(name: str) -> float:
    """Return the optional web mirror period for one ROS state topic."""

    try:
        return max(0.0, float(_STATE_MIRROR_MIN_PERIODS_S.get(str(name), 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _decode(encoded: str | dict[str, Any]) -> np.ndarray:
    # Older recordings wrap the base64 payload in an ``{encoding,data}``
    # object, while the direct sensor path sends the string itself.
    if isinstance(encoded, dict):
        encoded = encoded.get("data", "")
    image = PILImage.open(io.BytesIO(base64.b64decode(encoded)))
    return np.asarray(image)


def _patch_roslogging_findcaller_for_py311() -> None:
    """Avoid rosgraph's Python 3.11 ``findCaller`` recursion."""
    if sys.version_info < (3, 11):
        return
    try:
        import rosgraph.roslogging as roslogging
    except Exception:
        return
    if getattr(roslogging.RospyLogger.findCaller, "_physical_gateway_safe", False):
        return

    def _safe_find_caller(self, *args, **kwargs):
        result = logging.Logger.findCaller(self, *args, **kwargs)
        if len(result) == 3:
            return result[0], result[1], result[2], None
        return result

    _safe_find_caller._physical_gateway_safe = True
    roslogging.RospyLogger.findCaller = _safe_find_caller


class _NoopPublisher:
    """Publisher-shaped sink used when the legacy sensor lane is disabled."""

    def publish(self, *_args: Any, **_kwargs: Any) -> None:
        return


class _NoopBroadcaster:
    """Broadcaster-shaped sink so the disabled legacy lane owns no ROS topic."""

    def sendTransform(self, *_args: Any, **_kwargs: Any) -> None:
        return


# These fields are needed by the camera/web overlay, but are not consumed by
# semantic mapping or M1.  The sparse ``mask`` rows/columns remain in the
# authoritative detection envelope because they provide the visibility and
# segmentation evidence used by graph admission.  Keeping the polygon lists
# only on the raw YOLO report avoids serializing the same display geometry
# through ROS (and then parsing/copying it again in the mapper) at 10 Hz.
_DISPLAY_ONLY_DETECTION_FIELDS = frozenset(
    {
        "mask_polygon",
        "mask_polygons",
        "rgb_mask_polygons",
    }
)


def _semantic_detection_payload(item: dict[str, Any]) -> dict[str, Any]:
    """Return one detection record for semantic ROS consumers.

    This is intentionally a shallow projection: 3-D geometry, sparse mask
    evidence, labels, and tracking metadata retain their existing identity and
    numeric values.  Only presentation-only polygon lists are omitted.
    """

    return {
        key: value
        for key, value in item.items()
        if key not in {
            "camera_segment_points",
            "world_segment_points",
            "camera_segment_points_f32",
            "world_segment_points_f32",
            "_camera_segment_points_array",
            "_world_segment_points_array",
            "segment_point_count",
        }
        and key not in _DISPLAY_ONLY_DETECTION_FIELDS
    }


class PhysicalRosGateway:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.web_state_enabled = bool(getattr(args, "web_state_enabled", True))
        # The HTTP gateway may survive a ROS/navigation restart.  A process
        # scoped token lets the persistent renderer distinguish a new map
        # session even when its geometry starts with the same dimensions and
        # origin as the previous session.
        self._web_session_id = f"{os.getpid()}-{time.time_ns()}"
        # The direct WebSocket sensor bridge is authoritative in the current
        # architecture.  The old HTTP poller remains available for replay and
        # compatibility, but must not register a second RGB-D/PointCloud/TF
        # writer when ``legacy_http_sensor`` is disabled.
        configured_sensor_ros = getattr(args, "publish_sensor_ros", None)
        self.sensor_ros_enabled = (
            bool(getattr(args, "legacy_http_sensor", True))
            if configured_sensor_ros is None
            else str(configured_sensor_ros).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        rospy.loginfo(
            "physical ROS gateway presentation lane: raw_sensor_ros=%s "
            "(direct physical_sensor_ros_bridge remains authoritative)",
            self.sensor_ros_enabled,
        )
        self.args.occupancy_period = max(0.0, float(self.args.occupancy_period))
        self.args.room_grid_period = max(0.0, float(self.args.room_grid_period))
        self.last_seq = -1
        self.last_stamp = float("-inf")
        self._last_bridge_seq = -1
        self._active_bridge_session: str | None = None
        self.last_raw: dict[str, Any] | None = None
        self.world_frame = str(getattr(self.args, "world_frame", "tf_frame_map"))
        sensor_pub = rospy.Publisher if self.sensor_ros_enabled else lambda *_a, **_k: _NoopPublisher()
        self.rgb_pub = sensor_pub("/physical_nav/rgb/image_raw", Image, queue_size=1)
        self.detection_overlay_pub = rospy.Publisher(
            "/physical_nav/detections_overlay", Image, queue_size=1, latch=True
        )
        self.depth_pub = sensor_pub("/physical_nav/depth/image_raw", Image, queue_size=1)
        self.info_pub = sensor_pub("/physical_nav/camera_info", CameraInfo, queue_size=1)
        self.depth_info_pub = sensor_pub(
            "/physical_nav/depth_camera_info", CameraInfo, queue_size=1
        )
        # Latest-only transport with TCP_NODELAY keeps a fresh cloud from
        # waiting behind a large serialized PointCloud2 packet.
        self.cloud_pub = sensor_pub(
            "/physical_nav/points", PointCloud2, queue_size=1, tcp_nodelay=True
        )
        self.segmented_cloud_pub = rospy.Publisher(
            "/physical_nav/segmented_cloud", PointCloud2, queue_size=1
        )
        self.segmented_cloud_world_pub = rospy.Publisher(
            "/physical_nav/segmented_cloud_world", PointCloud2, queue_size=1
        )
        self.boxes_pub = rospy.Publisher("/physical_nav/boxes_3d", MarkerArray, queue_size=1)
        self.boxes_world_pub = rospy.Publisher(
            "/physical_nav/boxes_3d_world", MarkerArray, queue_size=1, latch=True
        )
        self.odom_pub = sensor_pub(
            "/physical_nav/odom", Odometry, queue_size=1, tcp_nodelay=True
        )
        self.detection_pub = rospy.Publisher("/physical_nav/detections", String, queue_size=1)
        # M1 needs public semantic names and a stable identity, while the raw
        # mapper should continue receiving untouched detector instances.  A
        # separate topic prevents M1-only identity hints from changing 3-D
        # tracking/merge behaviour.
        self.attribute_detection_pub = rospy.Publisher(
            "/physical_nav/attribute_detections", String, queue_size=1
        )
        broadcaster = tf2_ros.TransformBroadcaster if self.sensor_ros_enabled else lambda: _NoopBroadcaster()
        static_broadcaster = tf2_ros.StaticTransformBroadcaster if self.sensor_ros_enabled else lambda: _NoopBroadcaster()
        self.tf_broadcaster = broadcaster(); self.static_broadcaster = static_broadcaster()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0)); self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        visualization = rospy.get_param("~object_detection", {})
        if not isinstance(visualization, dict):
            visualization = {}
        # Segmented clouds are an optional RViz/Foxglove-only branch.  Keep
        # the authoritative detection/box topics independent from this
        # relatively expensive packed-point decode.  The connection check in
        # ``_publish_detection_debug`` additionally avoids doing the work
        # when the configured topic currently has no consumer.
        self._publish_debug_segmented_cloud = bool(
            visualization.get("publish_debug_segmented_cloud", False)
        )
        self._publish_debug_world_segmented_cloud = bool(
            visualization.get("publish_debug_world_segmented_cloud", False)
        )
        self._box_hold_s = max(
            0.0, float(visualization.get("debug_box_hold_s", args.box_hold_s))
        )
        self._box_match_distance_m = max(
            0.05,
            float(visualization.get("debug_box_match_distance_m", args.box_match_distance_m)),
        )
        self._box_smoothing_alpha = min(
            1.0,
            max(
                0.01,
                float(visualization.get("debug_box_smoothing_alpha", args.box_smoothing_alpha)),
            ),
        )
        self._box_min_confirmations = max(
            1,
            int(visualization.get("debug_box_min_confirmations", args.box_min_confirmations)),
        )
        self._world_box_tracks: dict[int, dict[str, Any]] = {}
        self._next_world_box_track_id = 1
        self._last_detection_receipt: tuple[Any, Any] | None = None
        self._static_sent = False; self._static_frames: set[str] = set(); self._static_transforms: dict[str, TransformStamped] = {}; self._depth_to_color_extrinsics: dict[str, Any] | None = None; self._lock = threading.Lock(); self._telemetry: dict[str, Any] = {}; self._telemetry_transport_session: str | None = None; self._telemetry_transport_connection = -1; self._last_direct_telemetry_mono = 0.0; self._direct_pose_hold_sec = 0.5; self._last_grid_post: dict[str, float] = {}; self._last_occupancy_post = 0.0
        # Capture-time TF is shared by all detections in one report and is
        # occasionally requested again by a replay/debug receipt.  Keep a
        # small bounded cache so a historical lookup is never repeated for the
        # same frame/stamp, while avoiding any latest-TF fallback.
        self._capture_transform_cache: dict[tuple[str, int], Any] = {}
        self._capture_transform_cache_lock = threading.Lock()
        # ROS subscriber callbacks must never perform TF/3-D mapping or wait
        # on the dashboard. Keep one newest detector report and process it in
        # a daemon worker; an old report has no value once a newer capture has
        # arrived.
        self._yolo_report_lock = threading.Lock()
        self._yolo_report_event = threading.Event()
        self._pending_yolo_report: str | None = None
        self._state_poll_lock = threading.Lock()
        threading.Thread(target=self._yolo_report_loop, name="physical-yolo-map", daemon=True).start()
        # All browser state posts use the same latest-only transport. This
        # prevents graph/plan callbacks from accumulating behind a slow HTTP
        # client and keeps the ROS callback queue responsive.
        self._state_post_lock = threading.Lock()
        self._state_post_event = threading.Event()
        self._pending_state_posts: dict[str, Any] = {}
        # Graph/consistency topics are latched and can replay the exact same
        # JSON whenever a subscriber reconnects.  Keep only the last wire
        # payload per name so an identical replay does not pay JSON parsing,
        # serialization, and localhost HTTP transport again.  This is scoped
        # to the optional web mirror; authoritative ROS topics are untouched.
        self._last_state_callback_payloads: dict[str, str] = {}
        self._last_state_mirror_post_mono: dict[str, float] = {}
        self._pending_state_mirror_payloads: dict[str, str] = {}
        self._state_callback_payload_lock = threading.Lock()
        threading.Thread(target=self._state_post_loop, name="physical-web-state", daemon=True).start()
        # Map callbacks must never wait on localhost HTTP/JSON serialization.
        # Keep one latest payload per map stage and let a daemon worker post it
        # asynchronously; stale OCC receipts are discarded instead of
        # accumulating behind a slow web process.
        self._grid_post_lock = threading.Lock()
        self._pending_grid_posts: dict[str, tuple[int, OccupancyGrid]] = {}
        # Revisions advance in the O(1) ROS callback, including for messages
        # that arrive while an older grid is being materialized.  The worker
        # checks them at every expensive boundary and drops obsolete work
        # before it can become another large localhost HTTP upload.
        self._grid_post_revisions: dict[str, int] = {}
        self._grid_post_event = threading.Event()
        threading.Thread(target=self._grid_post_loop, name="physical-grid-post", daemon=True).start()
        self._debug_cloud_lock = threading.Lock()
        self._pending_debug_cloud: tuple[
            list[dict[str, Any]],
            dict[str, Any],
            Any,
            list[dict[str, Any]],
            dict[int, dict[str, np.ndarray]],
        ] | None = None
        self._projection_cache: dict[str, Any] = {}
        # JPEG decoding/ROS publication for the debug overlay is presentation
        # work. Keep it latest-only so a slow Foxglove subscriber or a large
        # overlay cannot delay the authoritative detection/attribute topics.
        self._overlay_lock = threading.Lock()
        self._overlay_event = threading.Event()
        self._pending_overlay: dict[str, Any] | None = None
        threading.Thread(
            target=self._overlay_publish_loop,
            name="physical-detection-overlay",
            daemon=True,
        ).start()
        self._pose_timer = (
            rospy.Timer(rospy.Duration(0.05), self._refresh_pose_tf)
            if self.sensor_ros_enabled
            else None
        )
        # Detections originate in the non-ROS YOLOE worker and are republished
        # below for the existing mapper.  Do not subscribe to the same topic
        # here: that would feed our own message back into the HTTP state loop.
        for topic, name in (("/physical_nav/unified_graph", "graph"), ("/physical_nav/consistency", "consistency"),):
            rospy.Subscriber(topic, String, self._json_callback(name), queue_size=1)
        rospy.Subscriber("/physical_nav/yolo_report", String, self._yolo_report_callback, queue_size=1)
        # The direct sensor bridge publishes capture-time telemetry on ROS as
        # the authoritative local state stream.  Mirror that small message to
        # the optional web gateway instead of making the browser depend on a
        # second HTTP telemetry path.  This keeps the raw ROS/sensor chain
        # independent of the page and still lets the robot arrow update when
        # the direct bridge's best-effort HTTP mirror is unavailable.
        rospy.Subscriber(
            "/physical_nav/telemetry",
            String,
            self._telemetry_callback,
            queue_size=1,
        )
        # The direct sensor bridge timestamps Odometry with the same capture
        # stamp used by RGB-D/PointCloud2.  Mirror this compact pose into the
        # optional web state as a second, ROS-authoritative lane.  It fixes a
        # subtle dashboard failure mode in which a telemetry mirror can be
        # delayed or absent while TF/odom is still advancing normally.  The
        # RuntimeState freshness gate rejects an older packet, so this cannot
        # rewind a newer capture-time yaw.
        rospy.Subscriber(
            "/physical_nav/odom",
            Odometry,
            self._odom_callback,
            queue_size=1,
        )
        # Shadow navigation/decision outputs are observed for rendering and
        # evaluation only. They are never forwarded to a robot controller.
        for topic, name in (
            ("/explore_py/status", "explore_status"),
            ("/semantic_decision/candidates", "candidates"),
            ("/semantic_decision/selected_behavior", "selection"),
            ("/semantic_decision/execution_state", "execution_state"),
            ("/semantic_decision/behavior_feedback", "behavior_feedback"),
            ("/semantic_decision/decision_trace", "decision_trace"),
            ("/physical_nav/interaction_result", "interaction_result"),
            ("/physical_nav/mllm_events", "mllm_events"),
        ):
            rospy.Subscriber(topic, String, self._json_callback(name), queue_size=2)
        rospy.Subscriber("/explore_py/current_subgoal", PointStamped, self._subgoal_callback, queue_size=2)
        for topic, name in (
            (self.args.global_plan_topic, "global_plan"),
            (self.args.local_global_plan_topic, "local_global_plan"),
            (self.args.local_plan_topic, "local_plan"),
        ):
            if topic:
                rospy.Subscriber(topic, Path, self._plan_callback(name), queue_size=1)
        for topic, name in (
            (self.args.occupancy_grid_topic, "occupancy"),
            (self.args.room_grid_topic, "room_grid"),
            (self.args.global_costmap_topic, "global_costmap"),
            (self.args.local_costmap_topic, "local_costmap"),
        ):
            if topic:
                rospy.Subscriber(topic, OccupancyGrid, self._grid_callback(name), queue_size=1)
        self.timer = None
        # Use the resolved publisher gate, not the legacy poller flag.  A
        # caller can explicitly set ``publish_sensor_ros=false`` while
        # leaving the old HTTP sensor option enabled; in that case polling
        # and decoding frames here would still waste CPU and recreate the
        # duplicate raw-sensor lane we intentionally disabled.
        if self.sensor_ros_enabled and args.rate > 0:
            self.timer = rospy.Timer(rospy.Duration(1.0 / max(args.rate, 1e-3)), self._poll)
        # Detection/graph projection may take longer than one camera period.
        # Keep it off the RGB-D timer so 10 Hz image and point-cloud delivery
        # isn't serialized behind presentation/debug work.
        self._state_timer = None
        if bool(getattr(args, "legacy_http_state", True)) and args.state_period > 0:
            self._state_timer = rospy.Timer(rospy.Duration(max(args.state_period, 0.05)), self._poll_state)
        # Visualization is latest-frame-only and runs independently of the
        # state/detection processing path at the same configured frequency.
        self._debug_cloud_timer = rospy.Timer(
            rospy.Duration(max(args.state_period, 0.05)),
            self._publish_pending_debug_cloud,
        )

    def _post_state(self, name: str, value: Any) -> None:
        if not self.web_state_enabled:
            return
        with self._state_post_lock:
            self._pending_state_posts[str(name)] = value
            self._state_post_event.set()

    def _post_state_http(
        self,
        name: str,
        value: Any,
        *,
        is_current: Callable[[], bool] | None = None,
    ) -> bool:
        """Send one optional dashboard receipt without touching ROS state.

        This is deliberately kept outside ``_post_state``'s queueing method so
        large OccupancyGrid JSON bodies can use their own worker.  A slow or
        stopped browser may delay this call, but it can never delay a ROS
        subscriber callback or the small telemetry/detection queue.

        ``is_current`` is used only by the large-grid lane. It is checked on
        both sides of JSON serialization and immediately before opening the
        socket, so a newer same-name receipt cancels obsolete presentation
        work without changing ordinary small-state callers.
        """
        if is_current is not None and not is_current():
            return False
        payload = json.dumps(
            {"name": str(name), "value": value},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        if is_current is not None and not is_current():
            return False
        request = urllib.request.Request(
            self.args.web_url.rstrip("/") + "/api/ros-state",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        if is_current is not None and not is_current():
            return False
        with urllib.request.urlopen(request, timeout=.3) as response:
            response.read()
        return True

    def _flush_state_mirror_payloads(self) -> None:
        """Release presentation heartbeats whose mirror period elapsed."""

        now = time.monotonic()
        ready: list[tuple[str, str]] = []
        with self._state_callback_payload_lock:
            for name, payload in list(self._pending_state_mirror_payloads.items()):
                period = _state_mirror_min_period(name)
                last_post = self._last_state_mirror_post_mono.get(name, 0.0)
                if period > 0.0 and now - last_post < period:
                    continue
                self._pending_state_mirror_payloads.pop(name, None)
                self._last_state_mirror_post_mono[name] = now
                ready.append((name, payload))
        for name, payload in ready:
            try:
                self._post_state(name, json.loads(payload))
            except Exception as exc:
                rospy.logwarn_throttle(
                    5.0, "physical ROS deferred state %s: %s", name, exc
                )

    def _state_post_loop(self) -> None:
        while not rospy.is_shutdown():
            # Small state (especially capture-time telemetry) is independent
            # of the large-grid worker and should not be throttled to 2 Hz.
            # The event keeps this wait idle when no receipt is pending, so a
            # shorter timeout does not create a polling loop or extra ROS
            # serialization work.
            self._state_post_event.wait(_state_post_wait_seconds())
            self._flush_state_mirror_payloads()
            with self._state_post_lock:
                pending = self._pending_state_posts
                self._pending_state_posts = {}
                self._state_post_event.clear()
            for name, value in pending.items():
                try:
                    self._post_state_http(name, value)
                except Exception as exc:
                    # A single malformed visualization payload must not kill
                    # this latest-only worker and stop all later web state.
                    rospy.logwarn_throttle(5.0, "physical ROS state post %s: %s", name, exc)

    def _json_callback(self, name: str):
        def callback(msg: String) -> None:
            # In headless mode these topics have no consumer in this process;
            # skip even JSON parsing so the optional gateway cannot contend
            # with semantic mapping.  The detector report callback remains a
            # separate path because it feeds ROS semantic consumers.
            if not bool(getattr(self, "web_state_enabled", True)):
                return
            try:
                # Lightweight replay/unit-test stubs may construct the
                # gateway with ``object.__new__`` and only provide the old
                # de-duplication fields.  Keep the new rate-limit state lazy
                # so that compatibility callers retain the same contract.
                if not hasattr(self, "_last_state_mirror_post_mono"):
                    self._last_state_mirror_post_mono = {}
                if not hasattr(self, "_pending_state_mirror_payloads"):
                    self._pending_state_mirror_payloads = {}
                payload = str(msg.data)
                with self._state_callback_payload_lock:
                    if self._last_state_callback_payloads.get(name) == payload:
                        return
                    # One worker owns parsing and delivery order. Sending an
                    # immediate update here can race with a deferred older
                    # payload and rewind the graph shown by the browser.
                    self._last_state_callback_payloads[name] = payload
                    self._pending_state_mirror_payloads[name] = payload
                self._state_post_event.set()
            except Exception as exc: rospy.logwarn_throttle(5.0, "physical ROS state %s: %s", name, exc)
        return callback

    def _telemetry_callback(self, msg: String) -> None:
        """Forward the latest ROS telemetry snapshot to the optional web lane."""
        try:
            payload = json.loads(msg.data)
            nested = (
                isinstance(payload, dict)
                and isinstance(payload.get("telemetry"), dict)
            )
            value = dict(payload.get("telemetry")) if nested else dict(payload)
            # Keep private transport-generation metadata when the producer
            # places it beside the nested telemetry object.  Older producers
            # put it inside the object, which the copy above already keeps.
            if nested and isinstance(payload, dict):
                for key in ("_transport_session", "_transport_connection"):
                    if key in payload:
                        value[key] = payload[key]
            if not isinstance(value, dict) or not value:
                return
            session = value.get("_transport_session")
            session = str(session) if session not in (None, "") else None
            try:
                connection = int(value.get("_transport_connection", -1))
            except (TypeError, ValueError):
                connection = -1
            # Keep the compatibility pose refresh source coherent when the
            # legacy publisher lane is explicitly enabled.  In the normal
            # direct-bridge launch this field is only diagnostic; the sensor
            # bridge remains the sole /tf publisher.
            with self._lock:
                self._telemetry = {
                    key: item
                    for key, item in value.items()
                    if key not in {"_transport_session", "_transport_connection"}
                }
                # This is the authoritative capture-time pose lane.  Keep a
                # short monotonic hold so the compatibility /odom projection
                # below cannot race in after a newer telemetry packet and
                # replace its yaw with a packet that has no capture sequence.
                # The hold is deliberately finite: if telemetry stalls, odom
                # remains a safe presentation-only fallback.
                self._last_direct_telemetry_mono = time.monotonic()
                if session is not None:
                    self._telemetry_transport_session = session
                    self._telemetry_transport_connection = connection
            self._post_state("telemetry", value)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "physical ROS telemetry state: %s", exc)

    @staticmethod
    def _direct_pose_is_recent(
        last_direct_telemetry_mono: float,
        now_mono: float,
        hold_sec: float = 0.5,
    ) -> bool:
        """Return whether capture-time telemetry should win over odom.

        The comparison is intentionally monotonic and independent of either
        source's wall clock.  A non-positive/unknown receipt means that no
        direct pose has arrived yet, so the odom compatibility lane may be
        used.  Keeping this policy as a pure helper makes the callback race
        testable without starting ROS.
        """

        try:
            last = float(last_direct_telemetry_mono)
            now = float(now_mono)
            hold = max(0.0, float(hold_sec))
        except (TypeError, ValueError):
            return False
        age = now - last
        return (
            last > 0.0
            and math.isfinite(last)
            and math.isfinite(now)
            and 0.0 <= age <= hold
        )

    def _odom_callback(self, msg: Odometry) -> None:
        """Mirror capture-time odom pose to the presentation-only web lane.

        Do not publish or transform anything here: this callback is strictly
        a tiny state projection.  In particular, it must not become a second
        TF/odom writer (the direct sensor bridge is the sole owner).
        """
        try:
            now_mono = time.monotonic()
            with self._lock:
                direct_pose_recent = self._direct_pose_is_recent(
                    self._last_direct_telemetry_mono,
                    now_mono,
                    self._direct_pose_hold_sec,
                )
            if direct_pose_recent:
                # /physical_nav/telemetry carries the pose captured with the
                # RGB-D frame and transport generation.  Odom has neither a
                # capture sequence nor source-generation marker, so allowing
                # it to overwrite the direct lane causes the browser arrow
                # to stick or jump when the two callbacks cross in flight.
                return
            position = msg.pose.pose.position
            quaternion = msg.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (float(quaternion.w) * float(quaternion.z)
                        + float(quaternion.x) * float(quaternion.y)),
                1.0 - 2.0 * (float(quaternion.y) ** 2
                              + float(quaternion.z) ** 2),
            )
            stamp = float(msg.header.stamp.to_sec())
            if not math.isfinite(stamp) or stamp <= 0.0:
                stamp = time.time()
            linear = msg.twist.twist.linear
            value = {
                "received_at": stamp,
                "position": [
                    float(position.x), float(position.y), float(position.z)
                ],
                "velocity": [float(linear.x), float(linear.y), float(linear.z)],
                "yaw": float(yaw),
                "odom_frame": str(msg.header.frame_id or "tf_frame_odom"),
                "pose_source": "physical_nav_odom",
            }
            with self._lock:
                self._telemetry.update(value)
                session = self._telemetry_transport_session
                connection = self._telemetry_transport_connection
            # Odometry is a second, high-rate presentation lane.  Attach the
            # latest telemetry generation so a restarted source can advance
            # the web arrow even if the RGB-D mirror is temporarily absent.
            if session is not None:
                value["_transport_session"] = session
                value["_transport_connection"] = connection
            self._post_state("telemetry", value)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "physical ROS odom state: %s", exc)

    def _subgoal_callback(self, msg: PointStamped) -> None:
        self._post_state("current_subgoal", {"point": [float(msg.point.x), float(msg.point.y), float(msg.point.z)], "frame_id": str(msg.header.frame_id or "")})

    @staticmethod
    def _plan_payload(msg: Path) -> dict[str, Any]:
        poses = []
        for stamped in msg.poses:
            p = stamped.pose.position
            q = stamped.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            poses.append([float(p.x), float(p.y), float(yaw)])
        return {"frame_id": str(msg.header.frame_id or ""), "poses": poses}

    def _plan_callback(self, name: str):
        def callback(msg: Path) -> None:
            self._post_state(name, self._plan_payload(msg))
        return callback

    @staticmethod
    def _grid_payload(msg: OccupancyGrid, *, map_session: str = "") -> dict[str, Any]:
        origin = msg.info.origin
        payload = {
            "width": int(msg.info.width),
            "height": int(msg.info.height),
            "resolution": float(msg.info.resolution),
            "frame_id": str(getattr(msg.header, "frame_id", "") or ""),
            "origin": {
                "x": float(origin.position.x),
                "y": float(origin.position.y),
                "z": float(origin.position.z),
                "qx": float(origin.orientation.x),
                "qy": float(origin.orientation.y),
                "qz": float(origin.orientation.z),
                "qw": float(origin.orientation.w),
            },
            "data": list(msg.data),
        }
        # Keep replay/test payloads byte-for-byte compatible when no session
        # token is available; live ROS gateway instances always provide one.
        if str(map_session or "").strip():
            payload["map_session"] = str(map_session).strip()
        return payload

    def _grid_callback(self, name: str):
        def callback(msg: OccupancyGrid) -> None:
            # The dashboard mirror is optional.  When it is disabled, this
            # gateway must not even retain a ROS OccupancyGrid: converting its
            # data array to a Python list in ``_grid_post_loop`` is a large
            # copy that can contend with the authoritative mapping callbacks.
            if not bool(getattr(self, "web_state_enabled", True)):
                return
            with self._grid_post_lock:
                # Every receipt advances a cheap monotonic revision and
                # replaces the same-name pending slot. Rate limiting belongs
                # to the worker: throttling here would retain the first frame
                # in a period instead of the newest one.
                revisions = getattr(self, "_grid_post_revisions", None)
                if revisions is None:
                    revisions = self._grid_post_revisions = {}
                revision = int(revisions.get(name, 0)) + 1
                revisions[name] = revision
                self._pending_grid_posts[name] = (revision, msg)
                self._grid_post_event.set()
        return callback

    def _grid_post_period(self, name: str) -> float:
        period = (
            self.args.room_grid_period
            if name == "room_grid"
            else self.args.occupancy_period
        )
        return max(0.0, float(period))

    def _grid_revision_is_current(self, name: str, revision: int) -> bool:
        with self._grid_post_lock:
            revisions = getattr(self, "_grid_post_revisions", {})
            return int(revisions.get(name, -1)) == int(revision)

    def _claim_next_grid_post(
        self, *, now: float | None = None
    ) -> tuple[tuple[str, int, OccupancyGrid] | None, float]:
        """Claim one due grid without freezing the other latest slots.

        OCC is selected first when it is due. Other stages remain in the
        shared per-name slots while the claimed item is materialized/sent, so
        callbacks can replace them with fresher ROS messages in the meantime.
        The returned delay is the earliest time at which a retained item
        becomes eligible under its presentation-only rate limit.
        """

        current_time = time.monotonic() if now is None else float(now)
        with self._grid_post_lock:
            if not self._pending_grid_posts:
                self._grid_post_event.clear()
                self._last_grid_claimed_name = None
                return None, 0.5

            last_posts = getattr(self, "_last_grid_post", None)
            if last_posts is None:
                last_posts = self._last_grid_post = {}
            due_names: list[str] = []
            next_delays: list[float] = []
            for candidate in self._pending_grid_posts:
                remaining = self._grid_post_period(candidate) - (
                    current_time - float(last_posts.get(candidate, 0.0))
                )
                if remaining <= 0.0:
                    due_names.append(candidate)
                else:
                    next_delays.append(remaining)

            if not due_names:
                # Clear only while holding the same lock used by callbacks;
                # a callback arriving afterwards will set the event again.
                self._grid_post_event.clear()
                return None, min(next_delays, default=0.5)

            # OCC drives mapping visibility and is therefore the first item
            # claimed from a mixed batch.  Do not select it twice in a row
            # while another due grid is waiting, though: if localhost HTTP is
            # slower than the OCC period, a continuously replaced OCC slot
            # would otherwise starve room/global/local visualization forever.
            # Among the remaining maps, service the stage whose previous
            # attempt is oldest to keep the policy deterministic and fair.
            last_claimed = getattr(self, "_last_grid_claimed_name", None)
            selectable = due_names
            if last_claimed == "occupancy":
                non_occupancy = [
                    candidate
                    for candidate in due_names
                    if candidate != "occupancy"
                ]
                if non_occupancy:
                    selectable = non_occupancy
            selected = min(
                selectable,
                key=lambda candidate: (
                    0 if candidate == "occupancy" else 1,
                    float(last_posts.get(candidate, 0.0)),
                    candidate,
                ),
            )
            revision, msg = self._pending_grid_posts.pop(selected)
            last_posts[selected] = current_time
            self._last_grid_claimed_name = selected
            return (selected, int(revision), msg), 0.0

    def _grid_post_loop(self) -> None:
        wait_timeout = 0.5
        while not rospy.is_shutdown():
            self._grid_post_event.wait(wait_timeout)
            # A launch can disable the optional web mirror while this worker
            # is waking up.  Drop any receipt without materialising its full
            # OccupancyGrid payload in that mode.
            if not bool(getattr(self, "web_state_enabled", True)):
                with self._grid_post_lock:
                    self._pending_grid_posts.clear()
                    self._grid_post_event.clear()
                wait_timeout = 0.5
                continue

            while not rospy.is_shutdown():
                claimed, next_delay = self._claim_next_grid_post()
                if claimed is None:
                    wait_timeout = max(0.01, min(0.5, float(next_delay)))
                    break
                name, revision, msg = claimed
                try:
                    # Keep map JSON serialization and HTTP I/O off the shared
                    # telemetry/detection state worker.  Occupancy grids are
                    # intentionally large; they must not make a 10-Hz pose
                    # update wait behind a map upload.
                    self._post_grid_state(name, msg, revision=revision)
                except Exception as exc:
                    rospy.logwarn_throttle(5.0, "physical grid state %s: %s", name, exc)

    def _post_grid_state(
        self,
        name: str,
        msg: OccupancyGrid,
        *,
        revision: int | None = None,
    ) -> bool:
        """Materialize and mirror one map receipt on the dedicated map lane."""
        is_current = (
            None
            if revision is None
            else lambda: self._grid_revision_is_current(name, revision)
        )
        # Check immediately before and after ``list(msg.data)`` in
        # ``_grid_payload``. A callback that replaces this map during the copy
        # makes the result obsolete before JSON serialization begins.
        if is_current is not None and not is_current():
            return False
        payload = self._grid_payload(
            msg, map_session=getattr(self, "_web_session_id", "")
        )
        if is_current is not None and not is_current():
            return False
        if is_current is None:
            # Preserve the simple two-argument compatibility hook used by
            # replay/unit-test stubs that invoke this helper directly.
            self._post_state_http(name, payload)
            return True
        return self._post_state_http(name, payload, is_current=is_current)

    def _occupancy_callback(self, msg: Any) -> None:
        """Compatibility callback retained for small unit-test stubs."""
        now = time.monotonic()
        if now - self._last_occupancy_post < self.args.occupancy_period:
            return
        self._last_occupancy_post = now
        self._post_state(
            "occupancy",
            self._grid_payload(
                msg, map_session=getattr(self, "_web_session_id", "")
            ),
        )

    def _claim_frame(self, raw: dict[str, Any]) -> bool:
        """Accept each capture once, including across bridge reconnects.

        The dog bridge sequence is process-local and resets when that process
        restarts. Capture time is stable across such restarts and also lets us
        reject delayed packets from an older WebSocket session.
        """
        if not raw.get("rgb") or not raw.get("depth"):
            return False
        seq = int(raw.get("seq", -1))
        bridge_seq = int(raw.get("_bridge_seq", -1) or -1)
        bridge_session = raw.get("_transport_session")
        bridge_session = str(bridge_session) if bridge_session not in (None, "") else None
        active_bridge_session = getattr(self, "_active_bridge_session", None)
        last_bridge_seq = int(getattr(self, "_last_bridge_seq", -1))
        if bridge_session is not None and bridge_session != active_bridge_session:
            self._active_bridge_session = bridge_session
            last_bridge_seq = -1
            self._last_bridge_seq = last_bridge_seq
        if bridge_seq >= 0:
            # The local sensor bridge sequence is monotonic across source
            # stamp/Go2 sequence resets.  Use it whenever present; raw stamp
            # remains the compatibility path for old replay packets.
            if bridge_seq <= last_bridge_seq:
                return False
            self._last_bridge_seq = bridge_seq
            self.last_seq = seq
            try:
                self.last_stamp = max(self.last_stamp, float(raw.get("stamp", 0.0)))
            except (TypeError, ValueError):
                pass
            return True
        try:
            stamp = float(raw.get("stamp", 0.0))
        except (TypeError, ValueError):
            stamp = 0.0
        has_stamp = math.isfinite(stamp) and stamp > 0.0
        if has_stamp:
            if stamp <= self.last_stamp:
                return False
        elif seq <= self.last_seq:
            return False
        if self.last_seq >= 0 and seq < self.last_seq:
            rospy.loginfo(
                "physical sensor sequence reset: %d -> %d; accepting newer capture stamp %.6f",
                self.last_seq,
                seq,
                stamp,
            )
        self.last_seq = seq
        if has_stamp:
            self.last_stamp = stamp
        return True

    def _poll(self, _event: Any) -> None:
        try:
            with urllib.request.urlopen(self.args.web_url.rstrip("/") + "/api/raw-frame", timeout=.5) as response:
                raw = json.loads(response.read().decode())
            if not self._claim_frame(raw):
                return
            self.last_raw = raw
            self._publish(raw)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "physical raw-frame polling: %s", exc)

    def _poll_state(self, _event: Any) -> None:
        if not self._state_poll_lock.acquire(blocking=False):
            return
        try:
            self._publish_state()
        finally:
            self._state_poll_lock.release()

    def _refresh_pose_tf(self, _event: Any) -> None:
        """Refresh dynamic TF independently of the camera/WebSocket cadence."""
        telemetry = self._telemetry
        if telemetry:
            self._publish_pose(telemetry, rospy.Time.now())

    def _publish_state(self, state_override: dict[str, Any] | None = None) -> None:
        try:
            # ROS/RViz uses a dedicated projection containing bounded compact
            # point clouds. Browser dashboards keep polling the lighter
            # /api/state-summary and do not pay this transport cost.
            if state_override is None:
                with urllib.request.urlopen(self.args.web_url.rstrip("/") + "/api/ros-state", timeout=.6) as response: state = json.loads(response.read().decode())
            else:
                state = state_override
            detection_meta = state.get("detection_meta") or {
                "seq": state.get("seq", -1),
                "stamp": state.get("stamp", 0.0),
            }
            receipt = (
                detection_meta.get("seq", -1),
                detection_meta.get("stamp", 0.0),
            ) if isinstance(detection_meta, dict) else (-1, 0.0)
            detection_stamp = rospy.Time.from_sec(
                float(receipt[1] or state.get("frame_stamp", time.time()) or time.time())
            )
            # Keep the capture timestamp. Re-dating delayed detections to the
            # current wall time makes their boxes use a newer TF than the
            # point cloud and produces apparent motion/rotation in the map.
            if receipt == self._last_detection_receipt:
                # Do not count the same HTTP state snapshot as repeated YOLO
                # confirmations. Keep publishing held tracks so their timeout
                # remains observable even if the detector stops completely.
                self._publish_held_world_boxes(detection_stamp)
                return
            self._last_detection_receipt = receipt
            # The detector report and the RGB-D/cloud messages carry the
            # capture timestamp.  Looking up ``Time(0)`` here would apply the
            # robot's *current* pose to an older frame after it has moved or
            # rotated, which is exactly the box/cloud skew seen on the map.
            # Cache by source frame for this report (all detections from one
            # YOLO receipt share its capture stamp).
            transform_cache: dict[str, Any] = {}
            for item in state.get("detections", []):
                if not isinstance(item, dict):
                    continue
                source_frame = str(item.get("source_frame", "") or "")
                if source_frame not in transform_cache:
                    transform_cache[source_frame] = (
                        self._lookup_capture_transform(source_frame, detection_stamp)
                        if source_frame
                        else None
                    )
            # Mapping and the optional debug-cloud worker consume the same
            # packed segment samples. Decode each receipt once and pass the
            # short-lived arrays to both consumers instead of base64-decoding
            # the same payload again on the visualization timer.
            segment_cache: dict[int, dict[str, np.ndarray]] = {}
            detections = []
            for item in state.get("detections", []):
                if not isinstance(item, dict):
                    continue
                detections.append(
                    self._map_detection(
                        item,
                        transform_cache.get(str(item.get("source_frame", "") or "")),
                        segment_cache=segment_cache,
                    )
                )
            # Use the same history-smoothed portal geometry for semantic
            # mapping that is used by boxes_3d_world. Otherwise the point
            # cloud/marker follows the stable OBB while Graph receives the
            # raw PCA axis and can be rotated by roughly 90 degrees.
            stable_preview = self._stable_world_boxes(detections)
            self._apply_associated_portal_geometry(detections, stable_preview)
            # A geometry-capped detection is retained in ``detections`` for
            # the camera overlay, but it is not a valid 3-D observation.  Do
            # not send it to semantic mapping: the legacy ObjectMapStore
            # falls back to an all-zero position/size when world geometry is
            # absent, which creates a ghost object at the map origin.
            mapping_detections = [
                _semantic_detection_payload(item)
                for item in detections
                if not bool(item.get("geometry_skipped", False))
            ]
            detection_envelope = {
                "seq": receipt[0],
                "stamp": receipt[1],
                "detections": mapping_detections,
            }
            self.detection_pub.publish(json.dumps(detection_envelope, ensure_ascii=False, separators=(",", ":")))
            if self._publisher_has_subscribers(self.attribute_detection_pub):
                attribute_envelope = dict(detection_envelope)
                attribute_envelope["detections"] = [
                    self._attribute_detection(item) for item in mapping_detections
                ]
                self.attribute_detection_pub.publish(
                    json.dumps(attribute_envelope, ensure_ascii=False, separators=(",", ":"))
                )
            with self._debug_cloud_lock:
                # ``stable_preview`` already updates the temporal box track
                # once for this receipt.  Pass it to the debug worker so the
                # timer does not count the same frame a second time.
                self._pending_debug_cloud = (
                    detections,
                    transform_cache,
                    detection_stamp,
                    stable_preview,
                    segment_cache,
                )
            self._queue_detection_overlay(state)
            # Keep a compact, map-aligned evidence view for the LAN page while
            # preserving the raw YOLOE masks in ``detections``.
            compact = [
                {
                    key: value
                    for key, value in _semantic_detection_payload(item).items()
                    if key
                    not in {
                        "mask",
                        "camera_segment_points",
                        "world_segment_points",
                        "camera_segment_points_f32",
                        "world_segment_points_f32",
                        "_camera_segment_points_array",
                        "_world_segment_points_array",
                    }
                }
                for item in detections
            ]
            self._post_mapped_state({"seq": receipt[0], "stamp": receipt[1], "map_frame": self.world_frame, "detections": compact})
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "physical state polling: %s", exc)

    def _yolo_report_callback(self, msg: String) -> None:
        """Consume detector output directly from ROS; HTTP is only a mirror."""
        with self._yolo_report_lock:
            self._pending_yolo_report = str(msg.data)
            self._yolo_report_event.set()

    def _yolo_report_loop(self) -> None:
        while not rospy.is_shutdown():
            self._yolo_report_event.wait(0.5)
            with self._yolo_report_lock:
                payload = self._pending_yolo_report
                self._pending_yolo_report = None
                self._yolo_report_event.clear()
            if payload is None:
                continue
            if not self._state_poll_lock.acquire(blocking=False):
                # A prior mapping pass is still running. Keep the newest
                # payload for the next worker iteration instead of queueing
                # historical frames.
                with self._yolo_report_lock:
                    # The ROS callback may have installed a newer report
                    # between the worker's take and this lock reacquire. Do
                    # not overwrite that newer receipt with the older one.
                    if self._pending_yolo_report is None:
                        self._pending_yolo_report = payload
                    self._yolo_report_event.set()
                # A mapping pass can legitimately last tens of milliseconds.
                # Do not spin at 1 kHz while waiting for its lock.
                time.sleep(0.02)
                continue
            try:
                report = json.loads(payload)
                if isinstance(report, dict):
                    self._publish_state(report)
            except Exception as exc:
                rospy.logwarn_throttle(5.0, "physical YOLO ROS report: %s", exc)
            finally:
                self._state_poll_lock.release()

    @staticmethod
    def _attribute_detection(item: dict[str, Any]) -> dict[str, Any]:
        """Add the minimum stable/public fields required by physical M1."""
        result = dict(item)
        label = str(
            item.get("semantic_class")
            or item.get("raw_class")
            or item.get("class")
            or "object"
        ).strip().casefold()
        result.setdefault("semantic_name", label)
        result.setdefault("category", label)
        result.setdefault("name", label)
        result.setdefault("bbox_2d", item.get("bbox"))
        if result.get("visible_fraction") is None:
            bbox = item.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                area = max(1.0, abs(float(bbox[2]) - float(bbox[0])) * abs(float(bbox[3]) - float(bbox[1])))
                result["visible_fraction"] = min(
                    1.0, float(item.get("mask_area", 0) or 0) / area
                )
        if result.get("distance_m") is None:
            result["distance_m"] = item.get("depth_median_m")

        # Quantized map-space identity is stable across nearby frames but is
        # deliberately scoped to this M1-only topic.  It avoids one Qwen call
        # per image without forcing raw detector IDs into ObjectMapStore.
        center = item.get("world_position") or item.get("position") or {}
        try:
            if isinstance(center, dict):
                xyz = [float(center.get(axis, 0.0)) for axis in ("x", "y", "z")]
            else:
                xyz = [float(value) for value in center[:3]]
            cells = [int(round(value / 0.5)) for value in xyz]
            result.setdefault(
                "instance_id",
                "physical_{}_{}_{}_{}".format(label.replace(" ", "_"), *cells),
            )
        except (TypeError, ValueError, IndexError):
            result.setdefault(
                "instance_id",
                f"physical_{label.replace(' ', '_')}_{int(item.get('capture_seq', 0) or 0)}",
            )
        return result

    def _post_mapped_state(self, value: dict[str, Any]) -> None:
        # ``_post_state`` is already an O(1), latest-only dictionary update;
        # the HTTP worker drains that queue independently.  Spawning another
        # short-lived thread for every detector receipt only adds scheduling
        # pressure to the camera/OCC callbacks.
        self._post_state("mapped_detections", value)

    @staticmethod
    def _publisher_has_subscribers(publisher: Any) -> bool:
        """Conservatively gate compatibility-only ROS serialization."""

        try:
            return int(publisher.get_num_connections()) > 0
        except (AttributeError, TypeError, ValueError):
            # Unit/replay publishers may not expose ROS connection counts.
            return True

    @staticmethod
    def _point_dict(point: tuple[float, float, float]) -> dict[str, float]:
        return {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}

    @staticmethod
    def _point3(value: Any) -> tuple[float, float, float] | None:
        if isinstance(value, dict):
            try:
                return tuple(float(value.get(axis, 0.0)) for axis in ("x", "y", "z"))
            except (TypeError, ValueError):
                return None
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return tuple(float(item) for item in value[:3])
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _debug_color(label: str) -> tuple[int, int, int]:
        palette = (
            (245, 90, 70), (70, 200, 245), (80, 220, 120),
            (235, 190, 60), (180, 90, 235), (65, 150, 245),
            (235, 90, 170), (90, 220, 215),
        )
        token = str(label or "object")
        index = sum((offset + 1) * ord(char) for offset, char in enumerate(token))
        return palette[index % len(palette)]

    @staticmethod
    def _rgb_float(color: tuple[int, int, int]) -> float:
        red, green, blue = color
        packed = (int(red) << 16) | (int(green) << 8) | int(blue)
        return struct.unpack("f", struct.pack("I", packed))[0]

    @classmethod
    def _colored_cloud(cls, stamp: Any, frame: str, rows: list[tuple[float, float, float, float]]) -> PointCloud2:
        header = Header(stamp=stamp, frame_id=frame)
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]
        return pc2.create_cloud(header, fields, rows)

    @classmethod
    def _box_marker_array(
        cls,
        detections: list[dict[str, Any]],
        stamp: Any,
        frame: str,
        *,
        center_key: str,
        size_key: str,
        orientation_key: str | None = None,
        marker_size_key: str | None = None,
        persistent: bool = False,
    ) -> MarkerArray:
        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        markers = [clear]
        edges = (
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        )
        for index, detection in enumerate(detections):
            center = cls._point3(detection.get(center_key))
            size = cls._point3(detection.get(marker_size_key or size_key))
            if center is None or size is None:
                continue
            size = tuple(max(abs(float(value)), 0.01) for value in size)
            color = cls._debug_color(str(detection.get("semantic_class") or "object"))
            corners = [
                Point(
                    x=center[0] + sx * size[0] * 0.5,
                    y=center[1] + sy * size[1] * 0.5,
                    z=center[2] + sz * size[2] * 0.5,
                )
                for sx, sy, sz in (
                    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
                )
            ]
            box = Marker()
            box.header.stamp = stamp
            box.header.frame_id = frame
            box.ns = "physical_yoloe_boxes"
            track_id = int(detection.get("visualization_track_id", index))
            box.id = track_id * 2
            box.type = Marker.LINE_LIST
            box.action = Marker.ADD
            orientation = detection.get(orientation_key) if orientation_key else None
            if isinstance(orientation, (list, tuple)) and len(orientation) >= 4:
                box.pose.orientation.x = float(orientation[0])
                box.pose.orientation.y = float(orientation[1])
                box.pose.orientation.z = float(orientation[2])
                box.pose.orientation.w = float(orientation[3])
            else:
                box.pose.orientation.w = 1.0
            box.pose.position.x, box.pose.position.y, box.pose.position.z = center
            box.scale.x = 0.025
            box.color.r, box.color.g, box.color.b = [channel / 255.0 for channel in color]
            box.color.a = 0.95
            box.lifetime = rospy.Duration(0.0 if persistent else 0.75)
            for first, second in edges:
                # Points are local to the marker pose so RViz applies the
                # measured camera/base orientation instead of forcing every
                # box to be world-axis aligned.
                local_first = Point(
                    x=(corners[first].x - center[0]),
                    y=(corners[first].y - center[1]),
                    z=(corners[first].z - center[2]),
                )
                local_second = Point(
                    x=(corners[second].x - center[0]),
                    y=(corners[second].y - center[1]),
                    z=(corners[second].z - center[2]),
                )
                box.points.extend((local_first, local_second))
            markers.append(box)

            label = Marker()
            label.header = box.header
            label.ns = "physical_yoloe_labels"
            label.id = track_id * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = center[0]
            label.pose.position.y = center[1]
            label.pose.position.z = center[2] + size[2] * 0.5 + 0.08
            label.pose.orientation.w = 1.0
            label.scale.z = 0.12
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.lifetime = box.lifetime
            label.text = "%s %.2f" % (
                str(detection.get("semantic_class") or "object"),
                float(detection.get("confidence", 0.0) or 0.0),
            )
            markers.append(label)
        return MarkerArray(markers=markers)

    @staticmethod
    def _bbox_iou(first: Any, second: Any) -> float:
        if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)):
            return 0.0
        if len(first) < 4 or len(second) < 4:
            return 0.0
        left = max(float(first[0]), float(second[0]))
        top = max(float(first[1]), float(second[1]))
        right = min(float(first[2]), float(second[2]))
        bottom = min(float(first[3]), float(second[3]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, float(first[2]) - float(first[0])) * max(
            0.0, float(first[3]) - float(first[1])
        )
        second_area = max(0.0, float(second[2]) - float(second[0])) * max(
            0.0, float(second[3]) - float(second[1])
        )
        union = first_area + second_area - intersection
        return intersection / union if union > 1e-9 else 0.0

    @staticmethod
    def _aabb_iou(
        first_center: tuple[float, float, float],
        first_size: tuple[float, float, float],
        second_center: tuple[float, float, float],
        second_size: tuple[float, float, float],
    ) -> float:
        first_min = [first_center[i] - abs(first_size[i]) * 0.5 for i in range(3)]
        first_max = [first_center[i] + abs(first_size[i]) * 0.5 for i in range(3)]
        second_min = [second_center[i] - abs(second_size[i]) * 0.5 for i in range(3)]
        second_max = [second_center[i] + abs(second_size[i]) * 0.5 for i in range(3)]
        overlap = [
            max(0.0, min(first_max[i], second_max[i]) - max(first_min[i], second_min[i]))
            for i in range(3)
        ]
        intersection = overlap[0] * overlap[1] * overlap[2]
        first_volume = abs(first_size[0] * first_size[1] * first_size[2])
        second_volume = abs(second_size[0] * second_size[1] * second_size[2])
        union = first_volume + second_volume - intersection
        return intersection / union if union > 1e-9 else 0.0

    @staticmethod
    def _box_track_detection(detection: dict[str, Any]) -> dict[str, Any]:
        """Copy only metadata needed by the persistent marker-box track.

        A YOLO receipt may carry a full mask and packed segment cloud. Keeping
        those large payloads in every held track makes each 10-Hz smoothing
        update copy megabytes of presentation-only data and needlessly retains
        it for ``debug_box_hold_s``.  The marker/portal association consumes
        only the small semantic, bbox, confidence, pose, and orientation
        fields, so drop raw diagnostic payloads at this boundary.
        """
        large_keys = {
            "mask",
            "mask_polygons",
            "camera_segment_points",
            "world_segment_points",
            "camera_segment_points_f32",
            "world_segment_points_f32",
            "_camera_segment_points_array",
            "_world_segment_points_array",
        }
        return {
            key: value
            for key, value in detection.items()
            if key not in large_keys
        }

    @staticmethod
    def _box_track_family(label: Any) -> str:
        """Return a conservative association family for display tracks."""

        normalized = str(label or "object").strip().casefold()
        if normalized in {"door", "portal", "gate", "sliding door"}:
            return "portal"
        if normalized in {"fridge", "refrigerator", "locker"}:
            # Locker is the detector's provisional label for appliances that
            # M1 can later confirm as a refrigerator/water dispenser.
            return "refrigerator_candidate"
        return normalized

    def _apply_associated_portal_geometry(
        self,
        detections: list[dict[str, Any]],
        stable: list[dict[str, Any]],
    ) -> None:
        """Apply smoothing only through the exact frame-to-track association.

        Held tracks are useful for RViz, but an unmatched current door must
        never borrow the nearest held door's box. The association created by
        ``_stable_world_boxes`` is one-to-one and already passed its geometry
        gate; no nearest-neighbour search is repeated here.
        """

        stable_by_id = {
            int(item["visualization_track_id"]): item
            for item in stable
            if item.get("visualization_track_id") is not None
        }
        for detection in detections:
            track_id = detection.pop("_associated_visualization_track_id", None)
            if self._box_track_family(
                detection.get("semantic_class")
            ) != "portal" or track_id is None:
                continue
            stable_detection = stable_by_id.get(int(track_id))
            if stable_detection is None or self._box_track_family(
                stable_detection.get("semantic_class")
            ) != "portal":
                continue
            stable_center = self._point3(
                stable_detection.get("world_box3d_center")
            )
            stable_size = self._point3(
                stable_detection.get("world_box3d_size")
            )
            if stable_center is None or stable_size is None:
                continue
            orientation = list(
                stable_detection.get("world_box3d_orientation")
                or [0.0, 0.0, 0.0, 1.0]
            )
            yaw = stable_detection.get("world_box3d_yaw")
            detection.update(
                world_box3d_center=list(stable_center),
                world_box3d_size=list(stable_size),
                world_box3d_marker_size=list(stable_size),
                world_box3d_orientation=orientation,
                world_box3d_yaw=yaw,
                yaw=yaw,
                aabb_center=list(stable_center),
                aabb_size=list(stable_size),
                box3d_center=list(stable_center),
                box3d_size=list(stable_size),
            )

    def _stable_world_boxes(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Associate and smooth world-frame boxes for display only.

        The semantic mapper continues to consume raw detections.  This cache
        affects only ``boxes_3d_world`` so a missed YOLO frame or a short label
        flip does not make RViz/Foxglove markers jump or disappear.
        """
        now = time.monotonic()
        unmatched = set(self._world_box_tracks)
        alpha = self._box_smoothing_alpha
        for detection in sorted(
            detections,
            key=lambda item: float(item.get("confidence", 0.0) or 0.0),
            reverse=True,
        ):
            center = self._point3(detection.get("world_box3d_center"))
            size = self._point3(detection.get("world_box3d_size"))
            if center is None or size is None:
                continue
            if not all(math.isfinite(value) for value in center + size):
                continue
            size = tuple(max(0.01, abs(value)) for value in size)
            family = self._box_track_family(detection.get("semantic_class"))
            best_id = None
            best_score = float("inf")
            for track_id in unmatched:
                track = self._world_box_tracks[track_id]
                if track.get("family") != family:
                    continue
                old_center = track["center"]
                old_size = track["size"]
                distance = math.sqrt(sum((center[i] - old_center[i]) ** 2 for i in range(3)))
                overlap_3d = self._aabb_iou(center, size, old_center, old_size)
                overlap_2d = self._bbox_iou(detection.get("bbox"), track.get("bbox"))
                if distance > self._box_match_distance_m and overlap_3d < 0.05:
                    continue
                score = distance - 0.35 * overlap_3d - 0.15 * overlap_2d
                if score < best_score:
                    best_id, best_score = track_id, score
            if best_id is None:
                best_id = self._next_world_box_track_id
                self._next_world_box_track_id += 1
                self._world_box_tracks[best_id] = {
                    "center": center,
                    "size": size,
                    "bbox": detection.get("bbox"),
                    "hits": 0,
                    "last_seen": now,
                    "family": family,
                    "class_scores": {},
                    "detection": self._box_track_detection(detection),
                }
            else:
                unmatched.discard(best_id)
            track = self._world_box_tracks[best_id]
            detection["_associated_visualization_track_id"] = best_id
            if track["hits"] > 0:
                track["center"] = tuple(
                    (1.0 - alpha) * track["center"][i] + alpha * center[i]
                    for i in range(3)
                )
                track["size"] = tuple(
                    (1.0 - alpha) * track["size"][i] + alpha * size[i]
                    for i in range(3)
                )
            track["hits"] += 1
            track["last_seen"] = now
            track["bbox"] = detection.get("bbox")
            label = str(detection.get("semantic_class") or "object")
            scores = track["class_scores"]
            scores[label] = float(scores.get(label, 0.0)) + float(
                detection.get("confidence", 0.0) or 0.0
            )
            track["detection"] = self._box_track_detection(detection)
            if label.casefold() in {"door", "portal", "gate"}:
                orientation = detection.get("world_box3d_orientation")
                raw_yaw = None
                if isinstance(orientation, (list, tuple)) and len(orientation) >= 4:
                    raw_yaw = self._quaternion_yaw([float(value) for value in orientation[:4]])
                elif detection.get("world_box3d_yaw") is not None:
                    try:
                        raw_yaw = float(detection.get("world_box3d_yaw"))
                    except (TypeError, ValueError):
                        raw_yaw = None
                if raw_yaw is not None and math.isfinite(raw_yaw):
                    previous_yaw = track.get("yaw")
                    if previous_yaw is None:
                        track["yaw"] = raw_yaw
                    else:
                        candidates = [raw_yaw + index * (math.pi * 0.5) for index in range(-2, 3)]
                        aligned_yaw = min(candidates, key=lambda value: abs(0.5 * math.atan2(
                            math.sin(2.0 * (value - float(previous_yaw))),
                            math.cos(2.0 * (value - float(previous_yaw))),
                        )))
                        delta = 0.5 * math.atan2(
                            math.sin(2.0 * (aligned_yaw - float(previous_yaw))),
                            math.cos(2.0 * (aligned_yaw - float(previous_yaw))),
                        )
                        track["yaw"] = float(previous_yaw) if abs(delta) > math.pi * 0.25 else float(previous_yaw) + 0.20 * delta

        expired = [
            track_id
            for track_id, track in self._world_box_tracks.items()
            if now - float(track["last_seen"]) > self._box_hold_s
        ]
        for track_id in expired:
            del self._world_box_tracks[track_id]

        stable = []
        for track_id, track in sorted(self._world_box_tracks.items()):
            if int(track["hits"]) < self._box_min_confirmations:
                continue
            item = dict(track["detection"])
            item["visualization_track_id"] = track_id
            item["world_box3d_center"] = list(track["center"])
            item["world_box3d_size"] = list(track["size"])
            item["world_box3d_marker_size"] = list(track["size"])
            if track["class_scores"]:
                item["semantic_class"] = max(
                    track["class_scores"], key=track["class_scores"].get
                )
            if str(item.get("semantic_class") or "").casefold() in {"door", "portal", "gate"} and track.get("yaw") is not None:
                item["world_box3d_orientation"] = [
                    0.0,
                    0.0,
                    math.sin(float(track["yaw"]) * 0.5),
                    math.cos(float(track["yaw"]) * 0.5),
                ]
                item["world_box3d_yaw"] = float(track["yaw"])
            elif item.get("world_box3d_yaw") is not None:
                # Non-portal graph boxes must not retain the detector's
                # pre-TF yaw when the marker has already been fitted in the
                # map frame.
                item["yaw"] = float(item["world_box3d_yaw"])
            stable.append(item)
        return stable

    def _publish_detection_overlay(self, state: dict[str, Any]) -> None:
        # Direct ROS detector reports use the compact YOLO report contract
        # (``seq``, ``stamp``, ``camera_frame`` and ``overlay_jpeg`` at the
        # top level), whereas the legacy HTTP state mirror wraps the same
        # fields in ``detection_meta``.  Normalize both forms here.  Without
        # this fallback the direct 10 Hz path publishes boxes/points but never
        # publishes ``/physical_nav/detections_overlay`` when the web mirror
        # is disabled, making the ROS/Foxglove detection image appear frozen.
        meta = state.get("detection_meta")
        if not isinstance(meta, dict):
            meta = {
                "seq": state.get("seq", -1),
                "stamp": state.get("stamp", state.get("frame_stamp", 0.0)),
                "camera_frame": state.get("camera_frame", self.args.camera_frame),
                "overlay_jpeg": state.get("overlay_jpeg"),
            }
        elif not meta.get("overlay_jpeg") and state.get("overlay_jpeg"):
            # Be tolerant of a mixed report assembled by an older gateway.
            meta = dict(meta)
            meta.setdefault("seq", state.get("seq", -1))
            meta.setdefault("stamp", state.get("stamp", state.get("frame_stamp", 0.0)))
            meta.setdefault("camera_frame", state.get("camera_frame", self.args.camera_frame))
            meta["overlay_jpeg"] = state.get("overlay_jpeg")
        encoded = meta.get("overlay_jpeg") if isinstance(meta, dict) else None
        if not encoded and not self.web_state_enabled:
            # In headless mode the detector report is authoritative and the
            # optional dashboard binary endpoint does not exist.
            return
        try:
            if encoded:
                rgb = _decode(str(encoded))
            else:
                # The compact state summary intentionally excludes the large
                # base64 overlay so detection tracking cannot miss its 0.6 s
                # deadline. Fetch the existing binary debug image separately.
                with urllib.request.urlopen(
                    self.args.web_url.rstrip("/") + "/camera-overlay.jpg",
                    timeout=0.4,
                ) as response:
                    payload = response.read()
                rgb = np.asarray(
                    PILImage.open(io.BytesIO(payload)).convert("RGB")
                )
            if rgb.ndim == 3 and rgb.shape[2] >= 3:
                # PIL returns RGB for both transport paths; ROS publishes bgr8.
                rgb = np.ascontiguousarray(rgb[:, :, :3][:, :, ::-1])
            stamp = rospy.Time.from_sec(
                float(meta.get("stamp", state.get("frame_stamp", time.time())) or time.time())
            )
            frame = str(meta.get("camera_frame") or self.args.camera_frame)
            self.detection_overlay_pub.publish(_image_msg(rgb, "bgr8", stamp, frame))
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "physical detection overlay: %s", exc)

    def _queue_detection_overlay(self, state: dict[str, Any]) -> None:
        """Retain only the newest overlay receipt for the presentation worker."""
        with self._overlay_lock:
            self._pending_overlay = state
            self._overlay_event.set()

    def _overlay_publish_loop(self) -> None:
        while not rospy.is_shutdown():
            self._overlay_event.wait(0.5)
            with self._overlay_lock:
                state = self._pending_overlay
                self._pending_overlay = None
                self._overlay_event.clear()
            if state is None:
                continue
            try:
                self._publish_detection_overlay(state)
            except Exception as exc:
                rospy.logwarn_throttle(5.0, "physical detection overlay worker: %s", exc)

    def _publish_detection_debug(
        self,
        detections: list[dict[str, Any]],
        transform_cache: dict[str, Any],
        stamp: Any,
        stable_world: list[dict[str, Any]] | None = None,
        segment_cache: dict[int, dict[str, np.ndarray]] | None = None,
    ) -> None:
        started = time.perf_counter()
        publish_camera_cloud = self._debug_cloud_requested(
            self.segmented_cloud_pub, self._publish_debug_segmented_cloud
        )
        publish_world_cloud = self._debug_cloud_requested(
            self.segmented_cloud_world_pub,
            self._publish_debug_world_segmented_cloud,
        )
        camera_rows: list[tuple[float, float, float, float]] = []
        world_rows: list[tuple[float, float, float, float]] = []
        camera_frame = self.args.camera_frame
        for detection in detections:
            source_frame = str(detection.get("source_frame") or self.args.camera_frame)
            camera_frame = source_frame or camera_frame
            transform = transform_cache.get(source_frame)
            rgb = self._rgb_float(self._debug_color(str(detection.get("semantic_class") or "object")))
            # Do not decode packed segment points unless at least one of the
            # optional cloud outputs is actually enabled and subscribed.
            # Marker boxes below remain independent and are still published.
            if not (publish_camera_cloud or publish_world_cloud):
                continue
            cached = (segment_cache or {}).get(id(detection), {})
            camera_points = cached.get("camera")
            if camera_points is None:
                camera_points = self._segment_points_array(
                    detection, "camera_segment_points_f32", "camera_segment_points"
                )
            if camera_points.size:
                camera_points = camera_points[
                    np.isfinite(camera_points).all(axis=1)
                ]
            if camera_points.size:
                if publish_camera_cloud:
                    camera_rows.extend(
                        (
                            float(point[0]),
                            float(point[1]),
                            float(point[2]),
                            rgb,
                        )
                        for point in camera_points
                    )
                if publish_world_cloud and transform is not None:
                    world_points = self._rotate_points(
                        transform.transform.rotation, camera_points
                    )
                    world_points = world_points + np.asarray(
                        [
                            float(transform.transform.translation.x),
                            float(transform.transform.translation.y),
                            float(transform.transform.translation.z),
                        ],
                        dtype=np.float32,
                    )
                    world_rows.extend(
                        (
                            float(point[0]),
                            float(point[1]),
                            float(point[2]),
                            rgb,
                        )
                        for point in world_points
                    )
            if publish_world_cloud and transform is None:
                world_points = cached.get("world")
                if world_points is None:
                    world_points = self._segment_points_array(
                        detection, "world_segment_points_f32", "world_segment_points"
                    )
                if world_points.size:
                    world_points = world_points[
                        np.isfinite(world_points).all(axis=1)
                    ]
                    world_rows.extend(
                        (
                            float(point[0]),
                            float(point[1]),
                            float(point[2]),
                            rgb,
                        )
                        for point in world_points
                    )
        if publish_camera_cloud:
            self.segmented_cloud_pub.publish(
                self._colored_cloud(stamp, camera_frame, camera_rows)
            )
        if publish_world_cloud:
            self.segmented_cloud_world_pub.publish(
                self._colored_cloud(stamp, self.world_frame, world_rows)
            )
        self.boxes_pub.publish(self._box_marker_array(
            detections,
            stamp,
            camera_frame,
            center_key="camera_box3d_center",
            size_key="camera_box3d_size",
        ))
        if stable_world is None:
            stable_world = self._stable_world_boxes(detections)
        self.boxes_world_pub.publish(self._box_marker_array(
            stable_world,
            stamp,
            self.world_frame,
            center_key="world_box3d_center",
            size_key="world_box3d_size",
            orientation_key="world_box3d_orientation",
            marker_size_key="world_box3d_marker_size",
            persistent=True,
        ))
        rospy.loginfo_throttle(
            10.0,
            "segmented cloud timing: %.1f ms, camera_points=%d world_points=%d detections=%d enabled=%s/%s",
            (time.perf_counter() - started) * 1000.0,
            len(camera_rows), len(world_rows), len(detections),
            publish_camera_cloud,
            publish_world_cloud,
        )

    @staticmethod
    def _debug_cloud_requested(publisher: Any, configured: bool) -> bool:
        """Return whether an optional debug cloud should be materialized."""
        if not configured:
            return False
        get_connections = getattr(publisher, "get_num_connections", None)
        if not callable(get_connections):
            # Minimal replay/fake publishers do not expose connection counts;
            # preserve the explicitly enabled behavior for those callers.
            return True
        try:
            return int(get_connections()) > 0
        except Exception:
            # A transient ROS master lookup failure must not disable a
            # requested visualization stream permanently.
            return True

    def _publish_pending_debug_cloud(self, _event: Any) -> None:
        """Publish only the newest visualization payload."""
        with self._debug_cloud_lock:
            pending = self._pending_debug_cloud
            self._pending_debug_cloud = None
        if pending is not None:
            # The optional fourth item is the already-updated stable-box
            # snapshot.  The fifth item is the per-receipt packed-point cache;
            # legacy three/four-field tuples remain valid for replay callers.
            if len(pending) >= 5:
                self._publish_detection_debug(*pending[:5])
            else:
                self._publish_detection_debug(*pending)

    def _publish_held_world_boxes(self, stamp: Any) -> None:
        stable_world = self._stable_world_boxes([])
        self.boxes_world_pub.publish(
            self._box_marker_array(
                stable_world,
                stamp,
                self.world_frame,
                center_key="world_box3d_center",
                size_key="world_box3d_size",
                orientation_key="world_box3d_orientation",
                marker_size_key="world_box3d_marker_size",
                persistent=True,
            )
        )

    @staticmethod
    def _segment_points(
        detection: dict[str, Any], binary_key: str, legacy_key: str
    ) -> list[tuple[float, float, float]]:
        """Return packed segment samples in the legacy Python-list format.

        This helper remains part of the small compatibility surface used by
        old replay callers.  The hot mapping/debug paths use
        :meth:`_segment_points_array` below so they do not allocate one tuple
        and one Python float object per point on every detector receipt.
        """
        values = PhysicalRosGateway._segment_points_array(
            detection, binary_key, legacy_key
        )
        return [tuple(float(axis) for axis in row) for row in values]

    @staticmethod
    def _segment_points_array(
        detection: dict[str, Any], binary_key: str, legacy_key: str
    ) -> np.ndarray:
        """Decode compact segment samples as a contiguous ``(N, 3)`` array.

        YOLOE already serializes debug points as little-endian float32.  The
        previous gateway path decoded them into a Python list, iterated each
        point for TF rotation, and converted the list back to NumPy in
        ``_map_detection``.  At 10 Hz this creates thousands of short-lived
        Python objects and makes JSON/report handling compete with mapping.
        Keep the wire contract unchanged while doing the conversion once in
        C/NumPy.  Legacy list payloads are accepted for old recordings.
        """
        encoded = detection.get(binary_key)
        if encoded:
            try:
                values = np.frombuffer(
                    base64.b64decode(str(encoded)), dtype="<f4"
                )
                if values.size and values.size % 3 == 0:
                    return values.reshape(-1, 3)
            except Exception:
                pass
        legacy = detection.get(legacy_key) or []
        if not legacy:
            return np.empty((0, 3), dtype=np.float32)
        rows: list[tuple[float, float, float]] = []
        for value in legacy:
            point = PhysicalRosGateway._point3(value)
            if point is not None:
                rows.append(point)
        if not rows:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def _rotate_points(
        quaternion: Any, points: np.ndarray
    ) -> np.ndarray:
        """Rotate an ``N x 3`` point array by an xyzw quaternion.

        ``tf2`` is intentionally not used here because this gateway also runs
        in the minimal Conda/ROS replay environment.  The vector form is the
        same ``q * p * q^-1`` expansion as :meth:`_rotate_point`, but avoids a
        Python loop for every packed segment sample.
        """
        values = np.asarray(
            [
                float(quaternion.x),
                float(quaternion.y),
                float(quaternion.z),
                float(quaternion.w),
            ],
            dtype=np.float32,
        )
        vectors = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if vectors.size == 0:
            return vectors.reshape(0, 3)
        qvec = values[:3]
        twice_cross = 2.0 * np.cross(qvec[None, :], vectors)
        return vectors + values[3] * twice_cross + np.cross(qvec[None, :], twice_cross)

    @staticmethod
    def _rotate_point(quaternion: Any, point: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z, w = (float(quaternion.x), float(quaternion.y), float(quaternion.z), float(quaternion.w))
        px, py, pz = point
        # q * p * q^-1, expanded to avoid tf2_geometry_msgs/cv_bridge deps.
        tx = 2.0 * (y * pz - z * py); ty = 2.0 * (z * px - x * pz); tz = 2.0 * (x * py - y * px)
        return (px + w * tx + y * tz - z * ty, py + w * ty + z * tx - x * tz, pz + w * tz + x * ty - y * tx)

    @classmethod
    def _transform_camera_obb(
        cls,
        detection: dict[str, Any],
        transform: Any,
    ) -> dict[str, list[float]] | None:
        """Transform the detector's full-cloud OBB without refitting it.

        Packed segment points are intentionally bounded visualization data.
        They are not a sufficient statistic for the complete mask cloud and
        must not determine the semantic box size or principal axis.
        """

        center = cls._point3(detection.get("camera_obb_center"))
        size = cls._point3(detection.get("camera_obb_size"))
        orientation = detection.get("camera_obb_orientation")
        if center is None or size is None:
            return None
        if not isinstance(orientation, (list, tuple)) or len(orientation) < 4:
            return None
        try:
            camera_quaternion = tuple(float(value) for value in orientation[:4])
            transform_quaternion = tuple(
                float(getattr(transform.transform.rotation, axis))
                for axis in ("x", "y", "z", "w")
            )
            translation = tuple(
                float(getattr(transform.transform.translation, axis))
                for axis in ("x", "y", "z")
            )
        except (AttributeError, TypeError, ValueError):
            return None
        if not all(
            math.isfinite(value)
            for value in center + size + camera_quaternion
            + transform_quaternion + translation
        ):
            return None
        camera_norm = math.sqrt(sum(value * value for value in camera_quaternion))
        transform_norm = math.sqrt(sum(value * value for value in transform_quaternion))
        if camera_norm < 1e-9 or transform_norm < 1e-9:
            return None
        camera_quaternion = tuple(value / camera_norm for value in camera_quaternion)
        transform_quaternion = tuple(value / transform_norm for value in transform_quaternion)
        world_quaternion = _quat_multiply(transform_quaternion, camera_quaternion)
        world_norm = math.sqrt(sum(value * value for value in world_quaternion))
        if world_norm < 1e-9:
            return None
        world_quaternion = tuple(value / world_norm for value in world_quaternion)
        rotated_center = _quat_rotate(transform_quaternion, center)
        world_center = tuple(
            rotated_center[index] + translation[index] for index in range(3)
        )
        box_size = tuple(max(abs(float(value)), 0.01) for value in size)
        corners = []
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                for sz in (-0.5, 0.5):
                    local = (
                        sx * box_size[0], sy * box_size[1], sz * box_size[2],
                    )
                    rotated = _quat_rotate(world_quaternion, local)
                    corners.append(tuple(
                        world_center[index] + rotated[index] for index in range(3)
                    ))
        mins = tuple(min(corner[index] for corner in corners) for index in range(3))
        maxs = tuple(max(corner[index] for corner in corners) for index in range(3))
        aabb_center = tuple(
            (mins[index] + maxs[index]) * 0.5 for index in range(3)
        )
        aabb_size = tuple(
            max(maxs[index] - mins[index], 0.01) for index in range(3)
        )
        return {
            "obb_center": [float(value) for value in world_center],
            "obb_size": [float(value) for value in box_size],
            "obb_orientation": [float(value) for value in world_quaternion],
            "aabb_center": [float(value) for value in aabb_center],
            "aabb_size": [float(value) for value in aabb_size],
        }

    def _lookup_capture_transform(self, source_frame: str, stamp: Any) -> Any:
        """Resolve the sensor pose at the RGB-D capture time.

        Returning ``None`` on a historical lookup failure is intentional:
        ``_map_detection`` then uses the detector's capture-time world points
        instead of silently applying a newer robot pose.  A latest-TF
        fallback would make delayed YOLO reports appear to move with the
        robot and corrupt both boxes and the segmented world cloud.
        """
        if not source_frame:
            return None
        try:
            stamp_sec = float(stamp.to_sec())
        except (AttributeError, TypeError, ValueError):
            stamp_sec = 0.0
        if not math.isfinite(stamp_sec) or stamp_sec <= 0.0:
            rospy.logwarn_throttle(
                5.0,
                "physical detection has no valid capture stamp; using embedded world geometry",
            )
            return None
        cache_key = (source_frame, int(round(stamp_sec * 1_000_000_000.0)))
        cache = getattr(self, "_capture_transform_cache", None)
        cache_lock = getattr(self, "_capture_transform_cache_lock", None)
        if cache is not None:
            if cache_lock is None:
                cached = cache.get(cache_key)
            else:
                with cache_lock:
                    cached = cache.get(cache_key)
            if cached is not None:
                return cached
        try:
            # ``lookup_transform(..., 0.05s)`` serialized the mapping worker
            # behind every missing historical TF.  A zero-time can_transform
            # preserves the historical-time requirement but makes this lane
            # non-blocking; callers already retain embedded world geometry as
            # the correctness-preserving fallback when TF is unavailable.
            zero_timeout = rospy.Duration(0.0)
            can_transform = getattr(self.tf_buffer, "can_transform", None)
            if callable(can_transform) and not can_transform(
                self.world_frame,
                source_frame,
                stamp,
                zero_timeout,
            ):
                return None
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                source_frame,
                stamp,
                zero_timeout,
            )
            if cache is not None:
                if cache_lock is None:
                    cache[cache_key] = transform
                    if len(cache) > 64:
                        cache.pop(next(iter(cache)))
                else:
                    with cache_lock:
                        cache[cache_key] = transform
                        if len(cache) > 64:
                            cache.pop(next(iter(cache)))
            return transform
        except Exception as exc:
            rospy.logwarn_throttle(
                5.0,
                "capture-time TF unavailable for %s at %.3f; using embedded world geometry: %s",
                source_frame,
                stamp_sec,
                exc,
            )
            return None

    def _map_detection(
        self,
        detection: dict[str, Any],
        transform: Any = None,
        *,
        segment_cache: dict[int, dict[str, np.ndarray]] | None = None,
    ) -> dict[str, Any]:
        mapped = dict(detection)
        cached_segments: dict[str, np.ndarray] = {}
        if segment_cache is not None:
            segment_cache[id(mapped)] = cached_segments
        source_frame = str(detection.get("source_frame", "") or "")
        semantic_label = str(
            detection.get("semantic_class") or detection.get("raw_class") or ""
        ).strip().casefold()
        # Planar interaction targets need their complete visible extent. For
        # ordinary objects, the outer 2% of RGB-D mask points is frequently a
        # wall/floor leak and makes boxes metres larger than the object.
        bounds_low, bounds_high = (
            (0.02, 0.98)
            if semantic_label in {"door", "portal"}
            else (0.10, 0.90)
        )
        source_point = detection.get("camera_box3d_center") or detection.get("camera_position")
        camera_obb_contract = all(
            key in detection
            for key in (
                "camera_obb_center", "camera_obb_size", "camera_obb_orientation",
            )
        )
        transformed_obb = (
            self._transform_camera_obb(detection, transform)
            if camera_obb_contract and transform is not None
            else None
        )
        if transformed_obb is not None:
            obb_center = transformed_obb["obb_center"]
            obb_size = transformed_obb["obb_size"]
            obb_orientation = transformed_obb["obb_orientation"]
            aabb_center = transformed_obb["aabb_center"]
            aabb_size = transformed_obb["aabb_size"]
            mapped["world_box3d_center"] = list(obb_center)
            mapped["world_box3d_size"] = list(obb_size)
            mapped["world_box3d_marker_size"] = list(obb_size)
            mapped["world_box3d_orientation"] = list(obb_orientation)
            mapped["world_box3d_yaw"] = self._quaternion_yaw(obb_orientation)
            mapped["yaw"] = mapped["world_box3d_yaw"]
            mapped["world_position"] = self._point_dict(tuple(aabb_center))
            mapped["position"] = self._point_dict(tuple(aabb_center))
            mapped["aabb_center"] = list(aabb_center)
            mapped["aabb_size"] = list(aabb_size)
            mapped["box3d_center"] = list(obb_center)
            mapped["box3d_size"] = list(obb_size)
            # ``world_aabb_*_z`` carries the actual full-point extrema used
            # by class-height admission (not the 2--98% OBB extent).  Keep
            # the detector values when available so a refrigerator top is
            # not clipped back below its configured height threshold.
            mapped.setdefault(
                "world_aabb_min_z", float(aabb_center[2] - aabb_size[2] * 0.5)
            )
            mapped.setdefault(
                "world_aabb_max_z", float(aabb_center[2] + aabb_size[2] * 0.5)
            )
            mapped["map_frame"] = self.world_frame
            mapped["map_transform_status"] = "tf_camera_obb"
            mapped["map_transform_source_frame"] = source_frame
            return mapped
        if camera_obb_contract and transform is None:
            # Historical TF may arrive after the detector receipt.  The
            # detector's embedded world geometry was fitted from the same
            # complete cloud at capture time and is a better fallback than a
            # refit of its sparse debug sample.
            mapped["map_transform_status"] = "telemetry_full_obb_fallback"
            self._copy_world_fallback(mapped)
            return mapped
        # The world segmented cloud below is generated from these exact
        # camera-frame points and the TF transform.  Prefer their transformed
        # geometry for the box too, so a box can never be in the worker's
        # legacy telemetry frame while the cloud is in tf_frame_map.
        camera_segments = self._segment_points_array(
            detection, "camera_segment_points_f32", "camera_segment_points"
        )
        cached_segments["camera"] = camera_segments
        transformed_segments: np.ndarray = np.empty((0, 3), dtype=np.float32)
        if transform is not None and camera_segments.size:
            camera_segments = camera_segments[
                np.isfinite(camera_segments).all(axis=1)
            ]
            if camera_segments.size:
                transformed_array = self._rotate_points(
                    transform.transform.rotation, camera_segments
                )
                transformed_array = transformed_array + np.asarray(
                    [
                        float(transform.transform.translation.x),
                        float(transform.transform.translation.y),
                        float(transform.transform.translation.z),
                    ],
                    dtype=np.float32,
                )
                transformed_segments = transformed_array
        if transformed_segments.size:
            # Use the same finite transformed samples that are sent to
            # segmented_cloud_world. Quantiles reject an occasional depth
            # outlier while keeping the box centered on the visible cloud.
            values = np.asarray(transformed_segments, dtype=np.float32).reshape(-1, 3)
            mins = np.quantile(values, bounds_low, axis=0)
            maxs = np.quantile(values, bounds_high, axis=0)
            center = (mins + maxs) * 0.5
            size = np.maximum(maxs - mins, 0.01)
            obb_center, obb_size, obb_orientation = self._fit_world_obb(values)
            mapped["world_box3d_center"] = [float(axis) for axis in obb_center]
            mapped["world_box3d_size"] = [float(axis) for axis in obb_size]
            mapped["world_box3d_marker_size"] = [float(axis) for axis in obb_size]
            # ``size`` remains the world AABB for occupancy/semantic
            # clearance, while the marker gets a horizontal OBB fitted from
            # the exact TF-transformed segmented points.  Setting the marker
            # quaternion to identity here made the point cloud and its box
            # disagree whenever the door was rotated in the map.
            mapped["world_box3d_orientation"] = obb_orientation
            mapped["world_box3d_yaw"] = self._quaternion_yaw(obb_orientation)
            # The semantic/top-down path consumes ``yaw``; keep it aligned
            # with the orientation used by the world 3-D marker.
            mapped["yaw"] = mapped["world_box3d_yaw"]
            mapped["world_position"] = self._point_dict(tuple(float(axis) for axis in center))
            mapped["position"] = self._point_dict(tuple(float(axis) for axis in center))
            mapped["aabb_center"] = [float(axis) for axis in center]
            mapped["aabb_size"] = [float(axis) for axis in size]
            mapped["box3d_center"] = [float(axis) for axis in center]
            mapped["box3d_size"] = [float(axis) for axis in size]
            mapped["map_frame"] = self.world_frame
            mapped["map_transform_status"] = "tf_segment_points"
            mapped["map_transform_source_frame"] = source_frame
            return mapped
        # If TF is temporarily unavailable, segmented_cloud_world falls back
        # to the worker-provided world samples. Derive the fallback box from
        # those same samples instead of mixing them with a different worker
        # OBB center/size.
        world_segments = self._segment_points_array(
            detection, "world_segment_points_f32", "world_segment_points"
        )
        cached_segments["world"] = world_segments
        if world_segments.size:
            values = world_segments[np.isfinite(world_segments).all(axis=1)]
            if values.size:
                mins = np.quantile(values, bounds_low, axis=0)
                maxs = np.quantile(values, bounds_high, axis=0)
                center = (mins + maxs) * 0.5
                size = np.maximum(maxs - mins, 0.01)
                obb_center, obb_size, obb_orientation = self._fit_world_obb(values)
                mapped["world_box3d_center"] = [float(axis) for axis in obb_center]
                mapped["world_box3d_size"] = [float(axis) for axis in obb_size]
                mapped["world_box3d_marker_size"] = [float(axis) for axis in obb_size]
                mapped["world_box3d_orientation"] = obb_orientation
                mapped["world_box3d_yaw"] = self._quaternion_yaw(obb_orientation)
                mapped["yaw"] = mapped["world_box3d_yaw"]
                mapped["world_position"] = self._point_dict(tuple(float(axis) for axis in center))
                mapped["position"] = self._point_dict(tuple(float(axis) for axis in center))
                mapped["aabb_center"] = [float(axis) for axis in center]
                mapped["aabb_size"] = [float(axis) for axis in size]
                mapped["box3d_center"] = [float(axis) for axis in center]
                mapped["box3d_size"] = [float(axis) for axis in size]
                mapped["map_transform_status"] = "world_segment_points_fallback"
                return mapped
        if not source_frame or not isinstance(source_point, (dict, list, tuple)):
            mapped.setdefault("map_transform_status", "telemetry_fallback")
            self._copy_world_fallback(mapped)
            return mapped
        if isinstance(source_point, dict):
            try: point = (float(source_point.get("x", 0.0)), float(source_point.get("y", 0.0)), float(source_point.get("z", 0.0)))
            except (TypeError, ValueError):
                mapped["map_transform_status"] = "telemetry_fallback"; self._copy_world_fallback(mapped); return mapped
        elif len(source_point) >= 3:
            try: point = (float(source_point[0]), float(source_point[1]), float(source_point[2]))
            except (TypeError, ValueError):
                mapped["map_transform_status"] = "telemetry_fallback"; self._copy_world_fallback(mapped); return mapped
        else:
            mapped["map_transform_status"] = "telemetry_fallback"; self._copy_world_fallback(mapped); return mapped
        try:
            if transform is None:
                raise RuntimeError("sensor-to-map transform unavailable")
            translated = self._rotate_point(transform.transform.rotation, point)
            translated = (translated[0] + float(transform.transform.translation.x), translated[1] + float(transform.transform.translation.y), translated[2] + float(transform.transform.translation.z))
        except Exception:
            mapped["map_transform_status"] = "telemetry_fallback"
            self._copy_world_fallback(mapped)
            return mapped
        mapped["world_position"] = self._point_dict(translated)
        mapped["position"] = self._point_dict(translated)
        mapped["world_box3d_center"] = self._point_dict(translated)
        mapped["aabb_center"] = [translated[0], translated[1], translated[2]]
        mapped["box3d_center"] = [translated[0], translated[1], translated[2]]
        source_size = self._point3(detection.get("camera_box3d_size"))
        if source_size is not None:
            transformed_corners = []
            for sx in (-0.5, 0.5):
                for sy in (-0.5, 0.5):
                    for sz in (-0.5, 0.5):
                        corner = (point[0] + sx * abs(source_size[0]), point[1] + sy * abs(source_size[1]), point[2] + sz * abs(source_size[2]))
                        rotated = self._rotate_point(transform.transform.rotation, corner)
                        transformed_corners.append((rotated[0] + float(transform.transform.translation.x), rotated[1] + float(transform.transform.translation.y), rotated[2] + float(transform.transform.translation.z)))
            mins = tuple(min(corner[index] for corner in transformed_corners) for index in range(3))
            maxs = tuple(max(corner[index] for corner in transformed_corners) for index in range(3))
            map_center = tuple((mins[index] + maxs[index]) / 2.0 for index in range(3))
            map_size = tuple(max(maxs[index] - mins[index], 0.01) for index in range(3))
            mapped["world_box3d_center"] = self._point_dict(map_center)
            mapped["aabb_center"] = list(map_center)
            mapped["box3d_center"] = list(map_center)
            mapped["world_box3d_size"] = self._point_dict(map_size)
            # The marker must use the same transform-derived geometry as the
            # mapped box.  Do not overwrite it with the worker's legacy
            # telemetry-frame marker size: that value can be expressed in a
            # different frame and makes a correct point cloud appear to have
            # a displaced/inconsistent 3-D box.
            mapped["world_box3d_marker_size"] = list(map_size)
            mapped["world_box3d_orientation"] = list(
                mapped.get("world_box3d_orientation") or [0.0, 0.0, 0.0, 1.0]
            )
            mapped["world_box3d_yaw"] = self._quaternion_yaw(
                mapped["world_box3d_orientation"]
            )
            mapped["yaw"] = mapped["world_box3d_yaw"]
            mapped["aabb_size"] = list(map_size)
            mapped["box3d_size"] = list(map_size)
        mapped["map_frame"] = self.world_frame
        mapped["map_transform_status"] = "tf"
        mapped["map_transform_source_frame"] = source_frame
        return mapped

    @staticmethod
    def _quaternion_yaw(quaternion: list[float]) -> float:
        if len(quaternion) < 4:
            return 0.0
        x, y, z, w = (float(value) for value in quaternion[:4])
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    @staticmethod
    def _fit_world_obb(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """Fit a yaw-only world OBB to the same points used by the cloud."""
        points = np.asarray(values, dtype=np.float64)
        center = np.mean(points, axis=0)
        if points.shape[0] < 3:
            return center, np.maximum(np.ptp(points, axis=0), 0.01), [0.0, 0.0, 0.0, 1.0]
        centered = points - center
        covariance = centered[:, :2].T @ centered[:, :2] / max(1, points.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        horizontal = eigenvectors[:, int(np.argmax(eigenvalues))]
        yaw = math.atan2(float(horizontal[1]), float(horizontal[0]))
        if math.cos(yaw) < 0.0 or (abs(math.cos(yaw)) < 1e-6 and math.sin(yaw) < 0.0):
            yaw += math.pi
        c, s = math.cos(yaw), math.sin(yaw)
        axes = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        # ``axes`` is the local-to-world rotation.  Since points are stored
        # as row vectors, multiplying by ``axes`` projects world rows back to
        # the local OBB frame.  Keeping this convention makes the fitted
        # dimensions and quaternion describe the same box.
        projected = centered @ axes
        mins = np.quantile(projected, 0.02, axis=0)
        maxs = np.quantile(projected, 0.98, axis=0)
        local_center = (mins + maxs) * 0.5
        obb_center = center + axes @ local_center
        obb_size = np.maximum(maxs - mins, 0.01)
        orientation = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
        return obb_center, obb_size, orientation

    def _copy_world_fallback(self, mapped: dict[str, Any]) -> None:
        """Expose worker-computed telemetry world geometry when TF is late."""
        center = self._point3(
            mapped.get("world_box3d_center")
            or mapped.get("box3d_center")
            or mapped.get("aabb_center")
            or mapped.get("world_position")
        )
        size = self._point3(
            mapped.get("world_box3d_size")
            or mapped.get("box3d_size")
            or mapped.get("aabb_size")
        )
        if center is not None:
            mapped["world_box3d_center"] = list(center)
        if size is not None:
            mapped["world_box3d_size"] = list(size)
            # Keep the marker size in the fallback world frame.  Preserve the
            # worker OBB quaternion when TF is temporarily unavailable; do
            # not erase the angle with an identity quaternion.
            mapped["world_box3d_marker_size"] = list(size)

    def _publish(self, raw: dict[str, Any]) -> None:
        if isinstance(raw.get("telemetry"), dict):
            self._telemetry = dict(raw["telemetry"])
        rgb = _decode(raw["rgb"]); depth = _decode(raw["depth"])
        if rgb.ndim == 3 and rgb.shape[2] >= 3: rgb = np.ascontiguousarray(rgb[:, :, ::-1])
        if depth.dtype != np.uint16: depth = depth.astype(np.uint16)
        stamp = rospy.Time.from_sec(float(raw.get("stamp", time.time())))
        rgb_frame = str(raw.get("camera_frame", self.args.camera_frame))
        depth_frame = str(raw.get("depth_frame", rgb_frame))
        rgb_msg = _image_msg(rgb, "bgr8", stamp, rgb_frame)
        depth_msg = _image_msg(depth, "16UC1", stamp, depth_frame)
        # Keep the sensor capture timestamp.  Re-dating a delayed cloud to
        # ``now`` combines an old camera pose with the current TF pose and
        # bends the occupancy map when the robot turns.  GMapping now drops
        # clouds that exceed its age limit instead of accepting this mismatch.
        intr = raw.get("rgb_intrinsics") or raw.get("intrinsics", {})
        depth_intr = raw.get("depth_intrinsics") or intr
        info = CameraInfo(); info.header.stamp = stamp; info.header.frame_id = rgb_frame; info.width = int(intr.get("width", rgb.shape[1])); info.height = int(intr.get("height", rgb.shape[0])); info.K = [float(intr.get("fx", 0)), 0, float(intr.get("cx", 0)), 0, float(intr.get("fy", 0)), float(intr.get("cy", 0)), 0, 0, 1]
        distortion = [float(value) for value in (intr.get("distortion") or [])[:5]]
        info.D = distortion
        info.distortion_model = str(intr.get("distortion_model", "plumb_bob") or "plumb_bob")
        depth_info = CameraInfo()
        depth_info.header.stamp = stamp
        depth_info.header.frame_id = depth_frame
        depth_info.width = int(depth_intr.get("width", depth.shape[1]))
        depth_info.height = int(depth_intr.get("height", depth.shape[0]))
        depth_info.K = [
            float(depth_intr.get("fx", 0)), 0.0,
            float(depth_intr.get("cx", 0)), 0.0,
            float(depth_intr.get("fy", 0)),
            float(depth_intr.get("cy", 0)), 0.0, 0.0, 1.0,
        ]
        depth_info.D = [float(value) for value in (depth_intr.get("distortion") or [])[:5]]
        depth_info.distortion_model = str(
            depth_intr.get("distortion_model", "plumb_bob") or "plumb_bob"
        )
        self.rgb_pub.publish(rgb_msg); self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info); self.depth_info_pub.publish(depth_info)
        self._publish_cloud(depth, raw.get("depth_intrinsics") or intr, stamp, depth_frame); self._publish_pose(raw.get("telemetry", {}), stamp, depth_frame, raw.get("depth_to_color_extrinsics"))

    def _publish_cloud(self, depth: np.ndarray, intr: dict[str, Any], stamp: Any, frame: str) -> None:
        started = time.perf_counter()
        fx, fy, cx, cy = [_to_float(intr.get(k)) for k in ("fx", "fy", "cx", "cy")]; scale = _to_float(self.last_raw.get("depth_scale", .001) if self.last_raw else .001)
        if fx <= 0 or fy <= 0: return
        obstacle_depth = max(0.0, float(self.args.max_depth_m))
        no_return_depth = max(
            obstacle_depth + 1e-3,
            float(self.args.no_return_depth_m),
        )
        stride = max(1, int(self.args.point_stride))
        sampled = np.asarray(depth[::stride, ::stride], dtype=np.float32) * scale
        valid = sampled > 0.0
        obstacle_mask = valid & (sampled <= obstacle_depth)
        no_return_mask = valid & ~obstacle_mask
        # Preserve the bearing of a finite return beyond the mapping horizon,
        # but clamp its endpoint just outside the obstacle range. The mapper's
        # clear-only layer and costmap_2d can ray-trace it without drawing a
        # GMapping occupied endpoint.
        projected_z = np.where(no_return_mask, no_return_depth, sampled)
        cache_key = (depth.shape[0], depth.shape[1], stride, fx, fy, cx, cy)
        cached = self._projection_cache.get("global")
        if cached is None or cached[0] != cache_key:
            u = np.arange(0, depth.shape[1], stride, dtype=np.float32)
            v = np.arange(0, depth.shape[0], stride, dtype=np.float32)
            uu, vv = np.meshgrid(u, v)
            cached = (cache_key, (uu - cx) / fx, (vv - cy) / fy)
            self._projection_cache["global"] = cached
        _, x_factor, y_factor = cached
        xyz = np.column_stack((
            (x_factor * projected_z)[valid],
            (y_factor * projected_z)[valid],
            projected_z[valid],
        )).astype(np.float32, copy=False)
        obstacle_points = int(np.count_nonzero(obstacle_mask))
        no_return_points = int(np.count_nonzero(no_return_mask))
        header = Header(); header.stamp = stamp; header.frame_id = frame
        msg = PointCloud2(
            header=header, height=1, width=int(xyz.shape[0]),
            fields=[PointField("x", 0, PointField.FLOAT32, 1), PointField("y", 4, PointField.FLOAT32, 1), PointField("z", 8, PointField.FLOAT32, 1)],
            is_bigendian=False, point_step=12, row_step=int(xyz.shape[0] * 12),
            data=xyz.tobytes(order="C"), is_dense=False,
        )
        self.cloud_pub.publish(msg)
        rospy.loginfo_throttle(
            5.0,
            "mapping cloud: obstacle_points=%d no_return_points=%d obstacle_depth=%.2f no_return_depth=%.2f",
            obstacle_points,
            no_return_points,
            obstacle_depth,
            no_return_depth,
        )
        rospy.loginfo_throttle(
            10.0,
            "global cloud timing: %.1f ms, points=%d stride=%d",
            (time.perf_counter() - started) * 1000.0,
            int(xyz.shape[0]), stride,
        )

    def _publish_pose(
        self,
        telemetry: dict[str, Any],
        stamp: Any,
        depth_frame: str | None = None,
        depth_to_color_extrinsics: dict[str, Any] | None = None,
    ) -> None:
        if depth_to_color_extrinsics is not None:
            self._depth_to_color_extrinsics = dict(depth_to_color_extrinsics)
        elif self._depth_to_color_extrinsics is not None:
            depth_to_color_extrinsics = self._depth_to_color_extrinsics
        # Preserve the pose timestamp from the same camera capture.  The
        # transform buffer then provides the matching historical odom pose;
        # re-dating it to receipt time would pair an old pose with a new TF.
        position = telemetry.get("position", [0, 0, 0]); velocity = telemetry.get("velocity", [0, 0, 0]); yaw = _to_float(telemetry.get("yaw", telemetry.get("imu", {}).get("rpy", [0, 0, 0])[2] if telemetry.get("imu") else 0))
        quaternion = _telemetry_quaternion(telemetry)
        odom = Odometry(); odom.header.stamp = stamp; odom.header.frame_id = "tf_frame_odom"; odom.child_frame_id = "tf_frame_base_link"; odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = [_to_float(v) for v in position[:3]]; odom.twist.twist.linear.x, odom.twist.twist.linear.y = [_to_float(v) for v in velocity[:2]]
        if quaternion is None:
            quaternion = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = quaternion
        self.odom_pub.publish(odom)
        transform = TransformStamped(); transform.header.stamp = stamp; transform.header.frame_id = "tf_frame_odom"; transform.child_frame_id = "tf_frame_base_link"; transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = [_to_float(v) for v in position[:3]]; transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = quaternion; self.tf_broadcaster.sendTransform(transform)
        static_frames = [self.args.camera_frame]
        if depth_frame and depth_frame != self.args.camera_frame:
            static_frames.append(depth_frame)
        new_static = False
        color_frame = str(self.args.camera_frame)
        color_translation = (
            float(self.args.camera_x),
            float(self.args.camera_y),
            float(self.args.camera_z),
        )
        # REP-103 optical frame -> Go2 base: optical x=right, y=down,
        # z=forward maps to base x=forward, y=left, z=up.  This is a
        # coordinate convention, not an additional physical mounting tilt.
        mount = _quat_from_rpy(self.args.camera_roll, self.args.camera_pitch, self.args.camera_yaw)
        optical = (0.5, -0.5, 0.5, -0.5)  # base <- d435i_color_optical
        color_quaternion = _quat_multiply(mount, optical)
        for static_frame in static_frames:
            if static_frame == color_frame:
                frame_quaternion, frame_translation = color_quaternion, color_translation
            else:
                frame_quaternion, frame_translation = _depth_static_extrinsic(
                    color_quaternion,
                    color_translation,
                    static_frame,
                    color_frame,
                    depth_to_color_extrinsics,
                )
            previous = self._static_transforms.get(static_frame)
            static = TransformStamped()
            static.header.stamp = stamp
            static.header.frame_id = self.args.camera_parent
            static.child_frame_id = static_frame
            static.transform.translation.x, static.transform.translation.y, static.transform.translation.z = frame_translation
            static.transform.rotation.x, static.transform.rotation.y, static.transform.rotation.z, static.transform.rotation.w = frame_quaternion
            changed = previous is None or any(
                abs(float(current) - float(old)) > 1e-9
                for current, old in (
                    (static.transform.translation.x, previous.transform.translation.x),
                    (static.transform.translation.y, previous.transform.translation.y),
                    (static.transform.translation.z, previous.transform.translation.z),
                    (static.transform.rotation.x, previous.transform.rotation.x),
                    (static.transform.rotation.y, previous.transform.rotation.y),
                    (static.transform.rotation.z, previous.transform.rotation.z),
                    (static.transform.rotation.w, previous.transform.rotation.w),
                )
            )
            if changed:
                self._static_transforms[static_frame] = static
                new_static = True
            self._static_frames.add(static_frame)
        if new_static:
            # Keep the latched /tf_static message complete for reconnecting
            # listeners; StaticTransformBroadcaster does not merge calls.
            self.static_broadcaster.sendTransform(list(self._static_transforms.values()))
        self._static_sent = bool(self._static_frames)


def _to_float(value: Any) -> float:
    try: return float(value)
    except (TypeError, ValueError): return 0.0


def _telemetry_quaternion(telemetry: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Read live orientation as (x, y, z, w), preferring camera pose/IMU."""
    # This quaternion drives odom -> base_link.  A D435i motion quaternion is
    # in the camera motion-module frame and must never replace the Go2 body
    # attitude here.
    candidates = [telemetry.get("camera_pose"), telemetry.get("d435i_pose"), telemetry.get("pose"), telemetry.get("imu")]
    for source in candidates:
        if not isinstance(source, dict):
            continue
        raw = source.get("quaternion") or source.get("orientation")
        if isinstance(raw, dict):
            try:
                values = [_to_float(raw[key]) for key in ("x", "y", "z", "w")]
            except KeyError:
                continue
        elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
            values = [_to_float(value) for value in raw[:4]]
            if source is telemetry.get("imu"):
                values = [values[1], values[2], values[3], values[0]]
        else:
            continue
        norm = math.sqrt(sum(value * value for value in values))
        if norm > 1e-6 and math.isfinite(norm):
            return tuple(value / norm for value in values)
    return None


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy)


def _quat_multiply(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz)


def _quat_rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _rotation_matrix_to_quaternion(rotation: Any) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(max(trace + 1.0, 1e-12))
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        next_index, last_index = (index + 1) % 3, (index + 2) % 3
        scale = 2.0 * math.sqrt(
            max(
                1e-12,
                1.0
                + float(diagonal[index])
                - float(diagonal[next_index])
                - float(diagonal[last_index]),
            )
        )
        values = [0.0, 0.0, 0.0, 0.0]
        values[index] = 0.25 * scale
        values[3] = (matrix[last_index, next_index] - matrix[next_index, last_index]) / scale
        values[next_index] = (matrix[next_index, index] + matrix[index, next_index]) / scale
        values[last_index] = (matrix[last_index, index] + matrix[index, last_index]) / scale
        x, y, z, w = values
    values = np.asarray([x, y, z, w], dtype=np.float64)
    values /= max(float(np.linalg.norm(values)), 1e-12)
    return tuple(float(value) for value in values)


def _depth_static_extrinsic(
    color_quaternion: tuple[float, float, float, float],
    color_translation: tuple[float, float, float],
    depth_frame: str,
    color_frame: str,
    extrinsics: dict[str, Any] | None,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    """Compose base->depth from RealSense depth->color calibration."""
    if depth_frame == color_frame or not isinstance(extrinsics, dict):
        return color_quaternion, color_translation
    try:
        rotation = extrinsics.get("rotation")
        translation = extrinsics.get("translation")
        if rotation is None or translation is None or len(translation) < 3:
            return color_quaternion, color_translation
        q_dc = _rotation_matrix_to_quaternion(rotation)
        q_depth = _quat_multiply(color_quaternion, q_dc)
        offset = _quat_rotate(
            color_quaternion,
            (float(translation[0]), float(translation[1]), float(translation[2])),
        )
        t_depth = tuple(
            float(color_translation[index]) + float(offset[index])
            for index in range(3)
        )
        return q_depth, t_depth
    except (TypeError, ValueError, IndexError, OverflowError):
        return color_quaternion, color_translation


def _image_msg(array: np.ndarray, encoding: str, stamp: Any, frame: str) -> Image:
    array = np.ascontiguousarray(array); msg = Image(); msg.header.stamp = stamp; msg.header.frame_id = frame; msg.height = int(array.shape[0]); msg.width = int(array.shape[1]); msg.encoding = encoding; msg.is_bigendian = 0; msg.step = int(array.strides[0]); msg.data = array.tobytes(); return msg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--web-url", default="http://127.0.0.1:8765")
    p.add_argument("--rate", type=float, default=10.)
    p.add_argument("--state-period", type=float, default=.2)
    p.add_argument("--legacy-http-sensor", action="store_true")
    p.add_argument(
        "--publish-sensor-ros",
        action="store_true",
        default=None,
        help="enable the legacy gateway RGB-D/PointCloud/TF publisher",
    )
    p.add_argument("--legacy-http-state", action="store_true")
    p.add_argument("--web-state-enabled", action="store_true")
    # Keep ROS map publication independent from the HTTP dashboard. A full
    # OccupancyGrid is hundreds of KB when encoded as JSON; forwarding it at
    # every mapper tick starves the web server and semantic consumers. The
    # latest-only post at 5 Hz is sufficient for visualization and planning.
    p.add_argument("--occupancy-period", type=float, default=.2)
    p.add_argument("--room-grid-period", type=float, default=.5)
    p.add_argument("--point-stride", type=int, default=6)
    p.add_argument("--max-depth-m", type=float, default=8.)
    p.add_argument("--no-return-depth-m", type=float, default=8.05)
    p.add_argument("--world-frame", default="tf_frame_map")
    p.add_argument("--camera-frame", default="d435i_color_optical_frame")
    p.add_argument("--camera-parent", default="tf_frame_base_link")
    p.add_argument("--camera-x", type=float, default=.03)
    p.add_argument("--camera-y", type=float, default=0.)
    p.add_argument("--camera-z", type=float, default=.62)
    p.add_argument("--camera-roll", type=float, default=0.)
    p.add_argument("--camera-pitch", type=float, default=0.)
    p.add_argument("--camera-yaw", type=float, default=0.)
    p.add_argument("--box-hold-s", type=float, default=3.0)
    p.add_argument("--box-match-distance-m", type=float, default=.60)
    p.add_argument("--box-smoothing-alpha", type=float, default=1.0)
    p.add_argument("--box-min-confirmations", type=int, default=2)
    p.add_argument("--occupancy-grid-topic", default="/physical_nav/occupancy")
    p.add_argument("--room-grid-topic", default="/physical_nav/room_segment_grid")
    p.add_argument("--global-costmap-topic", default="/move_base/global_costmap/costmap")
    p.add_argument("--local-costmap-topic", default="/move_base/local_costmap/costmap")
    p.add_argument("--global-plan-topic", default="/move_base/GlobalPlanner/plan")
    p.add_argument("--local-global-plan-topic", default="/move_base/DWAPlannerROS/global_plan")
    p.add_argument("--local-plan-topic", default="/move_base/DWAPlannerROS/local_plan")
    args, _unknown = p.parse_known_args()
    # ROS launch appends __name/__log remappings; ignore those in the local
    # CLI parser so the gateway can also be run directly.
    _patch_roslogging_findcaller_for_py311()
    rospy.init_node("physical_ros_gateway", anonymous=False)
    for name in ("web_url", "rate", "state_period", "occupancy_period", "room_grid_period", "point_stride", "max_depth_m", "no_return_depth_m", "world_frame", "camera_frame", "camera_parent", "camera_x", "camera_y", "camera_z", "camera_roll", "camera_pitch", "camera_yaw", "box_hold_s", "box_match_distance_m", "box_smoothing_alpha", "box_min_confirmations", "occupancy_grid_topic", "room_grid_topic", "global_costmap_topic", "local_costmap_topic", "global_plan_topic", "local_global_plan_topic", "local_plan_topic", "legacy_http_sensor", "publish_sensor_ros", "legacy_http_state", "web_state_enabled"):
        setattr(args, name, rospy.get_param("~" + name, getattr(args, name)))
    PhysicalRosGateway(args); rospy.spin()


if __name__ == "__main__": main()
