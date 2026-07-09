import json
import threading
import time
from typing import Any

import numpy as np

from molmo_spaces.policy.base_policy import BasePolicy
from molmo_spaces.tasks.task import BaseMujocoTask


class RosBridgePolicy(BasePolicy):
    """
    ROS bridge policy for closed-loop external control.

    - Publishes observations to a ROS topic.
    - Receives actions from another ROS node.
    """

    def __init__(
        self,
        config,
        task: BaseMujocoTask | None = None,
        observation_topic: str = "/molmo_spaces/head_camera/image",
        action_topic: str = "/molmo_spaces/action",
        pointcloud_topic: str = "/registered_scan",
        camera_info_topic: str = "/molmo_spaces/head_camera/camera_info",
        depth_topic: str = "/molmo_spaces/head_camera/depth",
        action_timeout_s: float = 0.1,
        queue_size: int = 1,
        publish_pointcloud: bool = True,
        publish_camera_info: bool = True,
        depth_camera_name: str = "head_camera",
        pointcloud_frame_id: str = "tf_frame_lidar",
        optical_frame_id: str = "head_camera_optical_frame",
        depth_fov_deg: float = 90.0,
        depth_min_m: float = 0.1,
        depth_max_m: float = 30.0,
        pointcloud_stride: int = 2,
        pointcloud_roll_correction_deg: float = 0.0,
        odom_topic: str = "/odom",
        publish_odom: bool = True,
        map_frame_id: str = "tf_frame_map",
        odom_frame_id: str = "tf_frame_odom",
        base_frame_id: str = "tf_frame_base_link",
        lidar_offset_x_m: float = 0.0,
        lidar_offset_y_m: float = 0.0,
        lidar_offset_z_m: float = 1.6,
        lidar_calib_x_m: float = 0.0,
        lidar_calib_y_m: float = 0.0,
        lidar_calib_z_m: float = 0.0,
        lidar_calib_roll_deg: float = 0.0,
        lidar_calib_pitch_deg: float = 0.0,
        lidar_calib_yaw_deg: float = 0.0,
        allow_static_lidar_tf_fallback: bool = False,
        cmd_vel_topic: str = "/cmd_vel_stamped",
        cmd_vel_timeout_s: float = 0.5,
        cmd_vel_control_dt_s: float | None = None,
        cmd_vel_linear_gain: float = 1.0,
        map_warmup_skip_frames: int = 10,
        immediate_noop_after_publish: bool = False,
        timing_log_every_n_frames: int = 30,
        extra_image_topic: str = "/molmo_spaces/debug_front_camera/image",
        extra_image_camera_name: str = "debug_front_camera",
    ) -> None:
        super().__init__(config, task)
        self.observation_topic = observation_topic
        self.action_topic = action_topic
        self.pointcloud_topic = pointcloud_topic
        self.camera_info_topic = camera_info_topic
        self.depth_topic = depth_topic
        self.action_timeout_s = action_timeout_s
        self.queue_size = queue_size
        self.publish_pointcloud = publish_pointcloud
        self.publish_camera_info = publish_camera_info
        self.depth_camera_name = depth_camera_name
        self.pointcloud_frame_id = pointcloud_frame_id
        self.optical_frame_id = optical_frame_id
        self.depth_fov_deg = float(depth_fov_deg)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.pointcloud_stride = max(1, int(pointcloud_stride))
        self.pointcloud_roll_correction_deg = float(pointcloud_roll_correction_deg)
        self.odom_topic = odom_topic
        self.publish_odom = bool(publish_odom)
        self.map_frame_id = map_frame_id
        self.odom_frame_id = odom_frame_id
        self.base_frame_id = base_frame_id
        self.lidar_offset_x_m = float(lidar_offset_x_m)
        self.lidar_offset_y_m = float(lidar_offset_y_m)
        self.lidar_offset_z_m = float(lidar_offset_z_m)
        self.lidar_calib_x_m = float(lidar_calib_x_m)
        self.lidar_calib_y_m = float(lidar_calib_y_m)
        self.lidar_calib_z_m = float(lidar_calib_z_m)
        self.lidar_calib_roll_deg = float(lidar_calib_roll_deg)
        self.lidar_calib_pitch_deg = float(lidar_calib_pitch_deg)
        self.lidar_calib_yaw_deg = float(lidar_calib_yaw_deg)
        self.allow_static_lidar_tf_fallback = bool(allow_static_lidar_tf_fallback)
        self.cmd_vel_topic = cmd_vel_topic
        self.cmd_vel_timeout_s = float(cmd_vel_timeout_s)
        self.cmd_vel_linear_gain = max(0.0, float(cmd_vel_linear_gain))
        self.map_warmup_skip_frames = max(0, int(map_warmup_skip_frames))
        self.immediate_noop_after_publish = bool(immediate_noop_after_publish)
        self.timing_log_every_n_frames = max(0, int(timing_log_every_n_frames))
        self.extra_image_topic = extra_image_topic
        self.extra_image_camera_name = extra_image_camera_name
        if cmd_vel_control_dt_s is None:
            cfg_dt_ms = getattr(config, "policy_dt_ms", None)
            if cfg_dt_ms is not None and float(cfg_dt_ms) > 0.0:
                self.cmd_vel_control_dt_s = float(cfg_dt_ms) / 1000.0
            else:
                self.cmd_vel_control_dt_s = 0.1
        else:
            self.cmd_vel_control_dt_s = max(1e-3, float(cmd_vel_control_dt_s))
        # Default arm poses used when incoming action does not include arm groups.
        self.default_left_arm_qpos = np.array(
            [0.28, 0.0, 0.0, -0.64, 0.39, -0.26, -0.04], dtype=np.float32
        )
        self.default_right_arm_qpos = np.array(
            [0.28, 0.0, 0.0, -0.64, 0.39, -0.26, -0.04], dtype=np.float32
        )
        # Cache per-intrinsics projection lookup tables for depth->pointcloud.
        # This avoids rebuilding uv grids every frame.
        self._pointcloud_projection_cache: dict[
            tuple[int, int, float, float, float, float], tuple[np.ndarray, np.ndarray]
        ] = {}

        self._step_idx = 0
        self._latest_action: dict[str, Any] | None = None
        self._latest_action_step: int = -1
        self._last_consumed_action_step: int = -1
        self._latest_cmd_vel: np.ndarray | None = None
        self._latest_cmd_vel_mono_s: float = 0.0
        self._last_base_position_xyz: np.ndarray | None = None
        self._last_common_stamp_s: float | None = None
        self._base_position_jump_warn_m: float = 1.0
        self._timing_frame_count: int = 0
        self._timing_acc_ms: dict[str, float] = {
            "total": 0.0,
            "odom_tf": 0.0,
            "rgb_publish": 0.0,
            "depth_extract_intrinsics": 0.0,
            "depth_msg_publish": 0.0,
            "pointcloud_convert": 0.0,
            "pointcloud_publish": 0.0,
            "camera_info_publish": 0.0,
            "action_wait": 0.0,
            "postprocess_action": 0.0,
        }
        self._lock = threading.Lock()

        self._patch_rosgraph_logger_for_py311()
        import rospy
        from geometry_msgs.msg import TransformStamped, TwistStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo
        from sensor_msgs.msg import Image
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import String
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        self._rospy = rospy
        self._TransformStamped = TransformStamped
        self._TwistStamped = TwistStamped
        self._Image = Image
        self._CameraInfo = CameraInfo
        self._PointCloud2 = PointCloud2
        self._PointField = PointField
        self._String = String
        self._Odometry = Odometry
        if not rospy.core.is_initialized():
            rospy.init_node("molmo_spaces_ros_policy", anonymous=True, disable_signals=True)

        self._obs_pub = rospy.Publisher(self.observation_topic, Image, queue_size=self.queue_size)
        self._extra_image_pub = None
        if self.extra_image_topic and self.extra_image_camera_name:
            self._extra_image_pub = rospy.Publisher(self.extra_image_topic, Image, queue_size=self.queue_size)
        self._depth_pub = rospy.Publisher(self.depth_topic, Image, queue_size=self.queue_size)
        self._pointcloud_pub = rospy.Publisher(self.pointcloud_topic, PointCloud2, queue_size=self.queue_size)
        self._camera_info_pub = rospy.Publisher(
            self.camera_info_topic, CameraInfo, queue_size=self.queue_size
        )
        self._image_camera_info_pub = rospy.Publisher(
            f"{self.observation_topic}/camera_info", CameraInfo, queue_size=self.queue_size
        )
        self._depth_camera_info_pub = rospy.Publisher(
            f"{self.depth_topic}/camera_info", CameraInfo, queue_size=self.queue_size
        )
        self._odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=self.queue_size)
        self._static_tf_pub = StaticTransformBroadcaster()
        self._tf_broadcaster = TransformBroadcaster()
        self._publish_static_tfs()
        self._action_sub = rospy.Subscriber(self.action_topic, String, self._action_callback)
        self._cmd_vel_sub = rospy.Subscriber(self.cmd_vel_topic, TwistStamped, self._cmd_vel_callback)

    @staticmethod
    def _rotation_matrix_to_quaternion(rot: np.ndarray) -> tuple[float, float, float, float]:
        """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
        tr = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rot[2, 1] - rot[1, 2]) / s
            qy = (rot[0, 2] - rot[2, 0]) / s
            qz = (rot[1, 0] - rot[0, 1]) / s
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
        return float(qx), float(qy), float(qz), float(qw)

    def _build_optical_static_tf(self):
        """Build static TF from robot-centric frame to optical frame."""
        # parent(robot): x forward, y left, z up
        # child(optical): x right, y down, z forward
        # optical <- robot
        rot = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        qx, qy, qz, qw = self._rotation_matrix_to_quaternion(rot)
        tf_msg = self._TransformStamped()
        tf_msg.header.stamp = self._rospy.Time.now()
        tf_msg.header.frame_id = self.pointcloud_frame_id
        tf_msg.child_frame_id = self.optical_frame_id
        tf_msg.transform.translation.x = 0.0
        tf_msg.transform.translation.y = 0.0
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        return tf_msg

    def _build_base_to_lidar_static_tf(self):
        """Base->lidar is published dynamically from observation camera extrinsics."""
        return None

    def _publish_static_tfs(self) -> None:
        """
        Publish all static transforms in one latched TF message.
        In rospy tf2, each sendTransform() publishes only provided transforms.
        """
        tfs = []
        tfs.append(self._build_optical_static_tf())
        self._static_tf_pub.sendTransform(tfs)

    @staticmethod
    def _quat_wxyz_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
        """Convert scalar-first quaternion to a 3x3 rotation matrix."""
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rpy_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rot_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
            dtype=np.float64,
        )
        rot_y = np.array(
            [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
            dtype=np.float64,
        )
        rot_z = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return rot_z @ rot_y @ rot_x

    def _apply_lidar_calibration(self, T_base_lidar: np.ndarray) -> np.ndarray:
        """Apply a small tunable correction in the lidar frame."""
        if (
            abs(self.lidar_calib_x_m) < 1e-9
            and abs(self.lidar_calib_y_m) < 1e-9
            and abs(self.lidar_calib_z_m) < 1e-9
            and abs(self.lidar_calib_roll_deg) < 1e-9
            and abs(self.lidar_calib_pitch_deg) < 1e-9
            and abs(self.lidar_calib_yaw_deg) < 1e-9
        ):
            return T_base_lidar

        T_lidar_calib = np.eye(4, dtype=np.float64)
        T_lidar_calib[:3, 3] = np.array(
            [self.lidar_calib_x_m, self.lidar_calib_y_m, self.lidar_calib_z_m],
            dtype=np.float64,
        )
        T_lidar_calib[:3, :3] = self._rpy_to_rotmat(
            np.deg2rad(self.lidar_calib_roll_deg),
            np.deg2rad(self.lidar_calib_pitch_deg),
            np.deg2rad(self.lidar_calib_yaw_deg),
        )
        return T_base_lidar @ T_lidar_calib

    def _extract_lidar_pose_rel_base(self, observation: Any) -> np.ndarray | None:
        """Return lidar pose in base frame as a 4x4 transform."""
        obs_dict = self._extract_observation_dict(observation)
        if obs_dict is None:
            return None

        base_pose = self._extract_base_pose_from_observation(observation)
        if base_pose is None:
            return None

        sensor_params = obs_dict.get(f"sensor_param_{self.depth_camera_name}")
        if not isinstance(sensor_params, dict):
            return None

        cam2world_gl = sensor_params.get("cam2world_gl")
        if cam2world_gl is None:
            return None

        T_world_optical = np.asarray(cam2world_gl, dtype=np.float64)
        if T_world_optical.shape != (4, 4):
            return None

        px, py, pz = float(base_pose[0]), float(base_pose[1]), float(base_pose[2])
        qw, qx, qy, qz = (
            float(base_pose[3]),
            float(base_pose[4]),
            float(base_pose[5]),
            float(base_pose[6]),
        )
        T_world_base = np.eye(4, dtype=np.float64)
        T_world_base[:3, :3] = self._quat_wxyz_to_rotmat(qw, qx, qy, qz)
        T_world_base[:3, 3] = np.array([px, py, pz], dtype=np.float64)

        # The pointcloud is published in a robot-centric lidar/body frame:
        # x forward, y left, z up. The optical frame is related by the fixed
        # lidar->optical rotation used in _build_optical_static_tf().
        R_lidar_to_optical = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        T_optical_lidar = np.eye(4, dtype=np.float64)
        T_optical_lidar[:3, :3] = R_lidar_to_optical.T

        return np.linalg.inv(T_world_base) @ T_world_optical @ T_optical_lidar

    def _publish_base_to_lidar_tf(self, observation: Any, stamp) -> bool:
        if self.base_frame_id == self.pointcloud_frame_id:
            return True

        T_base_lidar = self._extract_lidar_pose_rel_base(observation)
        if T_base_lidar is None:
            if not self.allow_static_lidar_tf_fallback:
                self._rospy.logwarn_throttle(
                    2.0,
                    "RosBridgePolicy: sensor_param_%s missing; skip mapping observation instead of publishing fixed base->lidar TF.",
                    self.depth_camera_name,
                )
                return False

            self._rospy.logwarn_throttle(
                5.0,
                "RosBridgePolicy: sensor_param_%s missing; falling back to configured fixed base->lidar TF.",
                self.depth_camera_name,
            )
            T_base_lidar = np.eye(4, dtype=np.float64)
            T_base_lidar[:3, 3] = np.array(
                [self.lidar_offset_x_m, self.lidar_offset_y_m, self.lidar_offset_z_m],
                dtype=np.float64,
            )
            T_base_lidar = self._apply_lidar_calibration(T_base_lidar)
            qx, qy, qz, qw = self._rotation_matrix_to_quaternion(T_base_lidar[:3, :3])
            tf_msg = self._TransformStamped()
            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = self.base_frame_id
            tf_msg.child_frame_id = self.pointcloud_frame_id
            tf_msg.transform.translation.x = float(T_base_lidar[0, 3])
            tf_msg.transform.translation.y = float(T_base_lidar[1, 3])
            tf_msg.transform.translation.z = float(T_base_lidar[2, 3])
            tf_msg.transform.rotation.x = qx
            tf_msg.transform.rotation.y = qy
            tf_msg.transform.rotation.z = qz
            tf_msg.transform.rotation.w = qw
            self._tf_broadcaster.sendTransform(tf_msg)
            return False

        T_base_lidar = self._apply_lidar_calibration(T_base_lidar)
        qx, qy, qz, qw = self._rotation_matrix_to_quaternion(T_base_lidar[:3, :3])
        self._rospy.loginfo_throttle(
            2.0,
            (
                "RosBridgePolicy: publishing dynamic base->lidar TF from sensor_param_%s: "
                "xyz=(%.3f, %.3f, %.3f), quat=(%.4f, %.4f, %.4f, %.4f)"
            ),
            self.depth_camera_name,
            float(T_base_lidar[0, 3]),
            float(T_base_lidar[1, 3]),
            float(T_base_lidar[2, 3]),
            qx,
            qy,
            qz,
            qw,
        )
        tf_msg = self._TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.base_frame_id
        tf_msg.child_frame_id = self.pointcloud_frame_id
        tf_msg.transform.translation.x = float(T_base_lidar[0, 3])
        tf_msg.transform.translation.y = float(T_base_lidar[1, 3])
        tf_msg.transform.translation.z = float(T_base_lidar[2, 3])
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(tf_msg)
        return True

    @staticmethod
    def _patch_rosgraph_logger_for_py311() -> None:
        """
        Patch rosgraph's custom logger to avoid a known infinite-loop edge case.

        In some Python 3.11 environments with ROS Noetic, rosgraph's RospyLogger.findCaller
        can loop forever when walking stack frames if no matching frame is found and f_back is None.
        """
        try:
            import os
            import sys
            import inspect
            import rosgraph.roslogging as roslogging
        except Exception:
            # If rosgraph isn't importable yet, let rospy import handle errors.
            return

        if not hasattr(roslogging, "RospyLogger"):
            return

        def _safe_find_caller(self, *args, **kwargs):
            file_name, lineno, func_name = super(roslogging.RospyLogger, self).findCaller(
                *args, **kwargs
            )[:3]
            file_name = os.path.normcase(file_name)

            f = inspect.currentframe()
            if f is not None:
                f = f.f_back
            while hasattr(f, "f_code"):
                co = f.f_code
                filename = os.path.normcase(co.co_filename)
                if filename == file_name and f.f_lineno == lineno and co.co_name == func_name:
                    break
                if f.f_back:
                    f = f.f_back
                else:
                    # Critical fix: break instead of infinite loop.
                    break

            if f is None or not hasattr(f, "f_code"):
                if sys.version_info > (3, 2):
                    return file_name, lineno, func_name, None
                return file_name, lineno, func_name

            if f.f_back and f.f_code and f.f_code.co_name == "_base_logger":
                f = f.f_back
                if f.f_back:
                    f = f.f_back
            co = f.f_code
            func_name2 = co.co_name
            try:
                class_name = f.f_locals["self"].__class__.__name__
                func_name2 = f"{class_name}.{func_name2}"
            except KeyError:
                pass

            if sys.version_info > (3, 2):
                return co.co_filename, f.f_lineno, func_name2, None
            return co.co_filename, f.f_lineno, func_name2

        roslogging.RospyLogger.findCaller = _safe_find_caller

    def reset(self):
        self._step_idx = 0
        self._timing_frame_count = 0
        for key in self._timing_acc_ms:
            self._timing_acc_ms[key] = 0.0
        with self._lock:
            self._latest_action = None
            self._latest_action_step = -1
            self._last_consumed_action_step = -1
            self._latest_cmd_vel = None
            self._latest_cmd_vel_mono_s = 0.0
        self._last_base_position_xyz = None

    def _record_timing(self, stage_ms: dict[str, float]) -> None:
        if self.timing_log_every_n_frames <= 0:
            return
        self._timing_frame_count += 1
        for key in self._timing_acc_ms:
            self._timing_acc_ms[key] += float(stage_ms.get(key, 0.0))

        if self._timing_frame_count % self.timing_log_every_n_frames != 0:
            return

        n = float(self._timing_frame_count)
        avg = {k: self._timing_acc_ms[k] / n for k in self._timing_acc_ms}
        avg_total = max(avg["total"], 1e-6)
        fps = 1000.0 / avg_total
        self._rospy.loginfo(
            (
                "RosBridgePolicy timing avg over %d frames: total=%.2fms (%.2fHz), "
                "odom_tf=%.2fms, rgb_pub=%.2fms, depth_extract_intrinsics=%.2fms, "
                "depth_msg_pub=%.2fms, pcd_convert=%.2fms, pcd_pub=%.2fms, "
                "camera_info_pub=%.2fms, action_wait=%.2fms, postprocess=%.2fms"
            ),
            int(n),
            avg["total"],
            fps,
            avg["odom_tf"],
            avg["rgb_publish"],
            avg["depth_extract_intrinsics"],
            avg["depth_msg_publish"],
            avg["pointcloud_convert"],
            avg["pointcloud_publish"],
            avg["camera_info_publish"],
            avg["action_wait"],
            avg["postprocess_action"],
        )

    def _action_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self._rospy.logwarn("RosBridgePolicy: received non-JSON action payload, ignored.")
            return

        step = int(payload.get("step", -1))
        raw_action = payload.get("action", payload)
        action = self._coerce_action(raw_action)
        with self._lock:
            self._latest_action = action
            self._latest_action_step = step

    def _cmd_vel_callback(self, msg) -> None:
        twist = msg.twist
        cmd = np.array(
            [float(twist.linear.x), float(twist.linear.y), float(twist.angular.z)],
            dtype=np.float32,
        )
        with self._lock:
            self._latest_cmd_vel = cmd
            self._latest_cmd_vel_mono_s = time.monotonic()

    @staticmethod
    def _quat_wxyz_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _cmd_vel_to_base_action(self, cmd_vel: np.ndarray, observation: Any) -> dict[str, Any] | None:
        base_pose = self._extract_base_pose_from_observation(observation)
        if base_pose is None:
            return None

        px, py = float(base_pose[0]), float(base_pose[1])
        qw, qx, qy, qz = (
            float(base_pose[3]),
            float(base_pose[4]),
            float(base_pose[5]),
            float(base_pose[6]),
        )
        yaw = self._quat_wxyz_to_yaw(qw, qx, qy, qz)
        vx, vy, wz = float(cmd_vel[0]), float(cmd_vel[1]), float(cmd_vel[2])
        vx *= self.cmd_vel_linear_gain
        vy *= self.cmd_vel_linear_gain
        dt = self.cmd_vel_control_dt_s

        dx_world = (vx * np.cos(yaw) - vy * np.sin(yaw)) * dt
        dy_world = (vx * np.sin(yaw) + vy * np.cos(yaw)) * dt
        target_yaw = self._wrap_to_pi(yaw + wz * dt)

        return {
            "base": np.array([px + dx_world, py + dy_world, target_yaw], dtype=np.float32),
            "done": False,
        }

    def _coerce_action(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._coerce_action(v) for k, v in obj.items()}
        if isinstance(obj, list):
            if all(isinstance(v, (int, float)) for v in obj):
                return np.asarray(obj, dtype=np.float32)
            return [self._coerce_action(v) for v in obj]
        return obj

    def _sanitize_observation(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._sanitize_observation(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_observation(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._sanitize_observation(v) for v in obj]
        if isinstance(obj, np.ndarray):
            # Keep payloads bounded for ROS String transport.
            if obj.size <= 256:
                return obj.tolist()
            return {
                "__ndarray_summary__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "min": float(np.nanmin(obj)) if obj.size > 0 else None,
                "max": float(np.nanmax(obj)) if obj.size > 0 else None,
            }
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def _build_noop_action(self) -> dict[str, Any]:
        if self.task is not None:
            robot_view = self.task.env.current_robot.robot_view
            return {**robot_view.get_noop_ctrl_dict(["base"]), "done": False}
        return {"done": False}

    def _extract_image_from_observation(self, observation: Any) -> np.ndarray | None:
        obs_dict = self._extract_observation_dict(observation)
        if obs_dict is None:
            return None

        # Prefer common camera keys, then fallback to first image-like tensor.
        preferred_keys = ("head_camera", "exo_camera_1", "wrist_camera", "rgb", "image")
        for key in preferred_keys:
            value = obs_dict.get(key)
            if isinstance(value, np.ndarray) and value.ndim in (2, 3):
                return value

        for value in obs_dict.values():
            if isinstance(value, np.ndarray) and value.ndim in (2, 3):
                return value
        return None

    def _extract_named_image_from_observation(self, observation: Any, camera_name: str) -> np.ndarray | None:
        obs_dict = self._extract_observation_dict(observation)
        if obs_dict is None or not camera_name:
            return None
        value = obs_dict.get(camera_name)
        if isinstance(value, np.ndarray) and value.ndim in (2, 3):
            return value
        return None

    def _extract_observation_dict(self, observation: Any) -> dict[str, Any] | None:
        if isinstance(observation, list) and len(observation) > 0:
            observation = observation[0]
        if not isinstance(observation, dict):
            return None
        return observation

    def _extract_depth_from_observation(self, observation: Any) -> tuple[str, np.ndarray] | None:
        obs_dict = self._extract_observation_dict(observation)
        if obs_dict is None:
            return None

        preferred_key = f"{self.depth_camera_name}_depth"
        value = obs_dict.get(preferred_key)
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return preferred_key, value
        return None

    def _extract_intrinsics_from_observation(
        self, observation: Any, depth_key: str
    ) -> tuple[float, float, float, float] | None:
        obs_dict = self._extract_observation_dict(observation)
        if obs_dict is None:
            return None

        camera_name = depth_key[:-6] if depth_key.endswith("_depth") else depth_key
        sensor_key = f"sensor_param_{camera_name}"
        params = obs_dict.get(sensor_key)
        if not isinstance(params, dict):
            return None

        intrinsic_cv = params.get("intrinsic_cv")
        if intrinsic_cv is None:
            return None
        intrinsic = np.asarray(intrinsic_cv, dtype=np.float32)
        if intrinsic.shape != (3, 3):
            return None

        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        if fx <= 0 or fy <= 0:
            return None
        return fx, fy, cx, cy

    def _get_camera_fov_deg(self, camera_name: str) -> float | None:
        cam_cfg = getattr(getattr(self.config, "camera_config", None), "cameras", None)
        if cam_cfg is None:
            return None
        for cam in cam_cfg:
            if getattr(cam, "name", None) == camera_name:
                fov = getattr(cam, "fov", None)
                if fov is not None:
                    return float(fov)
        return None

    def _intrinsics_from_fov(
        self, camera_name: str, width: int, height: int
    ) -> tuple[float, float, float, float] | None:
        fov_deg = self._get_camera_fov_deg(camera_name)
        if fov_deg is None or width <= 0 or height <= 0:
            return None
        fov_rad = np.deg2rad(max(1e-3, fov_deg))
        # Treat configured fov as vertical FoV (matches sensor generation path).
        fy = (height * 0.5) / np.tan(fov_rad * 0.5)
        fx = fy
        cx = width * 0.5
        cy = height * 0.5
        return float(fx), float(fy), float(cx), float(cy)

    @staticmethod
    def _normalize_intrinsics_to_image_shape(
        intrinsics: tuple[float, float, float, float] | None,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float] | None:
        """Heuristically rescale intrinsics when source resolution mismatches depth image."""
        if intrinsics is None:
            return None
        fx, fy, cx, cy = intrinsics
        if width <= 0 or height <= 0:
            return None

        # Typical principal point should be near image center; if far away, assume mismatched resolution.
        target_cx = 0.5 * width
        target_cy = 0.5 * height
        sx = target_cx / cx if cx > 1e-6 else 1.0
        sy = target_cy / cy if cy > 1e-6 else 1.0

        # Apply only when mismatch is significant to avoid perturbing valid intrinsics.
        if abs(cx - target_cx) > 0.1 * width:
            fx *= sx
            cx *= sx
        if abs(cy - target_cy) > 0.1 * height:
            fy *= sy
            cy *= sy

        if fx <= 0 or fy <= 0:
            return None
        return fx, fy, cx, cy

    def _get_pointcloud_projection_lut(
        self, height: int, width: int, fx: float, fy: float, cx: float, cy: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return cached normalized projection maps used by depth->pointcloud:
          x_cam = proj_x * depth
          y_cam = proj_y * depth
        """
        key = (
            int(height),
            int(width),
            round(float(fx), 6),
            round(float(fy), 6),
            round(float(cx), 6),
            round(float(cy), 6),
        )
        cached = self._pointcloud_projection_cache.get(key)
        if cached is not None:
            return cached

        u = np.arange(width, dtype=np.float32)
        v = np.arange(height, dtype=np.float32)
        proj_x_row = (u - float(cx)) / max(float(fx), 1e-6)
        proj_y_col = (v - float(cy)) / max(float(fy), 1e-6)
        proj_x = np.broadcast_to(proj_x_row[None, :], (height, width)).copy()
        proj_y = np.broadcast_to(proj_y_col[:, None], (height, width)).copy()
        self._pointcloud_projection_cache[key] = (proj_x, proj_y)
        return proj_x, proj_y

    def _depth_to_pointcloud_msg(
        self,
        depth: np.ndarray,
        intrinsics: tuple[float, float, float, float] | None = None,
        stamp=None,
    ):
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32, copy=False)

        orig_h, orig_w = depth.shape
        stride = self.pointcloud_stride
        if self.pointcloud_stride > 1:
            depth = depth[::stride, ::stride]

        h, w = depth.shape
        if h == 0 or w == 0:
            return None

        if intrinsics is not None:
            fx, fy, cx, cy = intrinsics
            if stride > 1:
                fx /= stride
                fy /= stride
                cx /= stride
                cy /= stride
        else:
            fov_rad = np.deg2rad(max(1e-3, self.depth_fov_deg))
            fy = (h * 0.5) / np.tan(fov_rad * 0.5)
            fx = fy
            cx = (w - 1) * 0.5
            cy = (h - 1) * 0.5

        valid = np.isfinite(depth) & (depth >= self.depth_min_m) & (depth <= self.depth_max_m)
        if not np.any(valid):
            return None

        proj_x, proj_y = self._get_pointcloud_projection_lut(h, w, fx, fy, cx, cy)
        # Camera optical frame:
        # x_cam: right, y_cam: down, z_cam: forward
        z_cam = depth[valid]
        x_cam = proj_x[valid] * z_cam
        y_cam = proj_y[valid] * z_cam

        # Convert to robot-centric frame for mapping:
        # x: forward, y: left, z: up
        points = np.empty((z_cam.shape[0], 3), dtype=np.float32)
        points[:, 0] = z_cam
        points[:, 1] = -x_cam
        points[:, 2] = -y_cam

        # Optional roll correction around forward axis (robot +x).
        # Positive follows right-hand rule; if you observe clockwise tilt in view,
        # use a negative value to compensate.
        if abs(self.pointcloud_roll_correction_deg) > 1e-6:
            theta = np.deg2rad(self.pointcloud_roll_correction_deg)
            c, s = np.cos(theta), np.sin(theta)
            y_corr = points[:, 1] * c - points[:, 2] * s
            z_corr = points[:, 1] * s + points[:, 2] * c
            points[:, 1] = y_corr
            points[:, 2] = z_corr

        msg = self._PointCloud2()
        msg.header.stamp = stamp if stamp is not None else self._rospy.Time.now()
        msg.header.frame_id = self.pointcloud_frame_id
        msg.height = 1
        msg.width = int(points.shape[0])
        msg.fields = [
            self._PointField(name="x", offset=0, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name="y", offset=4, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name="z", offset=8, datatype=self._PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False
        msg.data = points.tobytes()
        return msg

    def _to_depth_msg(self, depth: np.ndarray, stamp=None):
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32, copy=False)
        msg = self._Image()
        msg.header.stamp = stamp if stamp is not None else self._rospy.Time.now()
        msg.header.frame_id = self.optical_frame_id
        msg.height = int(depth.shape[0])
        msg.width = int(depth.shape[1])
        msg.encoding = "32FC1"
        msg.is_bigendian = 0
        msg.step = int(depth.shape[1] * 4)
        msg.data = depth.tobytes()
        return msg

    def _build_camera_info_msg(
        self,
        width: int,
        height: int,
        intrinsics: tuple[float, float, float, float] | None,
        stamp=None,
    ):
        if intrinsics is None:
            return None
        fx, fy, cx, cy = intrinsics
        info = self._CameraInfo()
        info.header.stamp = stamp if stamp is not None else self._rospy.Time.now()
        info.header.frame_id = self.optical_frame_id
        info.width = int(width)
        info.height = int(height)
        info.distortion_model = "plumb_bob"
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _to_image_msg(self, frame: np.ndarray, stamp=None):
        img = frame
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                # Best effort normalization for float images.
                max_val = float(np.nanmax(img)) if img.size > 0 else 1.0
                if max_val <= 1.0:
                    img = np.clip(img, 0.0, 1.0) * 255.0
                img = np.nan_to_num(img, nan=0.0)
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            h, w = img.shape
            encoding = "mono8"
            step = w
            data = img.tobytes()
        elif img.ndim == 3 and img.shape[2] >= 3:
            # Keep first 3 channels as rgb8.
            img = img[:, :, :3]
            h, w, _ = img.shape
            encoding = "rgb8"
            step = w * 3
            data = img.tobytes()
        else:
            return None

        msg = self._Image()
        msg.header.stamp = stamp if stamp is not None else self._rospy.Time.now()
        msg.header.frame_id = self.optical_frame_id
        msg.height = h
        msg.width = w
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = step
        msg.data = data
        return msg

    def _extract_base_pose_from_observation(self, observation: Any) -> np.ndarray | None:
        obs_dict = self._extract_observation_dict(observation)
        if obs_dict is None:
            return None
        pose = obs_dict.get("robot_base_pose")
        if pose is None:
            return None
        pose_arr = np.asarray(pose, dtype=np.float32).reshape(-1)
        if pose_arr.size < 7:
            return None
        return pose_arr[:7]

    def _next_common_stamp(self):
        stamp = self._rospy.Time.now()
        stamp_s = float(stamp.to_sec())
        if self._last_common_stamp_s is not None and stamp_s <= self._last_common_stamp_s:
            # Keep publish timestamps strictly monotonic to avoid occasional time back-jumps.
            stamp_s = self._last_common_stamp_s + 1e-6
            stamp = self._rospy.Time.from_sec(stamp_s)
        self._last_common_stamp_s = stamp_s
        return stamp

    def _publish_odom_and_tf(self, observation: Any, stamp) -> bool:
        if not self.publish_odom:
            return True

        base_pose = self._extract_base_pose_from_observation(observation)
        if base_pose is None:
            self._rospy.logwarn_throttle(
                5.0,
                "RosBridgePolicy: robot_base_pose missing; /odom and odom->base TF not published.",
            )
            return False

        # robot_base_pose layout: [x, y, z, qw, qx, qy, qz]
        px, py, pz = float(base_pose[0]), float(base_pose[1]), float(base_pose[2])
        qw, qx, qy, qz = (
            float(base_pose[3]),
            float(base_pose[4]),
            float(base_pose[5]),
            float(base_pose[6]),
        )
        curr_pos = np.array([px, py, pz], dtype=np.float32)
        if self._last_base_position_xyz is not None:
            jump_dist = float(np.linalg.norm(curr_pos - self._last_base_position_xyz))
            if jump_dist > self._base_position_jump_warn_m:
                self._rospy.logwarn(
                    (
                        "RosBridgePolicy: detected base position jump > %.2fm (dist=%.3fm). "
                        "prev=(%.3f, %.3f, %.3f), curr=(%.3f, %.3f, %.3f), step=%d"
                    ),
                    self._base_position_jump_warn_m,
                    jump_dist,
                    float(self._last_base_position_xyz[0]),
                    float(self._last_base_position_xyz[1]),
                    float(self._last_base_position_xyz[2]),
                    px,
                    py,
                    pz,
                    self._step_idx,
                )
        self._last_base_position_xyz = curr_pos

        odom_msg = self._Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.odom_frame_id
        odom_msg.child_frame_id = self.base_frame_id
        odom_msg.pose.pose.position.x = px
        odom_msg.pose.pose.position.y = py
        odom_msg.pose.pose.position.z = pz
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        self._odom_pub.publish(odom_msg)

        tf_msg = self._TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.odom_frame_id
        tf_msg.child_frame_id = self.base_frame_id
        tf_msg.transform.translation.x = px
        tf_msg.transform.translation.y = py
        tf_msg.transform.translation.z = pz
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(tf_msg)
        return self._publish_base_to_lidar_tf(observation, stamp)

    def get_action(self, observation):
        frame_t0 = time.perf_counter()
        stage_ms = {k: 0.0 for k in self._timing_acc_ms}

        t0 = time.perf_counter()
        common_stamp = self._next_common_stamp()
        tf_ready = self._publish_odom_and_tf(observation, common_stamp)
        stage_ms["odom_tf"] = (time.perf_counter() - t0) * 1000.0

        skip_mapping_observation = self._step_idx < self.map_warmup_skip_frames or (not tf_ready)
        if skip_mapping_observation:
            if self._step_idx < self.map_warmup_skip_frames:
                self._rospy.loginfo_throttle(
                    2.0,
                    "RosBridgePolicy: map warmup active, skipping observation publish (%d/%d)",
                    self._step_idx + 1,
                    self.map_warmup_skip_frames,
                )
            elif not tf_ready:
                self._rospy.logwarn_throttle(
                    2.0,
                    "RosBridgePolicy: odom/tf not ready this frame, skip depth/pointcloud publish to keep timestamps aligned.",
                )

        if not skip_mapping_observation:
            t0 = time.perf_counter()
            frame = self._extract_image_from_observation(observation)
            if frame is not None:
                msg = self._to_image_msg(frame, stamp=common_stamp)
                if msg is not None:
                    self._obs_pub.publish(msg)
                else:
                    self._rospy.logwarn_throttle(2.0, "RosBridgePolicy: failed to encode image.")
            else:
                self._rospy.logwarn_throttle(
                    2.0,
                    "RosBridgePolicy: no image-like tensor found in observation."
                )
            if self._extra_image_pub is not None:
                extra_frame = self._extract_named_image_from_observation(observation, self.extra_image_camera_name)
                if extra_frame is not None:
                    extra_msg = self._to_image_msg(extra_frame, stamp=common_stamp)
                    if extra_msg is not None:
                        self._extra_image_pub.publish(extra_msg)
            stage_ms["rgb_publish"] = (time.perf_counter() - t0) * 1000.0

        if self.publish_pointcloud and not skip_mapping_observation:
            t0_depth_extract = time.perf_counter()
            depth_data = self._extract_depth_from_observation(observation)
            if depth_data is not None:
                depth_key, depth = depth_data
                camera_name = depth_key[:-6] if depth_key.endswith("_depth") else depth_key
                intrinsics = self._extract_intrinsics_from_observation(observation, depth_key)
                intrinsics = self._normalize_intrinsics_to_image_shape(
                    intrinsics, width=depth.shape[1], height=depth.shape[0]
                )
                # If principal-point scaling is anisotropic, prefer FoV-based intrinsics
                # to avoid aspect-ratio distortion in RViz DepthCloud.
                if intrinsics is not None:
                    fx, fy, _, _ = intrinsics
                    fx, fy, cx, cy = intrinsics
                    
                    ratio = fx / max(fy, 1e-6)
                    if ratio < 0.9 or ratio > 1.1:
                        fov_intrinsics = self._intrinsics_from_fov(
                            camera_name, width=depth.shape[1], height=depth.shape[0]
                        )
                        if fov_intrinsics is not None:
                            intrinsics = fov_intrinsics
                elif camera_name:
                    intrinsics = self._intrinsics_from_fov(
                        camera_name, width=depth.shape[1], height=depth.shape[0]
                    )
                # For the selected depth camera (default head_camera), enforce FoV-based
                # intrinsics to avoid mixed-resolution artifacts from sensor_param_*.
                if camera_name == self.depth_camera_name:
                    fov_intrinsics = self._intrinsics_from_fov(
                        camera_name, width=depth.shape[1], height=depth.shape[0]
                    )
                    if fov_intrinsics is not None:
                        intrinsics = fov_intrinsics
                stage_ms["depth_extract_intrinsics"] = (
                    time.perf_counter() - t0_depth_extract
                ) * 1000.0
                
                stamp = common_stamp
                t0_depth_msg = time.perf_counter()
                self._depth_pub.publish(self._to_depth_msg(depth, stamp=stamp))
                stage_ms["depth_msg_publish"] = (time.perf_counter() - t0_depth_msg) * 1000.0
                t0_pcd_convert = time.perf_counter()
                cloud_msg = self._depth_to_pointcloud_msg(depth, intrinsics=intrinsics, stamp=stamp)
                stage_ms["pointcloud_convert"] = (time.perf_counter() - t0_pcd_convert) * 1000.0
                if cloud_msg is not None:
                    t0_pcd_pub = time.perf_counter()
                    self._pointcloud_pub.publish(cloud_msg)
                    stage_ms["pointcloud_publish"] = (time.perf_counter() - t0_pcd_pub) * 1000.0
                    if self.publish_camera_info:
                        t0_cam_info = time.perf_counter()
                        info_msg = self._build_camera_info_msg(
                            width=depth.shape[1],
                            height=depth.shape[0],
                            intrinsics=intrinsics,
                            stamp=stamp,
                        )
                        if info_msg is not None:
                            self._camera_info_pub.publish(info_msg)
                            self._image_camera_info_pub.publish(info_msg)
                            self._depth_camera_info_pub.publish(info_msg)
                        stage_ms["camera_info_publish"] = (
                            time.perf_counter() - t0_cam_info
                        ) * 1000.0
                else:
                    self._rospy.logwarn_throttle(
                        2.0,
                        "RosBridgePolicy: depth found but no valid points for PointCloud2.",
                    )
            else:
                self._rospy.logwarn_throttle(
                    2.0,
                    "RosBridgePolicy: no depth tensor found; PointCloud2 not published.",
                )

        if self.immediate_noop_after_publish:
            # Debug mode: isolate simulator publish/step throughput from policy inference/wait time.
            t0_post = time.perf_counter()
            chosen_action = self._build_noop_action()
            if isinstance(chosen_action, dict):
                chosen_action.setdefault("done", False)
            stage_ms["postprocess_action"] = (time.perf_counter() - t0_post) * 1000.0
            stage_ms["total"] = (time.perf_counter() - frame_t0) * 1000.0
            self._record_timing(stage_ms)
            self._step_idx += 1
            return chosen_action

        t0_wait = time.perf_counter()
        deadline = time.monotonic() + self.action_timeout_s
        chosen_action = None
        while time.monotonic() < deadline and not self._rospy.is_shutdown():
            cmd_vel = None
            cmd_vel_ts = 0.0
            with self._lock:
                if self._latest_action is not None:
                    if self._latest_action_step < 0:
                        # Allow action payloads without explicit step field.
                        chosen_action = self._latest_action
                        self._last_consumed_action_step += 1
                        break
                    if self._latest_action_step > self._last_consumed_action_step:
                        chosen_action = self._latest_action
                        self._last_consumed_action_step = self._latest_action_step
                        break
                cmd_vel = self._latest_cmd_vel
                cmd_vel_ts = self._latest_cmd_vel_mono_s
            if cmd_vel is not None and (time.monotonic() - cmd_vel_ts) <= self.cmd_vel_timeout_s:
                cmd_action = self._cmd_vel_to_base_action(cmd_vel, observation)
                if cmd_action is not None:
                    chosen_action = cmd_action
                    break
            time.sleep(0.005)
        stage_ms["action_wait"] = (time.perf_counter() - t0_wait) * 1000.0

        if chosen_action is None:
            chosen_action = self._build_noop_action()
            self._rospy.logwarn_throttle(2.0, "RosBridgePolicy: action timeout, using noop action.")

        t0_post = time.perf_counter()
        if isinstance(chosen_action, dict):
            # Keep downstream navigation task behavior consistent.
            chosen_action.setdefault("done", False)
            if self.task is not None:
                robot_view = self.task.env.current_robot.robot_view
                if "base" not in chosen_action:
                    chosen_action["base"] = robot_view.get_noop_ctrl_dict(["base"])["base"]

                move_group_ids = set(robot_view.move_group_ids())
                if "left_arm" in move_group_ids and "left_arm" not in chosen_action:
                    noop_left = np.asarray(
                        robot_view.get_noop_ctrl_dict(["left_arm"])["left_arm"], dtype=np.float32
                    )
                    if noop_left.shape == self.default_left_arm_qpos.shape:
                        chosen_action["left_arm"] = self.default_left_arm_qpos.copy()
                    else:
                        self._rospy.logwarn_throttle(
                            5.0,
                            "RosBridgePolicy: left_arm default shape mismatch (%s vs %s), falling back to noop.",
                            str(self.default_left_arm_qpos.shape),
                            str(noop_left.shape),
                        )
                        chosen_action["left_arm"] = noop_left

                if "right_arm" in move_group_ids and "right_arm" not in chosen_action:
                    noop_right = np.asarray(
                        robot_view.get_noop_ctrl_dict(["right_arm"])["right_arm"], dtype=np.float32
                    )
                    if noop_right.shape == self.default_right_arm_qpos.shape:
                        chosen_action["right_arm"] = self.default_right_arm_qpos.copy()
                    else:
                        self._rospy.logwarn_throttle(
                            5.0,
                            "RosBridgePolicy: right_arm default shape mismatch (%s vs %s), falling back to noop.",
                            str(self.default_right_arm_qpos.shape),
                            str(noop_right.shape),
                        )
                        chosen_action["right_arm"] = noop_right
        stage_ms["postprocess_action"] = (time.perf_counter() - t0_post) * 1000.0
        stage_ms["total"] = (time.perf_counter() - frame_t0) * 1000.0
        self._record_timing(stage_ms)

        self._step_idx += 1
        return chosen_action

    def close(self):
        if hasattr(self, "_action_sub") and self._action_sub is not None:
            self._action_sub.unregister()
            self._action_sub = None
        if hasattr(self, "_cmd_vel_sub") and self._cmd_vel_sub is not None:
            self._cmd_vel_sub.unregister()
            self._cmd_vel_sub = None
