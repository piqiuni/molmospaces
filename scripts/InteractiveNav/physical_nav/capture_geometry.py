"""Shared capture-time geometry for the sensor adapter and detector.

Pure NumPy/Python: importing this module never loads ROS or starts a node.
The original private names are retained as compatibility exports by the bridge.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _insert_capture_telemetry(
    history,
    stamp: float,
    snapshot: dict[str, Any],
) -> None:
    """Insert one pose sample in timestamp order while retaining its bound.

    Capture packets are decoded on a different thread from the small telemetry
    envelopes, so a valid older capture sample can arrive after a newer live
    pose. A plain ``deque.append`` would either lose causal ordering or evict
    the wrong end of a full history. Equal timestamps describe the same pose
    instant; retain the latest receipt without consuming another history slot.
    """

    sample_stamp = float(stamp)
    entries = [
        (float(existing_stamp), existing_snapshot)
        for existing_stamp, existing_snapshot in history
        if float(existing_stamp) != sample_stamp
    ]
    entries.append((sample_stamp, dict(snapshot)))
    entries.sort(key=lambda item: item[0])
    maxlen = getattr(history, "maxlen", None)
    if maxlen is not None:
        maxlen = int(maxlen)
        entries = entries[-maxlen:] if maxlen > 0 else []
    history.clear()
    history.extend(entries)


def _nearest_capture_telemetry(history, stamp, max_delta_sec=0.15):
    """Select the latest causal pose; never wait or use a future sample."""
    try:
        capture = float(stamp)
        limit = float(max_delta_sec)
    except (TypeError, ValueError, OverflowError):
        return {}, None
    if not math.isfinite(capture) or capture <= 0. or not math.isfinite(limit) or limit < 0.:
        return {}, None
    best = None
    best_stamp = -float("inf")
    best_index = -1
    # Callback fixtures are not required to be pre-sorted. At an identical
    # timestamp the later receipt wins, matching duplicate replacement in the
    # production history.
    for index, item in enumerate(history):
        try:
            sample_stamp, snapshot = item
            sample_stamp = float(sample_stamp)
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            not isinstance(snapshot, dict)
            or not math.isfinite(sample_stamp)
            or sample_stamp <= 0.
            or sample_stamp > capture
        ):
            continue
        if sample_stamp > best_stamp or (
            sample_stamp == best_stamp and index > best_index
        ):
            best_stamp = sample_stamp
            best_index = index
            best = snapshot
    if best is None:
        return {}, None
    delta = capture - best_stamp
    if delta > limit:
        return {}, delta
    return dict(best), delta


def _telemetry_body_position(telemetry: Any) -> tuple[float, float, float] | None:
    if not isinstance(telemetry, dict):
        return None
    try:
        position = np.asarray(telemetry.get("position"), dtype=float)
        if position.ndim != 1 or position.size < 3 or not np.isfinite(position[:3]).all():
            return None
        return tuple(float(value) for value in position[:3])
    except (TypeError, ValueError, OverflowError):
        return None


def _telemetry_body_quaternion(telemetry: Any) -> tuple[float, float, float, float] | None:
    """One normalized base-link orientation for sensor TF and embedded boxes.

    Unitree IMU lists are wxyz; explicit mappings and other body-pose lists
    are xyzw. Camera pose/IMU fields cannot substitute for chassis orientation.
    Missing/invalid orientation stays missing, rather than becoming identity.
    """
    if not isinstance(telemetry, dict):
        return None
    for name in ("imu", "base_pose", "odom", "pose"):
        source = telemetry.get(name)
        if not isinstance(source, dict):
            continue
        for field in ("quaternion", "orientation"):
            raw = source.get(field)
            try:
                if isinstance(raw, dict):
                    values = [float(raw[key]) for key in ("x", "y", "z", "w")]
                elif isinstance(raw, (list, tuple, np.ndarray)) and len(raw) >= 4:
                    values = [float(value) for value in raw[:4]]
                    if name == "imu":
                        values = [values[1], values[2], values[3], values[0]]
                else:
                    continue
                norm = math.hypot(*values)
                if math.isfinite(norm) and norm > 1e-6:
                    return tuple(value / norm for value in values)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
    try:
        yaw = float(telemetry["yaw"] if "yaw" in telemetry else telemetry["imu"]["rpy"][2])
        if math.isfinite(yaw):
            return (0., 0., math.sin(yaw / 2.), math.cos(yaw / 2.))
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        pass
    return None


def _quat_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def _quat_multiply(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _quat_normalize(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Return a finite unit quaternion (xyzw), or identity on bad input."""
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        return (0.0, 0.0, 0.0, 1.0)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(float(value / norm) for value in values)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Interpret ROS/CLI bool parameters without treating ``"false"`` as true."""
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "off", "none"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _camera_imu_capture_match(telemetry, source_stamp, *, enabled=True, use_yaw=False, max_delta_sec=.15):
    """Check component receipts in the Go2 source clock, not host arrival time."""
    result = {"status": "unavailable", "accepted": True, "ages_sec": {},
              "continuity": {}, "max_delta_sec": max_delta_sec,
              "time_basis": "none"}
    if not enabled:
        result["status"] = "disabled"
        return result
    motion = telemetry.get("camera_imu") if isinstance(telemetry, dict) else None
    if not isinstance(motion, dict):
        return result
    if _coerce_bool(motion.get("imu_calibrating", False), False):
        result["status"] = "calibrating"
        return result
    if "component_received_at" not in motion:
        # Old recordings do not prove independent accel/gyro freshness.
        result["status"] = "legacy_unverified"
        return result
    required_components = ("accel", "gyro") if use_yaw else ("accel",)
    if "imu_ingress" in motion:
        try:
            streams = motion["imu_ingress"]["streams"]
            for component in required_components:
                health = streams[component]
                result["continuity"][component] = {
                    key: health.get(key)
                    for key in (
                        "healthy", "continuous_s", "effective_hz", "missing",
                        "duplicates", "late", "timestamp_regressions",
                    )
                }
                if health.get("healthy") is not True:
                    result.update(status="discontinuous", accepted=False)
                    return result
        except (TypeError, KeyError):
            result.update(status="invalid_continuity", accepted=False)
            return result
    time_field = "component_received_at"
    sample_times = motion.get("component_sample_at")
    sample_valid = motion.get("component_sample_time_valid")
    if (
        isinstance(sample_times, dict)
        and isinstance(sample_valid, dict)
        and all(sample_valid.get(component) is True for component in required_components)
    ):
        time_field = "component_sample_at"
        result["time_basis"] = "device_clock_mapped_sample"
    else:
        result["time_basis"] = "source_receipt"
    result.update(status="invalid_time", accepted=False)
    try:
        reference = float(source_stamp)
        limit = float(max_delta_sec)
        if not math.isfinite(reference) or reference <= 0. or not math.isfinite(limit) or limit < 0.:
            return result
        times = motion[time_field]
        for component in required_components:
            value = float(times[component])
            if not math.isfinite(value) or value <= 0.:
                return result
            result["ages_sec"][component] = reference - value
        if any(abs(age) > limit + 1e-9 for age in result["ages_sec"].values()):
            result["status"] = "stale"
        else:
            result.update(status="matched", accepted=True)
    except (TypeError, KeyError, ValueError, OverflowError):
        pass
    return result


def _camera_imu_correction_quaternion(
    telemetry: dict[str, Any] | None,
    *,
    enabled: bool = True,
    use_yaw: bool = False,
    max_roll_rad: float = 0.7,
    max_pitch_rad: float = 0.7,
    max_yaw_rad: float = 0.35,
) -> tuple[tuple[float, float, float, float], bool]:
    """Read a calibrated D435i correction and return ``(q, valid)``.

    ``go2_readonly_sensor_bridge`` publishes ``camera_imu.correction_rpy``
    after its gravity/gyro reference has converged.  The values are Euler
    angles in the D435/depth optical axes (x=right, y=down, z=forward), not
    Go2 ``base_link`` axes.  The caller converts this sensor-frame rotation to
    the parent frame before applying it to TF.  During calibration, with
    missing data, or for an implausible jump we deliberately return identity;
    this preserves the existing static extrinsic instead of injecting a bad
    TF into SLAM.

    Yaw is disabled by default because the D435i gyro yaw is relative and has
    no absolute heading reference.  Roll/pitch are gravity-observable and are
    therefore safe to use for pole sway correction once calibrated.
    """
    identity = (0.0, 0.0, 0.0, 1.0)
    if not enabled or not isinstance(telemetry, dict):
        return identity, False
    camera_imu = telemetry.get("camera_imu")
    if not isinstance(camera_imu, dict) or _coerce_bool(
        camera_imu.get("imu_calibrating", False), False
    ):
        return identity, False
    values = camera_imu.get("correction_rpy")
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return identity, False
    try:
        roll, pitch, yaw = (float(value) for value in values[:3])
    except (TypeError, ValueError):
        return identity, False
    if not all(math.isfinite(value) for value in (roll, pitch, yaw)):
        return identity, False
    limits = (max(0.0, float(max_roll_rad)), max(0.0, float(max_pitch_rad)),
              max(0.0, float(max_yaw_rad)))
    # Ignore the relative gyro yaw entirely unless explicitly requested.  A
    # long run can accumulate yaw drift far beyond ``max_yaw_rad``; that must
    # not invalidate otherwise useful gravity roll/pitch correction.
    if abs(roll) > limits[0] or abs(pitch) > limits[1] or (
        use_yaw and abs(yaw) > limits[2]
    ):
        return identity, False
    if not use_yaw:
        yaw = 0.0
    return _quat_normalize(_quat_rpy(roll, pitch, yaw)), True


def _camera_imu_parent_correction_quaternion(
    telemetry: dict[str, Any] | None,
    mount_quaternion: tuple[float, float, float, float],
    **kwargs: Any,
) -> tuple[tuple[float, float, float, float], bool]:
    """Convert a D435-frame correction into the TF parent frame.

    ``mount_quaternion`` maps the nominal camera optical frame into the TF
    parent (``base_link``).  A rotation measured around D435 axes must be
    conjugated by that nominal mount before it can be pre-multiplied onto the
    parent->camera transform:

    ``R_parent = R_mount R_d435 R_mountᵀ``.

    This is the axis substitution that is easy to miss when the camera is
    mounted with the REP-103 optical frame (camera roll maps mostly to Go2
    base yaw/side axes, and camera pitch does not map one-to-one to base
    pitch).  Invalid/calibrating samples retain identity.
    """
    sensor_quaternion, valid = _camera_imu_correction_quaternion(
        telemetry, **kwargs
    )
    if not valid:
        return (0.0, 0.0, 0.0, 1.0), False
    mount = _quat_normalize(mount_quaternion)
    inverse_mount = (-mount[0], -mount[1], -mount[2], mount[3])
    parent_quaternion = _quat_multiply(
        _quat_multiply(mount, sensor_quaternion), inverse_mount
    )
    return _quat_normalize(parent_quaternion), True


def _apply_camera_imu_correction(
    mount_quaternion: tuple[float, float, float, float],
    mount_translation: tuple[float, float, float],
    correction_quaternion: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    """Apply a parent-frame correction to a camera rigid transform.

    The correction acts on the complete parent<-camera transform (the ROS TF
    transform that maps camera coordinates into the parent), not only on its
    orientation: ``T' = T_correction * T_mount``.  In particular,
    ``t' = R_correction * t`` accounts for the horizontal displacement caused
    by the 0.98 m elevated camera when the two aluminium rails lean.
    """
    correction = _quat_normalize(correction_quaternion)
    mount = _quat_normalize(mount_quaternion)
    corrected_quaternion = _quat_normalize(_quat_multiply(correction, mount))
    corrected_translation = _quat_rotate(correction, mount_translation)
    return corrected_quaternion, corrected_translation


def _quat_rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a vector by an xyzw quaternion without ROS geometry helpers."""
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
    """Convert a RealSense row-major 3x3 rotation to an xyzw quaternion."""
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
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
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
    norm = max(float(np.linalg.norm(values)), 1e-12)
    return tuple(float(value / norm) for value in values)


def _depth_static_extrinsic(
    color_quaternion: tuple[float, float, float, float],
    color_translation: tuple[float, float, float],
    depth_frame: str,
    color_frame: str,
    extrinsics: dict[str, Any] | None,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    """Compose base->depth from base->color and RealSense depth->color data.

    ``get_extrinsics_to(color_profile)`` returns
    ``p_color = R_dc p_depth + t_dc``.  Therefore
    ``T_base_depth = T_base_color * T_color_depth``.  Older/replay packets may
    omit the calibration; in that case retaining the color mount transform is
    the least surprising fallback.
    """
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
