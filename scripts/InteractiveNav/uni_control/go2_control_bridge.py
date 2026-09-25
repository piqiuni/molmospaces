#!/usr/bin/env python3
"""Go2-side unified WebSocket bridge for discrete and continuous control."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import websocket

from control_protocol import (
    ControlCommand,
    DiscreteAction,
    SpeechCommand,
    parse_control_message,
    parse_lidar_message,
    parse_posture_message,
    parse_speech_message,
)


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float
    stamp: float


@dataclass
class VelocityTarget:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    deadline: float = 0.0
    seq: int = -1


@dataclass
class ActivePrimitive:
    action: DiscreteAction
    seq: int
    start_pose: Pose2D
    started_at: float
    deadline: float


@dataclass(frozen=True)
class SpeechJob:
    command: SpeechCommand
    expires_at: float


PRELOADED_INTERACTION_PROMPTS = (
    "您好，请帮我把前面的门打开，谢谢。",
    "您好，请帮我把前面的冰箱打开，谢谢。",
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Go2Driver:
    """Go2 telemetry, explicit auxiliary actions, and optional velocity control."""

    def __init__(
        self,
        interface: str,
        state_timeout_s: float,
        enable_motion: bool,
        motion_enable_attempts: int = 5,
        motion_enable_retry_s: float = 0.5,
        min_locomotion_height_m: float = 0.15,
    ) -> None:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import (
            ObstaclesAvoidClient,
        )
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
            LowState_,
            SportModeState_,
        )

        self._pose_lock = threading.Lock()
        self._pose: Optional[Pose2D] = None
        self._telemetry: Optional[dict[str, Any]] = None
        self._state_event = threading.Event()
        self._min_locomotion_height_m = max(0.0, float(min_locomotion_height_m))
        self._last_posture_block_log_at = 0.0
        # rt/lowstate publishes at the low-level rate (~500 Hz); refresh the
        # battery cache at 5 Hz so the DDS callback stays cheap.
        self._bms_lock = threading.Lock()
        self._bms: Optional[dict[str, Any]] = None
        self._bms_last_update = 0.0

        def on_state(message: SportModeState_) -> None:
            quaternion = message.imu_state.quaternion
            w, x, y, z = (float(value) for value in quaternion)
            yaw = math.atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
            with self._pose_lock:
                self._pose = Pose2D(
                    x=float(message.position[0]),
                    y=float(message.position[1]),
                    yaw=yaw,
                    stamp=time.monotonic(),
                )
                self._telemetry = {
                    "received_at": time.time(),
                    "error_code": int(message.error_code),
                    "mode": int(message.mode),
                    "progress": float(message.progress),
                    "gait_type": int(message.gait_type),
                    "foot_raise_height": float(message.foot_raise_height),
                    "position": [float(value) for value in message.position],
                    "body_height": float(message.body_height),
                    "velocity": [float(value) for value in message.velocity],
                    "yaw_speed": float(message.yaw_speed),
                    "range_obstacle": [float(value) for value in message.range_obstacle],
                    "foot_force": [int(value) for value in message.foot_force],
                    "imu": {
                        "quaternion": [float(value) for value in quaternion],
                        "gyroscope": [
                            float(value) for value in message.imu_state.gyroscope
                        ],
                        "accelerometer": [
                            float(value) for value in message.imu_state.accelerometer
                        ],
                        "rpy": [float(value) for value in message.imu_state.rpy],
                        "temperature": int(message.imu_state.temperature),
                    },
                }
            self._state_event.set()

        def on_lowstate(message: LowState_) -> None:
            now = time.monotonic()
            if now - self._bms_last_update < 0.1:
                return
            self._bms_last_update = now
            bms = message.bms_state
            voltage = float(message.power_v)
            current = float(message.power_a)
            snapshot = {
                "received_at": time.time(),
                "soc": int(bms.soc),
                "bms_current": int(bms.current),
                "cycle": int(bms.cycle),
                "cell_vol": [int(value) for value in bms.cell_vol],
                "bq_ntc": [int(value) for value in bms.bq_ntc],
                "mcu_ntc": [int(value) for value in bms.mcu_ntc],
                "voltage": voltage,
                "current": current,
                "power": voltage * current,
                "temperature_ntc1": int(message.temperature_ntc1),
                "temperature_ntc2": int(message.temperature_ntc2),
                "fan_frequency": [int(value) for value in message.fan_frequency],
            }
            with self._bms_lock:
                self._bms = snapshot

        ChannelFactoryInitialize(0, interface)
        self._lidar_publisher = ChannelPublisher("rt/utlidar/switch", String_)
        self._lidar_publisher.Init()
        self._lidar_command = std_msgs_msg_dds__String_()
        self._subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self._subscriber.Init(on_state, 10)
        if not self._state_event.wait(state_timeout_s):
            raise RuntimeError(
                f"no rt/sportmodestate received on {interface!r} within {state_timeout_s}s"
            )
        # Low-level state is best-effort: battery telemetry is useful but a
        # missed topic must not prevent the control bridge from starting.
        self._lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._lowstate_subscriber.Init(on_lowstate, 10)
        have_bms = False
        lowstate_deadline = time.monotonic() + 2.0
        while not have_bms and time.monotonic() < lowstate_deadline:
            with self._bms_lock:
                have_bms = self._bms is not None
            if not have_bms:
                time.sleep(0.05)
        if not have_bms:
            print("warning: no rt/lowstate received; battery telemetry unavailable")

        self._client = None
        self._motion_authorized = bool(enable_motion)
        self._velocity_control_enabled = False
        self._motion_lock = threading.RLock()
        self._last_enable_attempt = 0.0
        if enable_motion:
            self._client = ObstaclesAvoidClient()
            self._client.SetTimeout(3.0)
            self._client.Init()
            self.enable_velocity_control(
                attempts=motion_enable_attempts,
                retry_delay_s=motion_enable_retry_s,
            )
            # Never inherit a command left by an earlier bridge/process. A
            # newly started bridge must be stationary until a fresh command
            # arrives from the policy server.
            self._client.Move(0.0, 0.0, 0.0)

        # Posture calls are explicit keyboard actions and remain available even when
        # velocity motion is disabled by the launcher.
        self._sport_client = SportClient()
        self._sport_client.SetTimeout(5.0)
        self._sport_client.Init()

    def get_pose(self) -> Optional[Pose2D]:
        with self._pose_lock:
            if self._pose is None:
                return None
            return Pose2D(**vars(self._pose))

    def move(self, vx: float, vy: float, wz: float) -> None:
        if self._client is None:
            return
        with self._motion_lock:
            if not getattr(self, "_motion_authorized", True):
                return
            moving = abs(vx) > 1e-6 or abs(vy) > 1e-6 or abs(wz) > 1e-6
            if moving and self._min_locomotion_height_m > 0.0:
                with self._pose_lock:
                    telemetry = dict(self._telemetry or {})
                position = telemetry.get("position") or []
                height = float(position[2]) if len(position) >= 3 else 0.0
                if height < self._min_locomotion_height_m:
                    now = time.monotonic()
                    if now - self._last_posture_block_log_at >= 2.0:
                        print(
                            "motion blocked: Go2 is not in a locomotion-ready "
                            f"posture (body z={height:.3f} m, "
                            f"required>={self._min_locomotion_height_m:.3f} m)"
                        )
                        self._last_posture_block_log_at = now
                    return
            if moving and not self._velocity_control_enabled:
                # StandUp can briefly leave the API lease disabled. Recover
                # here as well as in set_posture so the first key press after
                # standing does not permanently kill the control loop.
                if time.monotonic() - self._last_enable_attempt < 0.20:
                    return
                try:
                    self.enable_velocity_control()
                except RuntimeError as exc:
                    print(f"waiting for motion control permission: {exc}")
                    return
            code = self._client.Move(vx, vy, wz)
            if code not in (None, 0) and moving:
                # A posture transition may invalidate an otherwise-valid
                # lease. Reacquire it once and retry the command.
                self._velocity_control_enabled = False
                try:
                    self.enable_velocity_control()
                    code = self._client.Move(vx, vy, wz)
                except RuntimeError as exc:
                    print(f"motion command deferred while locked: {exc}")
                    return
            if code not in (None, 0):
                print(f"motion command rejected with code {code}")

    def set_motion_enabled(self, enabled: bool) -> None:
        """Change the explicit output gate without rebuilding DDS or speech."""
        with self._motion_lock:
            self._motion_authorized = False
            if self._client is not None:
                self._client.Move(0.0, 0.0, 0.0)
            if not enabled:
                self._velocity_control_enabled = False
                if self._client is not None:
                    code = self._client.UseRemoteCommandFromApi(False)
                    if code not in (None, 0):
                        raise RuntimeError(f"motion lease release failed: {code}")
                return
            if self._client is None:
                from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient
                client = ObstaclesAvoidClient()
                client.SetTimeout(3.0)
                client.Init()
                self._client = client
            try:
                self.enable_velocity_control(attempts=1)
                code = self._client.Move(0.0, 0.0, 0.0)
                if code not in (None, 0):
                    raise RuntimeError(f"initial stop failed: {code}")
            except Exception:
                self._velocity_control_enabled = False
                with contextlib.suppress(Exception):
                    self._client.UseRemoteCommandFromApi(False)
                raise
            self._motion_authorized = True

    def enable_velocity_control(
        self,
        *,
        attempts: int = 1,
        retry_delay_s: float = 0.0,
    ) -> None:
        """Re-acquire API motion permission after a posture transition."""
        if self._client is None:
            raise RuntimeError(
                "velocity control is disabled; start the bridge with --enable-motion"
            )
        attempts = max(1, int(attempts))
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            self._last_enable_attempt = time.monotonic()
            try:
                self._enable_velocity_control_once()
                return
            except Exception as exc:
                self._velocity_control_enabled = False
                last_error = exc
                if attempt >= attempts:
                    break
                print(
                    "motion control enable attempt "
                    f"{attempt}/{attempts} failed: {exc}; retrying"
                )
                time.sleep(max(0.0, retry_delay_s))
        raise RuntimeError(
            f"motion control enable failed after {attempts} attempts: {last_error}"
        ) from last_error

    def _enable_velocity_control_once(self) -> None:
        # StandUp/StandDown can reset the obstacle-avoidance service switch.
        # Move() remains locked until this switch is enabled again.
        switch_result = self._client.SwitchGet()
        if not isinstance(switch_result, tuple) or len(switch_result) != 2:
            raise RuntimeError(f"unexpected SwitchGet result: {switch_result!r}")
        switch_code, switch_enabled = switch_result
        if switch_code not in (None, 0):
            raise RuntimeError(f"ObstaclesAvoidClient.SwitchGet failed with code {switch_code}")
        if not switch_enabled:
            switch_code = self._client.SwitchSet(True)
            if switch_code not in (None, 0):
                raise RuntimeError(
                    f"ObstaclesAvoidClient.SwitchSet(True) failed with code {switch_code}"
                )
            time.sleep(0.20)
        code = self._client.UseRemoteCommandFromApi(True)
        if code != 0:
            self._velocity_control_enabled = False
            raise RuntimeError(f"UseRemoteCommandFromApi(True) failed with code {code}")
        self._velocity_control_enabled = True

    def get_telemetry(self) -> Optional[dict[str, Any]]:
        with self._pose_lock:
            telemetry = copy.deepcopy(self._telemetry)
        if telemetry is None:
            return None
        with self._bms_lock:
            battery = copy.deepcopy(self._bms)
        if battery is not None:
            telemetry["battery"] = battery
        return telemetry

    def set_lidar_enabled(self, enabled: bool) -> None:
        self._lidar_command.data = "ON" if enabled else "OFF"
        self._lidar_publisher.Write(self._lidar_command)
        print(f"published rt/utlidar/switch={self._lidar_command.data}")

    def set_posture(self, action: str) -> None:
        # Stop any residual velocity before asking the sport service to change
        # posture.  StandUp can reset the obstacle-avoidance API ownership, so
        # explicitly reacquire it after the transition completes.
        with self._motion_lock:
            self.move(0.0, 0.0, 0.0)
            if self._client is not None:
                # Release the velocity command lease while the sport service
                # owns the posture transition.
                code = self._client.UseRemoteCommandFromApi(False)
                if code not in (None, 0):
                    raise RuntimeError(
                        f"UseRemoteCommandFromApi(False) failed with code {code}"
                    )
                self._velocity_control_enabled = False
            code = (
                self._sport_client.StandDown()
                if action == "stand_down"
                else self._sport_client.StandUp()
            )
            if code not in (None, 0):
                raise RuntimeError(f"SportClient.{action} failed with code {code}")
            if action == "stand_up":
                # StandUp reaches a static upright pose; BalanceStand is the
                # SDK transition into dynamic, walkable balance mode. Without
                # it the robot can look upright while remaining stand-locked.
                time.sleep(1.0)
                balance_error: Optional[int] = None
                for _ in range(8):
                    balance_code = self._sport_client.BalanceStand()
                    if balance_code in (None, 0):
                        balance_error = None
                        break
                    balance_error = int(balance_code)
                    time.sleep(0.25)
                if balance_error is not None:
                    print(
                        "BalanceStand did not return success; continuing with "
                        f"API lease recovery (code {balance_error})"
                    )
                if self._client is not None and self._motion_authorized:
                    # Wait for the posture transition and reacquire the lease.
                    # The move() fallback remains active if the SDK needs longer.
                    last_error: Optional[Exception] = None
                    for _ in range(24):
                        time.sleep(0.25)
                        try:
                            self.enable_velocity_control()
                            last_error = None
                            break
                        except RuntimeError as exc:
                            last_error = exc
                    if last_error is not None:
                        print(f"motion lease not ready after stand_up: {last_error}")
        print(f"applied posture={action}")

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self.move(0.0, 0.0, 0.0)
        finally:
            try:
                self._client.UseRemoteCommandFromApi(False)
                self._velocity_control_enabled = False
            except Exception as exc:
                print(f"warning: failed to release remote command mode: {exc}")


class DryRunDriver:
    """Small kinematic stand-in used for protocol tests without robot motion."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pose = Pose2D(0.0, 0.0, 0.0, time.monotonic())
        self._velocity = (0.0, 0.0, 0.0)
        self._updated = time.monotonic()

    def _integrate(self) -> None:
        now = time.monotonic()
        dt = min(max(now - self._updated, 0.0), 0.25)
        vx, vy, wz = self._velocity
        yaw = self._pose.yaw
        self._pose.x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        self._pose.y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        self._pose.yaw = wrap_angle(yaw + wz * dt)
        self._pose.stamp = now
        self._updated = now

    def get_pose(self) -> Pose2D:
        with self._lock:
            self._integrate()
            return Pose2D(**vars(self._pose))

    def move(self, vx: float, vy: float, wz: float) -> None:
        with self._lock:
            self._integrate()
            self._velocity = (vx, vy, wz)

    def get_telemetry(self) -> dict[str, Any]:
        pose = self.get_pose()
        with self._lock:
            velocity = self._velocity
        battery = {
            "received_at": time.time(),
            "simulated": True,
            "soc": 100,
            "bms_current": 0,
            "cycle": 0,
            "cell_vol": [0] * 15,
            "bq_ntc": [0, 0],
            "mcu_ntc": [0, 0],
            "voltage": 53.0,
            "current": 0.0,
            "power": 0.0,
            "temperature_ntc1": 0,
            "temperature_ntc2": 0,
            "fan_frequency": [0, 0, 0, 0],
        }
        return {
            "received_at": time.time(),
            "simulated": True,
            "position": [pose.x, pose.y, 0.0],
            "velocity": [velocity[0], velocity[1], 0.0],
            "yaw_speed": velocity[2],
            "imu": {"rpy": [0.0, 0.0, pose.yaw]},
            "battery": battery,
        }

    def set_lidar_enabled(self, enabled: bool) -> None:
        print(f"dry-run rt/utlidar/switch={'ON' if enabled else 'OFF'}")

    def set_posture(self, action: str) -> None:
        print(f"dry-run posture={action}")

    def close(self) -> None:
        self.move(0.0, 0.0, 0.0)


class MotionController:
    def __init__(
        self,
        args: argparse.Namespace,
        driver: Any,
        event_callback: Callable[[dict[str, Any]], None],
    ) -> None:
        self.args = args
        self.driver = driver
        self.event_callback = event_callback
        self.lock = threading.Lock()
        self.dispatch_lock = threading.RLock()
        self.motion_enabled = bool(getattr(args, "enable_motion", False)) or getattr(args, "state_source", "simulated") == "simulated"
        self.motion_enabled_at = 0.0
        self.stop_event = threading.Event()
        self.velocity = VelocityTarget()
        self.primitive: Optional[ActivePrimitive] = None
        self.active_mode = "idle"
        self.motion_suspended = False

    def submit(self, command: ControlCommand) -> bool:
        received_at = time.monotonic()
        # Consume and reject commands during SDK enable/disable rather than
        # building a socket backlog that could replay after the gate opens.
        if not self.dispatch_lock.acquire(blocking=False):
            return False
        try:
            if not self.motion_enabled or received_at < self.motion_enabled_at:
                return False
            return self._submit(command)
        finally:
            self.dispatch_lock.release()

    def _submit(self, command: ControlCommand) -> bool:
        cancelled: Optional[ActivePrimitive] = None
        with self.lock:
            self.motion_suspended = False
        if command.control_mode == "continuous":
            with self.lock:
                cancelled = self.primitive
                self.primitive = None
                self.velocity = VelocityTarget(
                    vx=command.vx,
                    vy=command.vy,
                    wz=command.wz,
                    deadline=time.monotonic() + command.ttl_ms / 1000.0,
                    seq=command.seq,
                )
                self.active_mode = "continuous"
        else:
            assert command.action is not None
            pose = self.driver.get_pose()
            if pose is None:
                raise RuntimeError("robot pose is unavailable")
            with self.lock:
                # A discrete key denotes one atomic step. Do not restart or
                # queue another step while one is still active; this prevents
                # keyboard auto-repeat/backlog from turning one press into
                # many consecutive steps. STOP always remains immediate.
                if (
                    command.action != DiscreteAction.STOP
                    and self.primitive is not None
                ):
                    print(
                        f"ignored discrete action {command.action.name}: "
                        f"{self.primitive.action.name} is still active"
                    )
                    return False
                cancelled = self.primitive
                self.velocity = VelocityTarget()
                if command.action == DiscreteAction.STOP:
                    self.primitive = None
                    self.active_mode = "idle"
                else:
                    timeout = (
                        self.args.forward_timeout
                        if command.action
                        in {
                            DiscreteAction.MOVE_FORWARD,
                            DiscreteAction.MOVE_BACKWARD,
                        }
                        else self.args.lateral_timeout
                        if command.action
                        in {DiscreteAction.MOVE_LEFT, DiscreteAction.MOVE_RIGHT}
                        else self.args.turn_timeout
                    )
                    started_at = time.monotonic()
                    self.primitive = ActivePrimitive(
                        action=command.action,
                        seq=command.seq,
                        start_pose=pose,
                        started_at=started_at,
                        deadline=started_at
                        + min(timeout, command.ttl_ms / 1000.0),
                    )
                    self.active_mode = "discrete"

        if cancelled is not None:
            self._emit_primitive(cancelled, "cancelled", "replaced_by_new_command")
        if command.control_mode == "discrete" and command.action == DiscreteAction.STOP:
            self.driver.move(0.0, 0.0, 0.0)
            self.event_callback(
                {
                    "v": 1,
                    "type": "status",
                    "seq": command.seq,
                    "control_mode": "discrete",
                    "action": DiscreteAction.STOP.name,
                    "state": "completed",
                }
            )
        return True

    def _emit_primitive(self, primitive: ActivePrimitive, state: str, detail: str = "") -> None:
        payload = {
            "v": 1,
            "type": "status",
            "seq": primitive.seq,
            "control_mode": "discrete",
            "action": primitive.action.name,
            "state": state,
        }
        if detail:
            payload["detail"] = detail
        self.event_callback(payload)

    def force_stop(self, reason: str) -> None:
        with self.lock:
            primitive = self.primitive
            self.primitive = None
            self.velocity = VelocityTarget()
            self.active_mode = "idle"
            self.motion_suspended = False
        self.driver.move(0.0, 0.0, 0.0)
        if primitive is not None:
            self._emit_primitive(primitive, "cancelled", reason)

    def suspend_motion(self, reason: str) -> None:
        with self.lock:
            primitive = self.primitive
            self.primitive = None
            self.velocity = VelocityTarget()
            self.active_mode = "posture"
            self.motion_suspended = True
        self.driver.move(0.0, 0.0, 0.0)
        if primitive is not None:
            self._emit_primitive(primitive, "cancelled", reason)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "active_mode": self.active_mode,
                "primitive": self.primitive.action.name if self.primitive else None,
                "motion_suspended": self.motion_suspended,
                "motion_enabled": self.motion_enabled,
            }

    def _finish_primitive(
        self,
        primitive: ActivePrimitive,
        state: str,
        detail: str = "",
    ) -> Tuple[float, float, float]:
        with self.lock:
            if self.primitive is primitive:
                self.primitive = None
                self.active_mode = "idle"
        self._emit_primitive(primitive, state, detail)
        return (0.0, 0.0, 0.0)

    def _primitive_target(
        self,
        primitive: ActivePrimitive,
        now: float,
    ) -> Tuple[float, float, float]:
        if now > primitive.deadline:
            return self._finish_primitive(primitive, "failed", "primitive_timeout")

        # Rotation is open-loop by default. On this Go2, the reported yaw
        # underestimates the physical turn, so yaw-based completion stops q/e
        # around 10 degrees. A short bounded timed command avoids waiting for
        # stale/biased yaw feedback and cannot spin indefinitely.
        if (
            primitive.action in {DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT}
            and self.args.turn_control_mode == "open_loop"
        ):
            if now - primitive.started_at >= self.args.turn_duration:
                return self._finish_primitive(primitive, "completed")
            direction = (
                1.0 if primitive.action == DiscreteAction.TURN_LEFT else -1.0
            )
            return (0.0, 0.0, direction * self.args.primitive_max_wz)

        pose = self.driver.get_pose()
        if pose is None or now - pose.stamp > self.args.max_state_age:
            return self._finish_primitive(primitive, "failed", "stale_or_missing_pose")

        if primitive.action in {
            DiscreteAction.MOVE_FORWARD,
            DiscreteAction.MOVE_BACKWARD,
            DiscreteAction.MOVE_LEFT,
            DiscreteAction.MOVE_RIGHT,
        }:
            dx = pose.x - primitive.start_pose.x
            dy = pose.y - primitive.start_pose.y
            start_yaw = primitive.start_pose.yaw
            forward_progress = dx * math.cos(start_yaw) + dy * math.sin(start_yaw)
            lateral_progress = -dx * math.sin(start_yaw) + dy * math.cos(start_yaw)
            if primitive.action in {
                DiscreteAction.MOVE_FORWARD,
                DiscreteAction.MOVE_BACKWARD,
            }:
                direction = (
                    1.0
                    if primitive.action == DiscreteAction.MOVE_FORWARD
                    else -1.0
                )
                progress = direction * forward_progress
                distance = self.args.forward_distance
                kp = self.args.forward_kp
                min_speed = self.args.primitive_min_vx
                max_speed = self.args.primitive_max_vx
            else:
                direction = (
                    1.0
                    if primitive.action == DiscreteAction.MOVE_LEFT
                    else -1.0
                )
                progress = direction * lateral_progress
                distance = self.args.lateral_distance
                kp = self.args.lateral_kp
                min_speed = self.args.primitive_min_vy
                max_speed = self.args.primitive_max_vy
            remaining = distance - progress
            if remaining <= self.args.forward_tolerance:
                return self._finish_primitive(primitive, "completed")
            yaw_error = wrap_angle(primitive.start_pose.yaw - pose.yaw)
            wz = clamp(
                self.args.heading_kp * yaw_error,
                -self.args.primitive_max_wz,
                self.args.primitive_max_wz,
            )
            speed = direction * clamp(kp * remaining, min_speed, max_speed)
            if primitive.action in {
                DiscreteAction.MOVE_FORWARD,
                DiscreteAction.MOVE_BACKWARD,
            }:
                return (speed, 0.0, wz)
            return (0.0, speed, wz)

        direction = 1.0 if primitive.action == DiscreteAction.TURN_LEFT else -1.0
        progress = direction * wrap_angle(pose.yaw - primitive.start_pose.yaw)
        remaining = math.radians(self.args.turn_angle_deg) - progress
        if remaining <= math.radians(self.args.turn_tolerance_deg):
            return self._finish_primitive(primitive, "completed")
        wz = direction * clamp(
            self.args.turn_kp * remaining,
            self.args.primitive_min_wz,
            self.args.primitive_max_wz,
        )
        return (0.0, 0.0, wz)

    def target(self, now: float) -> Tuple[float, float, float]:
        with self.lock:
            primitive = self.primitive
            velocity = VelocityTarget(**vars(self.velocity))
        if primitive is not None:
            return self._primitive_target(primitive, now)
        if now <= velocity.deadline:
            return (velocity.vx, velocity.vy, velocity.wz)
        with self.lock:
            if self.active_mode == "continuous":
                self.active_mode = "idle"
        return (0.0, 0.0, 0.0)

    def run(self) -> None:
        last_logged: Optional[Tuple[float, float, float]] = None
        last_applied: Optional[Tuple[float, float, float]] = None
        try:
            while not self.stop_event.wait(self.args.control_period):
                with self.dispatch_lock:
                    if not self.motion_enabled or self.snapshot()["motion_suspended"]:
                        last_applied = None
                        continue
                    target = self.target(time.monotonic())
                    moving = any(abs(value) > 1e-6 for value in target)
                    was_moving = bool(
                        last_applied
                        and any(abs(value) > 1e-6 for value in last_applied)
                    )
                    # Keep target selection and SDK dispatch atomic with a
                    # mode switch, so an old target cannot follow disable.
                    if moving or was_moving or last_applied is None:
                        self.driver.move(*target)
                    last_applied = target
                rounded = tuple(round(value, 3) for value in target)
                if rounded != last_logged:
                    print(
                        f"applied vx={target[0]:.3f} vy={target[1]:.3f} "
                        f"wz={target[2]:.3f} mode={self.snapshot()['active_mode']}"
                    )
                    last_logged = rounded
        except Exception as exc:
            print(f"control loop failed: {exc}")
            self.stop_event.set()
        finally:
            self.driver.move(0.0, 0.0, 0.0)


class SpeakerWorker:
    """Run persistent local TTS and AudioHub calls outside the motion loop."""

    def __init__(
        self,
        args: argparse.Namespace,
        event_callback: Callable[[dict[str, Any]], None],
    ) -> None:
        self.args = args
        self.event_callback = event_callback
        self.jobs: queue.Queue[SpeechJob] = queue.Queue(maxsize=args.speaker_queue_size)
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.state = "disabled" if args.speaker_backend == "disabled" else "idle"
        self.active_seq: Optional[int] = None
        self.requested_synthesis_backend = getattr(
            args, "speech_primary_backend", "edge"
        )
        self.active_synthesis_backend = self.requested_synthesis_backend
        self.fallback_backend = getattr(args, "speech_fallback_backend", "disabled")
        self.backend_error = ""
        self.last_synthesis_backend: Optional[str] = None
        self.matcha_synthesizer: Any = None
        self.persistent_speaker: Any = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        if self.args.speaker_backend != "disabled":
            if self.args.speaker_backend == "go2":
                self._initialize_synthesis_backend()
            self.thread.start()

    def _initialize_synthesis_backend(self) -> None:
        if self.requested_synthesis_backend != "matcha":
            print("speech synthesis backend: edge (online)", flush=True)
            return
        from go2_voice_intercom import MatchaTtsSynthesizer

        self._set_state("loading", None)
        started = time.monotonic()
        try:
            self.matcha_synthesizer = MatchaTtsSynthesizer(
                model_dir=Path(self.args.matcha_model_dir),
                vocoder=Path(self.args.matcha_vocoder),
                num_threads=self.args.tts_threads,
            )
        except Exception as exc:
            self.backend_error = f"{type(exc).__name__}: {exc}"
            if self.fallback_backend != "edge":
                self._set_state("failed", None)
                raise RuntimeError(
                    f"cannot load Matcha and Edge fallback is disabled: {exc}"
                ) from exc
            self.active_synthesis_backend = "edge"
            print(
                f"Matcha preload failed; using Edge fallback: {self.backend_error}",
                flush=True,
            )
        else:
            self.active_synthesis_backend = "matcha"
            print(
                "Matcha TTS loaded in "
                f"{time.monotonic() - started:.2f}s "
                f"({self.args.tts_threads} threads); Edge fallback enabled="
                f"{self.fallback_backend == 'edge'}",
                flush=True,
            )
        finally:
            if self.state != "failed":
                self._set_state("idle", None)

    def submit(self, command: SpeechCommand) -> int:
        if self.args.speaker_backend == "disabled":
            raise RuntimeError("speaker support is disabled")
        job = SpeechJob(
            command=command,
            expires_at=time.monotonic() + command.ttl_ms / 1000.0,
        )
        try:
            self.jobs.put_nowait(job)
        except queue.Full as exc:
            raise RuntimeError("speaker queue is full") from exc
        return self.jobs.qsize()

    def snapshot(self) -> dict[str, Any]:
        with self.state_lock:
            return {
                "speaker_state": self.state,
                "speaker_active_seq": self.active_seq,
                "speaker_queue_depth": self.jobs.qsize(),
                "speaker_synthesis_backend": self.active_synthesis_backend,
                "speaker_fallback_backend": self.fallback_backend,
                "speaker_last_synthesis_backend": self.last_synthesis_backend,
                "speaker_backend_error": self.backend_error or None,
            }

    def _set_state(self, state: str, seq: Optional[int]) -> None:
        with self.state_lock:
            self.state = state
            self.active_seq = seq

    def _send_final_status(
        self,
        command: SpeechCommand,
        state: str,
        *,
        detail: str = "",
        elapsed_s: float = 0.0,
        result: Optional[dict[str, object]] = None,
    ) -> None:
        payload: dict[str, Any] = {
            "v": 1,
            "type": "status",
            "seq": command.seq,
            "command_type": "speech",
            "state": state,
            "text_chars": len(command.text),
            "elapsed_s": round(elapsed_s, 3),
        }
        if detail:
            payload["detail"] = detail
        if result:
            for key in (
                "synthesis_backend",
                "requested_synthesis_backend",
                "fallback_from",
                "fallback_reason",
                "total_synthesis_seconds",
                "audio_cache_hit",
                "preloaded",
                "connection_reused",
                "audio_duration_seconds",
            ):
                if key in result:
                    payload[key] = result[key]
        self.event_callback(payload)

    def _speak(
        self,
        command: SpeechCommand,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> dict[str, object]:
        if self.args.speaker_backend == "simulated":
            time.sleep(0.02)
            return {"synthesis_backend": "simulated"}
        if self.persistent_speaker is not None and event_loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                self.persistent_speaker.speak(
                    command.text,
                    voice=command.voice,
                    volume=command.volume,
                    retain=command.text in PRELOADED_INTERACTION_PROMPTS,
                ),
                event_loop,
            )
            return future.result(timeout=max(30.0, command.ttl_ms / 1000.0))
        from go2_voice_intercom import speak_text_once

        result = asyncio.run(
            speak_text_once(
                command.text,
                voice=command.voice,
                robot_ip=self.args.robot_controller_ip,
                volume=command.volume,
                synthesis_backend=self.active_synthesis_backend,
                fallback_backend=(
                    self.fallback_backend
                    if self.active_synthesis_backend == "matcha"
                    else "disabled"
                ),
                matcha_synthesizer=self.matcha_synthesizer,
            )
        )
        with self.state_lock:
            backend = result.get("synthesis_backend")
            self.last_synthesis_backend = str(backend) if backend else None
        return result

    def run(self) -> None:
        event_loop: Optional[asyncio.AbstractEventLoop] = None
        event_loop_thread: Optional[threading.Thread] = None
        try:
            if self.args.speaker_backend == "go2":
                from go2_voice_intercom import PersistentSpeechSession

                event_loop = asyncio.new_event_loop()
                def run_event_loop() -> None:
                    assert event_loop is not None
                    asyncio.set_event_loop(event_loop)
                    event_loop.run_forever()

                event_loop_thread = threading.Thread(
                    target=run_event_loop,
                    name="go2-audiohub-event-loop",
                    daemon=True,
                )
                event_loop_thread.start()
                self.persistent_speaker = PersistentSpeechSession(
                    robot_ip=self.args.robot_controller_ip,
                    synthesis_backend=self.active_synthesis_backend,
                    fallback_backend=(
                        self.fallback_backend
                        if self.active_synthesis_backend == "matcha"
                        else "disabled"
                    ),
                    matcha_synthesizer=self.matcha_synthesizer,
                )
                self._set_state("preloading", None)
                try:
                    preload = asyncio.run_coroutine_threadsafe(
                        self.persistent_speaker.preload(
                            list(PRELOADED_INTERACTION_PROMPTS),
                            voice="zh-CN-XiaoxiaoNeural",
                        ),
                        event_loop,
                    ).result(timeout=120.0)
                    print(
                        "speech AudioHub persistent connection ready; "
                        f"preloaded={len(preload)} cache_hits="
                        f"{sum(bool(item.get('audio_cache_hit')) for item in preload)}",
                        flush=True,
                    )
                except Exception as exc:
                    # Keep the worker alive: the first real request retries the
                    # connection and can still synthesize/upload on demand.
                    print(f"speech preload failed; will retry on demand: {exc}", flush=True)
                finally:
                    self._set_state("idle", None)

            while not self.stop_event.is_set():
                try:
                    job = self.jobs.get(timeout=0.2)
                except queue.Empty:
                    continue
                command = job.command
                started_at = time.monotonic()
                if started_at >= job.expires_at:
                    self._send_final_status(command, "expired", detail="speech_ttl_expired")
                    self.jobs.task_done()
                    continue
                self._set_state("speaking", command.seq)
                print(f"speech seq={command.seq} started ({len(command.text)} chars)")
                try:
                    result = self._speak(command, event_loop)
                except Exception as exc:
                    elapsed = time.monotonic() - started_at
                    print(f"speech seq={command.seq} failed: {exc}")
                    self._send_final_status(
                        command,
                        "failed",
                        detail=str(exc),
                        elapsed_s=elapsed,
                    )
                else:
                    elapsed = time.monotonic() - started_at
                    print(
                        f"speech seq={command.seq} completed in {elapsed:.2f}s "
                        f"via {result.get('synthesis_backend', 'cached')} "
                        f"cache_hit={bool(result.get('audio_cache_hit'))}"
                    )
                    self._send_final_status(
                        command,
                        "completed",
                        elapsed_s=elapsed,
                        result=result,
                    )
                finally:
                    self._set_state("idle", None)
                    self.jobs.task_done()
        finally:
            if self.persistent_speaker is not None and event_loop is not None:
                with contextlib.suppress(Exception):
                    asyncio.run_coroutine_threadsafe(
                        self.persistent_speaker.close(), event_loop
                    ).result(timeout=5.0)
            if event_loop is not None:
                event_loop.call_soon_threadsafe(event_loop.stop)
            if event_loop_thread is not None:
                event_loop_thread.join(timeout=2.0)
            if event_loop is not None and not event_loop.is_running():
                event_loop.close()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


class UnifiedBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.driver = (
            Go2Driver(
                args.interface,
                args.state_timeout,
                args.enable_motion,
                args.motion_enable_attempts,
                args.motion_enable_retry_s,
                args.min_locomotion_height_m,
            )
            if args.state_source == "go2"
            else DryRunDriver()
        )
        self.ws_lock = threading.Lock()
        self.websocket: Any = None
        self.lidar_lock = threading.Lock()
        self.lidar_enabled: Optional[bool] = {
            "unknown": None,
            "on": True,
            "off": False,
        }[args.lidar_initial_state]
        self.controller = MotionController(args, self.driver, self.send_event)
        if args.speaker_backend == "auto":
            args.speaker_backend = "go2" if args.state_source == "go2" else "simulated"
        self.speaker = SpeakerWorker(args, self.send_event)

    def switch_motion(self, request: dict[str, Any]) -> dict[str, Any]:
        controller = self.controller
        with controller.dispatch_lock:
            if controller.stop_event.is_set():
                raise RuntimeError("control bridge is stopping")
            if request:
                enabled = request.get("enabled")
                if type(enabled) is not bool:
                    raise ValueError("enabled must be a boolean")
                limits = {key: float(request.get(key, getattr(self.args, key)))
                          for key in ("max_vx", "max_wz")}
                if any(not math.isfinite(value) or value <= 0 for value in limits.values()):
                    raise ValueError("motion limits must be finite and positive")
                unchanged = enabled == controller.motion_enabled and all(
                    value == getattr(self.args, key) for key, value in limits.items()
                )
                if not unchanged:
                    controller.motion_enabled = False
                    self.args.enable_motion = False
                    controller.force_stop("motion_mode_changed")
                    self.driver.set_motion_enabled(enabled)
                    for key, value in limits.items():
                        setattr(self.args, key, value)
                    controller.motion_enabled_at = time.monotonic()
                    controller.motion_enabled = enabled
                    self.args.enable_motion = enabled
            return {"mode": "enable_motion" if controller.motion_enabled else "speech_only",
                    "max_vx": self.args.max_vx, "max_wz": self.args.max_wz}

    def apply_lidar_action(self, action: str) -> bool:
        with self.lidar_lock:
            if action == "toggle":
                # A bridge restart cannot observe the current switch state. Choosing OFF
                # for the first toggle is deterministic and immediately stops rotation.
                enabled = False if self.lidar_enabled is None else not self.lidar_enabled
            else:
                enabled = action == "on"
            self.driver.set_lidar_enabled(enabled)
            self.lidar_enabled = enabled
            return enabled

    def lidar_snapshot(self) -> Optional[bool]:
        with self.lidar_lock:
            return self.lidar_enabled

    def telemetry_loop(self) -> None:
        telemetry_seq = 0
        while not self.controller.stop_event.wait(self.args.telemetry_period):
            telemetry = self.driver.get_telemetry()
            if telemetry is None:
                continue
            telemetry_seq += 1
            self.send_event(
                {
                    "v": 1,
                    "type": "telemetry",
                    "telemetry_seq": telemetry_seq,
                    "sport_mode_state": telemetry,
                    "lidar_enabled": self.lidar_snapshot(),
                    **self.speaker.snapshot(),
                    **self.controller.snapshot(),
                }
            )

    def send_event(self, payload: dict[str, Any]) -> None:
        with self.ws_lock:
            if self.websocket is None:
                return
            try:
                self.websocket.send(json.dumps(payload, separators=(",", ":")))
            except websocket.WebSocketException:
                pass

    def run(self) -> None:
        if self.args.enable_motion:
            mode = "VELOCITY + AUXILIARY CONTROL"
        elif self.args.state_source == "go2":
            mode = "TELEMETRY + AUXILIARY CONTROL (VELOCITY DISABLED)"
        else:
            mode = "SIMULATED DRY RUN"
        print(f"Connecting to {self.args.url} ({mode})")
        control_thread = threading.Thread(target=self.controller.run, daemon=True)
        telemetry_thread = threading.Thread(target=self.telemetry_loop, daemon=True)
        control_thread.start()
        telemetry_thread.start()
        self.speaker.start()
        try:
            while not self.controller.stop_event.is_set():
                ws = None
                last_seq = -1
                try:
                    ws = websocket.create_connection(self.args.url, timeout=5)
                    ws.settimeout(None)
                    with self.ws_lock:
                        self.websocket = ws
                    self.send_event(
                        {
                            "v": 1,
                            "type": "hello",
                            "role": "go2_bridge",
                            "mode": mode,
                            "supported_control_modes": ["discrete", "continuous"],
                            "supported_auxiliary_commands": [
                                "lidar",
                                "posture",
                                *(
                                    []
                                    if self.args.speaker_backend == "disabled"
                                    else ["speak"]
                                ),
                            ],
                            "lidar_enabled": self.lidar_snapshot(),
                            **self.speaker.snapshot(),
                        }
                    )
                    print("WebSocket connected")
                    while not self.controller.stop_event.is_set():
                        raw_message = ws.recv()
                        if not raw_message:
                            raise ConnectionError("WebSocket closed")
                        response: dict[str, Any]
                        try:
                            payload = json.loads(raw_message)
                            if payload.get("type") == "speak":
                                speech_command = parse_speech_message(
                                    payload,
                                    last_seq=last_seq,
                                    max_text_chars=self.args.max_speech_text_chars,
                                    max_ttl_ms=self.args.max_speech_ttl_ms,
                                )
                                queue_depth = self.speaker.submit(speech_command)
                                last_seq = speech_command.seq
                                response = {
                                    "v": 1,
                                    "type": "ack",
                                    "seq": speech_command.seq,
                                    "accepted": True,
                                    "command_type": "speech",
                                    "state": "queued",
                                    "queue_depth": queue_depth,
                                    **self.speaker.snapshot(),
                                }
                            elif payload.get("type") == "lidar":
                                lidar_command = parse_lidar_message(
                                    payload,
                                    last_seq=last_seq,
                                )
                                lidar_enabled = self.apply_lidar_action(
                                    lidar_command.action
                                )
                                last_seq = lidar_command.seq
                                response = {
                                    "v": 1,
                                    "type": "ack",
                                    "seq": lidar_command.seq,
                                    "accepted": True,
                                    "command_type": "lidar",
                                    "lidar_enabled": lidar_enabled,
                                    **self.controller.snapshot(),
                                }
                            elif payload.get("type") == "posture":
                                posture_command = parse_posture_message(
                                    payload,
                                    last_seq=last_seq,
                                )
                                self.controller.suspend_motion("posture_command")
                                self.driver.set_posture(posture_command.action)
                                last_seq = posture_command.seq
                                response = {
                                    "v": 1,
                                    "type": "ack",
                                    "seq": posture_command.seq,
                                    "accepted": True,
                                    "command_type": "posture",
                                    "posture_action": posture_command.action,
                                    **self.controller.snapshot(),
                                }
                            else:
                                command = parse_control_message(
                                    payload,
                                    last_seq=last_seq,
                                    max_discrete_ttl_ms=self.args.max_discrete_ttl_ms,
                                    max_continuous_ttl_ms=self.args.max_continuous_ttl_ms,
                                    max_vx=self.args.max_vx,
                                    max_vy=self.args.max_vy,
                                    max_wz=self.args.max_wz,
                                )
                                applied = self.controller.submit(command)
                                last_seq = command.seq
                                response = {
                                    "v": 1,
                                    "type": "ack",
                                    "seq": command.seq,
                                    "accepted": True,
                                    "applied": applied,
                                    "control_mode": command.control_mode,
                                    **self.controller.snapshot(),
                                }
                        except Exception as exc:
                            self.controller.force_stop("invalid_command")
                            response = {
                                "v": 1,
                                "type": "ack",
                                "accepted": False,
                                "error": str(exc),
                            }
                        self.send_event(response)
                except (ConnectionError, OSError, websocket.WebSocketException) as exc:
                    print(f"WebSocket unavailable: {exc}")
                finally:
                    self.controller.force_stop("websocket_disconnected")
                    with self.ws_lock:
                        self.websocket = None
                    if ws is not None:
                        ws.close()

                if self.args.once:
                    break
                if not self.controller.stop_event.wait(self.args.reconnect_delay):
                    print("Retrying WebSocket connection")
        finally:
            with self.controller.dispatch_lock:
                self.controller.stop_event.set()
                self.controller.force_stop("bridge_shutdown")
            control_thread.join(timeout=2.0)
            telemetry_thread.join(timeout=2.0)
            self.speaker.close()
            self.driver.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Go2 bridge for Habitat-style discrete actions and ROS velocities."
    )
    parser.add_argument("--url", default="ws://127.0.0.1:12333")
    parser.add_argument("--interface", default="eth0")
    parser.add_argument(
        "--state-source",
        choices=("simulated", "go2"),
        default="simulated",
    )
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument(
        "--lidar-initial-state",
        choices=("unknown", "on", "off"),
        default="unknown",
        help="assumed LiDAR state; with unknown, the first toggle sends OFF",
    )
    parser.add_argument("--state-timeout", type=float, default=5.0)
    parser.add_argument("--motion-enable-attempts", type=int, default=5)
    parser.add_argument("--motion-enable-retry-s", type=float, default=0.5)
    parser.add_argument(
        "--min-locomotion-height-m",
        type=float,
        default=0.15,
        help=(
            "reject non-zero velocity while reported body position z is below "
            "this threshold; use 0 to disable"
        ),
    )
    parser.add_argument(
        "--ready-file",
        default="",
        help="write this PID file only after hardware and motion initialization succeeds",
    )
    parser.add_argument("--max-state-age", type=float, default=0.5)
    parser.add_argument("--control-period", type=float, default=0.05)
    parser.add_argument("--telemetry-period", type=float, default=0.20)
    parser.add_argument("--max-discrete-ttl-ms", type=int, default=7000)
    parser.add_argument("--max-continuous-ttl-ms", type=int, default=500)
    parser.add_argument("--max-vx", type=float, default=0.25)
    parser.add_argument("--max-vy", type=float, default=0.0)
    parser.add_argument("--max-wz", type=float, default=0.40)
    parser.add_argument(
        "--speaker-backend",
        choices=("auto", "go2", "simulated", "disabled"),
        default="auto",
        help="auto selects Go2 AudioHub for real state and simulation otherwise",
    )
    parser.add_argument("--speaker-queue-size", type=int, default=3)
    parser.add_argument("--max-speech-text-chars", type=int, default=300)
    parser.add_argument("--max-speech-ttl-ms", type=int, default=120000)
    parser.add_argument("--robot-controller-ip", default="192.168.123.161")
    default_model_root = Path(__file__).resolve().parent / "vendor" / "local_tts" / "models"
    parser.add_argument(
        "--speech-primary-backend",
        choices=("matcha", "edge"),
        default="matcha",
        help="primary speech synthesizer; Matcha is local and Edge is online",
    )
    parser.add_argument(
        "--speech-fallback-backend",
        choices=("edge", "disabled"),
        default="edge",
        help="fallback used when Matcha cannot load or synthesize",
    )
    parser.add_argument(
        "--matcha-model-dir",
        default=str(default_model_root / "matcha-icefall-zh-baker"),
    )
    parser.add_argument(
        "--matcha-vocoder",
        default=str(default_model_root / "vocos-22khz-univ.onnx"),
    )
    parser.add_argument("--tts-threads", type=int, default=4)
    # Discrete forward/backward target remains the Habitat-style 0.25 m step.
    # The lateral primitive is intentionally shorter for indoor testing.
    parser.add_argument("--forward-distance", type=float, default=0.25)
    parser.add_argument("--lateral-distance", type=float, default=0.10)
    # Open-loop timed rotation is the default because the current yaw feedback
    # underestimates the physical turn. Double the previous target to 30 deg.
    parser.add_argument("--turn-angle-deg", type=float, default=30.0)
    parser.add_argument(
        "--turn-control-mode",
        choices=("open_loop", "closed_loop"),
        default="open_loop",
        help="rotation completion mode; open_loop uses a bounded timed command",
    )
    parser.add_argument(
        "--turn-duration",
        type=float,
        default=None,
        help="optional open-loop q/e duration; default derives from angle and speed",
    )
    parser.add_argument("--forward-tolerance", type=float, default=0.02)
    parser.add_argument("--turn-tolerance-deg", type=float, default=1.5)
    parser.add_argument("--primitive-min-vx", type=float, default=0.06)
    # Discrete primitives have their own speed limits.  Keep these separate
    # from --max-vx/--max-vy, which limit continuous commands.
    parser.add_argument("--primitive-max-vx", type=float, default=1.0)
    parser.add_argument("--primitive-min-vy", type=float, default=0.06)
    parser.add_argument("--primitive-max-vy", type=float, default=0.5)
    parser.add_argument("--primitive-min-wz", type=float, default=0.12)
    parser.add_argument("--primitive-max-wz", type=float, default=1.5)
    # With the 0.25 m / 0.10 m targets, these gains make the initial command
    # reach the requested 1.0 m/s / 0.5 m/s caps before tapering near target.
    # Keep the 1.0 m/s cap, but use a gentler nominal speed for a 0.25 m
    # primitive so odometry/control latency does not carry the robot to 0.30 m.
    parser.add_argument("--forward-kp", type=float, default=2.5)
    parser.add_argument("--lateral-kp", type=float, default=5.0)
    parser.add_argument("--heading-kp", type=float, default=2.0)
    parser.add_argument("--turn-kp", type=float, default=6.0)
    parser.add_argument("--forward-timeout", type=float, default=6.0)
    parser.add_argument("--lateral-timeout", type=float, default=5.0)
    parser.add_argument(
        "--turn-timeout",
        type=float,
        default=1.0,
        help="hard safety timeout for one q/e primitive (seconds)",
    )
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.primitive_min_vx <= 0 or args.primitive_max_vx < args.primitive_min_vx:
        parser.error("primitive vx limits must satisfy 0 < min <= max")
    if args.primitive_min_vy <= 0 or args.primitive_max_vy < args.primitive_min_vy:
        parser.error("primitive vy limits must satisfy 0 < min <= max")
    if args.primitive_min_wz <= 0 or args.primitive_max_wz < args.primitive_min_wz:
        parser.error("primitive wz limits must satisfy 0 < min <= max")
    if args.turn_duration is None:
        # Add a short allowance for gait startup/command latency. This keeps
        # turn-angle-deg meaningful even though completion is open-loop.
        args.turn_duration = (
            math.radians(args.turn_angle_deg) / args.primitive_max_wz + 0.25
        )
    if args.turn_duration <= 0 or args.turn_duration > args.turn_timeout:
        parser.error("turn duration must satisfy 0 < duration <= timeout")
    if args.telemetry_period < 0.05:
        parser.error("--telemetry-period must be at least 0.05 seconds")
    if args.enable_motion and args.state_source != "go2":
        parser.error("--enable-motion requires --state-source go2")
    if args.motion_enable_attempts <= 0 or args.motion_enable_retry_s < 0:
        parser.error("motion enable retries require attempts > 0 and delay >= 0")
    if not 1 <= args.speaker_queue_size <= 20:
        parser.error("--speaker-queue-size must be between 1 and 20")
    if not 1 <= args.tts_threads <= 8:
        parser.error("--tts-threads must be between 1 and 8")
    if not 1 <= args.max_speech_text_chars <= 2000:
        parser.error("--max-speech-text-chars must be between 1 and 2000")
    if not 1000 <= args.max_speech_ttl_ms <= 600000:
        parser.error("--max-speech-ttl-ms must be between 1000 and 600000")
    return args


def main() -> None:
    args = parse_args()
    bridge: Optional[UnifiedBridge] = None
    ready_path = Path(args.ready_file).expanduser() if args.ready_file else None
    ready_pid = str(os.getpid())
    motion_switch = None
    try:
        bridge = UnifiedBridge(args)
        if ready_path is not None:
            from motion_switch import MotionSwitchServer
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            motion_switch = MotionSwitchServer(ready_path, bridge.switch_motion)
            temporary = ready_path.with_name(f".{ready_path.name}.{ready_pid}.tmp")
            temporary.write_text(ready_pid + "\n", encoding="utf-8")
            temporary.replace(ready_path)
        bridge.run()
    except KeyboardInterrupt:
        print("Interrupted; stopping Go2 bridge")
        if bridge is not None:
            bridge.controller.stop_event.set()
    finally:
        if motion_switch is not None:
            motion_switch.close()
        if ready_path is not None:
            try:
                if ready_path.read_text(encoding="utf-8").strip() == ready_pid:
                    ready_path.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
