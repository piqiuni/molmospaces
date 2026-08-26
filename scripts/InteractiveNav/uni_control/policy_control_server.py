#!/usr/bin/env python3
"""Policy-side WebSocket server for discrete actions or continuous ROS cmd_vel."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import termios
import tty
from typing import Any, Optional

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from control_protocol import (
    DiscreteAction,
    make_continuous_message,
    make_discrete_message,
    make_lidar_message,
    make_posture_message,
    make_speech_message,
    parse_action,
)


class PolicyControlServer:
    def __init__(
        self,
        ack_timeout: float,
        telemetry_print_period: float,
        speech_timeout: float = 180.0,
    ) -> None:
        self.ack_timeout = ack_timeout
        self.telemetry_print_period = telemetry_print_period
        self.speech_timeout = speech_timeout
        self.bridge: Optional[ServerConnection] = None
        self.bridge_peer = ""
        self.connected = asyncio.Event()
        self.seq = 0
        self.send_lock = asyncio.Lock()
        self.ack_waiters: dict[int, asyncio.Future] = {}
        self.status_waiters: dict[int, asyncio.Future] = {}
        self.latest_telemetry: Optional[dict[str, Any]] = None
        self.telemetry_event = asyncio.Event()
        self.last_telemetry_print = 0.0
        self.text_entry_active = False

    async def handler(self, websocket: ServerConnection) -> None:
        if self.bridge is not None:
            await websocket.close(code=1013, reason="another Go2 bridge is active")
            return
        self.bridge = websocket
        self.bridge_peer = str(websocket.remote_address)
        self.connected.set()
        print(f"Go2 bridge connected: {self.bridge_peer}")
        try:
            async for raw_message in websocket:
                try:
                    message = json.loads(raw_message)
                except (TypeError, json.JSONDecodeError):
                    print(f"invalid bridge message: {raw_message!r}")
                    continue
                self._handle_bridge_message(message)
        except ConnectionClosed:
            pass
        finally:
            print(f"Go2 bridge disconnected: {self.bridge_peer}")
            self.bridge = None
            self.bridge_peer = ""
            self.connected.clear()
            error = ConnectionError("Go2 bridge disconnected")
            for waiter in list(self.ack_waiters.values()) + list(
                self.status_waiters.values()
            ):
                if not waiter.done():
                    waiter.set_exception(error)
            self.ack_waiters.clear()
            self.status_waiters.clear()

    def _handle_bridge_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "hello":
            print(json.dumps(message, ensure_ascii=False, separators=(",", ":")))
            return
        if message_type == "ack":
            seq = message.get("seq")
            waiter = self.ack_waiters.get(int(seq)) if seq is not None else None
            if waiter is not None and not waiter.done():
                waiter.set_result(message)
            if not message.get("accepted", False):
                print(f"command rejected: {message}")
            return
        if message_type == "status":
            seq = message.get("seq")
            waiter = self.status_waiters.get(int(seq)) if seq is not None else None
            if waiter is not None and not waiter.done():
                waiter.set_result(message)
            print(json.dumps(message, ensure_ascii=False, separators=(",", ":")))
            return
        if message_type == "telemetry":
            self.latest_telemetry = message
            self.telemetry_event.set()
            if self.text_entry_active:
                return
            now = asyncio.get_running_loop().time()
            if now - self.last_telemetry_print >= self.telemetry_print_period:
                state = message.get("sport_mode_state", {})
                print(
                    "telemetry "
                    f"position={state.get('position')} "
                    f"velocity={state.get('velocity')} "
                    f"yaw_speed={state.get('yaw_speed')} "
                    f"mode={message.get('active_mode')}"
                )
                self.last_telemetry_print = now
            return
        print(json.dumps(message, ensure_ascii=False, separators=(",", ":")))

    async def _send(self, payload: dict[str, Any]) -> None:
        if self.bridge is None:
            raise ConnectionError("no Go2 bridge connected")
        async with self.send_lock:
            await self.bridge.send(json.dumps(payload, separators=(",", ":")))

    async def publish_discrete(
        self,
        action: Any,
        ttl_ms: int,
        *,
        wait_for_completion: bool = False,
    ) -> dict[str, Any]:
        parsed_action = parse_action(action)
        self.seq += 1
        seq = self.seq
        loop = asyncio.get_running_loop()
        ack_waiter = loop.create_future()
        self.ack_waiters[seq] = ack_waiter
        if wait_for_completion:
            self.status_waiters[seq] = loop.create_future()
        try:
            await self._send(make_discrete_message(seq, parsed_action, ttl_ms))
            ack = await asyncio.wait_for(ack_waiter, timeout=self.ack_timeout)
            if not ack.get("accepted", False):
                raise RuntimeError(ack.get("error", "discrete command rejected"))
            if wait_for_completion:
                status = await asyncio.wait_for(
                    self.status_waiters[seq], timeout=self.ack_timeout + 10.0
                )
                return {"ack": ack, "status": status}
            return {"ack": ack}
        finally:
            self.ack_waiters.pop(seq, None)
            self.status_waiters.pop(seq, None)

    async def publish_continuous(
        self,
        vx: float,
        vy: float,
        wz: float,
        ttl_ms: int,
    ) -> int:
        self.seq += 1
        await self._send(make_continuous_message(self.seq, vx, vy, wz, ttl_ms))
        return self.seq

    async def publish_lidar(self, action: str = "toggle") -> dict[str, Any]:
        self.seq += 1
        seq = self.seq
        loop = asyncio.get_running_loop()
        ack_waiter = loop.create_future()
        self.ack_waiters[seq] = ack_waiter
        try:
            await self._send(make_lidar_message(seq, action))
            ack = await asyncio.wait_for(ack_waiter, timeout=self.ack_timeout)
            if not ack.get("accepted", False):
                raise RuntimeError(ack.get("error", "lidar command rejected"))
            return ack
        finally:
            self.ack_waiters.pop(seq, None)

    async def publish_posture(self, action: str) -> dict[str, Any]:
        self.seq += 1
        seq = self.seq
        loop = asyncio.get_running_loop()
        ack_waiter = loop.create_future()
        self.ack_waiters[seq] = ack_waiter
        try:
            await self._send(make_posture_message(seq, action))
            ack = await asyncio.wait_for(ack_waiter, timeout=self.ack_timeout)
            if not ack.get("accepted", False):
                raise RuntimeError(ack.get("error", "posture command rejected"))
            return ack
        finally:
            self.ack_waiters.pop(seq, None)

    async def publish_speech(
        self,
        text: str,
        *,
        voice: str = "zh-CN-XiaoxiaoNeural",
        volume: Optional[int] = None,
        ttl_ms: int = 30000,
        wait_for_completion: bool = True,
    ) -> dict[str, Any]:
        self.seq += 1
        seq = self.seq
        loop = asyncio.get_running_loop()
        ack_waiter = loop.create_future()
        self.ack_waiters[seq] = ack_waiter
        if wait_for_completion:
            self.status_waiters[seq] = loop.create_future()
        try:
            await self._send(
                make_speech_message(
                    seq,
                    text,
                    voice=voice,
                    volume=volume,
                    ttl_ms=ttl_ms,
                )
            )
            ack = await asyncio.wait_for(ack_waiter, timeout=self.ack_timeout)
            if not ack.get("accepted", False):
                raise RuntimeError(ack.get("error", "speech command rejected"))
            result = {"ack": ack}
            if wait_for_completion:
                result["status"] = await asyncio.wait_for(
                    self.status_waiters[seq], timeout=self.speech_timeout
                )
            return result
        finally:
            self.ack_waiters.pop(seq, None)
            self.status_waiters.pop(seq, None)


async def keyboard_source(server: PolicyControlServer, args: argparse.Namespace) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("keyboard source requires an interactive terminal")
    old_terminal = termios.tcgetattr(sys.stdin.fileno())
    read_task: Optional[asyncio.Task[str]] = None

    def print_controls() -> None:
        if args.control_mode == "discrete":
            print(
                "Keys: w/s=forward/backward, a/d=left/right, q/e=rotate, "
                "x=stop, l=toggle LiDAR, ,=lie down, .=stand up, t=speak text",
                flush=True,
            )
        else:
            print(
                "Keys: w=forward, a=left turn, d=right turn, s/space/x=stop, "
                "l=toggle LiDAR, ,=lie down, .=stand up, t=speak text, "
                "q=stop and quit",
                flush=True,
            )

    try:
        fd = sys.stdin.fileno()
        tty.setcbreak(fd)
        # Discard keys typed while the bridge was disconnected or restarting.
        termios.tcflush(fd, termios.TCIFLUSH)
        print_controls()
        while True:
            if read_task is None:
                read_task = asyncio.create_task(asyncio.to_thread(sys.stdin.read, 1))
            done, _ = await asyncio.wait({read_task}, timeout=5.0)
            if not done:
                print_controls()
                continue
            key = read_task.result()
            read_task = None
            if not key:
                return
            try:
                if key.lower() == "t":
                    termios.tcsetattr(
                        sys.stdin.fileno(), termios.TCSADRAIN, old_terminal
                    )
                    server.text_entry_active = True
                    try:
                        text = await asyncio.to_thread(input, "\nSpeech text> ")
                    except EOFError:
                        text = ""
                    finally:
                        server.text_entry_active = False
                        tty.setcbreak(sys.stdin.fileno())
                        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
                    if text.strip():
                        result = await server.publish_speech(
                            text,
                            voice=args.speech_voice,
                            volume=args.speech_volume,
                            ttl_ms=args.speech_ttl_ms,
                            wait_for_completion=args.wait_speech_completion,
                        )
                        ack = result["ack"]
                        print(
                            f"Speech queued: seq={ack.get('seq')} "
                            f"depth={ack.get('queue_depth')}"
                        )
                        if "status" in result:
                            print(
                                "Speech result: "
                                f"{result['status'].get('state')}"
                            )
                    continue
                if key.lower() == "l":
                    ack = await server.publish_lidar()
                    state = "ON" if ack.get("lidar_enabled") else "OFF"
                    print(f"LiDAR {state}")
                    continue
                if key in {",", "."}:
                    action = "stand_down" if key == "," else "stand_up"
                    ack = await server.publish_posture(action)
                    print(f"Posture {ack.get('posture_action', action)} accepted")
                    continue
                if args.control_mode == "discrete":
                    action = {
                        "w": DiscreteAction.MOVE_FORWARD,
                        "s": DiscreteAction.MOVE_BACKWARD,
                        "a": DiscreteAction.MOVE_LEFT,
                        "d": DiscreteAction.MOVE_RIGHT,
                        "q": DiscreteAction.TURN_LEFT,
                        "e": DiscreteAction.TURN_RIGHT,
                        "x": DiscreteAction.STOP,
                    }.get(key)
                    if action is not None:
                        await server.publish_discrete(action, args.discrete_ttl_ms)
                else:
                    velocity = {
                        "w": (args.keyboard_vx, 0.0, 0.0),
                        "a": (0.0, 0.0, args.keyboard_wz),
                        "d": (0.0, 0.0, -args.keyboard_wz),
                        "s": (0.0, 0.0, 0.0),
                        " ": (0.0, 0.0, 0.0),
                        "x": (0.0, 0.0, 0.0),
                    }.get(key)
                    if velocity is not None:
                        await server.publish_continuous(
                            *velocity, ttl_ms=args.continuous_ttl_ms
                        )
                if key == "q" and args.control_mode == "continuous":
                    await server.publish_continuous(
                        0.0, 0.0, 0.0, args.continuous_ttl_ms
                    )
                    await asyncio.sleep(0.1)
                    return
            except ConnectionError as exc:
                print(exc)
    finally:
        if read_task is not None and not read_task.done():
            read_task.cancel()
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_terminal)


async def stdin_source(server: PolicyControlServer, args: argparse.Namespace) -> None:
    print("Reading one control or speak JSON command per stdin line; EOF exits")
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            return
        line = line.strip()
        if not line:
            continue
        try:
            parsed: Any
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                parsed = line
            if isinstance(parsed, dict) and parsed.get("type") == "speak":
                result = await server.publish_speech(
                    parsed.get("text", ""),
                    voice=parsed.get("voice", "zh-CN-XiaoxiaoNeural"),
                    volume=parsed.get("volume"),
                    ttl_ms=int(parsed.get("ttl_ms", 30000)),
                    wait_for_completion=bool(parsed.get("wait", True)),
                )
                print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
                continue
            if args.control_mode == "discrete":
                if isinstance(parsed, dict):
                    parsed = parsed.get("action", parsed.get("discrete_action"))
                result = await server.publish_discrete(
                    parsed,
                    args.discrete_ttl_ms,
                    wait_for_completion=args.wait_discrete_completion,
                )
                print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
            else:
                if isinstance(parsed, dict):
                    velocity = parsed.get("velocity", parsed)
                    vx = velocity.get("vx", velocity.get("linear_x", 0.0))
                    vy = velocity.get("vy", velocity.get("linear_y", 0.0))
                    wz = velocity.get("wz", velocity.get("angular_z", 0.0))
                else:
                    vx, vy, wz = (float(value) for value in str(parsed).split())
                await server.publish_continuous(
                    float(vx), float(vy), float(wz), args.continuous_ttl_ms
                )
        except Exception as exc:
            print(f"input rejected: {exc}")


async def ros_source(server: PolicyControlServer, args: argparse.Namespace) -> None:
    if args.control_mode != "continuous":
        raise RuntimeError("ROS cmd_vel source requires --control-mode continuous")
    try:
        import rospy
        from geometry_msgs.msg import Twist
    except ImportError as exc:
        raise RuntimeError(
            "ROS source requires ROS 1 rospy and geometry_msgs in the current environment"
        ) from exc

    loop = asyncio.get_running_loop()
    latest: asyncio.Queue = asyncio.Queue(maxsize=1)

    def enqueue_velocity(value: tuple[float, float, float]) -> None:
        if latest.full():
            try:
                latest.get_nowait()
            except asyncio.QueueEmpty:
                pass
        latest.put_nowait(value)

    def on_cmd_vel(message: Twist) -> None:
        value = (
            float(message.linear.x),
            float(message.linear.y),
            float(message.angular.z),
        )
        loop.call_soon_threadsafe(enqueue_velocity, value)

    rospy.init_node(args.ros_node_name, anonymous=False, disable_signals=True)
    subscriber = rospy.Subscriber(args.cmd_vel_topic, Twist, on_cmd_vel, queue_size=1)
    print(f"Subscribed to ROS Twist topic: {args.cmd_vel_topic}")
    try:
        while not rospy.is_shutdown():
            try:
                vx, vy, wz = await asyncio.wait_for(latest.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            try:
                await server.publish_continuous(
                    vx, vy, wz, args.continuous_ttl_ms
                )
            except ConnectionError:
                pass
    finally:
        subscriber.unregister()
        if server.bridge is not None:
            await server.publish_continuous(0.0, 0.0, 0.0, args.continuous_ttl_ms)


async def run(args: argparse.Namespace) -> None:
    source = args.source
    if source == "auto":
        source = "keyboard" if args.control_mode == "discrete" else "ros"
    server = PolicyControlServer(
        ack_timeout=args.ack_timeout,
        telemetry_print_period=args.telemetry_print_period,
        speech_timeout=args.speech_timeout,
    )
    source_function = {
        "keyboard": keyboard_source,
        "stdin": stdin_source,
        "ros": ros_source,
    }[source]
    async with serve(server.handler, args.host, args.port):
        print(
            f"Listening on ws://{args.host}:{args.port}; "
            f"control_mode={args.control_mode}; source={source}"
        )
        await source_function(server, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Policy-side server for Habitat discrete actions or move_base cmd_vel."
    )
    parser.add_argument(
        "--control-mode",
        choices=("discrete", "continuous"),
        required=True,
    )
    parser.add_argument(
        "--source",
        choices=("auto", "keyboard", "stdin", "ros"),
        default="auto",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12333)
    parser.add_argument("--discrete-ttl-ms", type=int, default=6000)
    parser.add_argument("--continuous-ttl-ms", type=int, default=350)
    parser.add_argument("--wait-discrete-completion", action="store_true")
    parser.add_argument("--keyboard-vx", type=float, default=0.10)
    parser.add_argument("--keyboard-wz", type=float, default=0.30)
    parser.add_argument("--ack-timeout", type=float, default=3.0)
    parser.add_argument("--speech-timeout", type=float, default=180.0)
    parser.add_argument("--speech-voice", default="zh-CN-XiaoxiaoNeural")
    parser.add_argument(
        "--speech-volume",
        type=int,
        help="Go2 VUI volume 0-10; omitted keeps the current volume",
    )
    parser.add_argument("--speech-ttl-ms", type=int, default=60000)
    parser.add_argument(
        "--wait-speech-completion",
        action="store_true",
        help="block keyboard input until Go2 reports speech completion",
    )
    parser.add_argument("--telemetry-print-period", type=float, default=1.0)
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--ros-node-name", default="go2_policy_control_server")
    args = parser.parse_args()
    if not 100 <= args.continuous_ttl_ms <= 500:
        parser.error("--continuous-ttl-ms must be between 100 and 500")
    if not 100 <= args.discrete_ttl_ms <= 7000:
        parser.error("--discrete-ttl-ms must be between 100 and 7000")
    if args.source == "ros" and args.control_mode != "continuous":
        parser.error("--source ros requires --control-mode continuous")
    if not 10.0 <= args.speech_timeout <= 600.0:
        parser.error("--speech-timeout must be between 10 and 600 seconds")
    if args.speech_volume is not None and not 0 <= args.speech_volume <= 10:
        parser.error("--speech-volume must be between 0 and 10")
    if not 1000 <= args.speech_ttl_ms <= 120000:
        parser.error("--speech-ttl-ms must be between 1000 and 120000")
    return args


def main() -> None:
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        print("Interrupted; policy server stopped")


if __name__ == "__main__":
    main()
