#!/usr/bin/env python3
"""Read-only inventory of navigation-related interfaces on a Go2 computer."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
from typing import Any, Callable

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.vui.vui_client import VuiClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_


def query(name: str, function: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"name": name, "ok": True, "value": function()}
    except Exception as exc:
        return {"name": name, "ok": False, "error": repr(exc)}


def command_output(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip().splitlines(),
        "stderr": result.stderr.strip().splitlines(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--state-timeout", type=float, default=3.0)
    parser.add_argument("--skip-camera", action="store_true")
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.interface)
    results: list[dict[str, Any]] = []
    state_event = threading.Event()
    state_summary: dict[str, Any] = {}

    def on_state(message: SportModeState_) -> None:
        state_summary.update(
            position=list(message.position),
            velocity=list(message.velocity),
            yaw_speed=float(message.yaw_speed),
        )
        state_event.set()

    subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    subscriber.Init(on_state, 10)
    results.append(
        {
            "name": "dds.rt/sportmodestate",
            "ok": state_event.wait(args.state_timeout),
            "value": state_summary,
        }
    )

    obstacles = ObstaclesAvoidClient()
    obstacles.SetTimeout(3.0)
    obstacles.Init()
    results.append(query("sdk.obstacles_avoid.switch", obstacles.SwitchGet))

    vui = VuiClient()
    vui.SetTimeout(3.0)
    vui.Init()
    results.extend(
        [
            query("sdk.vui.switch", vui.GetSwitch),
            query("sdk.vui.volume", vui.GetVolume),
            query("sdk.vui.brightness", vui.GetBrightness),
        ]
    )

    if not args.skip_camera:
        video = VideoClient()
        video.SetTimeout(3.0)
        video.Init()

        def image_summary() -> dict[str, int]:
            code, data = video.GetImageSample()
            return {"code": int(code), "bytes": len(data) if data is not None else 0}

        results.append(query("sdk.front_camera.image_sample", image_summary))

    robot_state = RobotStateClient()
    robot_state.SetTimeout(3.0)
    robot_state.Init()

    def service_summary() -> dict[str, Any]:
        code, services = robot_state.ServiceList()
        return {
            "code": code,
            "services": [
                {"name": item.name, "status": item.status, "protect": item.protect}
                for item in (services or [])
            ],
        }

    results.append(query("sdk.robot_state.services", service_summary))
    results.append(query("os.pulseaudio.sinks", lambda: command_output(["pactl", "list", "short", "sinks"])))
    results.append(query("os.pulseaudio.sources", lambda: command_output(["pactl", "list", "short", "sources"])))
    results.append(query("os.usb", lambda: command_output(["lsusb"])))

    print(json.dumps({"interface": args.interface, "results": results}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
