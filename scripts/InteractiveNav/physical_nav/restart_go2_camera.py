#!/usr/bin/env python3
"""Restart only the running Go2 camera bridge, preserving unspecified options.

Example: python restart_go2_camera.py --rgb 1280x720@30 --depth 848x480@30
         --publish-fps 10
The local ROS stack must already be running. No motion commands are sent.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time


def mode(value):
    match = re.fullmatch(r"([1-9][0-9]*)[xX]([1-9][0-9]*)@([1-9][0-9]*)", value)
    if not match:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT@FPS, e.g. 1280x720@30")
    return tuple(map(int, match.groups()))


def replace_option(argv, name, value):
    result = []
    index = 0
    while index < len(argv):
        if argv[index] == name:
            index += 2
        elif argv[index].startswith(name + "="):
            index += 1
        else:
            result.append(argv[index])
            index += 1
    return result + [name, str(value)]


def command_for_modes(argv, rgb, depth, publish_fps):
    result = list(argv)
    for prefix, setting in (("color", rgb), ("depth", depth)):
        if setting:
            for field, value in zip(("width", "height", "fps"), setting):
                result = replace_option(result, "--" + prefix + "-" + field, value)
    if publish_fps is not None:
        result = replace_option(result, "--publish-fps", publish_fps)
    return result


def read_process(pid):
    try:
        argv = Path("/proc", str(pid), "cmdline").read_bytes().split(b"\0")
        return [os.fsdecode(value) for value in argv if value]
    except (OSError, ValueError):
        return []


def is_bridge_command(argv, bridge_path):
    if not argv or not re.fullmatch(r"python(?:[0-9.]+)?", Path(argv[0]).name):
        return False
    index = 1
    while index < len(argv) and argv[index] in ("-u", "-B", "-E", "-s", "-S", "-I", "-O", "-OO"):
        index += 1
    return index < len(argv) and argv[index] == bridge_path


def find_bridge(bridge_path):
    matches = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        argv = read_process(path.name)
        # Match an exact script argument, never a shell command containing it.
        if is_bridge_command(argv, bridge_path):
            matches.append((int(path.name), argv))
    if len(matches) != 1:
        raise RuntimeError("expected one running camera bridge, found %d; start the full service first or resolve duplicate bridges" % len(matches))
    return matches[0]


def remote_restart(args):
    pid, previous = find_bridge(args.bridge_path)
    command = command_for_modes(previous, args.rgb, args.depth, args.publish_fps)
    if args.dry_run:
        return {"dry_run": True, "old_pid": pid, "command": command}
    # Check the log destination before stopping a working camera.
    with open(args.log, "ab", buffering=0) as log:
        offset = log.tell()
        if read_process(pid) != previous:
            raise RuntimeError("camera process changed during inspection; retry")
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while read_process(pid) == previous:
            if time.monotonic() >= deadline:
                raise RuntimeError("camera did not stop within 10s; no forced kill performed")
            time.sleep(0.1)
        child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"),
        )
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if child.poll() is not None:
                fallback = subprocess.Popen(
                    previous, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                )
                return {"pid": fallback.pid, "old_pid": pid, "camera_started": False,
                        "previous_mode_relaunched": True, "command": previous,
                        "requested_command": command, "log": args.log,
                        "error": "requested mode exited with code %s" % child.returncode}
            with open(args.log, "rb") as reader:
                reader.seek(offset)
                text = reader.read(131072).decode("utf-8", errors="replace")
            if "D435i started" in text:
                return {"pid": child.pid, "old_pid": pid, "camera_started": True,
                        "command": command, "log": args.log}
            time.sleep(0.2)
        return {"pid": child.pid, "old_pid": pid, "camera_started": False,
                "command": command, "log": args.log}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=mode, help="RGB mode WIDTHxHEIGHT@FPS; omitted keeps current")
    parser.add_argument("--depth", type=mode, help="depth mode WIDTHxHEIGHT@FPS; omitted keeps current")
    parser.add_argument("--publish-fps", type=float, help="network send rate; omitted keeps current")
    parser.add_argument("--ssh-target", default=os.environ.get("PHYSICAL_NAV_GO2_SSH_TARGET", "unitree"))
    parser.add_argument("--bridge-path", default=os.environ.get("PHYSICAL_NAV_GO2_BRIDGE_PATH", "/home/unitree/physical_nav/go2_readonly_sensor_bridge.py"))
    parser.add_argument("--log", default=os.environ.get("PHYSICAL_NAV_GO2_BRIDGE_LOG", "/home/unitree/physical_nav/go2_readonly_sensor_bridge.log"))
    parser.add_argument("--timeout", type=float, default=30, help="seconds to wait for camera startup")
    parser.add_argument("--dry-run", action="store_true", help="inspect and print the command without restarting")
    parser.add_argument("--remote", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.publish_fps is not None and not 0 < args.publish_fps < float("inf"):
        parser.error("--publish-fps must be finite and positive")
    if not 0 < args.timeout <= 120:
        parser.error("--timeout must be in (0, 120]")
    try:
        if args.remote:
            print(json.dumps(remote_restart(args)), flush=True)
            return
        remote = ["python3", "-", "--remote", "--bridge-path", args.bridge_path,
                  "--log", args.log, "--timeout", str(args.timeout)]
        for name, setting in (("--rgb", args.rgb), ("--depth", args.depth)):
            if setting:
                remote += [name, "%dx%d@%d" % setting]
        if args.publish_fps is not None:
            remote += ["--publish-fps", str(args.publish_fps)]
        if args.dry_run:
            remote.append("--dry-run")
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", args.ssh_target,
             shlex.join(remote)], input=Path(__file__).read_text(), text=True,
            capture_output=True, timeout=args.timeout + 25, check=True,
        )
        data = json.loads(result.stdout)
        if not args.dry_run:
            runtime = Path(os.environ.get("PHYSICAL_NAV_ALL_RUNTIME_DIR", "/tmp/molmospaces-physical-nav-all-%s" % os.getuid()))
            runtime.mkdir(parents=True, exist_ok=True)
            # Keep the full launcher's later stop/restart ownership accurate.
            with tempfile.NamedTemporaryFile(mode="w", dir=runtime, delete=False) as stream:
                stream.write(str(data["pid"]) + "\n")
            os.replace(stream.name, runtime / "go2_bridge.pid")
            (runtime / "go2_bridge.owned").touch()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        if not args.dry_run and not data["camera_started"]:
            message = ("Requested mode failed; previous camera command relaunched."
                       if data.get("previous_mode_relaunched")
                       else "Camera startup not confirmed.")
            print(message + " Inspect the remote log.", file=sys.stderr)
            raise SystemExit(1)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.strip() or str(exc), file=sys.stderr)
        raise SystemExit(1)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print("Camera restart failed: %s" % exc, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
