#!/usr/bin/env python3
"""Owner-only local control socket for switching a running Go2 bridge."""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
from pathlib import Path


def socket_path(ready_file, pid):
    return Path(ready_file).parent / f"go2-motion-{int(pid)}.sock"


class MotionSwitchServer:
    def __init__(self, ready_file, callback):
        self.path = socket_path(ready_file, os.getpid())
        self.callback = callback
        self.stopping = threading.Event()
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(str(self.path))
        os.chmod(self.path, 0o600)
        self.socket.listen(2)
        self.socket.settimeout(0.2)
        self.thread = threading.Thread(target=self._run, name="motion-switch", daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopping.is_set():
            try:
                connection, _ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with connection:
                connection.settimeout(2.0)
                try:
                    raw = bytearray()
                    while b"\n" not in raw and len(raw) <= 4096:
                        chunk = connection.recv(1024)
                        if not chunk:
                            break
                        raw.extend(chunk)
                    if len(raw) > 4096 or b"\n" not in raw:
                        raise ValueError("invalid motion-switch request")
                    request = json.loads(raw.split(b"\n", 1)[0])
                    if not isinstance(request, dict):
                        raise ValueError("request must be an object")
                    result = {"ok": True, "pid": os.getpid(), **self.callback(request)}
                except Exception as exc:
                    result = {"ok": False, "pid": os.getpid(), "error": str(exc)}
                try:
                    connection.sendall((json.dumps(result) + "\n").encode())
                except OSError:
                    pass

    def close(self):
        self.stopping.set()
        self.socket.close()
        self.thread.join(timeout=2.0)
        self.path.unlink(missing_ok=True)


def request_switch(ready_file, request, timeout=25.0):
    pid = int(Path(ready_file).read_text().strip())
    os.kill(pid, 0)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(socket_path(ready_file, pid)))
        connection.sendall((json.dumps(request) + "\n").encode())
        raw = bytearray()
        while b"\n" not in raw and len(raw) <= 4096:
            chunk = connection.recv(1024)
            if not chunk:
                break
            raw.extend(chunk)
    result = json.loads(raw)
    if result.get("pid") != pid or not result.get("ok"):
        raise RuntimeError(result.get("error", "motion-switch identity mismatch"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--enabled", choices=("true", "false"))
    parser.add_argument("--max-vx", type=float)
    parser.add_argument("--max-wz", type=float)
    args = parser.parse_args()
    request = {}
    if args.enabled is not None:
        request["enabled"] = args.enabled == "true"
    for key in ("max_vx", "max_wz"):
        if getattr(args, key) is not None:
            request[key] = getattr(args, key)
    try:
        result = request_switch(args.ready_file, request)
    except (FileNotFoundError, ProcessLookupError, ConnectionRefusedError):
        # An old bridge can be migrated once by the launcher. Other failures
        # must not silently fall back to restarting an actuator.
        raise SystemExit(3)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
