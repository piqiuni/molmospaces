#!/usr/bin/env python3
"""Create the user-configurable SSH local port mapping for Qwen."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request


def ssh_command(args) -> list[str]:
    return ["ssh", "-N", "-T", "-p", str(args.ssh_port), "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=5", "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            "-L", f"127.0.0.1:{args.local_port}:{args.remote_bind}:{args.remote_port}",
            f"{args.user}@{args.host}"]


def owned_forward(argv: list[str], expected: list[str]) -> str | None:
    """Recognize only our dedicated SSH invocation, allowing a stale target port."""
    if len(argv) != len(expected) or not argv or Path(argv[0]).name != "ssh":
        return None
    index = expected.index("-L") + 1
    normalized = ["ssh", *argv[1:]]
    forward = normalized[index]
    prefix = expected[index].rsplit(":", 1)[0] + ":"
    if not forward.startswith(prefix) or not forward[len(prefix):].isdigit():
        return None
    normalized[index] = expected[index]
    return forward if normalized == expected else None


def process_args(pid: int) -> list[str]:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
    except (OSError, UnicodeError):
        return []


def model_endpoint_ready(port: int) -> bool:
    try:
        # Local health checks must not inherit a site HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=1.5) as response:
            payload = json.load(response)
        return bool(isinstance(payload, dict) and payload.get("data"))
    except (OSError, ValueError):
        return False


def matching_processes(expected: list[str]) -> list[tuple[int, list[str], str]]:
    matches = []
    for entry in Path("/proc").iterdir():
        try:
            if not entry.name.isdigit() or entry.stat().st_uid != os.getuid():
                continue
        except OSError:
            continue
        pid = int(entry.name)
        argv = process_args(pid)
        forward = owned_forward(argv, expected)
        if forward is not None:
            matches.append((pid, argv, forward))
    return matches


def verify_inference(port: int, model: str, timeout: float) -> None:
    """Probe actual generation, not just the model-list HTTP handler."""
    started = time.monotonic()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK only."}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            result = json.load(response)
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty assistant answer")
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Qwen inference health failed: {exc}") from exc
    print(f"Qwen inference=PASS model={model} elapsed={time.monotonic() - started:.2f}s",
          flush=True)


def local_port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def ensure_tunnel(args) -> int:
    """Reconcile a stale/orphaned *matching* forward; never kill an unknown listener."""
    print(f"Qwen remote={args.host}:{args.remote_port}; "
          f"health=http://127.0.0.1:{args.local_port}/v1/models (via SSH)", flush=True)
    expected = ssh_command(args)
    pid_file = Path(args.pid_file)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(pid_file) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        matches = matching_processes(expected)
        if len(matches) > 1:
            raise RuntimeError("multiple matching Qwen forwards; refusing ambiguous cleanup")
        if matches:
            pid, argv, forward = matches[0]
            if forward == expected[expected.index("-L") + 1]:
                if not model_endpoint_ready(args.local_port):
                    raise RuntimeError("matching Qwen tunnel exists, but /v1/models is unhealthy")
                pid_file.write_text(f"{pid}\n")
                print(f"Qwen health=PASS remote={args.host}:{args.remote_port} "
                      f"(pid={pid}, forward={forward})", flush=True)
                return pid
            if process_args(pid) != argv:
                raise RuntimeError("Qwen tunnel identity changed before replacement")
            print(f"replacing stale Qwen forward pid={pid}: {forward}", flush=True)
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 5.0
            while process_args(pid) == argv and time.monotonic() < deadline:
                time.sleep(0.05)
            if process_args(pid) == argv:
                raise RuntimeError("old Qwen SSH process did not stop; refusing force kill")
        if local_port_open(args.local_port):
            raise RuntimeError("Qwen local port is owned by an unrecognized listener; refusing cleanup")
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log:
            child = subprocess.Popen(expected, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=log, start_new_session=True)
        pid_file.write_text(f"{child.pid}\n")
        deadline = time.monotonic() + 15.0
        while child.poll() is None and time.monotonic() < deadline:
            if model_endpoint_ready(args.local_port):
                # An unrelated listener must not make a failed SSH bind look healthy.
                time.sleep(0.15)
                if child.poll() is None:
                    print(f"Qwen health=PASS remote={args.host}:{args.remote_port} "
                          f"(pid={child.pid})", flush=True)
                    return child.pid
            time.sleep(0.2)
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
        raise RuntimeError(f"Qwen tunnel failed health check; inspect {log_path}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ssh-port", type=int, default=41051)
    p.add_argument("--user", default="root")
    p.add_argument("--host", default="115.190.90.101")
    p.add_argument("--local-port", type=int, default=18080)
    p.add_argument("--remote-port", type=int, default=8100, help="Qwen HTTP port on the remote host")
    p.add_argument("--remote-bind", default="127.0.0.1")
    p.add_argument("--ensure", action="store_true", help="Reconcile and health-check a detached tunnel")
    p.add_argument("--pid-file")
    p.add_argument("--log-file")
    p.add_argument("--verify-inference", action="store_true")
    p.add_argument("--model", default="qwen3.6-35b-a3b-fp8")
    p.add_argument("--inference-timeout", type=float, default=8.0)
    args = p.parse_args(argv)
    if args.inference_timeout <= 0:
        p.error("--inference-timeout must be positive")
    if args.verify_inference and not args.ensure:
        p.error("--verify-inference requires --ensure")
    if args.ensure:
        if not args.pid_file or not args.log_file:
            p.error("--ensure requires --pid-file and --log-file")
        try:
            ensure_tunnel(args)
            if args.verify_inference:
                verify_inference(args.local_port, args.model, args.inference_timeout)
        except (RuntimeError, OSError) as exc:
            print(f"Qwen health=FAIL remote={args.host}:{args.remote_port}: {exc}", flush=True)
            raise
        return
    command = ssh_command(args)
    print("starting Qwen SSH tunnel:", " ".join(command), flush=True)
    # The supervisor's PID must own the listening SSH process itself. A
    # wrapper killed by SIGTERM could otherwise leave an old forward alive.
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
