#!/usr/bin/env python3
"""Create the user-configurable SSH local port mapping for Qwen."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ssh-port", type=int, default=41051)
    p.add_argument("--user", default="root")
    p.add_argument("--host", default="115.190.90.101")
    p.add_argument("--local-port", type=int, default=18080)
    p.add_argument("--remote-port", type=int, default=18080, help="Qwen HTTP port on the remote host")
    p.add_argument("--remote-bind", default="127.0.0.1")
    args = p.parse_args()
    target = f"{args.user}@{args.host}"
    command = ["ssh", "-N", "-T", "-p", str(args.ssh_port), "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=5", "-o", "ExitOnForwardFailure=yes",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
               "-L", f"127.0.0.1:{args.local_port}:{args.remote_bind}:{args.remote_port}", target]
    print("starting Qwen SSH tunnel:", " ".join(command[:-1] + [target]), flush=True)
    process = subprocess.Popen(command)
    try:
        process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=5)
        sys.exit(130)


if __name__ == "__main__":
    main()
