#!/usr/bin/env python3
"""Unbounded-ingress gateway leasing requests across YOLOE replicas."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class _Gateway:
    def __init__(self, workers: list[str], timeout_s: float) -> None:
        if not workers:
            raise ValueError("at least one worker endpoint is required")
        self.workers = [value.rstrip("/") for value in workers]
        self.timeout_s = float(timeout_s)
        # No ingress semaphore or queue-size limit: every accepted HTTP request
        # waits until one single-inflight replica is returned to this lease pool.
        self._idle: queue.Queue[str] = queue.Queue()
        for endpoint in self.workers:
            self._idle.put(endpoint)
        self._lock = threading.Lock()
        self.requests = 0
        self.failures = 0
        self.active = 0
        self.peak_active = 0

    def forward(self, body: bytes) -> tuple[int, bytes]:
        endpoint = self._idle.get()
        with self._lock:
            self.requests += 1
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            request = Request(
                endpoint + "/detect",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=self.timeout_s) as response:
                return int(response.status), response.read()
        except HTTPError as exc:
            with self._lock:
                self.failures += 1
            return int(exc.code), exc.read()
        except (OSError, URLError, TimeoutError) as exc:
            with self._lock:
                self.failures += 1
            payload = json.dumps(
                {"error": f"YOLOE replica request failed: {exc}", "detections": []},
                separators=(",", ":"),
            ).encode("utf-8")
            return HTTPStatus.BAD_GATEWAY, payload
        finally:
            with self._lock:
                self.active -= 1
            self._idle.put(endpoint)

    def health(self) -> dict[str, Any]:
        replicas = []
        for endpoint in self.workers:
            ready = False
            detail: dict[str, Any] = {}
            try:
                with urlopen(endpoint + "/health", timeout=2.0) as response:
                    detail = json.loads(response.read().decode("utf-8"))
                    ready = bool(detail.get("ready"))
            except Exception as exc:
                detail = {"error": str(exc)}
            replicas.append({"endpoint": endpoint, "ready": ready, "detail": detail})
        with self._lock:
            stats = {
                "requests": self.requests,
                "failures": self.failures,
                "active": self.active,
                "peak_active": self.peak_active,
            }
        return {
            "ready": all(item["ready"] for item in replicas),
            "ingress_concurrency_limit": None,
            "replica_count": len(replicas),
            "single_inflight_per_replica": True,
            "replicas": replicas,
            "stats": stats,
            "timestamp": time.time(),
        }


def _handler(gateway: _Gateway):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HabitatYOLOEGateway/1"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def _send(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send(HTTPStatus.NOT_FOUND, b'{"error":"not found"}')
                return
            body = json.dumps(gateway.health(), separators=(",", ":")).encode("utf-8")
            self._send(HTTPStatus.OK, body)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/detect":
                self._send(HTTPStatus.NOT_FOUND, b'{"error":"not found"}')
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 24 * 1024 * 1024:
                self._send(HTTPStatus.BAD_REQUEST, b'{"error":"invalid request size","detections":[]}')
                return
            status, response = gateway.forward(self.rfile.read(length))
            self._send(status, response)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12219)
    parser.add_argument("--worker", action="append", required=True)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args()
    gateway = _Gateway(args.worker, args.timeout_s)
    server = ThreadingHTTPServer((args.host, args.port), _handler(gateway))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
