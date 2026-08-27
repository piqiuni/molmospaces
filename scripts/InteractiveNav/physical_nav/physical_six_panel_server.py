#!/usr/bin/env python3
"""Local/LAN six-panel viewer and read-only sensor WebSocket gateway."""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from physical_protocol import validate_packet
from runtime_state import RuntimeState
from safety_gate import ReadOnlySafetyGate
from qwen_client import QwenClient

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - useful for protocol-only testing
    cv2 = None
    np = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np is None or np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _point_xy(value: Any) -> tuple[float, float] | None:
    """Read either graph point dictionaries or the mapper's XYZ arrays."""
    if isinstance(value, dict):
        return _safe_float(value.get("x")), _safe_float(value.get("y"))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _safe_float(value[0]), _safe_float(value[1])
    return None


def _decode(value: str, encoding: str) -> Any:
    raw = base64.b64decode(value)
    if cv2 is None:
        return raw
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not decode {encoding}")
    return image


class SixPanelRenderer:
    def __init__(self, state: RuntimeState, width: int = 640, height: int = 360) -> None:
        self.state, self.width, self.height = state, width, height

    def _placeholder(self, title: str) -> Any:
        panel = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        panel[:, :, :] = (32, 32, 32)
        cv2.putText(panel, title, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
        return panel

    def _image_panel(self, image: Any, title: str) -> Any:
        if image is None:
            return self._placeholder(title)
        if image.ndim == 2:
            valid = image > 0
            if np.any(valid):
                norm = np.zeros_like(image, dtype=np.uint8)
                clipped = np.clip(image.astype(np.float32), 0, np.percentile(image[valid], 98))
                norm = (clipped / max(1.0, float(clipped.max())) * 255).astype(np.uint8)
                image = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
            else:
                image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        panel = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        cv2.putText(panel, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
        return panel

    def _detection_panel(self, rgb: Any, detections: list[dict[str, Any]]) -> Any:
        panel = self._image_panel(rgb, "YOLOE-26l PF Seg")
        if rgb is None:
            return panel
        sx, sy = self.width / rgb.shape[1], self.height / rgb.shape[0]
        for det in detections:
            box = det.get("bbox") or det.get("bbox_2d")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(box)]
            color = (0, 255, 0) if float(det.get("confidence", 0)) >= .5 else (0, 165, 255)
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
            text = f"{det.get('semantic_class', det.get('class', '?'))} {float(det.get('confidence', 0)):.2f}"
            cv2.putText(panel, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1)
        return panel

    def _json_panel(self, title: str, value: Any) -> Any:
        panel = self._placeholder(title)
        lines = json.dumps(value, ensure_ascii=False, indent=2, default=str).splitlines()[:16]
        for i, line in enumerate(lines):
            cv2.putText(panel, line[:78], (12, 62 + i * 19), cv2.FONT_HERSHEY_PLAIN, 1.0, (220, 220, 220), 1)
        return panel

    def _map_panel(self, occupancy: Any, telemetry: dict[str, Any]) -> Any:
        if not isinstance(occupancy, dict) or not occupancy.get("data"):
            return self._json_panel("Occupancy / pose", telemetry)
        panel = np.full((self.height, self.width, 3), 127, dtype=np.uint8)
        width, height = int(occupancy.get("width", 0)), int(occupancy.get("height", 0)); values = np.asarray(occupancy.get("data", []), dtype=np.int16)
        if width > 0 and height > 0 and values.size >= width * height:
            grid = values[:width * height].reshape(height, width); image = np.full((height, width, 3), 127, dtype=np.uint8)
            image[grid == 0] = (235, 235, 235); image[grid > 50] = (30, 30, 30)
            panel = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        cv2.putText(panel, "Occupancy / pose", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
        pose = telemetry.get("position", [0, 0, 0]); cv2.putText(panel, f"pose={pose[:2]} yaw={telemetry.get('yaw', '?')}", (12, self.height - 15), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 100, 255), 1)
        return panel

    def _graph_panel(self, graph: dict[str, Any]) -> Any:
        panel = np.full((self.height, self.width, 3), 35, dtype=np.uint8); nodes = graph.get("nodes", []) if isinstance(graph, dict) else []; edges = graph.get("edges", []) if isinstance(graph, dict) else []
        positions = {}
        for index, node in enumerate(nodes):
            value = node.get("world_position") or node.get("position") or node.get("centroid") or node.get("aabb_center") or node.get("center") or {}
            point = _point_xy(value)
            x, y = point if point is not None else (float(index % 5), float(index // 5))
            positions[str(node.get("id", node.get("node_id", index)))] = (x, y)
        if positions:
            xs, ys = [p[0] for p in positions.values()], [p[1] for p in positions.values()]; minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
            def point(x: float, y: float) -> tuple[int, int]: return (int(40 + (x - minx) / max(1e-6, maxx - minx) * (self.width - 80)), int(self.height - 40 - (y - miny) / max(1e-6, maxy - miny) * (self.height - 80)))
            for edge in edges:
                a, b = positions.get(str(edge.get("source", edge.get("from", edge.get("src_id", ""))))), positions.get(str(edge.get("target", edge.get("to", edge.get("dst_id", "")))))
                if a and b: cv2.line(panel, point(*a), point(*b), (110, 110, 110), 1)
            for node in nodes:
                key = str(node.get("id", node.get("node_id", nodes.index(node)))); xy = positions.get(key); 
                if not xy: continue
                kind = str(node.get("type", node.get("node_type", "object"))); color = {"room": (255, 160, 30), "portal": (30, 220, 255), "container": (180, 80, 220), "support": (200, 200, 60)}.get(kind, (60, 220, 80)); px = point(*xy); cv2.circle(panel, px, 7, color, -1); cv2.putText(panel, str(node.get("label", key))[:18], (px[0] + 8, px[1]), cv2.FONT_HERSHEY_PLAIN, .9, color, 1)
        cv2.putText(panel, f"Global semantic graph  nodes={len(nodes)} edges={len(edges)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
        return panel

    def render(self) -> bytes:
        if cv2 is None:
            raise RuntimeError("physical viewer requires opencv-python and numpy")
        with self.state._lock:
            rgb, depth = self.state.rgb, self.state.depth
            detections, graph, consistency, occupancy = list(self.state.detections), dict(self.state.graph), dict(self.state.consistency), self.state.occupancy
            telemetry = dict(self.state.telemetry)
        panels = [self._image_panel(rgb, "Go2 / D435i RGB"), self._image_panel(depth, "D435i Depth"),
                  self._detection_panel(rgb, detections), self._map_panel(occupancy, telemetry),
                  self._graph_panel(graph),
                  self._json_panel("Perception-map consistency", consistency)]
        canvas = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("six-panel JPEG encoding failed")
        return bytes(encoded)


class _WebHandler(BaseHTTPRequestHandler):
    state: RuntimeState
    renderer: SixPanelRenderer
    gate: ReadOnlySafetyGate
    qwen_submit = None
    frame_lock = threading.Lock()
    latest_jpeg: bytes = b""

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _json(self, value: Any, status: int = 200) -> None:
        data = json.dumps(value, ensure_ascii=False, default=str).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/state":
            self._json({**self.state.snapshot(), "safety": self.gate.snapshot()}); return
        if path == "/api/raw-frame":
            self._json(self.state.raw_frame()); return
        if path == "/api/health":
            snapshot = self.state.snapshot(); self._json({"ok": bool(snapshot["frame_seq"] >= 0), "read_only": True, "frame_seq": snapshot["frame_seq"], "generated_at": snapshot["generated_at"]}); return
        if path == "/stream.mjpg":
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
            while True:
                with self.frame_lock: frame = self.latest_jpeg
                if frame:
                    try:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"); self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError): return
                time.sleep(.1)
        else:
            html = _HTML.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(html))); self.end_headers(); self.wfile.write(html)

    def do_POST(self) -> None:
        if self.path == "/api/ros-state":
            length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
            name, value = str(payload.get("name", "")), payload.get("value")
            if name not in {"detections", "mapped_detections", "graph", "consistency", "occupancy", "telemetry"}:
                self._json({"accepted": False, "error": "unsupported ROS state"}, 400); return
            self.state.update_topic(name, value); self._json({"accepted": True}); return
        if self.path == "/api/qwen":
            length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
            if self.qwen_submit is None: self._json({"error": "Qwen client is disabled"}, 503); return
            self._json(self.qwen_submit(payload), 202); return
        if self.path != "/api/teleop-intent": self._json({"error": "not found"}, 404); return
        length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
        self._json(self.gate.handle_intent(payload), 202)


_HTML = """<!doctype html><meta charset='utf-8'><title>Go2 Physical Interactive Navigation</title>
<style>body{font-family:monospace;background:#111;color:#eee;margin:12px}img{max-width:100%;border:1px solid #555}pre{white-space:pre-wrap;max-height:420px;overflow:auto;background:#1b1b1b;padding:10px}.grid{display:grid;grid-template-columns:2fr 1fr;gap:12px}.ok{color:#5f5}.warn{color:#fc3}</style>
<h2>Go2 Physical Interactive Navigation <span class='warn'>READ_ONLY_BLOCKED</span></h2>
<div class='grid'><div><img src='/stream.mjpg'></div><div><h3>状态 / Qwen / Graph</h3><pre id='state'>loading...</pre><input id='prompt' size='40' value='请分析当前全局语义图和感知一致性'><button onclick="askQwen()">请求 Qwen</button><br><button onclick="intent('STOP')">STOP（仅记录）</button><button onclick="intent('MOVE_FORWARD')">前进意图（阻断）</button></div></div>
<script>async function refresh(){try{let r=await fetch('/api/state');document.querySelector('#state').textContent=JSON.stringify(await r.json(),null,2)}catch(e){document.querySelector('#state').textContent=e}} async function intent(a){await fetch('/api/teleop-intent',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:a,source:'web'})});refresh()} async function askQwen(){await fetch('/api/qwen',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:document.querySelector('#prompt').value})});refresh()} setInterval(refresh,1000);refresh()</script>"""


def _compact_qwen_context(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Keep sparse masks out of the text prompt sent to the remote Qwen."""
    compact_detections: list[dict[str, Any]] = []
    detections = [item for item in snapshot.get("detections", []) if isinstance(item, dict)]
    detections.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    for detection in detections[:12]:
        item = {
            key: detection.get(key)
            for key in (
                "semantic_class", "confidence", "bbox", "mask_area", "depth_median_m",
                "world_position", "aabb_center", "aabb_size", "map_transform_status",
            )
            if key in detection
        }
        mask = detection.get("mask")
        if isinstance(mask, dict):
            item["mask_summary"] = {
                "area": mask.get("area", mask.get("mask_area", 0)),
                "rows": len(mask.get("rows", [])),
                "cols": len(mask.get("cols", [])),
            }
        compact_detections.append(item)

    graph = snapshot.get("graph")
    compact_graph: dict[str, Any] = {}
    if isinstance(graph, dict):
        compact_graph = {
            key: graph.get(key)
            for key in ("scene_id", "episode_id", "source_mode", "graph_revision", "timestamp", "module1_mode")
            if key in graph
        }
        compact_graph["nodes"] = []
        nodes = [item for item in graph.get("nodes", []) if isinstance(item, dict)]
        nodes.sort(key=lambda item: (bool(item.get("is_currently_visible", True)), float(item.get("last_seen") or 0.0)), reverse=True)
        for node in nodes[:16]:
            item = {
                key: node.get(key)
                for key in ("id", "type", "label", "centroid", "aabb_center", "aabb_size", "room_id")
                if key in node
            }
            interaction = node.get("interaction")
            if isinstance(interaction, dict):
                item["interaction"] = {
                    key: interaction.get(key)
                    for key in ("is_interactable", "interaction_mode", "capability", "state", "cost", "requires_interaction", "traversable")
                    if key in interaction
                }
            compact_graph["nodes"].append(item)
        compact_graph["edges"] = graph.get("edges", [])

    return {
        "frame_seq": snapshot.get("frame_seq", -1),
        "telemetry": _compact_telemetry(snapshot.get("telemetry", {})),
        "detections": compact_detections,
        "graph": compact_graph,
        "consistency": _compact_consistency(snapshot.get("consistency", {})),
    }


def _compact_consistency(consistency: Any) -> dict[str, Any]:
    if not isinstance(consistency, dict):
        return {}
    result = {
        key: consistency.get(key)
        for key in ("status", "counts", "detection_count", "graph_revision", "projection")
        if key in consistency
    }
    result["detections"] = []
    for item in consistency.get("detections", [])[:12]:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        result["detections"].append({
            "object_id": item.get("object_id", ""),
            "status": item.get("status", ""),
            "metrics": {
                key: metrics.get(key)
                for key in ("bbox_iou", "bbox_center_px", "map_projected_depth_abs_m", "map_distance_m", "map_z_abs_m", "rgbd_depth_lift_abs_m")
                if key in metrics
            },
            "reasons": item.get("reasons", []),
        })
    return result


def _compact_telemetry(telemetry: Any) -> dict[str, Any]:
    if not isinstance(telemetry, dict):
        return {}
    result = {
        key: telemetry.get(key)
        for key in ("position", "velocity", "yaw", "yaw_speed", "mode", "error_code", "body_height")
        if key in telemetry
    }
    battery = telemetry.get("battery")
    if isinstance(battery, dict):
        result["battery"] = {key: battery.get(key) for key in ("soc", "voltage", "current", "power") if key in battery}
    return result


class PhysicalGateway:
    def __init__(self, host: str, port: int, qwen_url: str = "", qwen_model: str = "qwen3.6-35b-a3b-fp8", camera_parent: str = "tf_frame_base_link", camera_x: float = 0.0, camera_y: float = 0.0, camera_z: float = 0.0, camera_roll: float = 0.0, camera_pitch: float = 0.0, camera_yaw: float = 0.0, qwen_auto_interval: float = 0.0) -> None:
        self.host, self.port = host, port
        self.state, self.gate = RuntimeState(), ReadOnlySafetyGate()
        self.renderer = SixPanelRenderer(self.state)
        self.qwen = QwenClient(qwen_url, qwen_model) if qwen_url else None
        self.camera_parent, self.camera_translation, self.camera_rpy = camera_parent, (camera_x, camera_y, camera_z), (camera_roll, camera_pitch, camera_yaw)
        self.state.set_calibration(
            parent_frame=camera_parent,
            translation_m=[camera_x, camera_y, camera_z],
            rpy_rad=[camera_roll, camera_pitch, camera_yaw],
            source="PHYSICAL_NAV_CAMERA_X/Y/Z/ROLL/PITCH/YAW",
            calibrated=any(abs(value) > 1e-12 for value in (camera_x, camera_y, camera_z, camera_roll, camera_pitch, camera_yaw)),
        )
        self.qwen_auto_interval = qwen_auto_interval
        self.http: ThreadingHTTPServer | None = None

    def receive(self, packet: dict[str, Any]) -> dict[str, Any]:
        validate_packet(packet)
        self.state.link_packet(str(packet.get("type", "")))
        if packet.get("type") == "hello":
            self.state.link_connected(packet)
            return {"type": "ack", "v": 1, "accepted": True, "read_only": True, "capabilities": ["rgb", "depth", "camera_info", "pose", "telemetry"]}
        if packet.get("type") in {"control", "cmd", "lidar", "posture", "speak", "teleop_intent"}:
            # Defensive boundary: even if an old policy client connects to this
            # port, no command is forwarded to Go2 or any local actuator.
            return self.gate.handle_intent(packet)
        if packet["type"] == "sensor_frame":
            rgb = _decode(packet["rgb"]["data"], "jpeg"); depth = _decode(packet["depth"]["data"], "png16")
            self.state.update_frame(rgb=rgb, depth=depth, rgb_b64=packet["rgb"]["data"], depth_b64=packet["depth"]["data"], depth_scale=float(packet.get("depth_scale", .001)), intrinsics=packet.get("intrinsics", {}), camera_frame=packet.get("camera_frame", ""), frame_seq=int(packet["seq"]), frame_stamp=float(packet["stamp"]), sync_ms=packet.get("color_depth_sync_ms"))
            # ROS publication is handled by physical_ros_gateway.py using the
            # /api/raw-frame endpoint. Keeping this process ROS-free avoids a
            # Python 3.13/ROS Noetic runtime conflict.
        elif packet["type"] == "telemetry":
            self.state.update_topic("telemetry", packet.get("telemetry", {}))
        return {"type": "ack", "v": 1, "seq": packet.get("seq", -1), "accepted": True, "read_only": True}

    def start_http(self) -> None:
        handler = type("PhysicalWebHandler", (_WebHandler,), {})
        handler.state, handler.renderer, handler.gate = self.state, self.renderer, self.gate
        def submit_qwen(payload: dict[str, Any]) -> dict[str, Any]:
            if self.qwen is None: return {"accepted": False, "error": "Qwen client is disabled"}
            user_prompt = str(payload.get("prompt", "请分析当前全局语义图与感知一致性"))
            snapshot = self.state.snapshot()
            context = _compact_qwen_context(snapshot)
            prompt = user_prompt + "\n\n当前实物平台状态(JSON)：\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            request = {"prompt": user_prompt, "context": context, "requested_at": time.time(), "model": self.qwen.model}
            self.state.add_qwen(request)
            def worker() -> None:
                result = self.qwen.chat(prompt, max_tokens=int(payload.get("max_tokens", 256)))
                self.state.add_qwen({}, {"completed_at": time.time(), "prompt": user_prompt, "result": result})
            threading.Thread(target=worker, daemon=True).start()
            return {"accepted": True, "queued_at": request["requested_at"]}
        # Store as a static callback; otherwise BaseHTTPRequestHandler binds
        # this closure as an instance method and adds an unwanted ``self``.
        handler.qwen_submit = staticmethod(submit_qwen)
        self.http = ThreadingHTTPServer((self.host, self.port), handler)
        def render_loop() -> None:
            while True:
                try:
                    frame = self.renderer.render()
                    with handler.frame_lock: handler.latest_jpeg = frame
                except Exception as exc:
                    self.state.last_error = str(exc)
                time.sleep(.1)
        threading.Thread(target=render_loop, daemon=True).start()
        if self.qwen is not None and self.qwen_auto_interval > 0:
            def qwen_loop() -> None:
                while True:
                    time.sleep(self.qwen_auto_interval)
                    snapshot = self.state.snapshot()
                    compact = _compact_qwen_context(snapshot)
                    prompt = "请根据当前全局语义图、检测结果和一致性诊断，简要报告空间关系异常和需要人工确认的对象。\n" + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
                    request = {"prompt": prompt, "requested_at": time.time(), "model": self.qwen.model, "source": "auto_graph_review"}
                    self.state.add_qwen(request)
                    result = self.qwen.chat(prompt, max_tokens=256)
                    self.state.add_qwen({}, {"completed_at": time.time(), "source": "auto_graph_review", "result": result})
            threading.Thread(target=qwen_loop, daemon=True).start()
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        print(f"physical six-panel web: http://{self.host}:{self.port}/", flush=True)


async def run_gateway(args: argparse.Namespace) -> None:
    import websockets
    gateway = PhysicalGateway(args.http_host, args.http_port, args.qwen_url, args.qwen_model, args.camera_parent, args.camera_x, args.camera_y, args.camera_z, args.camera_roll, args.camera_pitch, args.camera_yaw, args.qwen_auto_interval)
    gateway.start_http()
    async def handler(websocket: Any) -> None:
        try:
            async for raw in websocket:
                try:
                    packet = json.loads(raw); reply = gateway.receive(packet)
                except Exception as exc:
                    reply = {"type": "error", "accepted": False, "error": str(exc)}
                await websocket.send(json.dumps(reply, separators=(",", ":")))
        except websockets.exceptions.ConnectionClosed:
            # A browser/Go2 reconnect or a normal process shutdown can close
            # without a WebSocket close frame; it is not a sensor error.
            return
        finally:
            gateway.state.link_disconnected()
    async with websockets.serve(handler, args.ws_host, args.ws_port, max_size=args.max_message_mb * 1024 * 1024):
        print(f"physical sensor WebSocket: ws://{args.ws_host}:{args.ws_port}", flush=True)
        await __import__("asyncio").Future()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ws-host", default="0.0.0.0"); p.add_argument("--ws-port", type=int, default=12334)
    p.add_argument("--http-host", default="0.0.0.0"); p.add_argument("--http-port", type=int, default=8765)
    p.add_argument("--max-message-mb", type=int, default=16)
    p.add_argument("--qwen-url", default="", help="e.g. http://127.0.0.1:18080/v1 after SSH forwarding")
    p.add_argument("--qwen-model", default="qwen3.6-35b-a3b-fp8")
    p.add_argument("--qwen-auto-interval", type=float, default=0.0, help="seconds; 0 disables periodic graph review")
    p.add_argument("--camera-parent", default="tf_frame_base_link")
    p.add_argument("--camera-x", type=float, default=0.0); p.add_argument("--camera-y", type=float, default=0.0); p.add_argument("--camera-z", type=float, default=0.0); p.add_argument("--camera-roll", type=float, default=0.0); p.add_argument("--camera-pitch", type=float, default=0.0); p.add_argument("--camera-yaw", type=float, default=0.0)
    args = p.parse_args()
    try:
        import asyncio; asyncio.run(run_gateway(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__": main()
