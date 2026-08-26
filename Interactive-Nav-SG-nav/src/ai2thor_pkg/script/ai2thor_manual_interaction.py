#!/usr/bin/env python3
"""
AI2-THOR 手动交互录制脚本（不依赖 ROS）。

特点：
1) 键盘手动控制移动和交互
2) 自动录制左右拼接视频（左主视角，右 top-view）
3) 视频中叠加交互动作标注（动作名 + 成功/失败）
"""

import argparse
import queue
import json
import os
import sys
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np

import ai2thor.controller

try:
    import prior
except Exception:
    prior = None


def parse_args():
    default_output = os.path.join(
        os.getcwd(),
        "ai2thor_demo_output",
        datetime.now().strftime("%Y%m%d_%H%M%S_manual"),
    )
    parser = argparse.ArgumentParser(description="AI2-THOR manual interaction recorder")
    parser.add_argument("--scene-type", choices=["floorplan", "procthor"], default="floorplan")
    parser.add_argument("--scene", type=str, default="FloorPlan1")
    parser.add_argument("--procthor-dataset", type=str, default="procthor-10k")
    parser.add_argument("--procthor-split", type=str, default="train")
    parser.add_argument("--procthor-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=str, default=default_output)
    return parser.parse_args()


class ManualRecorder:
    def __init__(self, args):
        self.args = args
        os.makedirs(args.output_dir, exist_ok=True)
        self.action_log = []
        self.last_action_text = "READY"
        self.last_action_until = time.time()
        self.step_count = 0
        self.selected_object_id = None
        self.selected_object_name = "None"
        self.running = True
        self.key_queue: "queue.Queue[str]" = queue.Queue()
        self.controller = self._build_controller()
        self.event = self.controller.step(action="Pass")
        self._setup_top_view_camera()

        video_path = os.path.join(args.output_dir, "manual_side_by_side.mp4")
        self.video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.width * 2, args.height),
        )
        self.video_path = video_path
        self._start_terminal_key_listener()

    def _build_controller(self):
        scene = self.args.scene
        if self.args.scene_type == "procthor":
            if prior is None:
                raise RuntimeError("prior 未安装，请先执行: pip install prior")
            dataset = prior.load_dataset(self.args.procthor_dataset)
            scene = dataset[self.args.procthor_split][self.args.procthor_index]
        return ai2thor.controller.Controller(
            scene=scene,
            width=self.args.width,
            height=self.args.height,
            fieldOfView=self.args.fov,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            continuous=True,
            headless=self.args.headless,
            makeAgentsVisible=True,
        )

    def _visible_objects(self):
        objs = self.event.metadata.get("objects", [])
        return [o for o in objs if o.get("visible", False)]

    def _setup_top_view_camera(self):
        """Add map-view third-party camera for top-view recording."""
        event = self.controller.step(action="GetMapViewCameraProperties")
        if event.metadata.get("lastActionSuccess", False) and "actionReturn" in event.metadata:
            self.event = self.controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])
            print("已启用 top-view 录制")
        else:
            print("top-view 相机添加失败，将仅录制主视角视频")

    def _top_view_bgr(self):
        """Read latest top-view frame from third-party camera."""
        frame = None
        if hasattr(self.event, "third_party_camera_frames") and self.event.third_party_camera_frames:
            frame = self.event.third_party_camera_frames[-1]
        elif (
            hasattr(self.event, "events")
            and self.event.events
            and hasattr(self.event.events[0], "third_party_camera_frames")
            and self.event.events[0].third_party_camera_frames
        ):
            frame = self.event.events[0].third_party_camera_frames[-1]
        if frame is None:
            return np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)
        if frame.shape[1] != self.args.width or frame.shape[0] != self.args.height:
            frame = cv2.resize(frame, (self.args.width, self.args.height))
        # Ensure writable image for OpenCV drawing functions.
        return np.ascontiguousarray(frame).copy()

    def _find_visible(self, pred: Callable[[dict], bool]) -> Optional[dict]:
        candidates = [o for o in self._visible_objects() if pred(o)]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x.get("distance", 9999.0))
        return candidates[0]

    def _start_terminal_key_listener(self):
        """Terminal fallback: type key then Enter (e.g. w + Enter)."""
        if not sys.stdin or not sys.stdin.isatty():
            return

        def _worker():
            while self.running:
                try:
                    raw = input().strip().lower()
                except EOFError:
                    break
                if not raw:
                    continue
                self.key_queue.put(raw[0])

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _read_key(self) -> str:
        """Read key from OpenCV window first, then terminal fallback."""
        key_code = cv2.waitKey(1) & 0xFF
        if key_code != 255 and 32 <= key_code <= 126:
            return chr(key_code).lower()
        try:
            return self.key_queue.get_nowait()
        except queue.Empty:
            return ""

    def _set_action_text(self, text: str, hold: float = 2.0):
        self.last_action_text = text
        self.last_action_until = time.time() + hold

    def _log_action(self, action: str, success: bool, detail: str = "", error: str = ""):
        self.action_log.append(
            {
                "time": time.time(),
                "action": action,
                "success": bool(success),
                "detail": detail,
                "error": error,
            }
        )

    def do(self, action: str, detail: str = "", **kwargs):
        event = self.controller.step(action=action, **kwargs)
        success = bool(event.metadata.get("lastActionSuccess", False))
        error = event.metadata.get("errorMessage", "") if not success else ""
        status = "SUCCESS" if success else "FAILED"
        msg = f"{action} | {status}"
        if detail:
            msg += f" | {detail}"
        if error:
            msg += f" | {error}"
        print(msg)
        self._set_action_text(msg)
        self._log_action(action, success, detail=detail, error=error)
        self.event = event
        return event, success

    def _collect_interactable_hints(self, max_items: int = 6):
        """Collect current visible interactable objects for on-screen hints."""
        hints = []
        for o in self._visible_objects():
            tags = []
            if o.get("openable", False):
                tags.append("open/close")
            if o.get("pickupable", False):
                tags.append("pickup")
            if o.get("toggleable", False):
                tags.append("toggle")
            if o.get("breakable", False):
                tags.append("break")
            if o.get("sliceable", False):
                tags.append("slice")
            if o.get("dirtyable", False):
                tags.append("clean")
            if o.get("canFillWithLiquid", False):
                tags.append("fill")
            if o.get("isFilledWithLiquid", False):
                tags.append("empty")
            if o.get("receptacle", False):
                tags.append("put")

            if not tags:
                continue
            name = o.get("name", o.get("objectType", o.get("objectId", "object")))
            dist = o.get("distance", -1.0)
            hints.append((dist, f"{name}: {','.join(tags)}"))

        hints.sort(key=lambda x: x[0] if x[0] >= 0 else 9999.0)
        return [h[1][:66] for h in hints[:max_items]]

    def _object_actions(self, obj: dict):
        actions = []
        if obj.get("openable", False):
            actions.extend(["o=open", "c=close"])
        if obj.get("pickupable", False):
            actions.append("p=pickup")
        if obj.get("receptacle", False):
            actions.append("u=put")
        if obj.get("toggleable", False):
            actions.append("t=toggle")
        if obj.get("breakable", False):
            actions.append("b=break")
        if obj.get("sliceable", False):
            actions.append("l=slice")
        if obj.get("dirtyable", False):
            actions.append("r=clean")
        if obj.get("canFillWithLiquid", False):
            actions.append("f=fill")
        if obj.get("isFilledWithLiquid", False):
            actions.append("g=empty")
        return actions

    def _collect_selectable_objects(self, max_items: int = 9):
        items = []
        for o in self._visible_objects():
            acts = self._object_actions(o)
            if not acts:
                continue
            dist = o.get("distance", 9999.0)
            name = o.get("name", o.get("objectType", o.get("objectId", "object")))
            items.append((dist, {
                "objectId": o.get("objectId"),
                "name": name,
                "actions": acts,
                "distance": dist,
            }))
        items.sort(key=lambda x: x[0])
        return [x[1] for x in items[:max_items]]

    def _get_selected_visible_object(self):
        if not self.selected_object_id:
            return None
        for o in self._visible_objects():
            if o.get("objectId") == self.selected_object_id:
                return o
        return None

    def _print_step_actions(self, selectable_objects):
        selected = self._get_selected_visible_object()
        if selected is None:
            self.selected_object_id = None
            self.selected_object_name = "None"
        else:
            self.selected_object_name = selected.get("name", selected.get("objectId", "Unknown"))

        obj_summary = " | ".join(
            [f"{i+1}:{o['name']}({o['distance']:.2f}m)" for i, o in enumerate(selectable_objects)]
        )
        if not obj_summary:
            obj_summary = "No interactable objects visible"

        selected_actions = []
        if selected is not None:
            selected_actions = self._object_actions(selected)
        action_summary = ", ".join(selected_actions) if selected_actions else "None"
        global_actions = "y=throw, h=drop"

        line = (
            f"[Step {self.step_count}] Select[1-9] | Selected: {self.selected_object_name} | "
            f"ObjActions: {action_summary} | Global: {global_actions} | Visible: {obj_summary}"
        )
        print(line)

    def _draw_overlay(self, frame_bgr: np.ndarray, interactable_hints=None):
        cv2.putText(
            frame_bgr,
            "Manual Control: W/S/A/D move, Q/E look, O/C/P/U/T/B/L/R/F/G/Y/H interaction, X exit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        if time.time() <= self.last_action_until:
            cv2.rectangle(frame_bgr, (8, 38), (self.args.width - 8, 78), (20, 20, 20), -1)
            cv2.putText(
                frame_bgr,
                f"ACTION: {self.last_action_text}"[:140],
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

        hints = interactable_hints or []
        box_top = 86
        box_bottom = min(self.args.height - 8, box_top + 22 + 20 * (len(hints) + 1))
        cv2.rectangle(frame_bgr, (8, box_top), (self.args.width - 8, box_bottom), (20, 20, 20), -1)
        cv2.putText(
            frame_bgr,
            "INTERACTABLE NOW:",
            (15, box_top + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if hints:
            for i, line in enumerate(hints, start=1):
                cv2.putText(
                    frame_bgr,
                    f"- {line}",
                    (15, box_top + 18 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        else:
            cv2.putText(
                frame_bgr,
                "- none visible",
                (15, box_top + 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _frame_bgr(self):
        rgb = getattr(self.event, "frame", None)
        if rgb is None:
            return np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _inventory(self):
        return self.event.metadata.get("inventoryObjects", [])

    def _print_visible(self):
        print("\n=== 当前可见物体（最近 12 个）===")
        objs = self._visible_objects()
        for o in objs[:12]:
            print(
                f"- {o.get('name', o.get('objectId'))} | "
                f"type={o.get('objectType')} | dist={o.get('distance', -1):.2f}"
            )
        print("================================\n")

    def _do_interaction_by_key(self, key: str):
        selected = self._get_selected_visible_object()
        if key in {"y", "h"}:
            if key == "y":
                if self._inventory():
                    self.do("ThrowObject", detail="inventory", moveMagnitude=8.0, forceAction=True)
                else:
                    self._set_action_text("ThrowObject | FAILED | No held object")
            elif key == "h":
                if self._inventory():
                    self.do("DropHandObject", detail="inventory", forceAction=True)
                else:
                    self._set_action_text("DropHandObject | FAILED | No held object")
            return

        if selected is None:
            self._set_action_text("FAILED | Please select object first (keys 1-9)")
            return

        oid = selected.get("objectId")
        if key == "o":
            if selected.get("openable", False):
                self.do("OpenObject", detail=oid, objectId=oid, openness=1.0, forceAction=True)
            else:
                self._set_action_text("OpenObject | FAILED | Selected object not openable")
        elif key == "c":
            if selected.get("openable", False):
                self.do("CloseObject", detail=oid, objectId=oid, forceAction=True)
            else:
                self._set_action_text("CloseObject | FAILED | Selected object not openable")
        elif key == "p":
            if selected.get("pickupable", False):
                self.do("PickupObject", detail=oid, objectId=oid, forceAction=True)
            else:
                self._set_action_text("PickupObject | FAILED | Selected object not pickupable")
        elif key == "u":
            if self._inventory() and selected.get("receptacle", False):
                self.do("PutObject", detail=oid, objectId=oid, forceAction=True)
            else:
                self._set_action_text("PutObject | FAILED | Need held object + receptacle target")
        elif key == "t":
            if selected.get("toggleable", False):
                if selected.get("isToggled", False):
                    self.do("ToggleObjectOff", detail=oid, objectId=oid, forceAction=True)
                else:
                    self.do("ToggleObjectOn", detail=oid, objectId=oid, forceAction=True)
            else:
                self._set_action_text("ToggleObject | FAILED | Selected object not toggleable")
        elif key == "b":
            if selected.get("breakable", False):
                self.do("BreakObject", detail=oid, objectId=oid, forceAction=True)
            else:
                self._set_action_text("BreakObject | FAILED | Selected object not breakable")
        elif key == "l":
            if selected.get("sliceable", False):
                self.do("SliceObject", detail=oid, objectId=oid, forceAction=True)
            else:
                self._set_action_text("SliceObject | FAILED | Selected object not sliceable")
        elif key == "r":
            if selected.get("dirtyable", False):
                self.do("CleanObject", detail=oid, objectId=oid, forceAction=True)
            else:
                self._set_action_text("CleanObject | FAILED | Selected object not dirtyable")
        elif key == "f":
            if selected.get("canFillWithLiquid", False):
                self.do(
                    "FillObjectWithLiquid",
                    detail=oid,
                    objectId=oid,
                    fillLiquid="water",
                    forceAction=True,
                )
            else:
                self._set_action_text("FillObjectWithLiquid | FAILED | Selected object not fillable")
        elif key == "g":
            if selected.get("isFilledWithLiquid", False):
                self.do("EmptyLiquidFromObject", detail=oid, objectId=oid, forceAction=True)
            else:
                self._set_action_text("EmptyLiquidFromObject | FAILED | Selected object is not filled")

    def run(self):
        print("=== Manual Control Help ===")
        print("Move: W forward / S back / A left turn / D right turn / Q look up / E look down")
        print("Select object: keys 1-9 from visible interactable list in terminal")
        print("Interact with selected object: O open / C close / P pickup / U put / T toggle / B break")
        print("                              L slice / R clean / F fill / G empty")
        print("Global interact: Y throw held object / H drop held object")
        print("Other: I print visible objects / X exit")
        print("Tip: Click video window first; fallback is terminal input + Enter.")
        print("===========================")

        cv2.namedWindow("AI2-THOR Manual Interaction (Left: Main | Right: Top)", cv2.WINDOW_NORMAL)

        while True:
            self.event = self.controller.step(action="Pass")
            self.step_count += 1
            selectable_objects = self._collect_selectable_objects()
            self._print_step_actions(selectable_objects)
            interactable_hints = self._collect_interactable_hints()
            frame_bgr = self._frame_bgr()
            self._draw_overlay(frame_bgr, interactable_hints)

            top_view_bgr = self._top_view_bgr()
            self._draw_overlay(top_view_bgr, interactable_hints)

            cv2.putText(
                frame_bgr,
                "Main View",
                (10, self.args.height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                top_view_bgr,
                "Top View",
                (10, self.args.height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            side_by_side = np.hstack([frame_bgr, top_view_bgr])
            self.video_writer.write(side_by_side)
            cv2.imshow("AI2-THOR Manual Interaction (Left: Main | Right: Top)", side_by_side)

            key = self._read_key()
            if not key:
                continue

            if key == "x":
                self._set_action_text("Exit recording")
                break
            if key == "i":
                self._print_visible()
                continue

            if key in "123456789":
                idx = int(key) - 1
                if 0 <= idx < len(selectable_objects):
                    obj = selectable_objects[idx]
                    self.selected_object_id = obj["objectId"]
                    self.selected_object_name = obj["name"]
                    self._set_action_text(f"Selected: {self.selected_object_name}")
                else:
                    self._set_action_text("Select FAILED | index out of range")
                continue

            movement = {
                "w": ("MoveAhead", {"moveMagnitude": 0.25}),
                "s": ("MoveBack", {"moveMagnitude": 0.25}),
                "a": ("RotateLeft", {"degrees": 30}),
                "d": ("RotateRight", {"degrees": 30}),
                "q": ("LookUp", {"degrees": 30}),
                "e": ("LookDown", {"degrees": 30}),
            }
            if key in movement:
                action_name, kwargs = movement[key]
                self.do(action_name, **kwargs)
                continue

            if key in {"o", "c", "p", "u", "t", "b", "l", "r", "f", "g", "y", "h"}:
                self._do_interaction_by_key(key)

    def close(self):
        self.running = False
        self.video_writer.release()
        self.controller.stop()
        cv2.destroyAllWindows()
        with open(os.path.join(self.args.output_dir, "manual_action_log.json"), "w", encoding="utf-8") as f:
            json.dump(self.action_log, f, indent=2, ensure_ascii=False)
        print("Side-by-side video saved:", self.video_path)
        print("Action log saved:", os.path.join(self.args.output_dir, "manual_action_log.json"))


def main():
    args = parse_args()
    print("Output directory:", args.output_dir)
    demo = ManualRecorder(args)
    try:
        demo.run()
    finally:
        demo.close()


if __name__ == "__main__":
    main()
