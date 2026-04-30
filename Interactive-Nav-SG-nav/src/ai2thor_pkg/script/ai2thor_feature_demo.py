#!/usr/bin/env python3
"""
AI2-THOR feature demo script (no ROS required).

What it demonstrates:
1) Scene loading (FloorPlan or ProcTHOR)
2) Agent movement and camera control
3) Reachable points and teleport navigation
4) RGB / Depth / Instance segmentation / Top view capture
5) Automatic video export for presentation recording
"""

import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

import ai2thor.controller

try:
    import prior
except Exception:
    prior = None


class VideoRecorder:
    def __init__(self, output_dir: str, fps: int = 12):
        self.output_dir = output_dir
        self.fps = fps
        self.writers: Dict[str, cv2.VideoWriter] = {}
        self.action_log: List[dict] = []
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_writer(self, name: str, frame: np.ndarray) -> cv2.VideoWriter:
        if name in self.writers:
            return self.writers[name]
        h, w = frame.shape[:2]
        path = os.path.join(self.output_dir, f"{name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        self.writers[name] = writer
        return writer

    def write_frame(self, name: str, frame_bgr: np.ndarray):
        writer = self._get_writer(name, frame_bgr)
        writer.write(frame_bgr)

    def _get_primary_event(self, event, active_agent_id: int = 0):
        if hasattr(event, "frame"):
            return event
        if hasattr(event, "events") and event.events:
            idx = min(max(int(active_agent_id), 0), len(event.events) - 1)
            return event.events[idx]
        return event

    def record_event(self, event, note: str = "", active_agent_id: int = 0):
        primary_event = self._get_primary_event(event, active_agent_id)

        rgb = getattr(primary_event, "frame", None)
        if rgb is not None:
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if note:
                cv2.putText(
                    rgb_bgr,
                    note[:80],
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
            self.write_frame("rgb", rgb_bgr)

        if hasattr(primary_event, "depth_frame") and primary_event.depth_frame is not None:
            depth = primary_event.depth_frame.astype(np.float32)
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            self.write_frame("depth", depth_color)

        if hasattr(primary_event, "instance_segmentation_frame") and primary_event.instance_segmentation_frame is not None:
            seg_rgb = primary_event.instance_segmentation_frame
            seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)
            self.write_frame("instance_segmentation", seg_bgr)

        if hasattr(primary_event, "third_party_camera_frames") and primary_event.third_party_camera_frames:
            # AI2-THOR third-party frame is usually BGR
            top_view_bgr = primary_event.third_party_camera_frames[-1]
            self.write_frame("top_view", top_view_bgr)
        elif (
            hasattr(event, "events")
            and event.events
            and hasattr(event.events[0], "third_party_camera_frames")
            and event.events[0].third_party_camera_frames
        ):
            top_view_bgr = event.events[0].third_party_camera_frames[-1]
            self.write_frame("top_view", top_view_bgr)

        # Multi-agent: save each agent's RGB stream if available
        if hasattr(event, "events") and event.events and len(event.events) > 1:
            for i, agent_event in enumerate(event.events):
                if hasattr(agent_event, "frame") and agent_event.frame is not None:
                    agent_rgb_bgr = cv2.cvtColor(agent_event.frame, cv2.COLOR_RGB2BGR)
                    self.write_frame(f"rgb_agent_{i}", agent_rgb_bgr)

    def log_action(self, action: str, success: bool, error: str = "", extra: Optional[dict] = None):
        item = {
            "time": time.time(),
            "action": action,
            "success": bool(success),
            "error": error,
            "extra": extra or {},
        }
        self.action_log.append(item)

    def close(self):
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()
        self.compose_multi_agent_grid_video()
        with open(os.path.join(self.output_dir, "action_log.json"), "w", encoding="utf-8") as f:
            json.dump(self.action_log, f, indent=2)

    def compose_multi_agent_grid_video(self):
        """Compose rgb_agent_*.mp4 into a grid video for presentation."""
        agent_files = sorted(
            [
                os.path.join(self.output_dir, f)
                for f in os.listdir(self.output_dir)
                if f.startswith("rgb_agent_") and f.endswith(".mp4")
            ]
        )
        if len(agent_files) < 2:
            return

        captures = [cv2.VideoCapture(p) for p in agent_files]
        try:
            if not captures or not captures[0].isOpened():
                return

            base_w = int(captures[0].get(cv2.CAP_PROP_FRAME_WIDTH)) or 720
            base_h = int(captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = captures[0].get(cv2.CAP_PROP_FPS) or float(self.fps)
            n = len(captures)
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))
            out_w = cols * base_w
            out_h = rows * base_h

            out_path = os.path.join(self.output_dir, "multi_agent_grid.mp4")
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (out_w, out_h),
            )

            while True:
                frames = []
                for cap in captures:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        writer.release()
                        print("Multi-agent grid video generated:", out_path)
                        return
                    if frame.shape[1] != base_w or frame.shape[0] != base_h:
                        frame = cv2.resize(frame, (base_w, base_h))
                    frames.append(frame)

                canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                for i, frame in enumerate(frames):
                    r = i // cols
                    c = i % cols
                    y0, y1 = r * base_h, (r + 1) * base_h
                    x0, x1 = c * base_w, (c + 1) * base_w
                    canvas[y0:y1, x0:x1] = frame
                    cv2.putText(
                        canvas,
                        f"Agent {i}",
                        (x0 + 10, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                writer.write(canvas)
        finally:
            for cap in captures:
                cap.release()


class AI2ThorFeatureDemo:
    def __init__(self, args):
        self.args = args
        self.active_agent_id = 0
        self.controller = self._build_controller()
        self.recorder = VideoRecorder(args.output_dir, fps=args.fps)
        self.last_event = self.controller.step(action="Pass")
        self._add_top_view_camera()
        self._snapshot("init")

    def _build_controller(self):
        scene = self.args.scene
        if self.args.scene_type == "procthor":
            if prior is None:
                raise RuntimeError("prior is not installed. Please `pip install prior`.")
            dataset = prior.load_dataset(self.args.procthor_dataset)
            scene = dataset[self.args.procthor_split][self.args.procthor_index]

        return ai2thor.controller.Controller(
            scene=scene,
            width=self.args.width,
            height=self.args.height,
            fieldOfView=self.args.fov,
            agentCount=self.args.agent_count,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            continuous=True,
            headless=self.args.headless,
            visibilityDistance=30.0,
            makeAgentsVisible=True,
        )

    def _add_top_view_camera(self):
        event = self.controller.step(action="GetMapViewCameraProperties")
        if event.metadata.get("lastActionSuccess", False):
            self.last_event = self.controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])
        else:
            self.last_event = event

    def _snapshot(self, note: str):
        self.last_event = self.controller.step(action="Pass", agentId=self.active_agent_id)
        self.recorder.record_event(self.last_event, note, active_agent_id=self.active_agent_id)
        self._record_wait(self.args.snapshot_wait, f"{note} (snapshot)")

    def _record_wait(self, duration: float, note: str = ""):
        if duration <= 0:
            return
        if not self.args.record_during_wait:
            time.sleep(duration)
            return

        interval = 1.0 / max(1e-6, self.args.wait_capture_fps)
        end_time = time.time() + duration
        while time.time() < end_time:
            self.last_event = self.controller.step(action="Pass", agentId=self.active_agent_id)
            self.recorder.record_event(self.last_event, note, active_agent_id=self.active_agent_id)
            remain = end_time - time.time()
            if remain <= 0:
                break
            time.sleep(min(interval, remain))

    def _do(self, action: str, note: str = "", wait: Optional[float] = None, **kwargs):
        if "agentId" not in kwargs:
            kwargs["agentId"] = self.active_agent_id
        event = self.controller.step(action=action, **kwargs)
        success = bool(event.metadata.get("lastActionSuccess", False))
        error = event.metadata.get("errorMessage", "") if not success else ""
        self.recorder.log_action(action, success, error, extra=kwargs)
        self.recorder.record_event(event, note or action, active_agent_id=self.active_agent_id)
        self.last_event = event
        msg = f"[{action}] success={success}"
        if not success and error:
            msg += f" | error={error}"
        print(msg)
        delay = self.args.step_wait if wait is None else wait
        self._record_wait(delay, f"{note or action} (hold)")
        return event

    def _objects(self) -> List[dict]:
        return self.last_event.metadata.get("objects", [])

    def _inventory(self) -> List[dict]:
        return self.last_event.metadata.get("inventoryObjects", [])

    def _find_object(self, predicate: Callable[[dict], bool], visible_first: bool = True) -> Optional[dict]:
        objs = self._objects()
        if visible_first:
            visible = [o for o in objs if o.get("visible", False)]
            for obj in visible:
                if predicate(obj):
                    return obj
        for obj in objs:
            if predicate(obj):
                return obj
        return None

    def _print_scene_summary(self):
        objs = self._objects()
        visible = [o for o in objs if o.get("visible", False)]
        print("\n=== Scene summary ===")
        print(f"scene: {self.args.scene_type} | objects(total/visible): {len(objs)}/{len(visible)}")
        for o in visible[:12]:
            print(
                f"- {o.get('name', o['objectId'])} | type={o.get('objectType')} "
                f"| dist={o.get('distance', -1):.2f} | id={o.get('objectId')}"
            )
        print("=====================\n")

    def demo_movement_and_camera(self):
        print("\n[Demo] movement and camera")
        self._do("MoveAhead", moveMagnitude=0.25)
        self._do("MoveBack", moveMagnitude=0.25)
        self._do("MoveLeft", moveMagnitude=0.25)
        self._do("MoveRight", moveMagnitude=0.25)
        self._do("RotateLeft", degrees=45)
        self._do("RotateRight", degrees=45)
        self._do("LookUp", degrees=30)
        self._do("LookDown", degrees=30)
        self._do("RotateLook", horizon=0)

    def demo_navigation(self):
        print("\n[Demo] navigation")
        event = self._do("GetReachablePositions")
        positions = event.metadata.get("actionReturn", []) or []
        print(f"reachable positions: {len(positions)}")
        if positions:
            for idx in random.sample(range(len(positions)), k=min(3, len(positions))):
                self._do("Teleport", position=positions[idx], forceAction=True)

    def demo_multi_agent(self):
        if self.args.agent_count <= 1:
            return
        print(f"\n[Demo] multi-agent (count={self.args.agent_count})")
        for agent_id in range(self.args.agent_count):
            self.active_agent_id = agent_id
            self._do("RotateRight", degrees=45, note=f"agent {agent_id} rotate")
            self._do("MoveAhead", moveMagnitude=0.25, note=f"agent {agent_id} move")
        self.active_agent_id = 0

    def _pickup_some_object(self) -> bool:
        pickup_obj = self._find_object(lambda o: o.get("pickupable", False))
        if not pickup_obj:
            return False
        event = self._do("PickupObject", objectId=pickup_obj["objectId"], forceAction=True)
        return bool(event.metadata.get("lastActionSuccess", False))

    def demo_interactions(self):
        print("\n[Demo] object interactions")
        interaction_wait = self.args.interaction_wait

        def interact(action_name: str, **kwargs):
            return self._do(action_name, wait=interaction_wait, **kwargs)

        # Open / Close
        openable = self._find_object(lambda o: o.get("openable", False))
        if openable:
            interact("OpenObject", objectId=openable["objectId"], openness=1.0, forceAction=True)
            interact("CloseObject", objectId=openable["objectId"], forceAction=True)

        # Toggle
        toggleable = self._find_object(lambda o: o.get("toggleable", False))
        if toggleable:
            interact("ToggleObjectOn", objectId=toggleable["objectId"], forceAction=True)
            interact("ToggleObjectOff", objectId=toggleable["objectId"], forceAction=True)

        # Pickup / Put / Throw / Drop
        if self._pickup_some_object():
            receptacle = self._find_object(lambda o: o.get("receptacle", False) and not o.get("pickupable", False))
            if receptacle:
                interact("PutObject", objectId=receptacle["objectId"], forceAction=True)
            if self._pickup_some_object():
                interact("ThrowObject", moveMagnitude=8.0, forceAction=True)

        if self._pickup_some_object():
            interact("DropHandObject", forceAction=True)

        # Fill / Empty
        fillable = self._find_object(lambda o: o.get("canFillWithLiquid", False))
        if fillable:
            interact("FillObjectWithLiquid", objectId=fillable["objectId"], fillLiquid="water", forceAction=True)
            interact("EmptyLiquidFromObject", objectId=fillable["objectId"], forceAction=True)

        # Dirty / Clean
        dirtyable = self._find_object(lambda o: o.get("dirtyable", False))
        if dirtyable:
            interact("DirtyObject", objectId=dirtyable["objectId"], forceAction=True)
            interact("CleanObject", objectId=dirtyable["objectId"], forceAction=True)

        # Break
        breakable = self._find_object(lambda o: o.get("breakable", False))
        if breakable:
            interact("BreakObject", objectId=breakable["objectId"], forceAction=True)

        # Slice (needs knife in many scenes; try anyway)
        sliceable = self._find_object(lambda o: o.get("sliceable", False))
        if sliceable:
            interact("SliceObject", objectId=sliceable["objectId"], forceAction=True)

        # Cook
        cookable = self._find_object(lambda o: o.get("cookable", False))
        if cookable:
            interact("CookObject", objectId=cookable["objectId"], forceAction=True)

        # Push / Pull
        moveable = self._find_object(lambda o: o.get("moveable", False))
        if moveable:
            interact("PushObject", objectId=moveable["objectId"], moveMagnitude=80.0, forceAction=True)
            interact("PullObject", objectId=moveable["objectId"], moveMagnitude=80.0, forceAction=True)

        # Use up
        usable = self._find_object(lambda o: o.get("canBeUsedUp", False))
        if usable:
            interact("UseUpObject", objectId=usable["objectId"], forceAction=True)

        # Create object (may fail depending on task configuration)
        interact("CreateObject", objectType="Apple", forceAction=True)

    def run(self):
        self._print_scene_summary()
        self.demo_movement_and_camera()
        self.demo_multi_agent()
        self.demo_navigation()
        self._snapshot("final")

    def close(self):
        self.recorder.close()
        self.controller.stop()


def parse_args():
    default_output = os.path.join(
        os.getcwd(),
        "ai2thor_demo_output",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    parser = argparse.ArgumentParser(description="AI2-THOR all-in-one feature demo script")
    parser.add_argument("--scene-type", choices=["floorplan", "procthor"], default="floorplan")
    parser.add_argument("--scene", type=str, default="FloorPlan1", help="FloorPlan scene name")
    parser.add_argument("--procthor-dataset", type=str, default="procthor-10k")
    parser.add_argument("--procthor-split", type=str, default="train")
    parser.add_argument("--procthor-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--step-wait", type=float, default=1.0, help="wait seconds after each action")
    parser.add_argument("--snapshot-wait", type=float, default=0.6, help="wait seconds after each snapshot frame")
    parser.add_argument("--interaction-wait", type=float, default=2.2, help="wait seconds after each interaction action")
    parser.add_argument("--wait-capture-fps", type=float, default=6.0, help="sampling fps while waiting")
    parser.add_argument(
        "--no-record-during-wait",
        action="store_true",
        help="if set, waiting period will not record extra frames",
    )
    parser.add_argument("--headless", action="store_true", help="run without Unity window")
    parser.add_argument("--agent-count", type=int, default=1, help="number of agents in scene")
    parser.add_argument("--output-dir", type=str, default=default_output)
    args = parser.parse_args()
    args.record_during_wait = not args.no_record_during_wait
    return args


def main():
    args = parse_args()
    print("Output directory:", args.output_dir)
    demo = AI2ThorFeatureDemo(args)
    try:
        demo.run()
    finally:
        demo.close()
    print("Done. Videos and action log are saved in:", args.output_dir)


if __name__ == "__main__":
    main()
