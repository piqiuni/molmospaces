#!/usr/bin/env python3
"""
English narration AI2-THOR feature demo script (no ROS required).

Features:
1) Automatically runs common AI2-THOR feature demos
2) Prints narration lines by stage
3) Generates narration_cn.txt for subtitle/voice-over reference
"""

import argparse
import os
import time
from datetime import datetime
from typing import List, Tuple

from ai2thor_feature_demo import AI2ThorFeatureDemo


class AI2ThorFeatureDemoCN(AI2ThorFeatureDemo):
    def __init__(self, args):
        super().__init__(args)
        self.t0 = time.time()
        self.narration_records: List[Tuple[float, str]] = []

    def _sec(self) -> float:
        return time.time() - self.t0

    def say(self, text: str, pause: float = None):
        """Print narration and save timestamps."""
        if pause is None:
            pause = self.args.narration_wait
        t = self._sec()
        self.narration_records.append((t, text))
        print(f"[Narration {t:06.2f}s] {text}")
        # Save one frame with text overlay for easier alignment.
        self._snapshot(text)
        time.sleep(max(0.0, pause))

    def run_with_narration(self):
        # 0. Opening
        self.say("Hello everyone. This is an AI2-THOR feature demonstration. We start with scene initialization.")
        self._print_scene_summary()

        # 1. Visual sensing
        self.say("The output includes RGB, depth, instance segmentation, and top view for perception and debugging.")
        self._snapshot("RGB / Depth / Seg / TopView")

        # 2. Basic movement
        self.say("Next, we demonstrate basic agent motion: forward/back/left/right movement, turning, and camera control.")
        self.demo_movement_and_camera()

        if self.args.agent_count > 1:
            self.say(f"This scene uses {self.args.agent_count} agents. Next is a turn-based multi-agent motion demo.")
            self.demo_multi_agent()

        # 3. Reachable points and navigation
        self.say("Now we query reachable points and navigate by teleporting to different reachable locations.")
        self.demo_navigation()

        # 4. Ending
        self.say("The demo is complete. All steps have been saved as videos and logs for reporting and editing.", pause=0.8)
        self._snapshot("demo finished")

    def save_narration_file(self):
        path = os.path.join(self.args.output_dir, "narration_cn.txt")
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("AI2-THOR narration timeline\n")
            f.write("=" * 36 + "\n")
            for t, text in self.narration_records:
                f.write(f"[{t:06.2f}s] {text}\n")
        print("Narration script saved:", path)


def parse_args():
    default_output = os.path.join(
        os.getcwd(),
        "ai2thor_demo_output",
        datetime.now().strftime("%Y%m%d_%H%M%S_cn"),
    )

    parser = argparse.ArgumentParser(description="AI2-THOR automatic demo script with English narration")
    parser.add_argument("--scene-type", choices=["floorplan", "procthor"], default="floorplan")
    parser.add_argument("--scene", type=str, default="FloorPlan1", help="FloorPlan scene name")
    parser.add_argument("--procthor-dataset", type=str, default="procthor-10k")
    parser.add_argument("--procthor-split", type=str, default="train")
    parser.add_argument("--procthor-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--step-wait", type=float, default=1.0, help="Wait time after each action in seconds")
    parser.add_argument("--snapshot-wait", type=float, default=0.6, help="Wait time after each snapshot in seconds")
    parser.add_argument("--interaction-wait", type=float, default=2.2, help="Wait time after each interaction action")
    parser.add_argument("--wait-capture-fps", type=float, default=6.0, help="Frame sampling FPS during wait periods")
    parser.add_argument("--no-record-during-wait", action="store_true", help="Disable frame recording during wait periods")
    parser.add_argument("--narration-wait", type=float, default=1.8, help="Pause after each narration sentence")
    parser.add_argument("--headless", action="store_true", help="Run without Unity window")
    parser.add_argument("--agent-count", type=int, default=1, help="Number of agents in the scene")
    parser.add_argument("--output-dir", type=str, default=default_output)
    args = parser.parse_args()
    args.record_during_wait = not args.no_record_during_wait
    return args


def main():
    args = parse_args()
    print("Output directory:", args.output_dir)
    demo = AI2ThorFeatureDemoCN(args)
    try:
        demo.run_with_narration()
        demo.save_narration_file()
    finally:
        demo.close()
    print("Done. Videos and narration script are generated.")


if __name__ == "__main__":
    main()
