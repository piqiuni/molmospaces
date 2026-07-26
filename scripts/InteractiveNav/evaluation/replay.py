"""Initial-state replay validation for InteractiveNav V3 episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# InteractiveNav collection runs headless with EGL.  Set this before importing
# any module that can create a MuJoCo renderer so replay works on a worker node
# without an X11 DISPLAY.  Users may still explicitly override it beforehand.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import interactive_nav_v3

from .compatibility import compatible_episode_payload
from .metrics import joint_open_fraction, target_metrics


def _scene_args(episode: dict[str, Any], output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        scene_dataset=str(episode["scene_dataset"]),
        data_split=str(episode["data_split"]),
        robot=str(episode["robot"]["robot_name"]),
        variant="base",
        seed=0,
        output_dir=output_dir,
    )


def replay_initial_state(episode: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Load a scene once and prove recorded V3 initial articulation state."""

    interactive_nav_v3.validate_interactive_nav_v3_episode(
        episode, expected_domains=list(episode["interactive_nav"]["interaction_domains"])
    )
    episode, compatibility = compatible_episode_payload(episode, output_dir / "compatibility")
    ctx = probe.load_scene_context(_scene_args(episode, output_dir), int(episode["house_index"]))
    try:
        state_application = probe.apply_episode_scene_state(ctx.env, episode)
        expected = {
            str(row["joint_name"]): float(row["position"])
            for row in episode.get("scene_modifications", {}).get("articulation_states", [])
        }
        actual = {name: probe.joint_value_by_name(ctx.env, name) for name in expected}
        max_error = max((abs(actual[name] - value) for name, value in expected.items()), default=0.0)
        interactions = episode["interactive_nav"].get("interactions", [])
        fractions = {row["interaction_id"]: joint_open_fraction(ctx.env, row) for row in interactions}
        initial_expected = {
            row["interaction_id"]: float(row["initial_state"]["joint_fraction"])
            for row in interactions
        }
        fraction_error = max(
            (abs(fractions[key] - value) for key, value in initial_expected.items()), default=0.0
        )
        target_name = episode["interactive_nav"]["target"]["selected_instance"]
        visibility = float(ctx.env.check_visibility("head_camera", target_name))
        return {
            "schema_version": "interactive_nav_v3_replay_validation_v1",
            "case_id": episode["interactive_nav"]["case_id"],
            "house_index": int(episode["house_index"]),
            "state_application": state_application,
            "runtime_compatibility": compatibility,
            "articulation_max_abs_error": max_error,
            "interaction_fraction_max_abs_error": fraction_error,
            "interaction_fractions": fractions,
            "target_initial_visibility_fraction": visibility,
            "passed": bool(max_error <= 1e-6 and fraction_error <= 1e-6),
        }
    finally:
        probe.close_context(ctx)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--episode-indices", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    episodes = json.loads(args.benchmark.read_text())
    rows = []
    for index in args.episode_indices:
        rows.append(replay_initial_state(episodes[index], args.output_dir / f"episode_{index:04d}"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "replay_validation.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(output)
    return 0 if all(row["passed"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
