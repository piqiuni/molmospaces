"""Capture closed/open external-camera pairs for selected gallery scenes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import explore_molmo_interactions as emi
from scripts.InteractiveNav.capture_open_articulation_gallery import (
    TARGETS,
    GalleryTarget,
    choose_container_pose,
    choose_door_pose,
    find_container,
    find_door,
    render_target,
    scene_args,
    set_container_open,
    write_json,
)


CONTACT_SHEET_SELECTION = {
    1: "door_single",
    2: "door_double",
    3: "fridge_5",
    4: "fridge_14",
    7: "drawer_4",
    8: "drawer_6",
    9: "drawer_12",
}
SCENE56_SELECTION = {
    5: "fridge_15",
    6: "fridge_19",
}
DRAWER56_SELECTION = {
    5: "drawer_4",
    6: "drawer_6",
}
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "scripts/InteractiveNav/output/articulation_state_pairs_20260721_side_rear_v2"
)
SCENE56_OUTPUT = REPO_ROOT / "scripts/InteractiveNav/output/articulation_state_pairs_20260722_scene56_right_side"
DRAWER56_OUTPUT = REPO_ROOT / "scripts/InteractiveNav/output/articulation_state_pairs_20260722_drawers56_right_side_initial"


def selected_targets(selection: str) -> tuple[dict[int, str], list[GalleryTarget], str]:
    if selection == "scene56_right":
        contact_selection = SCENE56_SELECTION
        camera_profile = "right_side_rear"
    elif selection == "drawer56_right":
        contact_selection = DRAWER56_SELECTION
        camera_profile = "right_side_rear"
    elif selection == "all":
        contact_selection = CONTACT_SHEET_SELECTION
        camera_profile = "default"
    else:
        raise ValueError(f"Unknown selection: {selection}")
    by_id = {target.gallery_id: target for target in TARGETS}
    targets = []
    for _, gallery_id in sorted(contact_selection.items()):
        if camera_profile == "right_side_rear":
            profile = "right_side_rear_close" if gallery_id == "drawer_4" else camera_profile
        else:
            profile = "rear_shoulder_close" if gallery_id.startswith("drawer_") else "rear_shoulder"
        targets.append(replace(by_id[gallery_id], camera_profile=profile))
    return contact_selection, targets, camera_profile


def set_container_state(
    ctx: probe.LoadedContext,
    rec: dict[str, Any],
    joint_indices: tuple[int, ...],
    state: str,
) -> None:
    probe.set_all_articulation_joints_closed(ctx.env, rec, rec["joints"])
    if state == "open":
        set_container_open(ctx, rec, joint_indices)
    elif state != "closed":
        raise ValueError(f"Unsupported articulation state: {state}")


def restore_initial_robot_posture(ctx: probe.LoadedContext) -> None:
    """Keep every static frame at the reset RBY1 posture, without policy motion."""
    probe.apply_default_arm_pose(ctx.env)
    probe.apply_default_head_pose(ctx.env, ctx.initial_head_qpos)
    probe.apply_default_torso_pose(ctx.env, ctx.initial_torso_qpos)


def capture_house(
    house_index: int,
    targets: list[GalleryTarget],
    args: argparse.Namespace,
    contact_selection: dict[int, str],
) -> list[dict[str, Any]]:
    ctx = None
    rows: list[dict[str, Any]] = []
    try:
        ctx = probe.load_scene_context(scene_args(args.seed, args.variant), house_index)
        records, containers = probe.collect_scene_records(ctx)
        doorway_analysis = None
        doorway_records: list[dict[str, Any]] = []
        if any(target.kind == "door" for target in targets):
            emi.ensure_runtime_dependencies()
            doorway_analysis = emi.collect_runtime_doorway_analysis(ctx.env)
            doorway_records = emi.collect_interactive_door_root_object_records(
                ctx.env,
                doorway_analysis,
            )

        for target in targets:
            if target.kind == "door":
                if doorway_analysis is None:
                    raise RuntimeError("Doorway analysis was not initialized")
                rec = find_door(records, doorway_records, target.asset_id)
                robot_pose, pose_meta = choose_door_pose(ctx, doorway_analysis, rec)
            else:
                rec = find_container(containers, target.asset_id)
                robot_pose, pose_meta = choose_container_pose(
                    ctx,
                    rec,
                    target.joint_indices,
                    target.camera_profile,
                )

            sheet_number = next(
                number
                for number, gallery_id in contact_selection.items()
                if gallery_id == target.gallery_id
            )
            for state in ("closed", "open"):
                restore_initial_robot_posture(ctx)
                if target.kind == "door":
                    rec["door_transition"] = emi.set_door_root_state(
                        ctx.env,
                        doorway_analysis,
                        rec["name"],
                        state,
                    )
                else:
                    set_container_state(ctx, rec, target.joint_indices, state)
                image_path = args.output_dir / (
                    f"{sheet_number:02d}_{target.gallery_id}__{state}__"
                    f"h{house_index}__{target.asset_id.lower()}.png"
                )
                row = render_target(
                    ctx,
                    target,
                    rec,
                    robot_pose,
                    image_path,
                    pose_meta,
                )
                row.update(
                    {
                        "contact_sheet_number": sheet_number,
                        "articulation_state": state,
                        "paired_scene_id": target.gallery_id,
                        "scene_dataset": "procthor-10k",
                        "data_split": "train",
                        "robot_posture": "initial_default_rby1",
                    }
                )
                rows.append(row)
                print(
                    f"captured #{sheet_number} {target.gallery_id} {state}: {image_path}",
                    flush=True,
                )
    finally:
        if ctx is not None:
            probe.close_context(ctx)
    return rows


def save_contact_sheet(
    output_dir: Path,
    rows: list[dict[str, Any]],
    title: str = "Closed → open pairs · right-rear external camera",
) -> Path:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["contact_sheet_number"]),
            0 if row["articulation_state"] == "closed" else 1,
        ),
    )
    row_count = max(1, math.ceil(len(ordered) / 2))
    fig, axes = plt.subplots(row_count, 2, figsize=(14, 4.0 * row_count), squeeze=False)
    for ax, row in zip(axes.reshape(-1), ordered, strict=True):
        ax.imshow(plt.imread(output_dir / row["image"]))
        ax.set_title(
            f"#{row['contact_sheet_number']} {row['gallery_id']} · "
            f"{row['articulation_state']}",
            fontsize=11,
        )
        ax.axis("off")
    for ax in axes.reshape(-1)[len(ordered) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = output_dir / "contact_sheet_closed_open.png"
    fig.savefig(path, dpi=145, facecolor="white")
    plt.close(fig)
    path.chmod(0o644)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="base")
    parser.add_argument(
        "--selection",
        choices=("all", "scene56_right", "drawer56_right"),
        default="all",
        help="Scene subset and camera profile to capture",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output_dir == DEFAULT_OUTPUT:
        if args.selection == "scene56_right":
            args.output_dir = SCENE56_OUTPUT
        elif args.selection == "drawer56_right":
            args.output_dir = DRAWER56_OUTPUT
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contact_selection, targets, camera_profile = selected_targets(args.selection)
    grouped: dict[int, list[GalleryTarget]] = {}
    for target in targets:
        grouped.setdefault(target.house_index, []).append(target)
    rows: list[dict[str, Any]] = []
    for house_index in sorted(grouped):
        rows.extend(capture_house(house_index, grouped[house_index], args, contact_selection))
    if args.selection == "drawer56_right":
        title = "Closed → open pairs · robot-right external camera · drawer scenes #5/#6"
    elif args.selection == "scene56_right":
        title = "Closed → open pairs · robot-right external camera · scenes #5/#6"
    else:
        title = "Closed → open pairs · right-rear external camera"
    contact_sheet = save_contact_sheet(args.output_dir, rows, title)
    manifest = {
        "schema_version": "articulation_state_pairs_v1",
        "source_contact_sheet": str(
            REPO_ROOT
            / (
                "scripts/InteractiveNav/output/articulation_state_pairs_20260721_side_rear_v2/"
                "contact_sheet_closed_open.png"
                if args.selection == "drawer56_right"
                else "scripts/InteractiveNav/output/open_articulation_gallery_20260720_v1/contact_sheet.png"
            )
        ),
        "selected_contact_sheet_numbers": sorted(contact_selection),
        "camera_profiles": {
            "scene_selection": args.selection,
            "profile": camera_profile,
        },
        "state_order": ["closed", "open"],
        "image_count": len(rows),
        "no_policy_rollout": True,
        "robot_posture": "initial_default_rby1",
        "contact_sheet": contact_sheet.name,
        "rows": rows,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "image_count": len(rows),
                "contact_sheet": contact_sheet.name,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    expected_count = 4 if args.selection in {"scene56_right", "drawer56_right"} else 14
    return 0 if len(rows) == expected_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
