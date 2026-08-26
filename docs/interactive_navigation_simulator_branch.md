# Interactive navigation simulator branch

`interactive-nav/sim` is based on the current upstream `main` and contains only
simulator-facing interaction support. It intentionally excludes the ROS
navigation/semantic decision implementation in `codex/exp-setting`.

## Interface policy

`InteractiveNavSimulatorPolicy` is a safe demonstration policy. Its default
mode scans the scene and lists available operations without changing scene
state. Explicit `InteractionDemoCommand` entries demonstrate opening a door or
one selected drawer/cabinet joint.

The simulator seam is `SimulatorInteractionInterface`:

- `scan()` lists doors, containers, joints, ranges, positions, and normalized
  open fractions.
- `open_door(name, fraction)` moves only the door hinge.
- `set_joint_open_fraction(name, joint_index, fraction)` moves exactly one
  hinge or slider, including one drawer in a multi-drawer cabinet.

## Real-scene smoke test

`scripts/smoke_test_interaction_interface.py` attaches RBY1 to a local MuJoCo
scene, scans its articulations, and searches around each target for a robot pose
that is collision-free with the target both closed and open. It then checks one
door through `open_door` and every joint of the largest detected container
through `set_joint_open_fraction`. Each target runs through open fractions
`0, 0.5, 1, 0`, and container sibling joints must remain unchanged.

The scene XML must be in a layout where its relative mesh and texture paths
resolve. Run from an environment where this checkout is installed, or set
`PYTHONPATH` to the checkout:

```bash
python scripts/smoke_test_interaction_interface.py \
  --scene-xml <SCENE_XML> \
  --metadata <OPTIONAL_SCENE_METADATA_JSON> \
  --output <RESULT_JSON>
```

The command exits nonzero on a missing interface category, an invalid state
transition, sibling-joint movement, or failure to find a collision-free pose.

## Synchronization rule

Changes flow in one direction:

```text
origin/main -> interactive-nav/sim -> codex/exp-setting
```

Merge or cherry-pick simulator commits from this branch into
`codex/exp-setting`. Do not merge `codex/exp-setting` back into this branch;
that would reintroduce the full navigation algorithm history. The root
`.gitignore` also prevents untracked copies of `Interactive-Nav-SG-nav/` and
`scripts/InteractiveNav/` from being added here. Git ignore rules do not remove
tracked files during a reverse merge, so directionality remains the actual
isolation guarantee.
