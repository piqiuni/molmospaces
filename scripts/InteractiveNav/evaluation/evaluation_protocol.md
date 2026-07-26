# InteractiveNav V3 evaluation protocol

This evaluator is intentionally separate from `molmo_spaces/evaluation`.  It
uses the same JSON episode representation and the same `JsonEvalTaskSampler`,
but V3 interaction supervision remains evaluator-only.

The canonical public entry point is:

```bash
MUJOCO_GL=egl python scripts/InteractiveNav/evaluate_interactive_nav_v3.py \
  --benchmark <benchmark.json-or-directory> \
  --output-dir <evaluation-output> \
  --policy noop --workers 1
```

It routes to `scripts/InteractiveNav/evaluation/benchmark_runner.py`; it does
not invoke or modify `molmo_spaces/evaluation`.  The older `runner.py` is a
compatibility implementation and is not the source of formal benchmark
results.

## Fixed input

The production validation benchmark is the balanced 3000-episode V3 JSON:

```text
scripts/InteractiveNav/output/interactive_nav_v3_nav_benchmark_val_light_3000_manifest_v3/balanced/benchmark.json
```

It contains 1000 Channel, 1000 Container and 1000 Mixed episodes.  The
evaluator must not modify it.  Each episode's `scene_modifications` is the
authoritative initial object/articulation state.

### Runtime compatibility gate

Before an episode is formally scored, the evaluator checks every frozen oracle
terminal waypoint labelled `satisfy_nav_to_obj_success` against the live
position of the V3 `selected_instance`.  If none can satisfy the recorded
distance threshold (including waypoint tolerance), the trace is complete but
`scoring_eligible=false` and no formal metric is aggregated for that row.

This catches a currently confirmed construction error in part of the Channel
benchmark: the path goal was sampled for a nearest same-category candidate,
while `selected_instance` retained the original source object name.  For
example, episode 0's frozen terminal goal is 6.60 m from its declared toilet.
This is neither a policy failure nor an evaluator relaxation opportunity; the
benchmark records need repairing/rebuilding before those rows can support a
formal score.  `--no-require-runtime-goal-consistency` permits a diagnostic
integration rollout, but its row remains scoring-ineligible.

## Policy-visible information

Policies receive first-person observations, the task language, elapsed time and
their own action history.  They do not receive `interactive_nav`, selected
instance identifiers, controlling joints, interaction ids, oracle plans,
segmentation masks, or validation evidence.  `scripted_oracle` is the only
exception; all its results are marked `uses_oracle_gt=true` and may only be
reported as an execution upper bound.

## Terminal conditions

`overall_success` requires all of:

1. the V3 selected instance is visible in `head_camera` and its planar base
   distance is below the recorded threshold;
2. every interaction in at least one valid oracle plan reaches 0.8 semantic
   open fraction;
3. all prerequisite interactions were executed before their dependent action;
4. for `interaction_requirement=unnecessary`, no interaction action occurred.

The evaluator additionally reports navigation success, required interaction
success, sequence success, wrong interaction count, path length, individual
interaction readbacks, and terminal reason.

Interactions use the existing physical `ForceJointController` with the robot
base and upper body locked during the force schedule.  Its readback-based
direction adaptation is preserved; evaluator code must not overwrite the
controller's selected sign.  A joint that remains below 0.8 after the bounded
two-second force schedule is honestly reported as a failed interaction.

## Reproducibility and parallelism

One episode owns one MuJoCo context.  `--workers N` starts up to N independent
processes; this is required because MuJoCo renderers and episode state are not
thread safe.  `--resume` only skips episode directories whose completed trace
has the identical run signature (benchmark hash plus evaluation configuration).
A partial, failed, or differently configured trace is rerun.  `ros_bridge` is
kept single-worker because it attaches to a stateful ROS master.

## Required validation order

1. `replay.py`: V3 schema plus initial joint readback.
2. `noop`: verifies false-positive prevention and trace/report generation.
3. `scripted_oracle`: force execution and prerequisite upper-bound validation.
4. multi-process noop/oracle smoke: checks isolation and deterministic merge.
5. `ros_bridge`: current navigation system integration; its ROS launch remains
   external so the evaluator never creates a competing ROS master.

## Current ROS navigation stack

The current interactive-navigation algorithm is the ROS stack rooted at
`nav_pkg/molmospaces_nav_system.launch`: `explore_py_node.py` selects
frontiers, `move_base` produces `/cmd_vel`, and `nav_pkg/relay_node` converts
that stream to `/cmd_vel_stamped` for the evaluator-side bridge.  Start the
algorithm without its normal simulator in one terminal, then run one evaluator
process against the same ROS master:

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate mlspaces
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch \
  start_sim:=false start_mapping:=true start_nav:=true start_explore_py:=true \
  start_semantic_mapping:=false
```

```bash
MUJOCO_GL=egl python scripts/InteractiveNav/evaluate_interactive_nav_v3.py \
  --benchmark scripts/InteractiveNav/output/interactive_nav_v3_nav_benchmark_val_light_3000_manifest_v3/balanced/benchmark.json \
  --output-dir scripts/InteractiveNav/output/v3_eval_current_ros_5x100 \
  --policy ros_bridge --ros-action-timeout-s 1.0 \
  --no-require-runtime-goal-consistency \
  --workers 1 --episode-indices 0 366 1000 1002 2000 --max-steps 100
```

`--policy ros_bridge` is the equivalent short form for the default topics and
settings. Use a finite positive `--ros-action-timeout-s` for bounded benchmark
rollouts: `0` intentionally means wait indefinitely for a fresh command. Both
forms attach to the existing ROS graph; neither starts a ROS
master or gives it an evaluator task, V3 record, object identifier, joint name,
oracle plan, segmentation image, or realtime GT topic.

Do not overwrite `PYTHONPATH` after sourcing the ROS setup script: conda's
Python needs `/opt/ros/noetic/lib/python3/dist-packages` to import `rospy`.
The public evaluator entry point already adds the repository root itself.  For
the ROS bridge only, the evaluator enables live `head_camera` depth locally so
the frozen RGB-only light benchmark can publish the point cloud required by the
navigation graph; the benchmark JSON remains unchanged.
