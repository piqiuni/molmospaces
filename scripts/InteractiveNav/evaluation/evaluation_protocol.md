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

The formal validation input is the runtime-qualified v1.1 release:

```text
scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_val_release_v1_1/benchmark/benchmark.json
```

The source candidate contained 3000 episodes.  After the frozen runtime quality
gate, the formal scoring denominator is 2968: 1000 Channel, 976 Container and
992 Mixed.  The 32 excluded candidate rows remain in
`scoring/scoring_manifest.jsonl`; they are not policy failures and must not be
included in aggregate metrics.  The evaluator must not modify the formal JSON.
Each episode's `scene_modifications` is the authoritative initial
object/articulation state.

### Runtime compatibility gate

Before formal scoring, protocol v4 checks the live selected target against any
frozen terminal goal, authoritative robot start pose, every recorded
articulation state, all interaction object/joint bindings, initial target
visibility when specified, and critical scene-name compatibility.  A failed
check records `scoring_eligible=false` and is excluded from formal aggregation.

The repaired candidate changed 344 Channel target identities whose original
`selected_instance` did not match the path endpoint.  All 1000 repaired Channel
episodes passed the runtime gate.  The final 32 exclusions are 24 Container and
8 Mixed episodes whose targets became initially visible in the current runtime,
violating the strict V3 `visibility_fraction > 0` condition.  The release
manifest records the candidate hash, formal hash, protocol signature and exact
exclusion reason counts.

## Policy-visible information

The ordinary policy interface receives first-person observations, the task
language, elapsed time and its own action history.  It does not receive
`interactive_nav`, selected-instance identifiers, controlling joints,
interaction ids, oracle plans, or validation evidence.  `scripted_oracle` is
the only exception; all its results are marked `uses_oracle_gt=true` and may
only be reported as an execution upper bound.

`ros_object_goal_rule` is a separate, explicitly restricted-GT evaluation
mode for the current ROS object-goal rule stack.  Per ROS frame it receives
only the compact semantic-minimal record
`{id, name, bbox_2d, mask_rle, box_3d}`, where `id` is an opaque,
episode-local token such as `obj_000017`.  The evaluator rejects source object
names, joint names/indices, joint values, open/closed state, container
relations, visibility privilege, oracle records, and task-selected instance
IDs at this boundary.  The RLE remains compact on the ROS wire and semantic
mapping counts it without materialising a dense mask.

## Terminal conditions

The rollout endpoint is `task_success`: the frozen V3 selected instance is
visible in `head_camera` and its planar base distance is below the recorded
threshold.  This is checked privately by the benchmark and ends the episode as
soon as it holds.

`interaction_conditioned_success` requires `task_success` plus all of:

1. every interaction in at least one valid oracle plan reaches 0.8 semantic
   open fraction;
2. all prerequisite interactions were executed before their dependent action;
3. for `interaction_requirement=unnecessary`, no interaction action occurred.

For backward compatibility, result field `success` is the same as
`interaction_conditioned_success`.  Reports also expose both rates explicitly,
along with navigation success, required interaction success, sequence success,
wrong interaction count, path length, and terminal reason.

Interactions use the existing physical `ForceJointController` with the robot
base and upper body locked during the force schedule.  Its readback-based
direction adaptation is preserved; evaluator code must not overwrite the
controller's selected sign.  A joint that remains below 0.8 after the bounded
two-second force schedule is honestly reported as a failed interaction.  In
restricted object-goal mode the method sends only `open(opaque_instance_id)`.
The evaluator privately resolves the object to its force-policy joints, checks
the generic final open postcondition, and returns only `completed`/`failed`.
The benchmark does not score controller substeps, but it still checks final V3
postconditions and prerequisite order for `interaction_conditioned_success`.

## Reproducibility and parallelism

One episode owns one MuJoCo context.  `--workers N` starts up to N independent
processes; this is required because MuJoCo renderers and episode state are not
thread safe.  `--resume` only skips episode directories whose completed trace
has the identical run signature (benchmark hash, evaluation configuration, and
evaluator protocol implementation).  A partial, failed, or differently
configured trace is rerun.  `ros_bridge` and `ros_object_goal_rule` are kept
single-worker because they attach to a stateful ROS master.

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
  --benchmark scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_val_release_v1_1/benchmark/benchmark.json \
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

## Current rule object-goal evaluation

The House-7 command's `METHOD=object_goal_rule` identifies the current rule
algorithm configuration, but it is not the V3 evaluator entry point: that
script starts its own fixed-house simulator.  For frozen V3 episodes use
`--policy ros_object_goal_rule`; it creates the simulator from each benchmark
episode and dynamically publishes the public language goal plus restricted GT.

Start the ROS algorithm without its simulator and without its built-in realtime
GT publisher:

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate mlspaces
export REPO=/home/user/ldl/molmospaces-exp-setting
source "$REPO/Interactive-Nav-SG-nav/devel/setup.zsh"
# These ROS Python packages use source-tree imports; append rather than replace
# PYTHONPATH so conda can still import rospy and the ROS message packages.
export PYTHONPATH="$REPO/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts:$REPO/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts:$REPO/Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts:$REPO/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts:$PYTHONPATH"
roslaunch nav_pkg molmospaces_nav_system.launch \
  start_sim:=false start_mapping:=true start_nav:=true \
  start_explore:=false start_explore_py:=true \
  start_semantic_mapping:=true semantic_source:=realtime_gt \
  publish_realtime_gt:=false start_semantic_decision:=true \
  semantic_decision_config_override_file:=$REPO/scripts/InteractiveNav/configs/semantic_decision/object_goal_v3_evaluator.yaml
```

Then run one evaluator process against that ROS master:

```bash
MUJOCO_GL=egl python scripts/InteractiveNav/evaluate_interactive_nav_v3.py \
  --benchmark scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_val_release_v1_1/benchmark/benchmark.json \
  --output-dir scripts/InteractiveNav/output/v3_eval_object_goal_rule_smoke \
  --policy ros_object_goal_rule --workers 1 \
  --ros-action-timeout-s 1.0 --max-steps 500 --max-episodes 5
```

The evaluator owns `/semantic_decision/target`,
`/semantic_mapping/gt_observations`, `/semantic_decision/interaction_command`,
and `/semantic_mapping/interaction_result` for this mode.  Topic names can be
overridden with the corresponding `--ros-*-topic` options.

The V3 evaluator override uses `module3=direct_atomic`.  The method sends only
`open(opaque_object_id)` and receives only high-level success/failure from the
evaluator-owned force skill.  After a successful command, the decision node may
record an episode-local `command_outcome_belief` from its own requested action;
this prevents duplicate open commands but is neither a simulator state read nor
joint feedback.  The default ROS configuration remains `rule_verified`.
