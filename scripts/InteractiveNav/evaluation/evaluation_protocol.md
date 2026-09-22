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

The default evaluation input is the repaired v1.2 release:

```text
scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_val_release_v1_2/benchmark/benchmark.json
```

The repaired input contains 3000 episodes: 1000 each for Channel, Container and
Mixed. The historical v1.1 release had 2968 episodes; its quality-gate signature
must not be reused for v1.2. The evaluator must not modify the benchmark JSON.
Each episode's `scene_modifications` is the authoritative initial
object/articulation state.

### Runtime compatibility gate

Before formal scoring, protocol v13 checks the live selected target against any
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

The public target context also carries `success_distance_threshold_m`, the
task's distance criterion (1.5 m in the current release), without the selected
target position or identity. The method compares this against the observed
target position, not the distance to a container approach waypoint. The
evaluator independently verifies the selected instance and strict distance
inequality. Door frame/leaf render components belonging to one private asset
instance share a public door identity; their combined mask does not expose the
underlying component or joint names.

A currently visible, reliably observed target already within the public arrival
criterion remains eligible for completion verification even when its unused
navigation standoff is blocked. This candidate requests verification without
navigation; distant or currently invisible targets retain clearance checks.

The current benchmark interaction profile bypasses refrigerator open-sweep
preflight and uses ten observation steps after fully opening each drawer.
An object-goal completion claim during opening is deferred until that drawer
has completed these ten observation steps. A verified terminal interrupts the
remaining scan before closing or restoring the view; its public final state is
open. Pure exploration continues the full open-observe-close scan. Physical
open success and required-interaction scoring remain separate from target
discovery; a partly open drawer is not credited merely because its target can
already be seen.

## Terminal conditions

For the restricted ROS object-goal policy, raw simulator visibility or proximity
never ends a rollout.  The algorithm must first publish
`goal_status=SUCCEEDED` with `detail.reason=target_goal_succeeded` for the active
episode and an object-goal mission.  The evaluator then accepts the declaration
only when both conditions hold:

1. for `any_candidate` tasks, the evaluator first selects the currently nearest
   frozen candidate using the same ordered tie-break as the native NavToObj
   metric; that candidate's opaque id must have appeared in a frame actually
   published through the configured restricted-GT visibility/box filters;
2. the evaluator-private planar distance for that same selected opaque target is
   below the recorded success threshold at declaration time.

This split keeps the target instance and container relationship private while
requiring the evaluated algorithm to find and declare the object.  A one- or
two-pixel raw segmentation sliver that was filtered from the public frame cannot
produce `target_found`.  Likewise, the private target observation made during a
drawer-open macro is diagnostic only: the macro does not break the rollout, and
completion still requires the algorithm's public declaration.  A declaration
without matching public evidence or range is recorded as
`target_claim_unverified` and receives no task credit.

The goal-status vocabulary is intentionally strict.  Only `SUCCEEDED` with the
exact reason `target_goal_succeeded` and an object-goal mission can enter the
target verifier.  Generic behavior/action `SUCCEEDED` messages are ignored.
Exploration failure ends the rollout only through the dedicated
`EXPLORATION_EXHAUSTED`, `EXPLORATION_STALLED`, or `STARTUP_SCAN_FAILED` status.

For non-restricted compatibility policies, the historical private target check
remains both the pre-action rollout endpoint and the final navigation score.

`interaction_conditioned_success` requires `task_success` plus all of:

1. every interaction in at least one valid oracle plan reaches 0.8 semantic
   open fraction;
2. all prerequisite interactions were executed before their dependent action;
3. for `interaction_requirement=unnecessary`, no interaction action occurred.

For backward compatibility, result field `success` is the same as
`interaction_conditioned_success`.  Reports also expose both rates explicitly,
along with navigation success, required interaction success, sequence success,
wrong interaction count, path length, and terminal reason.

### ROS step accounting and command liveness

`--max-steps` and each dynamic `effective_max_steps` are budgets of evaluator-
applied actions, not ROS polling attempts.  A base command, view command,
interaction macro, or algorithm-requested explicit `observe` consumes one
applied step.  `observe(reason=ros_bridge_no_fresh_action)` means only that the
bridge's bounded wait produced no new command: it refreshes/publishes the next
observation but does not step MuJoCo and does not consume the applied-action
budget.

`decision_step` remains the causal observation/recording sequence, so these
waits still receive a distinct index.  Results expose both `step_count`
(recordable observation turns) and `applied_action_step_count`, plus
`no_fresh_action_count` and the full `policy_termination` diagnostic.

A single or intermittent no-fresh return never ends an episode.  By default,
90 continuous wall-clock seconds without a fresh action or evaluator-consumed
interaction ends it as `ros_bridge_command_starvation`; any real command resets
that interval.  The 90-second default leaves headroom for one 30-second M2
timeout, a 1-second backoff, and one bounded retry.
`--ros-command-starvation-timeout-s 0` disables this wall guard.
An independent default hard limit of four observation turns per applied-action
budget prevents an unbounded loop even when the wall guard is disabled; configure
it with `--ros-observation-turn-multiplier`.

When the final applied action exhausts the budget, the evaluator allows a bounded
zero-simulation-step goal-status drain (five seconds by default, configured by
`--ros-final-goal-status-drain-timeout-s`).  This accepts a declaration caused by
the final published frame without granting another action or overriding an
already-established policy/exploration terminal.

Restricted interactions cross an explicit public/private seam.  A command may
contain only an episode-local opaque object id plus public approach pose,
M1-grounded drawer regions, public type hints, and portal-aperture evidence.  The
opaque id must have a fresh 3-D box in an already-published restricted frame, and
the robot must satisfy both that public-box distance gate and the command's
ordinary approach-pose/yaw contract.  Guessed, stale, too-far, unsupported, and
unresolved commands are returned and counted as attempts; they cannot remotely
operate an unobserved object.  Before that public provenance exists, a privately
registered ID, an unregistered ID, and a guessed public alias receive the same
neutral `interaction_not_visible` contract; feedback cannot be used to enumerate
hidden articulation capability or the task container.

The evaluator then resolves a live articulation capability privately, but the
hidden oracle recipe never chooses the joints, changes physical success, or
changes feedback.  All live joints registered for that public object are passed
once to the same `prepare_articulation_state_force` /
`complete_articulation_force` fast group backend used by ordinary interaction
navigation, with the robot locked and the same 0.8 readback postcondition.  The
ordinary refrigerator open-sweep preflight is also reused; an unsafe stance
applies no force and returns a retryable `unsafe_open_sweep` result with a public
retreat recommendation.

Drawer skills never enumerate hidden slide joints when M1 supplies no actionable
region.  `drawer_scan` uses the low drawer view and performs a per-region
open-observe-close cleanup macro.  `drawer_open` uses the same low view, opens
each grounded drawer separately, publishes a fresh frame while it remains open,
restores the view, and verifies every selected drawer remains open.  The frozen
recipe is consulted only after execution for private scoring and prerequisite
order.

Feedback is reconstructed from a strict public allow-list matching the ordinary
bridge: physical status, semantic pre/post state, interaction capability,
interactable/retryable flags, public failure taxonomy, verification source,
pose validation, and aggregate step/cost fields.  It never contains source
names, joint names/indices/fractions, forces, qpos, or oracle/effect ids.  A
physically successful extra object therefore reports `SUCCEEDED`; task relevance
is scored only evaluator-side.  Non-articulated portals become `static_open`
only from public absent-leaf plus open-connectivity evidence; otherwise they fail
closed as unavailable/static-closed/blocked.

The restricted evaluator intentionally uses the ordinary bridge's fast backend.
It reproduces the physical group/readback contract and the drawer/public-feedback
lifecycle above, but does not emulate every smooth-mode per-task-step hold or
transition schedule.  This timing distinction is evaluator-private and cannot
change task relevance or public success semantics.

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
  --benchmark scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_val_release_v1_2/benchmark/benchmark.json \
  --output-dir scripts/InteractiveNav/output/v3_eval_current_ros_5x100 \
  --policy ros_bridge --ros-action-timeout-s 1.0 \
  --no-require-runtime-goal-consistency \
  --workers 1 --episode-indices 0 366 1000 1002 2000 --max-steps 100
```

`--policy ros_bridge` is the equivalent short form for the default topics and
settings. Use a finite positive `--ros-action-timeout-s` for bounded benchmark
rollouts: this is one bridge wait, not an episode terminal timeout.  `0`
intentionally blocks that individual call indefinitely and therefore prevents
the evaluator's command-starvation guard from regaining control. Both
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
  --benchmark scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_val_release_v1_2/benchmark/benchmark.json \
  --output-dir scripts/InteractiveNav/output/v3_eval_object_goal_rule_smoke \
  --policy ros_object_goal_rule --workers 1 \
  --ros-action-timeout-s 1.0 --max-steps 500 --max-episodes 5
```

The evaluator owns `/semantic_decision/target`,
observes `/semantic_decision/goal_status`,
`/semantic_mapping/gt_observations`, `/semantic_decision/interaction_command`,
and `/semantic_mapping/interaction_result` for this mode.  Topic names can be
overridden with the corresponding `--ros-*-topic` options.

The V3 evaluator override uses `module3=direct_atomic`.  The method sends an
opaque-object command with its public approach/visual evidence and receives the
same allow-listed state/capability/retry/failure contract described above.  The
evaluator does not disclose the hidden target container, selected instance,
oracle recipe, source name, or joint state.  After a successful command, the
decision node may record an episode-local `command_outcome_belief` from its own
requested action; this prevents duplicate open commands but is not a simulator
state read.  The default ROS configuration remains `rule_verified`.

## Post-hoc goal-equivalence scoring (v2)

`rescore_benchmark_goals.py` produces a separate, versioned result set. It does
not mutate the released benchmark, the original results, the online v17 terminal
verifier, or the policy's observations. Report this as a changed evaluation
definition, not an algorithm improvement or a replacement for frozen-instance SR.

Only an eligible, completed episode with a saved category-level `verified`
success claim can enter the relaxed layers. That existing verifier requires
public observation evidence for the same opaque instance and robot-to-object
distance below 1.5 m. The alternative must have the same metadata category.
Explicit unique/attribute grounding is not relaxed automatically. The report
keeps four independent endpoints: `exact_instance_success`,
`category_goal_success`, `interaction_contract_goal_success`, and
`interactive_episode_success`.

Category equivalence is accepted when either:

- The alternative's metadata ancestor is the target container and its frozen
  position lies inside the closed-container AABB, excluding the top 1 cm; room
  IDs must also match. Metadata `parent` alone is insufficient because it also
  represents objects supported *on* furniture.
- Its planar position is within 0.30 m of the original target, room IDs match,
  and both have the same support/parent or both are outside known containers.
  This is a target-to-target tolerance; the robot's arrival threshold is unchanged.
- It is another same-category object in the same observed room. This includes
  a target inside a container and a found object on a surface, but excludes a
  known different-container substitution and inconsistent container geometry.

An accepted same-container, near, or same-room candidate is **Category-only**
unless the dataset's frozen `identity_contract` or a reviewed candidate-specific
interaction-plan audit proves the complete interaction contract. A shared room or
door topology is not, by itself, that proof. Objects in another room, a known
different required container, or without verified public evidence remain failures.

The original `nav_success`, `task_success`, `success`,
`interaction_conditioned_success`, SPL, Total Cost, interaction precision,
eligibility, executed path, steps, and terminal facts are never rewritten.
Thus finding a nearby CD without opening the required drawer can raise
`category_goal_success` while leaving the strict navigation and
interaction-conditioned scores unchanged. No Category-SPL is computed because
the candidate-specific reference path is unavailable; `spl_original_reference`
is only a fixed-reference diagnostic.

Legacy v17 identity recovery uses scene metadata/XML, the deterministic sorted
registry construction, and a cross-check against the recorded full channel
alias set. Double-leaf door roots have no skill alias. When recordings exist,
the accepted public opaque ID's geometry is also cross-checked. Missing or
inconsistent evidence fails closed. Reports include input hashes and per-episode
reasons; original files are preserved and existing report directories are never
overwritten. No simulation, model inference, asset downloads, or installation
is performed by the rescoring command.

### Mixed quality audit subsets

`audit_mixed_benchmark_run.py` consumes an existing goal-equivalence report and
audits **all** mixed episodes, including successes. It does not edit any success
label. Four explicitly named subsets retain their own denominators:

1. `all_planned`: unchanged goal-equivalence scores.
2. `original_eligible`: the existing runtime consistency gate.
3. `scene_valid`: additionally quarantine a strict original-target completion
   with only portal interactions while the frozen task requires opening a closed
   container to reveal that target. This is a runtime counterexample to interaction
   necessity, not proof of a particular mesh or physics defect. Relaxed-category
   goal completion alone is insufficient for this exclusion.
4. `scene_and_execution_valid`: additionally quarantine a matching public/private
   drawer-scan command that scanned all regions and closed successfully but failed
   the checked pose-restoration convergence bound. This establishes an execution
   fault, not that the policy would otherwise have found its intended target.

The quarantine is specific to these evaluation attempts and requires revalidation
or rerunning before reuse; it never deletes benchmark data. Private transient GT
visibility without public completion, navigation/planning warnings, and shutdown
exceptions are recorded for review but do not automatically exclude a scene.
System-wide planning/actionlib races must not be used to select only failed
episodes for removal. These post-hoc conditional subsets are diagnostic and must
be reported alongside the unfiltered score, not as an unbiased algorithm gain.
