# Habitat ObjectNav-v2 Module-1 detector / Module-2 adapter

This is a new, external adapter.  It does not modify `Interactive-Nav-SG-nav`
or the existing ROS interactive-navigation implementation.

It runs the current InteractiveNav **Module-2** MLLM selector
(`semantic_decision_py_pkg.model_policy.ModelPolicyClient`) on candidates made
from public Habitat Challenge 2023 RGB-D/GPS/compass/ObjectGoal observations. The
ObjectGoal task-category integer is mapped through the public six-category task
vocabulary before it is sent to the MLLM. The adapter
only creates `EXPLORE` / `NAVIGATE` candidates and only emits Habitat
`velocity_control` / `velocity_stop` actions; `INTERACT` is rejected at both
candidate and action boundaries.  A separate public-RGB goal-visibility query
uses the current shared InteractiveNav MLLM transport to return a target box.
The adapter estimates range from depth inside that public-RGB box, approaches
with continuous navigation, and requires two close, centered confirmations
before emitting `velocity_stop`; it does not access semantic GT or interaction
state.  This is a conservative public-observation proxy, not an oracle for the
v2 valid-viewpoint success condition.

## Habitat-v2 Module-1 + Module-2 profiles

[`module1_ros_yoloe_m2_navigation_only.yaml`](../configs/habitat_objectnav_v2/module1_ros_yoloe_m2_navigation_only.yaml)
is the preferred profile when the original ROS detector node and local YOLOE
weights are available. It makes the boundary explicit:

```text
Habitat public RGB-D + camera intrinsics
  → loopback HTTP relay (127.0.0.1:12188)
  → original ROS object_detection_node.py
  → original ExternalHttpDetector + yoloe_local provider
  → public 2-D detection + same-frame depth
  → public target standoff + frontier candidates
  → original Module-2 ModelPolicyClient selects NAVIGATE / EXPLORE
  → Habitat velocity_control / velocity_stop only
```

The ROS graph starts only the original `object_detection_node.py`; it does not
start `semantic_decision.launch`, the executor, attribute inference, semantic
mapping, room inference, or any Module-3 interaction node. Habitat remains in
the Challenge Python environment. The relay accepts RGB/depth/intrinsics only
and returns only `{label, raw_label, confidence, bbox, source_model}`. The
profile disables Module-3 with a fail-closed check, permits only `EXPLORE` and
`NAVIGATE`, and rejects all interaction actions at the relay, candidate, and
action seams.

It asserts—without overriding—the official v2 task definition:

- distance is measured to `VIEW_POINTS`;
- success radius remains **0.1 m**;
- `velocity_stop` is required for a success.

The profile's 0.1 m `route_goal_reached_distance_m` is only the controller's
public local-standoff tolerance. It is not a claim that the detector knows the
official valid viewpoint. The default detector-only profile therefore disables
its uncalibrated STOP proxy; use its evaluation as an integration/grounding
measurement until a separately calibrated public stopping policy is available.

Start the original ROS detector with the local prompt-free YOLOE checkpoint.
The launcher uses a separate ROS master on `13518` and HTTP relay on `12188`,
and puts ROS/Torch/Ultralytics caches and logs under `/home/ldl/tmp`:

```bash
YOLOE_GPU_ID=0 \
ROS_MASTER_PORT=13518 \
MODULE1_RELAY_PORT=12188 \
YOLOE_MODEL_PATH=/home/ldl/molmospaces-exp-setting/detection_models/yoloe/weights/yoloe-26x-seg-pf.pt \
bash /home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/habitat_v2_adapter/start_ros_module1_yoloe.sh
```

The launcher activates `/home/ldl/conda_envs/ros-noetic`, which must contain
compatible `torch`, `torchvision`, and `ultralytics` packages. YOLOE is
prompt-free: it predicts from RGB using its built-in vocabulary; the adapter
then filters its predicted classes through the public ObjectGoal aliases. It
does not transmit an ObjectGoal word prompt to the detector.

Validate the profile before creating a Habitat environment:

```bash
EGL_PLATFORM=surfaceless \
  PYTHONPATH=/home/ldl/molmospaces-exp-setting/scripts/InteractiveNav:/home/ldl/habitat-objectnav/src/habitat-lab/habitat-lab \
  /home/ldl/conda_envs/habitat-challenge-2023/bin/python \
  /home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/habitat_v2_adapter/evaluate.py \
  --adapter-config /home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/configs/habitat_objectnav_v2/module1_ros_yoloe_m2_navigation_only.yaml \
  --validate-adapter-config
```

For a focused, auditable v2 diagnostic, add `--public-trace` and
`--posthoc-step-metrics`. The latter writes official distance metrics only after
each action, in a file that is never provided to the policy or either Module:

```bash
EGL_PLATFORM=surfaceless \
TMPDIR=/home/ldl/tmp/habitat-objectnav-ros-yoloe \
XDG_CACHE_HOME=/home/ldl/.cache/habitat-objectnav-ros-yoloe \
PYTHONPYCACHEPREFIX=/home/ldl/.cache/habitat-objectnav-ros-yoloe/pycache \
PYTHONPATH=/home/ldl/molmospaces-exp-setting/scripts/InteractiveNav:/home/ldl/habitat-objectnav/src/habitat-lab/habitat-lab \
/home/ldl/conda_envs/habitat-challenge-2023/bin/python \
  /home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/habitat_v2_adapter/evaluate.py \
  --adapter-config /home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/configs/habitat_objectnav_v2/module1_ros_yoloe_m2_navigation_only.yaml \
  --scene-id 00803-k1cupFYWXJ6 --scene-count 1 --episodes-per-scene 1 \
  --max-steps 400 --max-episode-seconds 500 --gpu-id 0 \
  --public-trace --posthoc-step-metrics \
  --output-dir /home/ldl/outputs/habitat_objectnav_v2_m2/ros_yoloe_00803
```

[`module1_detector_m2_navigation_only.yaml`](../configs/habitat_objectnav_v2/module1_detector_m2_navigation_only.yaml)
is retained as the older, non-ROS YOLOv7 sidecar profile for comparison only.

An optional external GroundingDINO adapter is available as a separate deep
module. It runs in the existing VLFM environment and accepts only an RGB JPEG
plus a category caption; the Challenge process calls it through loopback JSON.
The worker is never enabled by default because focused 00803 and 00808
diagnostics did not yet show a metric improvement. To smoke-test it without
downloading anything:

```bash
CUDA_VISIBLE_DEVICES=1 bash /home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/habitat_v2_adapter/start_grounding_dino.sh --port 12182
```

Then add `--grounding-dino-endpoint http://127.0.0.1:12182` to a focused
evaluation. The local tokenizer is
`Interactive-Nav-SG-nav/data/models/bert-base-uncased`; the worker uses the
already-present `groundingdino_swint_ogc.pth` and does not fetch Hugging Face
files. The repeatable `--scene-id 00803-k1cupFYWXJ6` option is intended for
such diagnostics; it does not change the default ten-scene slice.

## Required layout

```text
/home/ldl/habitat-objectnav/src/habitat-lab/       # challenge-2023 source
/home/ldl/habitat-objectnav/data/
  datasets/objectnav/hm3d/objectnav_hm3d_v2/val/val.json.gz
```

All outputs are written to `/home/ldl/outputs/habitat_objectnav_v2_m2/`.
Credentials for Matterport downloads must be supplied outside this repository.

The evaluator defaults to the read-only authorized shared installation available
on this host:

```text
/vepfs-wxy/memVLN/data/scene_datasets/hm3d/val/hm3d-val-habitat-v0.2
/vepfs-wxy/memVLN/data/scene_datasets/hm3d/val/hm3d_annotated_val_basis.scene_dataset_config.json
```

It rewrites each public episode scene path to these explicit paths rather than
creating a local asset overlay. Confirm that this shared installation is
licensed for your project before using it. For another authorized installation,
pass `--scene-root` and `--scene-dataset-config` explicitly.

After accepting Matterport's HM3D terms, download the required validation
assets from an interactive local terminal (the password is not echoed; a
permission-600 temporary authentication file is removed when the script exits):

```bash
bash /home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/habitat_v2_adapter/download_hm3d_val.sh
```

## Evaluation

After the official `habitat-sim-challenge-2023==0.2.3` environment is ready:

```bash
EGL_PLATFORM=surfaceless \
TMPDIR=/home/ldl/tmp/habitat-objectnav-eval \
XDG_CACHE_HOME=/home/ldl/.cache/habitat-objectnav-eval \
PYTHONPYCACHEPREFIX=/home/ldl/.cache/habitat-objectnav-eval/pycache \
PYTHONPATH=/home/ldl/molmospaces-exp-setting/scripts/InteractiveNav:/home/ldl/habitat-objectnav/src/habitat-lab/habitat-lab \
/home/ldl/conda_envs/habitat-challenge-2023/bin/python \
  /home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/habitat_v2_adapter/evaluate.py \
  --adapter-config /home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/configs/habitat_objectnav_v2/module1_ros_yoloe_m2_navigation_only.yaml \
  --scene-count 10 --episodes-per-scene 1 --gpu-id 0 \
  --max-episode-seconds 500
```

This profile requires the detector-only ROS relay from the preceding section. The
evaluator writes the resolved profile to `effective_adapter_config.yaml` with
the metrics, so each run records the enabled Module-1 transport, original
Module-2 selector, and Module-3 fail-closed state.

The runner selects one episode from each of ten unique HM3D validation scene
IDs and writes its exact slice to `selection_manifest.json`, along with
`episodes.jsonl`, `summary.json`, and a request-level MLLM log.  This is a
10-episode, scene-stratified smoke evaluation—not a replacement for the full
public validation metric.  `summary.json` also reports per-role MLLM request
success/failure counts and policy decision provenance; by default the run fails
if it did not record at least one successful model-backed Module-2 selection.
Use `--allow-no-mllm-success` only for transport diagnostics.
Each new run preserves an existing request log as
`mllm_requests.previous[.N].jsonl` before starting a fresh evidence file.
The default 500-second wall-clock budget is retained from the official v2
configuration; setting it to `0` makes a latency-relaxed local diagnostic and
must not be reported as the official benchmark protocol.

`summary.json` also includes `vision_policy_stats`, allowing a zero-success run
to be distinguished from an MLLM transport failure. This is a research baseline,
not a published ObjectNav policy; do not compare its 10-episode smoke number
directly with full-validation leaderboard results.

`--persistent-target-tracking` is an additional **off-by-default diagnostic**.
It only promotes a public RGB-D surface estimate after a spatially separated,
geometrically consistent re-observation, then offers a normal `NAVIGATE`
standoff alongside `EXPLORE` candidates to Module-2. It neither creates an
oracle target pose nor changes the stop criterion. Keep it off unless a focused
run reports an actual promotion and improves official metrics.

## Current measured status

The current stable 10-scene / 10-episode run is
[`run-20260817T085119Z`](/home/ldl/outputs/habitat_objectnav_v2_m2/current_m2_public_ten/run-20260817T085119Z/summary.json).
It used the official 1,000-step and 500-second limits and completed all ten
episodes with 159 model-backed Module-2 selections and 1,104 successful MLLM
requests (zero request failures), without oracle observations or interaction
actions.  Its measured SR/SPL is still `0.0/0.0`; it is an integration baseline,
not evidence of an improved ObjectNav policy.

Two safeguards were added after that baseline: the optional public RGB-D track
cannot become a target without a cross-view geometric inlier, and short visual
approach takes a public-depth/bounded-control circuit breaker instead of owning
an episode indefinitely. Focused runs confirmed the safeguards execute without
oracle inputs or false STOP, but have not yet raised official success. A
GroundingDINO verifier also rejected the long-lived `00808` chair boxes rather
than confirming them, but did not improve that scene; keep it opt-in.

The remaining limiting component is calibrated goal grounding and close-range
viewpoint selection. The locally served general MLLM can return useful boxes,
but it is not a trained open-vocabulary detector and did not reliably satisfy
the v2 `<0.1 m` valid-viewpoint criterion. Keep any detector or visual-controller
variant opt-in until a controlled multi-scene run shows a nonzero official
`success`/`spl`; do not treat detector positives or MLLM boxes as proof of the
v2 valid-viewpoint condition.
