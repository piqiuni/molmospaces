#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/ldl/molmospaces-exp-setting
INTERACTIVE_NAV_ROOT="$PROJECT_ROOT/Interactive-Nav-SG-nav"
ROS_ENV=/home/ldl/conda_envs/ros-noetic
MASTER_PORT="${ROS_MASTER_PORT:-13530}"
BRIDGE_PORT="${FULL_STACK_BRIDGE_PORT:-12230}"
RUNTIME_ROOT="${HABITAT_FULL_STACK_RUNTIME_ROOT:-/home/ldl/tmp/habitat-full-stack/master-${MASTER_PORT}}"
OVERRIDE="$PROJECT_ROOT/scripts/InteractiveNav/configs/habitat_objectnav_v2/ros_full_stack_override.yaml"
MAPPING_DEFAULT="$INTERACTIVE_NAV_ROOT/src/semantic_mapping_py_pkg/config/default.yaml"
EXPLORE_DEFAULT="$INTERACTIVE_NAV_ROOT/src/explore_py_pkg/config/explore_py.yaml"
DECISION_DEFAULT="$INTERACTIVE_NAV_ROOT/src/semantic_decision_py_pkg/config/default.yaml"
RECORDER_OUTPUT_DIR="${HABITAT_FULL_STACK_RECORDER_OUTPUT_DIR:-}"

mkdir -p "$RUNTIME_ROOT/logs/ros" "$RUNTIME_ROOT/cache" "$RUNTIME_ROOT/tmp" "$RUNTIME_ROOT/ros"
export TMPDIR="$RUNTIME_ROOT/tmp"
export XDG_CACHE_HOME="$RUNTIME_ROOT/cache"
export ROS_HOME="$RUNTIME_ROOT/ros"
export ROS_LOG_DIR="$RUNTIME_ROOT/logs/ros"
source /home/ldl/miniconda3/etc/profile.d/conda.sh
set +u
conda activate "$ROS_ENV"
source "$INTERACTIVE_NAV_ROOT/devel/setup.bash"
set -u
export PYTHONPATH="$INTERACTIVE_NAV_ROOT/src/semantic_mapping_py_pkg/scripts:$INTERACTIVE_NAV_ROOT/src/semantic_decision_py_pkg/scripts:$INTERACTIVE_NAV_ROOT/src/explore_py_pkg/scripts:${PYTHONPATH:-}"
export ROS_MASTER_URI="http://127.0.0.1:${MASTER_PORT}"
export ROS_IP=127.0.0.1

declare -a PIDS=()
CLEANED_UP=0
cleanup() {
  if [[ "$CLEANED_UP" -eq 1 ]]; then
    return
  fi
  CLEANED_UP=1
  for pid in "${PIDS[@]:-}"; do
    kill -INT -- "-$pid" 2>/dev/null || true
  done
  # Give recorders/ffmpeg time to write their trailer before terminating any
  # ROS child that did not react to SIGINT.
  for _ in $(seq 1 15); do
    alive=0
    for pid in "${PIDS[@]:-}"; do
      if kill -0 -- "-$pid" 2>/dev/null; then
        alive=1
        break
      fi
    done
    [[ "$alive" -eq 0 ]] && return
    sleep 1
  done
  for pid in "${PIDS[@]:-}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

setsid roscore -p "$MASTER_PORT" >"$RUNTIME_ROOT/logs/roscore.log" 2>&1 &
PIDS+=("$!")
for _ in $(seq 1 30); do
  rosparam list >/dev/null 2>&1 && break
  sleep 1
done
rosparam list >/dev/null 2>&1 || { echo "roscore did not become ready" >&2; exit 3; }

setsid roslaunch semantic_mapping_py_pkg semantic_mapping_py.launch \
  config_file:="$MAPPING_DEFAULT" override_config_file:="$OVERRIDE" \
  attribute_model_name:=disabled \
  start_object_detection:=true start_room_attribute:=true \
  start_semantic_mapping:=true start_attribute_inference:=false \
  subscribe_pointcloud:=false \
  object_detection_backend:=yoloe_pf_box3d \
  object_detection_provider:=external_http \
  object_detection_model_path:=disabled \
  object_detection_class_mapping:="$PROJECT_ROOT/scripts/InteractiveNav/configs/habitat_objectnav_v2/objectnav_v2_yoloe_class_mapping.json" \
  object_detection_confidence_threshold:=0.35 object_detection_device:=cpu \
  publish_debug_markers:=false publish_debug_segmented_cloud:=false \
  publish_debug_detection_image:=false publish_debug_world_markers:=false \
  publish_debug_world_segmented_cloud:=false \
  >"$RUNTIME_ROOT/logs/mapping.log" 2>&1 &
PIDS+=("$!")

setsid roslaunch explore_py_pkg explore_py.launch \
  config_file:="$EXPLORE_DEFAULT" override_config_file:="$OVERRIDE" \
  external_control_enabled:=true \
  >"$RUNTIME_ROOT/logs/explore.log" 2>&1 &
PIDS+=("$!")

setsid roslaunch "$PROJECT_ROOT/scripts/InteractiveNav/habitat_v2_adapter/habitat_full_candidate.launch" \
  config_file:="$DECISION_DEFAULT" override_config_file:="$OVERRIDE" \
  >"$RUNTIME_ROOT/logs/candidate.log" 2>&1 &
PIDS+=("$!")

setsid python "$PROJECT_ROOT/scripts/InteractiveNav/habitat_v2_adapter/ros_full_stack_bridge.py" \
  --port "$BRIDGE_PORT" --defer-step-sync-to-diagnostic \
  >"$RUNTIME_ROOT/logs/bridge.log" 2>&1 &
PIDS+=("$!")

if [[ -n "$RECORDER_OUTPUT_DIR" ]]; then
  mkdir -p "$RECORDER_OUTPUT_DIR"
  setsid python "$INTERACTIVE_NAV_ROOT/src/explore_py_pkg/scripts/record_explore_debug.py" \
    --output-dir "$RECORDER_OUTPUT_DIR" \
    --occupancy-grid-topic /struct_mapping/occ_map \
    --raw-occupancy-grid-topic /struct_mapping/raw_occ_map \
    --global-plan-topic /habitat_v2/policy/global_plan \
    --local-global-plan-topic /habitat_v2/policy/global_plan \
    --local-plan-topic /habitat_v2/policy/local_plan \
    --global-costmap-topic /struct_mapping/occ_map \
    --local-costmap-topic /struct_mapping/occ_map \
    --image-topic /habitat_v2/full/rgb \
    --video-step-sync-topic /habitat_v2/full/step_sync \
    --odom-topic /odom \
    --map-frame habitat_v2_map \
    --odom-frame habitat_v2_map \
    --semantic-video \
    --no-gt-perception-overlay \
    --external-perception-overlay \
    --external-detections-topic /semantic_mapping/object_detections \
    --semantic-focus-mode object_goal \
    --task-target-topic /semantic_decision/target \
    --semantic-panel6-mode external \
    --semantic-panel6-image-topic /habitat_v2/diagnostics/topdown_goal_occ \
    --first-person-video \
    --first-person-video-with-map \
    --first-person-video-capture-mode step \
    --first-person-video-width-px 480 \
    --first-person-video-fps 10 \
    --video-occ-crop-margin-m 1.5 \
    --video-occ-lower-margin-m 0.5 \
    --runtime-video-encode \
    --first-person-video-h264 \
    --no-external-video \
    --video-save-panel-frames \
    --no-video-save-composite-frames \
    >"$RUNTIME_ROOT/logs/original_recorder.log" 2>&1 &
  PIDS+=("$!")
  # Importing the original recorder in the ROS conda environment can take tens
  # of seconds on a busy multi-worker host; do not start Habitat before its RGB
  # subscriber is genuinely registered.
  for _ in $(seq 1 90); do
    recorder_ready="$(rosnode list 2>/dev/null | grep -Fx /explore_py_debug_recorder || true)"
    [[ -n "$recorder_ready" ]] && break
    sleep 1
  done
  [[ -n "${recorder_ready:-}" ]] || { echo "original recorder did not register with ROS" >&2; exit 5; }
fi

for _ in $(seq 1 45); do
  health="$(curl --silent --max-time 1 "http://127.0.0.1:${BRIDGE_PORT}/health" || true)"
  [[ "$health" == *'"ready":true'* ]] && break
  sleep 1
done
[[ "${health:-}" == *'"ready":true'* ]] || { echo "full ROS bridge did not become ready" >&2; exit 4; }
echo "Full navigation ROS stack ready: http://127.0.0.1:${BRIDGE_PORT}"
echo "ROS master=${ROS_MASTER_URI}; Module-3 executor is not started"
wait "${PIDS[-1]}"
