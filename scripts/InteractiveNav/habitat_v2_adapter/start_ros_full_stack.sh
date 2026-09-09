#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/ldl/molmospaces-exp-setting
INTERACTIVE_NAV_ROOT="$PROJECT_ROOT/Interactive-Nav-SG-nav"
ROS_ENV=/home/ldl/conda_envs/ros-noetic
MASTER_PORT="${ROS_MASTER_PORT:-13530}"
BRIDGE_PORT="${FULL_STACK_BRIDGE_PORT:-12230}"
BRIDGE_UPDATE_WAIT_S="${FULL_STACK_BRIDGE_UPDATE_WAIT_S:-4.0}"
SIM_DT_S="${HABITAT_SIM_DT_S:-0.1}"
DETECTOR_STEP_INTERVAL="${HABITAT_DETECTOR_STEP_INTERVAL:-2}"
YOLOE_CONFIDENCE_THRESHOLD="${FULL_STACK_YOLOE_CONFIDENCE_THRESHOLD:-0.35}"
RUNTIME_ROOT="${HABITAT_FULL_STACK_RUNTIME_ROOT:-/home/ldl/tmp/habitat-full-stack/master-${MASTER_PORT}}"
YOLO_SERVICE_PORT="${FULL_STACK_YOLO_PORT:-$((BRIDGE_PORT + 10))}"
YOLO_MODEL_PATH="${FULL_STACK_YOLO_MODEL_PATH:-/home/ldl/.cache/habitat-detector-replay/weights/yolo26x-seg.pt}"
YOLO_MODEL_MODE="${FULL_STACK_YOLO_MODEL_MODE:-yolo_closed}"
OBJECT_DETECTION_BACKEND="${FULL_STACK_OBJECT_DETECTION_BACKEND:-yoloe_pf_box3d}"
YOLO_RUNTIME_ROOT="${FULL_STACK_YOLO_RUNTIME_ROOT:-$RUNTIME_ROOT/yolo}"
OVERRIDE="$PROJECT_ROOT/scripts/InteractiveNav/configs/habitat_objectnav_v2/ros_full_stack_override.yaml"
MAPPING_DEFAULT="$INTERACTIVE_NAV_ROOT/src/semantic_mapping_py_pkg/config/default.yaml"
EXPLORE_DEFAULT="$INTERACTIVE_NAV_ROOT/src/explore_py_pkg/config/explore_py.yaml"
DECISION_DEFAULT="$INTERACTIVE_NAV_ROOT/src/semantic_decision_py_pkg/config/default.yaml"
NAV_OVERRIDE="$PROJECT_ROOT/scripts/InteractiveNav/configs/habitat_objectnav_v2/original_ros_nav_override.yaml"
RECORDER_OUTPUT_DIR="${HABITAT_FULL_STACK_RECORDER_OUTPUT_DIR:-}"

mkdir -p "$RUNTIME_ROOT/logs/ros" "$RUNTIME_ROOT/cache" "$RUNTIME_ROOT/tmp" "$RUNTIME_ROOT/ros" "$YOLO_RUNTIME_ROOT/logs"
export TMPDIR="$RUNTIME_ROOT/tmp"
export XDG_CACHE_HOME="$RUNTIME_ROOT/cache"
export ROS_HOME="$RUNTIME_ROOT/ros"
export ROS_LOG_DIR="$RUNTIME_ROOT/logs/ros"
if [[ "${HABITAT_SKIP_CONDA_ACTIVATE:-0}" == "1" ]]; then
  # Useful on hosts where conda's solver/activation hook is delayed by load;
  # the requested ROS environment is already selected by the caller.
  export PATH="$ROS_ENV/bin:$PATH"
  export CONDA_PREFIX="$ROS_ENV"
else
  source /home/ldl/miniconda3/etc/profile.d/conda.sh
  set +u
  conda activate "$ROS_ENV"
  set -u
fi
set +u
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

if [[ ! -f "$YOLO_MODEL_PATH" ]]; then
  echo "YOLO checkpoint not found: $YOLO_MODEL_PATH" >&2
  exit 2
fi

setsid python "$PROJECT_ROOT/scripts/InteractiveNav/habitat_v2_adapter/yoloe_worker_server.py" \
  --port "$YOLO_SERVICE_PORT" --replica-id 0 \
  --model-path "$YOLO_MODEL_PATH" --model-mode "$YOLO_MODEL_MODE" \
  --prompt-list "chair,bed,potted plant,toilet,tv,couch" \
  --confidence-threshold "$YOLOE_CONFIDENCE_THRESHOLD" \
  --class-mapping "$PROJECT_ROOT/scripts/InteractiveNav/configs/habitat_objectnav_v2/objectnav_v2_yoloe_class_mapping.json" \
  --device cuda:0 >"$YOLO_RUNTIME_ROOT/logs/yolo.log" 2>&1 &
PIDS+=("$!")
for _ in $(seq 1 180); do
  yolo_health="$(curl --silent --max-time 1 "http://127.0.0.1:${YOLO_SERVICE_PORT}/health" || true)"
  [[ "$yolo_health" == *'"ready":true'* ]] && break
  sleep 1
done
[[ "${yolo_health:-}" == *'"ready":true'* ]] || { echo "dedicated YOLO did not become ready" >&2; exit 3; }

ACTIVE_OVERRIDE="$RUNTIME_ROOT/ros_full_stack_override.yaml"
sed "s#^  external_url:.*#  external_url: http://127.0.0.1:${YOLO_SERVICE_PORT}/detect#" \
  "$OVERRIDE" >"$ACTIVE_OVERRIDE"

setsid roscore -p "$MASTER_PORT" >"$RUNTIME_ROOT/logs/roscore.log" 2>&1 &
PIDS+=("$!")
for _ in $(seq 1 30); do
  rosparam list >/dev/null 2>&1 && break
  sleep 1
done
rosparam list >/dev/null 2>&1 || { echo "roscore did not become ready" >&2; exit 3; }

# Habitat replaces only the simulator/robot driver.  Mapping, global planning,
# local planning and velocity generation stay in the original ROS code stack.
setsid roslaunch struct_mapping_pkg slam_gmapping.launch \
  mapping_mode:=odom_locked mapping_scan_source:=pointcloud \
  mapping_scan_topic:=/molmo_spaces/organized_depth_scan \
  scan_filter_tolerance_sec:=0.0 \
  >"$RUNTIME_ROOT/logs/struct_mapping.log" 2>&1 &
PIDS+=("$!")

setsid roslaunch nav_pkg nav.launch \
  override_config_file:="$NAV_OVERRIDE" \
  global_map_topic:=/struct_mapping/occ_map \
  global_planner_allow_unknown:=true \
  base_local_planner:=dwa_local_planner/DWAPlannerROS \
  move_base_respawn:=false \
  >"$RUNTIME_ROOT/logs/navigation.log" 2>&1 &
PIDS+=("$!")

setsid roslaunch semantic_mapping_py_pkg semantic_mapping_py.launch \
  config_file:="$MAPPING_DEFAULT" override_config_file:="$ACTIVE_OVERRIDE" \
  attribute_model_name:=disabled \
  start_object_detection:=true start_room_attribute:=true \
  start_semantic_mapping:=true start_attribute_inference:=false \
  subscribe_pointcloud:=true \
  object_detection_backend:="$OBJECT_DETECTION_BACKEND" \
  object_detection_provider:=external_http \
  object_detection_model_path:=disabled \
  object_detection_class_mapping:="$PROJECT_ROOT/scripts/InteractiveNav/configs/habitat_objectnav_v2/objectnav_v2_yoloe_class_mapping.json" \
  object_detection_confidence_threshold:="$YOLOE_CONFIDENCE_THRESHOLD" object_detection_device:=cpu \
  publish_debug_markers:=false publish_debug_segmented_cloud:=false \
  publish_debug_detection_image:=false publish_debug_world_markers:=false \
  publish_debug_world_segmented_cloud:=false \
  >"$RUNTIME_ROOT/logs/mapping.log" 2>&1 &
PIDS+=("$!")

setsid roslaunch explore_py_pkg explore_py.launch \
  config_file:="$EXPLORE_DEFAULT" override_config_file:="$ACTIVE_OVERRIDE" \
  external_control_enabled:=true \
  >"$RUNTIME_ROOT/logs/explore.log" 2>&1 &
PIDS+=("$!")

setsid roslaunch "$PROJECT_ROOT/scripts/InteractiveNav/habitat_v2_adapter/habitat_full_candidate.launch" \
  config_file:="$DECISION_DEFAULT" override_config_file:="$ACTIVE_OVERRIDE" \
  >"$RUNTIME_ROOT/logs/candidate.log" 2>&1 &
PIDS+=("$!")

export HABITAT_DIRECT_DETECTOR_URL="http://127.0.0.1:${YOLO_SERVICE_PORT}"
BRIDGE_RECORDER_ARGS=()
if [[ -n "$RECORDER_OUTPUT_DIR" ]]; then
  # The evaluator publishes panel 6 after policy.act(); hold the original
  # recorder's step latch until that exact-step diagnostic is available.
  BRIDGE_RECORDER_ARGS+=(--defer-step-sync-to-diagnostic)
fi
setsid python "$PROJECT_ROOT/scripts/InteractiveNav/habitat_v2_adapter/ros_full_stack_bridge.py" \
  --port "$BRIDGE_PORT" --update-wait-s "$BRIDGE_UPDATE_WAIT_S" \
  --sim-dt-s "$SIM_DT_S" \
  --detector-step-interval "$DETECTOR_STEP_INTERVAL" \
  --original-ros-navigation \
  --map-frame tf_frame_map --odom-frame tf_frame_odom \
  --base-frame tf_frame_base_link --lidar-frame tf_frame_lidar \
  --camera-frame tf_frame_camera \
  "${BRIDGE_RECORDER_ARGS[@]}" \
  >"$RUNTIME_ROOT/logs/bridge.log" 2>&1 &
PIDS+=("$!")

if [[ -n "$RECORDER_OUTPUT_DIR" ]]; then
  mkdir -p "$RECORDER_OUTPUT_DIR"
  setsid python "$INTERACTIVE_NAV_ROOT/src/explore_py_pkg/scripts/record_explore_debug.py" \
    --output-dir "$RECORDER_OUTPUT_DIR" \
    --occupancy-grid-topic /struct_mapping/occ_map \
    --raw-occupancy-grid-topic /struct_mapping/occ_map \
    --global-plan-topic /move_base/OrientedGlobalPlanner/plan \
    --local-global-plan-topic /move_base/DWAPlannerROS/global_plan \
    --local-plan-topic /move_base/DWAPlannerROS/local_plan \
    --global-costmap-topic /move_base/global_costmap/costmap \
    --local-costmap-topic /move_base/local_costmap/costmap \
    --image-topic /habitat_v2/full/rgb \
    --video-step-sync-topic /habitat_v2/full/step_sync \
    --odom-topic /odom \
    --map-frame tf_frame_map \
    --odom-frame tf_frame_odom \
    --semantic-video \
    --no-gt-perception-overlay \
    --external-perception-overlay \
    --semantic-focus-mode object_goal \
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
