#!/usr/bin/env bash
# Start only the original InteractiveNav object-detection node with YOLOE.
# The process tree intentionally excludes Module-2 decision launch files and
# every Module-3 executor/interaction node.  All logs/cache locations are under
# /home/ldl; the Habitat evaluator talks to the relay only over loopback HTTP.
set -euo pipefail

PROJECT_ROOT=/home/ldl/molmospaces-exp-setting
INTERACTIVE_NAV_ROOT="$PROJECT_ROOT/Interactive-Nav-SG-nav"
ROS_ENV=/home/ldl/conda_envs/ros-noetic
MODEL_PATH="${YOLOE_MODEL_PATH:-$PROJECT_ROOT/detection_models/yoloe/weights/yoloe-26x-seg-pf.pt}"
MASTER_PORT="${ROS_MASTER_PORT:-13518}"
RELAY_PORT="${MODULE1_RELAY_PORT:-12188}"
GPU_ID="${YOLOE_GPU_ID:-0}"
RUNTIME_ROOT="${HABITAT_MODULE1_RUNTIME_ROOT:-/home/ldl/tmp/habitat-module1-yoloe}"
OVERRIDE_CONFIG="$PROJECT_ROOT/scripts/InteractiveNav/configs/habitat_objectnav_v2/ros_module1_yoloe_override.yaml"
CLASS_MAPPING="$PROJECT_ROOT/scripts/InteractiveNav/configs/habitat_objectnav_v2/objectnav_v2_yoloe_class_mapping.json"
DEFAULT_CONFIG="$INTERACTIVE_NAV_ROOT/src/semantic_mapping_py_pkg/config/default.yaml"

mkdir -p "$RUNTIME_ROOT/logs/ros" "$RUNTIME_ROOT/cache" "$RUNTIME_ROOT/ultralytics" "$RUNTIME_ROOT/ros"
export TMPDIR="$RUNTIME_ROOT/tmp"
export XDG_CACHE_HOME="$RUNTIME_ROOT/cache"
export TORCH_HOME="$RUNTIME_ROOT/cache/torch"
export YOLO_CONFIG_DIR="$RUNTIME_ROOT/ultralytics"
export ROS_HOME="$RUNTIME_ROOT/ros"
export ROS_LOG_DIR="$RUNTIME_ROOT/logs/ros"
mkdir -p "$TMPDIR"

source /home/ldl/miniconda3/etc/profile.d/conda.sh
# The generated RoboStack activation hook reads cross-build variables without
# defaults.  Temporarily relax nounset only while Conda sources that hook.
set +u
conda activate "$ROS_ENV"
source "$INTERACTIVE_NAV_ROOT/devel/setup.bash"
set -u
export PYTHONPATH="$INTERACTIVE_NAV_ROOT/src/semantic_mapping_py_pkg/scripts:${PYTHONPATH:-}"
export ROS_MASTER_URI="http://127.0.0.1:${MASTER_PORT}"
export ROS_IP=127.0.0.1

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "YOLOE checkpoint is missing: $MODEL_PATH" >&2
  echo "Place the authorized yoloe-26x-seg-pf.pt there, then rerun this launcher." >&2
  exit 2
fi
if ! python - <<'PY'
import torch  # noqa: F401
from ultralytics import YOLOE  # noqa: F401
PY
then
  echo "ROS Noetic environment is missing torch and/or ultralytics YOLOE support." >&2
  echo "Install compatible dependencies in $ROS_ENV before launching the original detector node." >&2
  exit 3
fi
if ss -ltn "sport = :${MASTER_PORT}" | grep -q LISTEN; then
  echo "ROS master port ${MASTER_PORT} is already in use; choose ROS_MASTER_PORT." >&2
  exit 4
fi
if ss -ltn "sport = :${RELAY_PORT}" | grep -q LISTEN; then
  echo "HTTP relay port ${RELAY_PORT} is already in use; choose MODULE1_RELAY_PORT." >&2
  exit 5
fi

cleanup() {
  # Each worker is its own session, so terminate its complete process group
  # (including rosmaster/node children) rather than leaving orphan ROS workers.
  [[ -n "${RELAY_PID:-}" ]] && kill -- "-$RELAY_PID" 2>/dev/null || true
  [[ -n "${LAUNCH_PID:-}" ]] && kill -- "-$LAUNCH_PID" 2>/dev/null || true
  [[ -n "${MASTER_PID:-}" ]] && kill -- "-$MASTER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

setsid roscore -p "$MASTER_PORT" >"$RUNTIME_ROOT/logs/roscore.log" 2>&1 &
MASTER_PID=$!
for _ in $(seq 1 30); do
  rosparam list >/dev/null 2>&1 && break
  sleep 1
done
rosparam list >/dev/null 2>&1 || { echo "roscore did not become ready" >&2; exit 6; }

CUDA_VISIBLE_DEVICES="$GPU_ID" setsid roslaunch semantic_mapping_py_pkg semantic_mapping_py.launch \
  config_file:="$DEFAULT_CONFIG" \
  override_config_file:="$OVERRIDE_CONFIG" \
  attribute_model_name:=disabled \
  start_object_detection:=true \
  start_room_attribute:=false \
  start_semantic_mapping:=false \
  start_attribute_inference:=false \
  subscribe_pointcloud:=false \
  object_detection_backend:=external_http \
  object_detection_provider:=yoloe_local \
  object_detection_model_path:="$MODEL_PATH" \
  object_detection_class_mapping:="$CLASS_MAPPING" \
  object_detection_confidence_threshold:=0.35 \
  object_detection_device:=cuda:0 \
  publish_debug_markers:=false \
  publish_debug_segmented_cloud:=false \
  publish_debug_detection_image:=false \
  publish_debug_world_markers:=false \
  publish_debug_world_segmented_cloud:=false \
  >"$RUNTIME_ROOT/logs/object_detection.log" 2>&1 &
LAUNCH_PID=$!

setsid python "$PROJECT_ROOT/scripts/InteractiveNav/habitat_v2_adapter/ros_module1_detector_bridge.py" \
  --port "$RELAY_PORT" >"$RUNTIME_ROOT/logs/http_relay.log" 2>&1 &
RELAY_PID=$!

for _ in $(seq 1 30); do
  health="$(curl --silent --max-time 1 "http://127.0.0.1:${RELAY_PORT}/health" || true)"
  if [[ "$health" == *'"ready": true'* ]]; then
    break
  fi
  kill -0 "$LAUNCH_PID" 2>/dev/null || { echo "object detector launch exited; see $RUNTIME_ROOT/logs/object_detection.log" >&2; exit 7; }
  kill -0 "$RELAY_PID" 2>/dev/null || { echo "HTTP relay exited; see $RUNTIME_ROOT/logs/http_relay.log" >&2; exit 8; }
  sleep 1
done
if [[ "${health:-}" != *'"ready": true'* ]]; then
  echo "detector-only relay did not become ready; see $RUNTIME_ROOT/logs" >&2
  exit 9
fi
echo "Detector-only Module-1 ready check: http://127.0.0.1:${RELAY_PORT}/health"
echo "ROS master: ${ROS_MASTER_URI}; model: ${MODEL_PATH}; physical GPU: ${GPU_ID}"
wait "$RELAY_PID"
