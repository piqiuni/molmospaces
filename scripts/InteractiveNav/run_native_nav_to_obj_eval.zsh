#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
USER_CACHE_ROOT=${USER_CACHE_ROOT:-/home/user}
MLSPACES_CACHE_DIR=${MLSPACES_CACHE_DIR:-${USER_CACHE_ROOT}/.cache/molmo-spaces-resources}
MLSPACES_ASSETS_DIR=${MLSPACES_ASSETS_DIR:-/home/user/ldl/molmospaces/assets}
BENCHMARK_DIR=${BENCHMARK_DIR:-${MLSPACES_ASSETS_DIR}/benchmarks/molmospaces-bench-v2/procthor-10k/NavToObjDataGenConfig/NavToObjProcthor10kBench_20260112_json_benchmark}
EPISODE_IDX=${EPISODE_IDX:-0}
OUTPUT_ROOT=${1:-${REPO_ROOT}/outputs/native_nav_to_obj_eval_$(date +%Y%m%d_%H%M%S)}
DEBUG_DIR=${DEBUG_DIR:-${OUTPUT_ROOT}/debug}
ROS_SETUP=${ROS_SETUP:-${REPO_ROOT}/Interactive-Nav-SG-nav/devel/setup.zsh}
ROS_MASTER_URI=${ROS_MASTER_URI:-http://127.0.0.1:11601}
RUN_ROS_MASTER_URI=${ROS_MASTER_URI}
SEMANTIC_DECISION_CONFIG=${SEMANTIC_DECISION_CONFIG:-${REPO_ROOT}/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/config/default.yaml}
SEMANTIC_DECISION_OVERRIDE=${SEMANTIC_DECISION_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/object_goal_runtime.yaml}
SEMANTIC_MAPPING_OVERRIDE=${SEMANTIC_MAPPING_OVERRIDE:-}
EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_controlled_explore.yaml}
ROUTE_NAV_CONFIG=${ROUTE_NAV_CONFIG:-${SCRIPT_DIR}/configs/semantic_decision/semantic_interaction_nav.yaml}
SIM_TIMEOUT_S=${SIM_TIMEOUT_S:-1500}
VIDEO_FPS=${VIDEO_FPS:-15}
VIDEO_PANEL_WIDTH_PX=${VIDEO_PANEL_WIDTH_PX:-640}
VIDEO_ENCODER_PRESET=${VIDEO_ENCODER_PRESET:-ultrafast}
ENABLE_RECORDING=${ENABLE_RECORDING:-true}
SKIP_OFFLINE_VIDEO=${SKIP_OFFLINE_VIDEO:-false}
FILTER_MISSING_SCENE_OBJECTS=${FILTER_MISSING_SCENE_OBJECTS:-false}
TASK_HORIZON_STEPS=${TASK_HORIZON_STEPS:-}
ENABLE_ATTRIBUTE_INFERENCE=${ENABLE_ATTRIBUTE_INFERENCE:-false}

mkdir -p "${OUTPUT_ROOT}" "${DEBUG_DIR}" "${OUTPUT_ROOT}/ros_home/log"
export ROS_MASTER_URI
export ROS_IP=${ROS_IP:-127.0.0.1}
export ROS_HOSTNAME=${ROS_HOSTNAME:-127.0.0.1}
export MLSPACES_CACHE_DIR
export MLSPACES_ASSETS_DIR
export ROS_HOME="${OUTPUT_ROOT}/ros_home"
export ROS_LOG_DIR="${OUTPUT_ROOT}/ros_home/log"
export NATIVE_NAV_DEBUG_DIR="${DEBUG_DIR}"
if [[ "${FILTER_MISSING_SCENE_OBJECTS}" == true ]]; then
  export NATIVE_NAV_FILTER_MISSING_SCENE_OBJECTS=1
fi

set +u
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate mlspaces
set -u
source "${ROS_SETUP}"
# Some ROS setup files restore the default master URI.  Re-apply the run-local
# value after sourcing so the simulator, nodes, recorder, and evaluator share
# the same isolated master.
export ROS_MASTER_URI="${RUN_ROS_MASTER_URI}"
export ROS_PACKAGE_PATH="${REPO_ROOT}/Interactive-Nav-SG-nav/src:/opt/ros/noetic/share"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts:${REPO_ROOT}/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts:${REPO_ROOT}/Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts:${REPO_ROOT}/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts:${SCRIPT_DIR}:${PYTHONPATH:-}"

cleanup_process() {
  local pid="${1:-}"
  local grace_s="${2:-15}"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return
  fi
  kill -INT "${pid}" 2>/dev/null || true
  local max_attempts=$((grace_s * 2))
  local attempt=1
  while (( attempt <= max_attempts )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      return
    fi
    sleep 0.5
    attempt=$((attempt + 1))
  done
  kill -TERM "${pid}" 2>/dev/null || true
  sleep 1
  kill -KILL "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  cleanup_process "${LAUNCH_PID:-}" 20
  cleanup_process "${RECORDER_PID:-}" 20
  cleanup_process "${ROSCORE_PID:-}" 10
}

on_signal() {
  trap - EXIT INT TERM
  cleanup
  exit 130
}

trap cleanup EXIT
trap on_signal INT TERM

MASTER_PORT=${ROS_MASTER_URI##*:}
MASTER_PORT=${MASTER_PORT%%/*}
roscore -p "${MASTER_PORT}" >"${OUTPUT_ROOT}/roscore.log" 2>&1 &
ROSCORE_PID=$!

MASTER_READY=false
for _attempt in {1..120}; do
  if timeout 1s rosparam list >/dev/null 2>&1; then
    MASTER_READY=true
    break
  fi
  sleep 0.25
done
if [[ "${MASTER_READY}" != true ]]; then
  print -u2 -- "ROS master did not become ready"
  exit 3
fi

RECORDER_PID=""
if [[ "${ENABLE_RECORDING}" == true ]]; then
  PYTHONUNBUFFERED=1 python -u \
    "${REPO_ROOT}/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts/record_explore_debug.py" \
    --output-dir "${DEBUG_DIR}" \
    --occupancy-grid-topic /semantic_mapping/planning_occ_map \
    --image-topic /molmo_spaces/head_camera/image \
    --video-step-sync-topic /molmo_spaces/step_sync \
    --first-person-video-capture-mode step \
    --semantic-video \
    --first-person-video-with-map \
    --first-person-video-fps "${VIDEO_FPS}" \
    --first-person-video-width-px "${VIDEO_PANEL_WIDTH_PX}" \
    --video-frame-job-queue-size 4 \
    --artifact-write-queue-size 4 \
    --video-history-size 16 \
    --image-queue-size 4 \
    --video-global-panel-scale 1.8 \
    --runtime-video-encode \
    --first-person-video-h264-preset "${VIDEO_ENCODER_PRESET}" \
    --no-external-video \
    --no-video-save-panel-frames \
    --no-first-person-video-h264 \
    >"${OUTPUT_ROOT}/recorder.log" 2>&1 &
  RECORDER_PID=$!
  sleep 1
fi

roslaunch \
  "${REPO_ROOT}/Interactive-Nav-SG-nav/src/nav_pkg/launch/molmospaces_nav_system.launch" \
  start_sim:=false \
  start_mapping:=true \
  mapping_mode:=odom_locked \
  start_semantic_mapping:=true \
  semantic_source:=realtime_gt \
  semantic_attribute_inference:="${ENABLE_ATTRIBUTE_INFERENCE}" \
  publish_realtime_gt:=false \
  start_nav:=true \
  start_explore:=false \
  start_explore_py:=true \
  explore_py_config_override_file:="${EXPLORE_PY_CONFIG_OVERRIDE}" \
  start_semantic_decision:=true \
  semantic_decision_config_file:="${SEMANTIC_DECISION_CONFIG}" \
  semantic_decision_config_override_file:="${SEMANTIC_DECISION_OVERRIDE}" \
  semantic_config_override_file:="${SEMANTIC_MAPPING_OVERRIDE}" \
  nav_config_override_file:="${ROUTE_NAV_CONFIG}" \
  exploration_only:=false \
  randomize_camera:=false \
  robot:=rby1 \
  scene_dataset:=procthor-10k \
  data_split:=val \
  house_ind:=0 \
  house_inds:=0 \
  target_types:=television \
  task_horizon:=500 \
  scene_timeout_s:="${SIM_TIMEOUT_S}" \
  output_dir:="${OUTPUT_ROOT}/ros_system" \
  >"${OUTPUT_ROOT}/roslaunch.log" 2>&1 &
LAUNCH_PID=$!

set +e
FILTER_MISSING_SCENE_OBJECTS_ARG=()
if [[ "${FILTER_MISSING_SCENE_OBJECTS}" == true ]]; then
  FILTER_MISSING_SCENE_OBJECTS_ARG=(--filter-missing-scene-objects)
fi
TASK_HORIZON_STEPS_ARG=()
if [[ -n "${TASK_HORIZON_STEPS}" ]]; then
  TASK_HORIZON_STEPS_ARG=(--task-horizon-steps "${TASK_HORIZON_STEPS}")
fi
python -u "${SCRIPT_DIR}/run_native_nav_to_obj_eval.py" \
  --benchmark-dir "${BENCHMARK_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --debug-dir "${DEBUG_DIR}" \
  --episode-idx "${EPISODE_IDX}" \
  "${FILTER_MISSING_SCENE_OBJECTS_ARG[@]}" \
  "${TASK_HORIZON_STEPS_ARG[@]}"
EVAL_RC=$?
set -e

cleanup_process "${LAUNCH_PID:-}" 20
LAUNCH_PID=""
if [[ -n "${RECORDER_PID}" ]]; then
  cleanup_process "${RECORDER_PID}" 20
  RECORDER_PID=""
fi

if [[ "${EVAL_RC}" -eq 0 ]] && [[ "${ENABLE_RECORDING}" == true ]] && [[ "${SKIP_OFFLINE_VIDEO}" != true ]]; then
  python "${SCRIPT_DIR}/build_semantic_video_offline.py" \
    --scene-dir "${OUTPUT_ROOT}" \
    --debug-dir "${DEBUG_DIR}" \
    --fps "${VIDEO_FPS}" \
    --state-alignment latest \
    --output-stem overview_6panel \
    >"${OUTPUT_ROOT}/offline_video.log" 2>&1 || true
  if [[ ! -f "${OUTPUT_ROOT}/videos/overview_6panel.mp4" ]] \
    && [[ -f "${DEBUG_DIR}/videos/overview_6panel.mp4" ]]; then
    mkdir -p "${OUTPUT_ROOT}/videos"
    mv "${DEBUG_DIR}/videos/overview_6panel.mp4" "${OUTPUT_ROOT}/videos/overview_6panel.mp4"
  fi
fi

exit "${EVAL_RC}"
