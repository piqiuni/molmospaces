#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
USER_CACHE_ROOT=${USER_CACHE_ROOT:-${HOME}}
MLSPACES_CACHE_DIR=${MLSPACES_CACHE_DIR:-${USER_CACHE_ROOT}/.cache/molmo-spaces-resources}
MLSPACES_ASSETS_DIR=${MLSPACES_ASSETS_DIR:-${REPO_ROOT}/assets}
BENCHMARK_DIR=${BENCHMARK_DIR:-${MLSPACES_ASSETS_DIR}/benchmarks/molmospaces-bench-v2/procthor-10k/NavToObjDataGenConfig/NavToObjProcthor10kBench_20260112_json_benchmark}
EPISODE_IDX=${EPISODE_IDX:-0}
OUTPUT_ROOT=${1:-${REPO_ROOT}/outputs/native_nav_to_obj_eval_$(date +%Y%m%d_%H%M%S)}
DEBUG_DIR=${DEBUG_DIR:-${OUTPUT_ROOT}/debug}
ROS_SETUP=${ROS_SETUP:-${REPO_ROOT}/Interactive-Nav-SG-nav/devel/setup.bash}
CONDA_SH=${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-mlspaces}
MLSPACES_PYTHON=${MLSPACES_PYTHON:-}
ROS_MASTER_URI=${ROS_MASTER_URI:-http://127.0.0.1:11601}
RUN_ROS_MASTER_URI=${ROS_MASTER_URI}
SEMANTIC_DECISION_CONFIG=${SEMANTIC_DECISION_CONFIG:-${REPO_ROOT}/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/config/default.yaml}
SEMANTIC_DECISION_OVERRIDE=${SEMANTIC_DECISION_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/object_goal_runtime.yaml}
SEMANTIC_MAPPING_OVERRIDE=${SEMANTIC_MAPPING_OVERRIDE:-}
EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/native_nav_to_obj_spin_bootstrap.yaml}
ROUTE_NAV_CONFIG=${ROUTE_NAV_CONFIG:-${SCRIPT_DIR}/configs/semantic_decision/native_nav_to_obj_dwa_recovery.yaml}
NATIVE_NAV_LOCAL_COSTMAP_INFLATION_RADIUS=${NATIVE_NAV_LOCAL_COSTMAP_INFLATION_RADIUS:-0.15}
SIM_TIMEOUT_S=${SIM_TIMEOUT_S:-1500}
NAV_STACK_READY_TIMEOUT_S=${NAV_STACK_READY_TIMEOUT_S:-60}
VIDEO_FPS=${VIDEO_FPS:-15}
VIDEO_PANEL_WIDTH_PX=${VIDEO_PANEL_WIDTH_PX:-640}
VIDEO_ENCODER_PRESET=${VIDEO_ENCODER_PRESET:-ultrafast}
ENABLE_RECORDING=${ENABLE_RECORDING:-true}
SKIP_OFFLINE_VIDEO=${SKIP_OFFLINE_VIDEO:-false}
NATIVE_NAV_RECORD_VIDEOS=${NATIVE_NAV_RECORD_VIDEOS:-true}
NATIVE_NAV_ACTION_TIMEOUT_S=${NATIVE_NAV_ACTION_TIMEOUT_S:-0.5}
NATIVE_NAV_MAX_CONSECUTIVE_ACTION_TIMEOUTS=${NATIVE_NAV_MAX_CONSECUTIVE_ACTION_TIMEOUTS:-0}
NATIVE_NAV_DYNAMIC_HORIZON=${NATIVE_NAV_DYNAMIC_HORIZON:-false}
NATIVE_NAV_DYNAMIC_HORIZON_MIN_STEPS=${NATIVE_NAV_DYNAMIC_HORIZON_MIN_STEPS:-360}
NATIVE_NAV_DYNAMIC_HORIZON_BASE_STEPS=${NATIVE_NAV_DYNAMIC_HORIZON_BASE_STEPS:-240}
NATIVE_NAV_DYNAMIC_HORIZON_STEPS_PER_METER=${NATIVE_NAV_DYNAMIC_HORIZON_STEPS_PER_METER:-45}
FILTER_MISSING_SCENE_OBJECTS=${FILTER_MISSING_SCENE_OBJECTS:-false}
GMAPPING_OVERLAY_PRELOAD=${GMAPPING_OVERLAY_PRELOAD:-true}
TASK_HORIZON_STEPS=${TASK_HORIZON_STEPS:-}
ROS_HOUSE_IND=${ROS_HOUSE_IND:-0}
ROS_TARGET_TYPES=${ROS_TARGET_TYPES:-television}
ROS_TASK_HORIZON=${ROS_TASK_HORIZON:-${TASK_HORIZON_STEPS:-500}}
ENABLE_ATTRIBUTE_INFERENCE=${ENABLE_ATTRIBUTE_INFERENCE:-false}
SEMANTIC_MODEL_ENV_FILE=${SEMANTIC_MODEL_ENV_FILE:-${SEMANTIC_DECISION_ENV_FILE:-}}
SEMANTIC_MODEL_METRICS_PATH=${SEMANTIC_MODEL_METRICS_PATH:-${OUTPUT_ROOT}/mllm_metrics.jsonl}

if [[ -n "${SEMANTIC_MODEL_ENV_FILE}" ]] && [[ ! -f "${SEMANTIC_MODEL_ENV_FILE}" ]]; then
  printf 'Semantic model env file does not exist: %s\n' "${SEMANTIC_MODEL_ENV_FILE}" >&2
  exit 2
fi

RUNTIME_TMPDIR=${TMPDIR:-${OUTPUT_ROOT}/tmp}
mkdir -p "${OUTPUT_ROOT}" "${DEBUG_DIR}" "${OUTPUT_ROOT}/ros_home/log" "${RUNTIME_TMPDIR}"
export ROS_MASTER_URI
export ROS_IP=${ROS_IP:-127.0.0.1}
export ROS_HOSTNAME=${ROS_HOSTNAME:-127.0.0.1}
export TMPDIR="${RUNTIME_TMPDIR}"
export TMP="${RUNTIME_TMPDIR}"
export TEMP="${RUNTIME_TMPDIR}"
export MLSPACES_CACHE_DIR
export MLSPACES_ASSETS_DIR
export ROS_HOME="${OUTPUT_ROOT}/ros_home"
export ROS_LOG_DIR="${OUTPUT_ROOT}/ros_home/log"
export NATIVE_NAV_DEBUG_DIR="${DEBUG_DIR}"
export NATIVE_NAV_RECORD_VIDEOS
export NATIVE_NAV_ACTION_TIMEOUT_S
export NATIVE_NAV_MAX_CONSECUTIVE_ACTION_TIMEOUTS
export NATIVE_NAV_DYNAMIC_HORIZON
export NATIVE_NAV_DYNAMIC_HORIZON_MIN_STEPS
export NATIVE_NAV_DYNAMIC_HORIZON_BASE_STEPS
export NATIVE_NAV_DYNAMIC_HORIZON_STEPS_PER_METER
export SEMANTIC_MODEL_METRICS_PATH
if [[ -n "${SEMANTIC_MODEL_ENV_FILE}" ]]; then
  export SEMANTIC_DECISION_ENV_FILE="${SEMANTIC_MODEL_ENV_FILE}"
fi
if [[ "${FILTER_MISSING_SCENE_OBJECTS}" == true ]]; then
  export NATIVE_NAV_FILTER_MISSING_SCENE_OBJECTS=1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  printf 'Conda setup file does not exist: %s\n' "${CONDA_SH}" >&2
  exit 2
fi
if [[ ! -f "${ROS_SETUP}" ]]; then
  printf 'ROS setup file does not exist: %s\n' "${ROS_SETUP}" >&2
  exit 2
fi

set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
if [[ -z "${MLSPACES_PYTHON}" ]]; then
  MLSPACES_PYTHON="$(command -v python)"
fi
if [[ ! -x "${MLSPACES_PYTHON}" ]]; then
  printf 'MlSpaces Python executable does not exist: %s\n' "${MLSPACES_PYTHON}" >&2
  exit 2
fi
source "${ROS_SETUP}"
set -u
# Some ROS setup files restore the default master URI.  Re-apply the run-local
# value after sourcing so the simulator, nodes, recorder, and evaluator share
# the same isolated master.
export ROS_MASTER_URI="${RUN_ROS_MASTER_URI}"
export ROS_PACKAGE_PATH="${REPO_ROOT}/Interactive-Nav-SG-nav/src:${ROS_PACKAGE_PATH:-}"
export LD_LIBRARY_PATH="${REPO_ROOT}/Interactive-Nav-SG-nav/devel/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts:${REPO_ROOT}/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts:${REPO_ROOT}/Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts:${REPO_ROOT}/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts:${SCRIPT_DIR}:${PYTHONPATH:-}"

# RoboStack's generated binaries can retain a DT_RPATH to its underlay.  Load
# the workspace's paired GMapping libraries in the ROS process so the custom
# odometry-locked overlay is used instead of an ABI-incompatible underlay.
ROS_LAUNCH_LD_PRELOAD=${LD_PRELOAD:-}
if [[ "${GMAPPING_OVERLAY_PRELOAD}" == true ]]; then
  GMAPPING_UTILS_LIBRARY="${REPO_ROOT}/Interactive-Nav-SG-nav/devel/lib/libutils.so"
  GMAPPING_GRIDFASTSLAM_LIBRARY="${REPO_ROOT}/Interactive-Nav-SG-nav/devel/lib/libgridfastslam.so"
  if [[ -f "${GMAPPING_UTILS_LIBRARY}" && -f "${GMAPPING_GRIDFASTSLAM_LIBRARY}" ]]; then
    ROS_LAUNCH_LD_PRELOAD="${GMAPPING_UTILS_LIBRARY}:${GMAPPING_GRIDFASTSLAM_LIBRARY}${ROS_LAUNCH_LD_PRELOAD:+:${ROS_LAUNCH_LD_PRELOAD}}"
  fi
fi

cleanup_process() {
  local pid="${1:-}"
  local grace_s="${2:-15}"
  if [[ -z "${pid}" ]]; then
    return
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
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
if ss -ltnH "sport = :${MASTER_PORT}" | grep -q .; then
  printf 'ROS master port is already occupied: %s\n' "${ROS_MASTER_URI}" >&2
  exit 3
fi
roscore -p "${MASTER_PORT}" >"${OUTPUT_ROOT}/roscore.log" 2>&1 &
ROSCORE_PID=$!

sleep 0.5
if ! kill -0 "${ROSCORE_PID}" 2>/dev/null; then
  printf 'ROS master failed to start on %s; the port may be occupied\n' "${ROS_MASTER_URI}" >&2
  exit 3
fi
if grep -Fq "roscore cannot run as another roscore/master is already running" \
  "${OUTPUT_ROOT}/roscore.log"; then
  printf 'ROS master port is already occupied: %s\n' "${ROS_MASTER_URI}" >&2
  exit 3
fi

MASTER_READY=false
for _attempt in {1..120}; do
  if ! kill -0 "${ROSCORE_PID}" 2>/dev/null; then
    printf 'ROS master exited while starting on %s\n' "${ROS_MASTER_URI}" >&2
    exit 3
  fi
  if grep -Fq "roscore cannot run as another roscore/master is already running" \
    "${OUTPUT_ROOT}/roscore.log"; then
    printf 'ROS master port is already occupied: %s\n' "${ROS_MASTER_URI}" >&2
    exit 3
  fi
  if timeout 1s rosparam list >/dev/null 2>&1; then
    MASTER_READY=true
    break
  fi
  sleep 0.25
done
if [[ "${MASTER_READY}" != true ]]; then
  printf 'ROS master did not become ready\n' >&2
  exit 3
fi

RECORDER_PID=""
if [[ "${ENABLE_RECORDING}" == true ]]; then
  PYTHONUNBUFFERED=1 python -u \
    "${REPO_ROOT}/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts/record_explore_debug.py" \
    --output-dir "${DEBUG_DIR}" \
    --occupancy-grid-topic /semantic_mapping/planning_occ_map \
    --raw-occupancy-grid-topic /struct_mapping/occ_map \
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

env LD_PRELOAD="${ROS_LAUNCH_LD_PRELOAD}" roslaunch \
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
  local_costmap_inflation_radius:="${NATIVE_NAV_LOCAL_COSTMAP_INFLATION_RADIUS}" \
  exploration_only:=false \
  randomize_camera:=false \
  robot:=rby1 \
  scene_dataset:=procthor-10k \
  data_split:=val \
  house_ind:="${ROS_HOUSE_IND}" \
  house_inds:="${ROS_HOUSE_IND}" \
  target_types:="${ROS_TARGET_TYPES}" \
  task_horizon:="${ROS_TASK_HORIZON}" \
  scene_timeout_s:="${SIM_TIMEOUT_S}" \
  output_dir:="${OUTPUT_ROOT}/ros_system" \
  >"${OUTPUT_ROOT}/roslaunch.log" 2>&1 &
LAUNCH_PID=$!

# Do not wait for scans, maps, candidates, or a velocity command here: those
# require the evaluator's first observation.  Only wait for the static ROS
# control stack so the first observation cannot race node/action-server setup.
NAV_STACK_READY=false
NAV_STACK_READY_DEADLINE=$((SECONDS + NAV_STACK_READY_TIMEOUT_S))
while (( SECONDS < NAV_STACK_READY_DEADLINE )); do
  if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    printf 'ROS launch exited before the native navigation stack became ready\n' >&2
    tail -n 80 "${OUTPUT_ROOT}/roslaunch.log" >&2 || true
    exit 4
  fi
  NODE_LIST=$(timeout 1s rosnode list 2>/dev/null || true)
  REQUIRED_NODES_READY=true
  for _node in /move_base /semantic_mapping_py /explore_py /semantic_candidate_node /semantic_rule_decision_node /semantic_behavior_executor /relay_node; do
    if ! grep -Fxq "${_node}" <<<"${NODE_LIST}"; then
      REQUIRED_NODES_READY=false
      break
    fi
  done
  if [[ "${REQUIRED_NODES_READY}" == true ]] \
    && timeout 1s rosnode ping -c 1 /move_base >/dev/null 2>&1 \
    && timeout 1s rostopic info /cmd_vel 2>/dev/null | grep -Fq "/relay_node"; then
    NAV_STACK_READY=true
    break
  fi
  sleep 0.25
done
if [[ "${NAV_STACK_READY}" != true ]]; then
  printf 'Native navigation ROS stack did not become ready within %ss\n' "${NAV_STACK_READY_TIMEOUT_S}" >&2
  tail -n 80 "${OUTPUT_ROOT}/roslaunch.log" >&2 || true
  exit 4
fi
sleep 0.5

set +e
FILTER_MISSING_SCENE_OBJECTS_ARG=()
if [[ "${FILTER_MISSING_SCENE_OBJECTS}" == true ]]; then
  FILTER_MISSING_SCENE_OBJECTS_ARG=(--filter-missing-scene-objects)
fi
TASK_HORIZON_STEPS_ARG=()
if [[ -n "${TASK_HORIZON_STEPS}" ]]; then
  TASK_HORIZON_STEPS_ARG=(--task-horizon-steps "${TASK_HORIZON_STEPS}")
fi
"${MLSPACES_PYTHON}" -u "${SCRIPT_DIR}/run_native_nav_to_obj_eval.py" \
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
  "${MLSPACES_PYTHON}" "${SCRIPT_DIR}/build_semantic_video_offline.py" \
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
