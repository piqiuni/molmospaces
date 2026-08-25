#!/usr/bin/env bash
set -euo pipefail

# Compatibility for the small set of legacy zsh-style status writes below.
print() {
  local fd=1
  if [[ "${1:-}" == "-u2" ]]; then fd=2; shift; fi
  if [[ "${1:-}" == "-r" ]]; then shift; fi
  if [[ "${1:-}" == "--" ]]; then shift; fi
  printf '%s\n' "$@" >&"$fd"
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
ROUTE_CONFIG=${ROUTE_CONFIG:-${SCRIPT_DIR}/configs/semantic_decision/house7_force_routes.yaml}
VIDEO_BUILDER=${VIDEO_BUILDER:-${SCRIPT_DIR}/build_semantic_video_offline.py}
RECORDER_DRAIN_HELPER=${RECORDER_DRAIN_HELPER:-${SCRIPT_DIR}/wait_for_recorder_drain.py}
ROUTE_ID=${2:-${ROUTE_ID:-house7_force_route_01}}
HOUSE_IND=${HOUSE_IND:-7}
USE_FIXED_ROUTE=${USE_FIXED_ROUTE:-true}
SCENE_SEED=${SCENE_SEED:-${HOUSE_IND}}
METHOD=${METHOD:-interactive_rule}
OUTPUT_DIR=${1:-${REPO_ROOT}/outputs/house7_${METHOD}_${ROUTE_ID}_$(date +%Y%m%d_%H%M%S)}
if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${PWD}/${OUTPUT_DIR}"
fi
ROS_SETUP=${ROS_SETUP:-${REPO_ROOT}/Interactive-Nav-SG-nav/devel/setup.bash}
ROS_MASTER_URI=${ROS_MASTER_URI:-http://127.0.0.1:11501}
RUN_ROS_MASTER_URI=${ROS_MASTER_URI}
TASK_HORIZON=${TASK_HORIZON:-1000}
POINTCLOUD_STRIDE=${POINTCLOUD_STRIDE:-1}
# Preserve organized RGB-D continuity for the global SLAM scan.  The flattened
# PointCloud2 path cannot distinguish supported far/no-return pixels from sparse
# depth-edge samples and can leave a visible wedge unknown even while the local
# rolling costmap has already ray-traced it free.
MAPPING_SCAN_SOURCE=${MAPPING_SCAN_SOURCE:-organized_depth}
MAPPING_SCAN_TOPIC=${MAPPING_SCAN_TOPIC:-/molmo_spaces/organized_depth_scan}
SCAN_FILTER_TOLERANCE_SEC=${SCAN_FILTER_TOLERANCE_SEC:-0.03}
DEPTH_SCAN_ARGS=""
case "${MAPPING_SCAN_SOURCE}" in
  pointcloud)
    ;;
  organized_depth)
    DEPTH_SCAN_ARGS="--publish_depth_scan --depth_scan_topic ${MAPPING_SCAN_TOPIC}"
    ;;
  *)
    print -u2 -- "Unsupported MAPPING_SCAN_SOURCE=${MAPPING_SCAN_SOURCE}; expected pointcloud or organized_depth"
    exit 2
    ;;
esac
VIDEO_FPS=${VIDEO_FPS:-15}
if [[ -z "${VIDEO_PANEL_WIDTH_PX:-}" ]]; then
  if [[ "${USE_FIXED_ROUTE}" == true ]]; then
    VIDEO_PANEL_WIDTH_PX=640
  else
    VIDEO_PANEL_WIDTH_PX=480
  fi
fi
VIDEO_FRAME_JOB_QUEUE_SIZE=${VIDEO_FRAME_JOB_QUEUE_SIZE:-${TASK_HORIZON}}
VIDEO_FRAME_QUEUE_OVERFLOW=${VIDEO_FRAME_QUEUE_OVERFLOW:-block}
ARTIFACT_WRITE_QUEUE_SIZE=${ARTIFACT_WRITE_QUEUE_SIZE:-64}
ARTIFACT_WRITE_WORKERS=${ARTIFACT_WRITE_WORKERS:-4}
ARTIFACT_WRITE_OVERFLOW=${ARTIFACT_WRITE_OVERFLOW:-block}
VIDEO_HISTORY_SIZE=${VIDEO_HISTORY_SIZE:-256}
SEMANTIC_VIDEO_MAX_OBJECT_NODES=${SEMANTIC_VIDEO_MAX_OBJECT_NODES:-64}
VIDEO_ROOM_PANEL_SCALE=${VIDEO_ROOM_PANEL_SCALE:-1.5}
VIDEO_SEMANTIC_XY_PANEL_SCALE=${VIDEO_SEMANTIC_XY_PANEL_SCALE:-1.8}
VIDEO_SEMANTIC_XY_LABEL_MODE=${VIDEO_SEMANTIC_XY_LABEL_MODE:-interaction_target_only}
# Keep the complete known-map context visible in the six-panel output.  The
# recorder persists this setting so rebuilding the video later keeps the view.
VIDEO_SEMANTIC_XY_OVERVIEW_INSET=${VIDEO_SEMANTIC_XY_OVERVIEW_INSET:-false}
IMAGE_QUEUE_SIZE=${IMAGE_QUEUE_SIZE:-64}
OBSERVATION_QUEUE_SIZE=${OBSERVATION_QUEUE_SIZE:-16}
VIDEO_ENCODER_PRESET=${VIDEO_ENCODER_PRESET:-ultrafast}
EXTERNAL_VIDEO_WIDTH_PX=${EXTERNAL_VIDEO_WIDTH_PX:-1024}
EXTERNAL_VIDEO_OVERLAY=${EXTERNAL_VIDEO_OVERLAY:-true}
PAPER_FRAME_EXPORTS=${PAPER_FRAME_EXPORTS:-false}
STEP_SYNC_QUEUE_SIZE=${STEP_SYNC_QUEUE_SIZE:-2048}
STEP_SYNC_CAPTURE_EVERY=${STEP_SYNC_CAPTURE_EVERY:-1}
STEP_SYNC_IMAGE_CACHE_SIZE=${STEP_SYNC_IMAGE_CACHE_SIZE:-${TASK_HORIZON}}
EXTRA_IMAGE_QUEUE_SIZE=${EXTRA_IMAGE_QUEUE_SIZE:-16}
TIMING_LOG_EVERY_N_STEPS=${TIMING_LOG_EVERY_N_STEPS:-50}
RECORDER_PERFORMANCE_LOG_EVERY_N_FRAMES=${RECORDER_PERFORMANCE_LOG_EVERY_N_FRAMES:-50}
RECORDER_DRAIN_TIMEOUT_S=${RECORDER_DRAIN_TIMEOUT_S:-7200}
RECORDER_DRAIN_POLL_S=${RECORDER_DRAIN_POLL_S:-0.5}
RECORDER_DRAIN_PROGRESS_S=${RECORDER_DRAIN_PROGRESS_S:-30}
RECORDER_DRAIN_STALL_TIMEOUT_S=${RECORDER_DRAIN_STALL_TIMEOUT_S:-180}
RECORDER_SHUTDOWN_GRACE_S=${RECORDER_SHUTDOWN_GRACE_S:-600}
GT_STEP_INTERVAL=${GT_STEP_INTERVAL:-3}
# Keep semantic object discovery more conservative than the 8 m mapping scan,
# but let an actually visible container be recognised before it has already
# fallen out of a practical interaction-staging range.  Occlusion, box-area,
# and consecutive-frame gates still apply.
GT_MAX_DISTANCE_M=${GT_MAX_DISTANCE_M:-5.0}
GT_MIN_VISIBLE_PIXELS=${GT_MIN_VISIBLE_PIXELS:-16}
GT_MIN_VISIBLE_FRACTION=${GT_MIN_VISIBLE_FRACTION:-0.20}
GT_REQUIRED_CONSECUTIVE_OBSERVATIONS=${GT_REQUIRED_CONSECUTIVE_OBSERVATIONS:-2}
# Explicit rule-oracle mode: derive a container approach normal from live
# simulator joints. It never loads a scene-specific world pose.
GT_EMIT_INTERACTION_APPROACH_AXIS=${GT_EMIT_INTERACTION_APPROACH_AXIS:-}
# The bridge now waits for a real first-map bootstrap, so do not discard the
# first ten simulator observations.  Override only for isolated legacy tests.
MAP_WARMUP_SKIP_FRAMES=${MAP_WARMUP_SKIP_FRAMES:-0}
STEP_READY_WARMUP_SKIP_FRAMES=${STEP_READY_WARMUP_SKIP_FRAMES:-0}
GT_ROI_X_MIN_RATIO=${GT_ROI_X_MIN_RATIO:-0.10}
GT_ROI_X_MAX_RATIO=${GT_ROI_X_MAX_RATIO:-0.90}
GT_MIN_FORWARD_COSINE=${GT_MIN_FORWARD_COSINE:-0.15}
# Keep the launch-time override consistent with the checked-in global/local
# costmap configs.  A stale 0.30 default here silently defeated the requested
# 0.40 m local inflation in every house-run smoke.
LOCAL_COSTMAP_INFLATION_RADIUS=${LOCAL_COSTMAP_INFLATION_RADIUS:-0.45}
SIM_TIMEOUT_S=${SIM_TIMEOUT_S:-1200}
ROUTE_NAV_CONFIG=${ROUTE_NAV_CONFIG:-${SCRIPT_DIR}/configs/semantic_decision/semantic_interaction_nav.yaml}
EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-}
SEMANTIC_DECISION_CONFIG=${SEMANTIC_DECISION_CONFIG:-${REPO_ROOT}/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/config/default.yaml}
SEMANTIC_DECISION_OVERRIDE=${SEMANTIC_DECISION_OVERRIDE:-}
COMPLETION_CONFIRMATIONS=${COMPLETION_CONFIRMATIONS:-3}
COMPLETION_POST_HOLD_STEPS=${COMPLETION_POST_HOLD_STEPS:-}
INITIAL_DOOR_STATE=${INITIAL_DOOR_STATE:-closed}
FORCE_CLOSE_CONTAINERS=${FORCE_CLOSE_CONTAINERS:-false}
CLEAN_INTERMEDIATE=${CLEAN_INTERMEDIATE:-false}
ENABLE_RECORDING=${ENABLE_RECORDING:-true}
ENABLE_EXTERNAL_VIDEO=${ENABLE_EXTERNAL_VIDEO:-false}
EXTERNAL_IMAGE_TOPIC=${EXTERNAL_IMAGE_TOPIC:-/molmo_spaces/debug_front_camera/image}
DEBUG_FOLLOW_CAMERA_OFFSET=${DEBUG_FOLLOW_CAMERA_OFFSET:--1.45,1.30,1.90}
DEBUG_FOLLOW_CAMERA_LOOKAT_OFFSET=${DEBUG_FOLLOW_CAMERA_LOOKAT_OFFSET:-0.05,0.40,1.38}
DEBUG_FOLLOW_CAMERA_FOV_DEG=${DEBUG_FOLLOW_CAMERA_FOV_DEG:-65.0}
if [[ -z "${INTERACTION_EXECUTION_MODE:-}" ]]; then
  if [[ -n "${DRAWER_EXECUTION_MODE:-}" ]]; then
    INTERACTION_EXECUTION_MODE=${DRAWER_EXECUTION_MODE}
  # Full semantic/MLLM interaction runs must keep the same physical action
  # timing with or without video recording.  In particular, a drawer scan is
  # an open -> low-view dwell -> close macro, not a one-step state flip.
  elif [[ "${METHOD}" == full_mllm_exploration || "${METHOD}" == semantic_interaction_* || "${ENABLE_RECORDING}" == true ]]; then
    INTERACTION_EXECUTION_MODE=smooth
  else
    INTERACTION_EXECUTION_MODE=fast
  fi
fi
if [[ -z "${DRAWER_EXECUTION_MODE:-}" ]]; then
  DRAWER_EXECUTION_MODE=${INTERACTION_EXECUTION_MODE}
fi
if [[ -z "${INTERACTION_TRANSITION_STEPS:-}" ]]; then
  INTERACTION_TRANSITION_STEPS=${DRAWER_TRANSITION_STEPS:-5}
fi
if [[ -z "${DRAWER_TRANSITION_STEPS:-}" ]]; then
  DRAWER_TRANSITION_STEPS=${INTERACTION_TRANSITION_STEPS}
fi
# Each M1-grounded drawer front gets a short but visible dwell while the head
# is low.  The scan macro then closes it before advancing to the next front.
DRAWER_OBSERVATION_STEPS=${DRAWER_OBSERVATION_STEPS:-3}
ENABLE_ATTRIBUTE_INFERENCE=${ENABLE_ATTRIBUTE_INFERENCE:-false}
SEMANTIC_ATTRIBUTE_MODEL_NAME=${SEMANTIC_ATTRIBUTE_MODEL_NAME:-}
# Portal observations now include bounded visual morphology and aperture
# evidence.  A 256-token structured reply can still truncate a multi-region
# drawer/front observation; use the same 384-token M1 budget as V3 while
# leaving room/M2/M3 budgets independently configured.
SEMANTIC_ATTRIBUTE_MAX_OUTPUT_TOKENS=${SEMANTIC_ATTRIBUTE_MAX_OUTPUT_TOKENS:-384}
# Keep this unset until the method is selected.  M1 can emit grounded drawer
# regions and is invoked at a safe observation pose, so it gets its own bounded
# request budget rather than inheriting the shorter M2/M3 decision timeout.
SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S=${SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S:-}
SEMANTIC_MAPPING_OVERRIDE=${SEMANTIC_MAPPING_OVERRIDE:-}
SEMANTIC_MODEL_ENV_FILE=${SEMANTIC_MODEL_ENV_FILE:-${REPO_ROOT}/.env}
RAW_OCCUPANCY_GRID_TOPIC=${RAW_OCCUPANCY_GRID_TOPIC:-/struct_mapping/occ_map}
SKIP_DEBUG_RECORDER=${SKIP_DEBUG_RECORDER:-false}
SKIP_OFFLINE_VIDEO=${SKIP_OFFLINE_VIDEO:-false}
SKIP_COVERAGE=${SKIP_COVERAGE:-false}
ENABLE_COSTMAP_LATENCY_PROBE=${ENABLE_COSTMAP_LATENCY_PROBE:-false}
COSTMAP_LATENCY_PROBE_OUTPUT_DIR=${COSTMAP_LATENCY_PROBE_OUTPUT_DIR:-${OUTPUT_DIR}/costmap_latency}

case "${METHOD}" in
  semantic_interaction_exploration)
    START_SEMANTIC_DECISION=true
    COMPLETION_MODE=semantic
    FORCE_CLOSE_CONTAINERS=true
    SEMANTIC_DECISION_OVERRIDE=${SEMANTIC_DECISION_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/interactive_exploration.yaml}
    EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_controlled_explore.yaml}
    ;;
  semantic_interaction_object_goal)
    START_SEMANTIC_DECISION=true
    COMPLETION_MODE=semantic
    FORCE_CLOSE_CONTAINERS=true
    COMPLETION_POST_HOLD_STEPS=${COMPLETION_POST_HOLD_STEPS:-10}
    SEMANTIC_DECISION_OVERRIDE=${SEMANTIC_DECISION_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/object_goal_runtime.yaml}
    EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_controlled_explore.yaml}
    ;;
  frontier_only)
    START_SEMANTIC_DECISION=false
    COMPLETION_MODE=frontier
    ;;
  interactive_rule)
    START_SEMANTIC_DECISION=true
    COMPLETION_MODE=semantic
    EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_controlled_explore.yaml}
    ;;
  container_exploration)
    START_SEMANTIC_DECISION=true
    COMPLETION_MODE=semantic
    FORCE_CLOSE_CONTAINERS=true
    SEMANTIC_DECISION_OVERRIDE=${SEMANTIC_DECISION_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/interactive_exploration.yaml}
    EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_controlled_explore.yaml}
    ;;
  object_goal_rule)
    START_SEMANTIC_DECISION=true
    COMPLETION_MODE=semantic
    FORCE_CLOSE_CONTAINERS=true
    COMPLETION_POST_HOLD_STEPS=${COMPLETION_POST_HOLD_STEPS:-10}
    SEMANTIC_DECISION_OVERRIDE=${SEMANTIC_DECISION_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/object_goal_fridge.yaml}
    EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_controlled_explore.yaml}
    ;;
  object_goal_model_mock)
    START_SEMANTIC_DECISION=true
    COMPLETION_MODE=semantic
    FORCE_CLOSE_CONTAINERS=true
    COMPLETION_POST_HOLD_STEPS=${COMPLETION_POST_HOLD_STEPS:-10}
    SEMANTIC_DECISION_OVERRIDE=${SEMANTIC_DECISION_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/object_goal_fridge_model_mock.yaml}
    EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_controlled_explore.yaml}
    ;;
  object_goal_runtime)
    START_SEMANTIC_DECISION=true
    COMPLETION_MODE=semantic
    FORCE_CLOSE_CONTAINERS=true
    COMPLETION_POST_HOLD_STEPS=${COMPLETION_POST_HOLD_STEPS:-10}
    SEMANTIC_DECISION_OVERRIDE=${SEMANTIC_DECISION_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/object_goal_runtime.yaml}
    EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_controlled_explore.yaml}
    ;;
  full_mllm_exploration)
    START_SEMANTIC_DECISION=true
    COMPLETION_MODE=semantic
    FORCE_CLOSE_CONTAINERS=true
    ENABLE_ATTRIBUTE_INFERENCE=true
    BYPASS_UNSAFE_OPEN_SWEEP=${BYPASS_UNSAFE_OPEN_SWEEP:-false}
    MLLM_DECISION_TIMEOUT_S=${MLLM_DECISION_TIMEOUT_S:-3.0}
    SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S=${SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S:-8.0}
    export SEMANTIC_MODEL_TIMEOUT_S="${MLLM_DECISION_TIMEOUT_S}"
    SEMANTIC_DECISION_OVERRIDE=${SEMANTIC_DECISION_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/full_mllm_interactive_exploration.yaml}
    SEMANTIC_MAPPING_OVERRIDE=${SEMANTIC_MAPPING_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/full_mllm_mapping.yaml}
    EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_controlled_explore.yaml}
    ;;
  full_mllm_object_goal)
    START_SEMANTIC_DECISION=true
    COMPLETION_MODE=semantic
    FORCE_CLOSE_CONTAINERS=true
    COMPLETION_POST_HOLD_STEPS=${COMPLETION_POST_HOLD_STEPS:-10}
    ENABLE_ATTRIBUTE_INFERENCE=true
    BYPASS_UNSAFE_OPEN_SWEEP=${BYPASS_UNSAFE_OPEN_SWEEP:-false}
    MLLM_DECISION_TIMEOUT_S=${MLLM_DECISION_TIMEOUT_S:-3.0}
    SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S=${SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S:-8.0}
    export SEMANTIC_MODEL_TIMEOUT_S="${MLLM_DECISION_TIMEOUT_S}"
    SEMANTIC_DECISION_OVERRIDE=${SEMANTIC_DECISION_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/full_mllm_object_goal_runtime.yaml}
    SEMANTIC_MAPPING_OVERRIDE=${SEMANTIC_MAPPING_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/full_mllm_mapping.yaml}
    EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_controlled_explore.yaml}
    ;;
  full_mllm_object_goal_apple)
    START_SEMANTIC_DECISION=true
    COMPLETION_MODE=semantic
    FORCE_CLOSE_CONTAINERS=true
    COMPLETION_POST_HOLD_STEPS=${COMPLETION_POST_HOLD_STEPS:-10}
    ENABLE_ATTRIBUTE_INFERENCE=true
    BYPASS_UNSAFE_OPEN_SWEEP=${BYPASS_UNSAFE_OPEN_SWEEP:-false}
    MLLM_DECISION_TIMEOUT_S=${MLLM_DECISION_TIMEOUT_S:-3.0}
    SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S=${SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S:-8.0}
    export SEMANTIC_MODEL_TIMEOUT_S="${MLLM_DECISION_TIMEOUT_S}"
    SEMANTIC_DECISION_OVERRIDE=${SEMANTIC_DECISION_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/full_mllm_object_goal_apple.yaml}
    SEMANTIC_MAPPING_OVERRIDE=${SEMANTIC_MAPPING_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/full_mllm_mapping.yaml}
    EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_controlled_explore.yaml}
    RUNTIME_TARGET_MODE=${RUNTIME_TARGET_MODE:-none}
    ;;
  *)
    print -u2 -- "Unsupported METHOD=${METHOD}; use semantic_interaction_exploration, semantic_interaction_object_goal, frontier_only, interactive_rule, container_exploration, object_goal_rule, object_goal_model_mock, object_goal_runtime, full_mllm_exploration, full_mllm_object_goal, or full_mllm_object_goal_apple"
    exit 2
    ;;
esac

BYPASS_UNSAFE_OPEN_SWEEP=${BYPASS_UNSAFE_OPEN_SWEEP:-false}
SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S=${SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S:-8.0}

COMPLETION_POST_HOLD_STEPS=${COMPLETION_POST_HOLD_STEPS:-0}

mkdir -p "${OUTPUT_DIR}/sim" "${OUTPUT_DIR}/debug" "${OUTPUT_DIR}/ros_home/log"
if [[ "${ENABLE_RECORDING}" == true ]]; then
  mkdir -p "${OUTPUT_DIR}/videos"
fi
export ROS_MASTER_URI
export ROS_IP=${ROS_IP:-127.0.0.1}
export ROS_HOSTNAME=${ROS_HOSTNAME:-127.0.0.1}
export ROS_HOME="${OUTPUT_DIR}/ros_home"
export ROS_LOG_DIR="${OUTPUT_DIR}/ros_home/log"
export SEMANTIC_DECISION_ENV_FILE="${SEMANTIC_MODEL_ENV_FILE}"
export SEMANTIC_MODEL_METRICS_PATH="${OUTPUT_DIR}/mllm_metrics.jsonl"

set +u
CONDA_SH=${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}
# Batch workers can be launched by a service account while the shared Conda
# installation belongs to the experiment user.  Prefer an explicit CONDA_SH,
# but otherwise discover the active `conda` executable before rejecting the
# run instead of assuming that $HOME owns the installation.
if [[ ! -f "${CONDA_SH}" ]]; then
  CONDA_BIN=$(command -v conda 2>/dev/null || true)
  if [[ -n "${CONDA_BIN}" ]]; then
    CONDA_SH_CANDIDATE="$(cd -- "$(dirname -- "${CONDA_BIN}")/.." && pwd)/etc/profile.d/conda.sh"
    if [[ -f "${CONDA_SH_CANDIDATE}" ]]; then
      CONDA_SH="${CONDA_SH_CANDIDATE}"
    fi
  fi
fi
if [[ ! -f "${CONDA_SH}" ]]; then
  printf '%s\n' "Missing conda initialization script: ${CONDA_SH}" >&2
  exit 2
fi
source "${CONDA_SH}"
CONDA_ENV=${CONDA_ENV:-mlspaces}
conda activate "${CONDA_ENV}"
PYTHON_BIN=${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}
if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf '%s\n' "Missing MolmoSpaces Python executable: ${PYTHON_BIN}" >&2
  exit 2
fi
MLSPACES_SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
source "${ROS_SETUP}"
ROS_SOURCE_DIR=${ROS_SOURCE_DIR:-$(cd -- "$(dirname -- "${ROS_SETUP}")/../src" && pwd)}
if [[ ! -d "${ROS_SOURCE_DIR}" ]]; then
  printf '%s\n' "Missing ROS source directory: ${ROS_SOURCE_DIR}" >&2
  exit 2
fi
set -u
# Keep each batch worker on its explicitly isolated ROS master after sourcing
# the workspace setup, while loading OpenCV/MuJoCo from the MolmoSpaces env.
export ROS_MASTER_URI="${RUN_ROS_MASTER_URI}"
export ROS_PACKAGE_PATH="${ROS_SOURCE_DIR}:${ROS_PACKAGE_PATH#*:}"
export PYTHONPATH="${ROS_SOURCE_DIR}/semantic_mapping_py_pkg/scripts:${ROS_SOURCE_DIR}/semantic_decision_py_pkg/scripts:${ROS_SOURCE_DIR}/semantic_mllm_py_pkg/scripts:${ROS_SOURCE_DIR}/explore_py_pkg/scripts:${MLSPACES_SITE_PACKAGES}:${PYTHONPATH:-}"

python() { "${PYTHON_BIN}" "$@"; }

# A decision override may opt in to the dynamically derived rule-oracle axis.
# An explicit environment value still wins, which makes the mode easy to turn
# off in a reproduced run.  This is deliberately not a scene geometry lookup.
if [[ -z "${GT_EMIT_INTERACTION_APPROACH_AXIS}" && -n "${SEMANTIC_DECISION_OVERRIDE}" && -f "${SEMANTIC_DECISION_OVERRIDE}" ]]; then
  GT_EMIT_INTERACTION_APPROACH_AXIS=$(python -c 'import sys,yaml; data=yaml.safe_load(open(sys.argv[1])) or {}; value=(data.get("runtime") or {}).get("rule_oracle_gt_interaction_axis", False); print("true" if bool(value) else "false")' "${SEMANTIC_DECISION_OVERRIDE}")
fi
GT_EMIT_INTERACTION_APPROACH_AXIS=${GT_EMIT_INTERACTION_APPROACH_AXIS:-false}
if [[ "${GT_EMIT_INTERACTION_APPROACH_AXIS}" == true && "${METHOD}" != interactive_rule ]]; then
  print -u2 -- "GT_EMIT_INTERACTION_APPROACH_AXIS is restricted to METHOD=interactive_rule"
  exit 2
fi

ROS_DEVEL_ROOT=$(dirname -- "${ROS_SETUP}")
for ROS_EXECUTABLE in \
  struct_mapping_pkg/slam_gmapping \
  struct_mapping_pkg/voronoi_mapping_node \
  nav_pkg/relay_node; do
  if [[ ! -x "${ROS_DEVEL_ROOT}/lib/${ROS_EXECUTABLE}" ]]; then
    print -u2 -- "Missing ROS executable: ${ROS_DEVEL_ROOT}/lib/${ROS_EXECUTABLE}"
    print -u2 -- "Build the workspace first: cd ${REPO_ROOT}/Interactive-Nav-SG-nav && catkin_make --pkg struct_mapping_pkg nav_pkg"
    exit 3
  fi
done

FIXED_ROUTE_ARGS=""
if [[ "${USE_FIXED_ROUTE}" == true ]]; then
  ROUTE_FIELDS=$(python -c 'import sys,yaml; p=yaml.safe_load(open(sys.argv[1])); r=next(x for x in p["routes"] if x["route_id"]==sys.argv[2]); print("{}\t{}".format(r["seed"], ",".join(str(v) for v in r["start_xyyaw"])))' "${ROUTE_CONFIG}" "${ROUTE_ID}")
  IFS=$'\t' read -r SCENE_SEED ROBOT_XYYAW <<< "${ROUTE_FIELDS}"
  FIXED_ROUTE_ARGS="--fixed_robot_xyyaw ${ROBOT_XYYAW}"
fi

signal_process_tree() {
  local signal="$1"
  local pid="$2"
  local child=""
  [[ -n "${pid}" ]] || return
  while IFS= read -r child; do
    [[ -n "${child}" ]] && signal_process_tree "${signal}" "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill "-${signal}" "${pid}" 2>/dev/null || true
}

cleanup_process() {
  local pid="${1:-}"
  local grace_s="${2:-20}"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return
  fi
  signal_process_tree INT "${pid}"
  local max_attempts=$((grace_s * 2))
  local _attempt=1
  while (( _attempt <= max_attempts )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      return
    fi
    sleep 0.5
    _attempt=$((_attempt + 1))
  done
  signal_process_tree TERM "${pid}"
  sleep 1
  signal_process_tree KILL "${pid}"
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  cleanup_process "${LAUNCH_PID:-}" 20
  cleanup_process "${RECORDER_PID:-}" 20
  cleanup_process "${COSTMAP_PROBE_PID:-}" 10
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
roscore -p "${MASTER_PORT}" >"${OUTPUT_DIR}/roscore.log" 2>&1 &
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

COSTMAP_PROBE_PID=""
if [[ "${ENABLE_COSTMAP_LATENCY_PROBE}" == true ]]; then
  PYTHONUNBUFFERED=1 python -u "${REPO_ROOT}/scripts/InteractiveNav/measure_costmap_latency.py" \
    --output-dir "${COSTMAP_LATENCY_PROBE_OUTPUT_DIR}" \
    >"${OUTPUT_DIR}/costmap_latency_probe.log" 2>&1 &
  COSTMAP_PROBE_PID=$!
  sleep 0.25
fi

RECORDER_PID=""
if [[ "${SKIP_DEBUG_RECORDER}" != true ]]; then
  if [[ "${ENABLE_RECORDING}" == true ]]; then
    EXTERNAL_VIDEO_ARGS=(--no-external-video)
    EXTERNAL_VIDEO_OVERLAY_ARG=--external-video-overlay
    if [[ "${EXTERNAL_VIDEO_OVERLAY}" != true ]]; then
      EXTERNAL_VIDEO_OVERLAY_ARG=--no-external-video-overlay
    fi
    if [[ "${ENABLE_EXTERNAL_VIDEO}" == true ]]; then
      EXTERNAL_VIDEO_ARGS=(--external-image-topic "${EXTERNAL_IMAGE_TOPIC}" --external-video)
    fi
    PAPER_FRAME_EXPORT_ARGS=(--no-paper-frame-exports)
    if [[ "${PAPER_FRAME_EXPORTS}" == true ]]; then
      PAPER_FRAME_EXPORT_ARGS=(
        --paper-frame-exports
        --video-step-sync-topic /molmo_spaces/step_sync
        --step-sync-queue-size "${STEP_SYNC_QUEUE_SIZE}"
      )
    fi
    RECORDER_SEMANTIC_XY_INSET_ARGS=(--no-video-semantic-xy-overview-inset)
    if [[ "${VIDEO_SEMANTIC_XY_OVERVIEW_INSET}" == true ]]; then
      RECORDER_SEMANTIC_XY_INSET_ARGS=(--video-semantic-xy-overview-inset)
    fi
    PYTHONUNBUFFERED=1 python -u "${REPO_ROOT}/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts/record_explore_debug.py" \
      --output-dir "${OUTPUT_DIR}/debug" \
      --occupancy-grid-topic /semantic_mapping/planning_occ_map \
      --raw-occupancy-grid-topic "${RAW_OCCUPANCY_GRID_TOPIC}" \
      --projected-scan-topic "${MAPPING_SCAN_TOPIC}" \
      --first-person-video-capture-mode step \
      --video-step-sync-topic /molmo_spaces/step_sync \
      --step-capture-ack-topic /molmo_spaces/step_capture_ack \
      --step-sync-capture-every "${STEP_SYNC_CAPTURE_EVERY}" \
      --step-sync-image-cache-size "${STEP_SYNC_IMAGE_CACHE_SIZE}" \
      --semantic-video \
      --semantic-video-max-object-nodes "${SEMANTIC_VIDEO_MAX_OBJECT_NODES}" \
      --first-person-video-with-map \
      --first-person-video-fps "${VIDEO_FPS}" \
      --first-person-video-width-px "${VIDEO_PANEL_WIDTH_PX}" \
      --external-video-width-px "${EXTERNAL_VIDEO_WIDTH_PX}" \
      "${EXTERNAL_VIDEO_OVERLAY_ARG}" \
      "${PAPER_FRAME_EXPORT_ARGS[@]}" \
      --video-frame-job-queue-size "${VIDEO_FRAME_JOB_QUEUE_SIZE}" \
      --video-frame-queue-overflow "${VIDEO_FRAME_QUEUE_OVERFLOW}" \
      --artifact-write-queue-size "${ARTIFACT_WRITE_QUEUE_SIZE}" \
      --artifact-write-workers "${ARTIFACT_WRITE_WORKERS}" \
      --artifact-write-overflow "${ARTIFACT_WRITE_OVERFLOW}" \
      --performance-log-every-n-frames "${RECORDER_PERFORMANCE_LOG_EVERY_N_FRAMES}" \
      --video-history-size "${VIDEO_HISTORY_SIZE}" \
      --image-queue-size "${IMAGE_QUEUE_SIZE}" \
      --video-global-panel-scale 1.8 \
      --video-room-panel-scale "${VIDEO_ROOM_PANEL_SCALE}" \
      --video-semantic-xy-panel-scale "${VIDEO_SEMANTIC_XY_PANEL_SCALE}" \
      --video-semantic-xy-label-mode "${VIDEO_SEMANTIC_XY_LABEL_MODE}" \
      "${RECORDER_SEMANTIC_XY_INSET_ARGS[@]}" \
      --no-runtime-video-encode \
      --offline-video-only \
      --first-person-video-h264-preset "${VIDEO_ENCODER_PRESET}" \
      "${EXTERNAL_VIDEO_ARGS[@]}" \
      --video-save-panel-frames \
      --video-save-composite-frames \
      --no-first-person-video-h264 \
      >"${OUTPUT_DIR}/recorder.log" 2>&1 &
  else
    PYTHONUNBUFFERED=1 python -u "${REPO_ROOT}/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts/record_explore_debug.py" \
      --output-dir "${OUTPUT_DIR}/debug" \
      --occupancy-grid-topic /semantic_mapping/planning_occ_map \
      --raw-occupancy-grid-topic "${RAW_OCCUPANCY_GRID_TOPIC}" \
      --projected-scan-topic "${MAPPING_SCAN_TOPIC}" \
      --no-first-person-video \
      --no-first-person-video-with-map \
      --no-semantic-video \
      --no-external-video \
      >"${OUTPUT_DIR}/recorder.log" 2>&1 &
  fi
  RECORDER_PID=$!
  sleep 1
fi

if [[ -z "${RUNTIME_TARGET_MODE:-}" ]]; then
  RUNTIME_TARGET_MODE=none
  if [[ "${METHOD}" == object_goal_runtime || "${METHOD}" == semantic_interaction_object_goal || "${METHOD}" == full_mllm_object_goal ]]; then
    RUNTIME_TARGET_MODE=random_far_container_object
  fi
fi
RUNTIME_TARGET_SELECTION_INPUT_PATH=${RUNTIME_TARGET_SELECTION_INPUT_PATH:-}
RUNTIME_TARGET_SELECTION_INPUT_ARGS=""
if [[ -n "${RUNTIME_TARGET_SELECTION_INPUT_PATH}" ]]; then
  RUNTIME_TARGET_SELECTION_INPUT_ARGS="--runtime_target_selection_input_path ${RUNTIME_TARGET_SELECTION_INPUT_PATH}"
fi
if [[ "${ENABLE_RECORDING}" == true ]]; then
  SIM_CAPTURE_ARGS="--observation_queue_size ${OBSERVATION_QUEUE_SIZE} --step_frame_dir ${OUTPUT_DIR}/sim_step_frames --step_frame_queue_size 4"
else
  SIM_CAPTURE_ARGS="--observation_queue_size 1"
fi
PUBLISH_DEBUG_FRONT_CAMERA=false
DEBUG_CAMERA_ARGS=""
if [[ "${ENABLE_EXTERNAL_VIDEO}" == true ]]; then
  PUBLISH_DEBUG_FRONT_CAMERA=true
  DEBUG_CAMERA_ARGS="--debug_front_camera_offset=${DEBUG_FOLLOW_CAMERA_OFFSET} --debug_front_camera_lookat_offset=${DEBUG_FOLLOW_CAMERA_LOOKAT_OFFSET} --debug_front_camera_fov_deg=${DEBUG_FOLLOW_CAMERA_FOV_DEG}"
fi
STEP_READY_BARRIER_ENABLED=${STEP_READY_BARRIER_ENABLED:-true}
# Keep readiness smoke tests bounded; the simulator remains fail-open after
# this wait and the outer SIM_TIMEOUT_S is the hard process guard.
STEP_READY_TIMEOUT_S=${STEP_READY_TIMEOUT_S:-2.0}
# Only the one-time first-map bootstrap may wait longer; all later simulator
# steps keep the bounded 2s fail-open protection above.
STEP_READY_BOOTSTRAP_TIMEOUT_S=${STEP_READY_BOOTSTRAP_TIMEOUT_S:-10.0}
# The recorder acknowledges each queued raw step snapshot before the simulator
# advances. Enable this only when the offline recorder is actually running.
if [[ "${ENABLE_RECORDING}" == true ]] && [[ "${SKIP_DEBUG_RECORDER}" != true ]]; then
  STEP_CAPTURE_ACK_BARRIER_ENABLED=${STEP_CAPTURE_ACK_BARRIER_ENABLED:-true}
else
  STEP_CAPTURE_ACK_BARRIER_ENABLED=${STEP_CAPTURE_ACK_BARRIER_ENABLED:-false}
fi
STEP_CAPTURE_ACK_TIMEOUT_S=${STEP_CAPTURE_ACK_TIMEOUT_S:-2.0}
# At 5 Hz, command collection cannot keep the historical 0.5s wait; 0.2s
# keeps the bridge cadence aligned with policy_dt_ms while remaining overrideable.
ACTION_TIMEOUT_S=${ACTION_TIMEOUT_S:-0.2}
GT_INTERACTION_AXIS_ARGS="--realtime_gt_emit_interaction_approach_axis ${GT_EMIT_INTERACTION_APPROACH_AXIS}"
SIM_EXTRA_ARGS="--seed ${SCENE_SEED} ${FIXED_ROUTE_ARGS} --initial_door_state ${INITIAL_DOOR_STATE} --enable_force_interaction true --force_interaction_close_all_containers_on_prepare ${FORCE_CLOSE_CONTAINERS} --force_interaction_bypass_unsafe_open_sweep ${BYPASS_UNSAFE_OPEN_SWEEP} --force_interaction_log_path ${OUTPUT_DIR}/force_interaction_events.json --force_interaction_execution_mode ${INTERACTION_EXECUTION_MODE} --force_interaction_transition_steps ${INTERACTION_TRANSITION_STEPS} --force_interaction_drawer_execution_mode ${DRAWER_EXECUTION_MODE} --force_interaction_drawer_transition_steps ${DRAWER_TRANSITION_STEPS} --force_interaction_drawer_observation_steps ${DRAWER_OBSERVATION_STEPS} --realtime_gt_step_interval ${GT_STEP_INTERVAL} --realtime_gt_min_visible_pixels ${GT_MIN_VISIBLE_PIXELS} --realtime_gt_min_visible_fraction ${GT_MIN_VISIBLE_FRACTION} --realtime_gt_required_consecutive_observations ${GT_REQUIRED_CONSECUTIVE_OBSERVATIONS} --realtime_gt_max_distance_m ${GT_MAX_DISTANCE_M} ${GT_INTERACTION_AXIS_ARGS} --action_timeout_s ${ACTION_TIMEOUT_S} --pointcloud_stride ${POINTCLOUD_STRIDE} ${DEPTH_SCAN_ARGS} --step_capture_ack_topic /molmo_spaces/step_capture_ack --step_capture_ack_barrier_enabled ${STEP_CAPTURE_ACK_BARRIER_ENABLED} --step_capture_ack_timeout_s ${STEP_CAPTURE_ACK_TIMEOUT_S} --step_ready_barrier_enabled ${STEP_READY_BARRIER_ENABLED} --step_ready_warmup_skip_frames ${STEP_READY_WARMUP_SKIP_FRAMES} --step_ready_timeout_s ${STEP_READY_TIMEOUT_S} --step_ready_bootstrap_timeout_s ${STEP_READY_BOOTSTRAP_TIMEOUT_S} --map_warmup_skip_frames ${MAP_WARMUP_SKIP_FRAMES} ${SIM_CAPTURE_ARGS} ${DEBUG_CAMERA_ARGS} --extra_image_queue_size ${EXTRA_IMAGE_QUEUE_SIZE} --require_move_base_active_for_cmd_vel false --no-retain_task_history --runtime_target_selection_mode ${RUNTIME_TARGET_MODE} --runtime_target_selection_top_k 3 --runtime_target_selection_path ${OUTPUT_DIR}/target_selection.json ${RUNTIME_TARGET_SELECTION_INPUT_ARGS} --completion_mode ${COMPLETION_MODE} --completion_confirmations ${COMPLETION_CONFIRMATIONS} --completion_post_hold_steps ${COMPLETION_POST_HOLD_STEPS} --completion_status_path ${OUTPUT_DIR}/completion_status.json --step_log_every_n_steps ${TIMING_LOG_EVERY_N_STEPS} --timing_log_every_n_frames ${TIMING_LOG_EVERY_N_STEPS} --sim_timing_log_every_n_steps ${TIMING_LOG_EVERY_N_STEPS}"

roslaunch "${REPO_ROOT}/Interactive-Nav-SG-nav/src/nav_pkg/launch/molmospaces_nav_system.launch" \
  start_sim:=true \
  start_mapping:=true \
  mapping_mode:=odom_locked \
  mapping_scan_source:="${MAPPING_SCAN_SOURCE}" \
  mapping_scan_topic:="${MAPPING_SCAN_TOPIC}" \
  scan_filter_tolerance_sec:="${SCAN_FILTER_TOLERANCE_SEC}" \
  start_semantic_mapping:=true \
  semantic_source:=realtime_gt \
  publish_realtime_gt:=true \
  start_nav:=true \
  start_explore:=false \
  start_explore_py:=true \
  explore_py_config_override_file:="${EXPLORE_PY_CONFIG_OVERRIDE}" \
  start_semantic_decision:="${START_SEMANTIC_DECISION}" \
  semantic_attribute_inference:="${ENABLE_ATTRIBUTE_INFERENCE}" \
  semantic_attribute_model_name:="${SEMANTIC_ATTRIBUTE_MODEL_NAME}" \
  semantic_attribute_max_output_tokens:="${SEMANTIC_ATTRIBUTE_MAX_OUTPUT_TOKENS}" \
  semantic_attribute_request_timeout_s:="${SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S}" \
  semantic_decision_config_file:="${SEMANTIC_DECISION_CONFIG}" \
  semantic_decision_config_override_file:="${SEMANTIC_DECISION_OVERRIDE}" \
  semantic_config_override_file:="${SEMANTIC_MAPPING_OVERRIDE}" \
  nav_config_override_file:="${ROUTE_NAV_CONFIG}" \
  local_costmap_inflation_radius:="${LOCAL_COSTMAP_INFLATION_RADIUS}" \
  exploration_only:=true \
  randomize_camera:=false \
  publish_debug_front_camera:="${PUBLISH_DEBUG_FRONT_CAMERA}" \
  robot:=rby1 \
  scene_dataset:=procthor-10k \
  data_split:=train \
  house_ind:="${HOUSE_IND}" \
  house_inds:="${HOUSE_IND}" \
  task_horizon:="${TASK_HORIZON}" \
  scene_timeout_s:="${SIM_TIMEOUT_S}" \
  max_consecutive_action_timeouts:=0 \
  output_dir:="${OUTPUT_DIR}/sim" \
  sim_extra_args:="${SIM_EXTRA_ARGS}" \
  >"${OUTPUT_DIR}/roslaunch.log" 2>&1 &
LAUNCH_PID=$!

set +e
timeout --signal=INT "${SIM_TIMEOUT_S}s" tail --pid="${LAUNCH_PID}" -f /dev/null
LAUNCH_WAIT_RC=$?
set -e
if [[ "${LAUNCH_WAIT_RC}" -eq 124 ]]; then
  cleanup_process "${LAUNCH_PID}"
  LAUNCH_PID=""
  print -u2 -- "Navigation launch timed out after ${SIM_TIMEOUT_S}s"
  exit 124
fi
if [[ "${LAUNCH_WAIT_RC}" -ne 0 ]] && [[ "${LAUNCH_WAIT_RC}" -ne 130 ]]; then
  print -u2 -- "Navigation launch failed with ${LAUNCH_WAIT_RC}"
  exit "${LAUNCH_WAIT_RC}"
fi
set +e
wait "${LAUNCH_PID}"
LAUNCH_EXIT=$?
set -e
LAUNCH_PID=""
if [[ "${LAUNCH_EXIT}" -ne 0 ]] && [[ "${LAUNCH_EXIT}" -ne 130 ]]; then
  # The finite simulator is a required roslaunch child.  Once it finishes
  # cleanly, roslaunch shuts down the remaining graph and returns 1 even
  # though the episode loop completed.  Preserve real launch failures, but
  # allow the recorder/video post-processing to run for that normal case.
  if grep -Fq "Worker 0 completed processing assigned houses" "${OUTPUT_DIR}/roslaunch.log" \
    && grep -Fq "Completed 1 houses, skipped 0 houses" "${OUTPUT_DIR}/roslaunch.log"; then
    print -u2 -- "Navigation simulator completed; accepting roslaunch exit ${LAUNCH_EXIT} for post-processing"
  else
    print -u2 -- "Navigation launch exited with ${LAUNCH_EXIT}"
    exit "${LAUNCH_EXIT}"
  fi
fi

if [[ -n "${COSTMAP_PROBE_PID}" ]]; then
  kill -TERM "${COSTMAP_PROBE_PID}" 2>/dev/null || true
  wait "${COSTMAP_PROBE_PID}" 2>/dev/null || true
  COSTMAP_PROBE_PID=""
fi

RECORDER_DRAIN_STATUS=0
if [[ -n "${RECORDER_PID}" ]] && [[ "${ENABLE_RECORDING}" == true ]]; then
  "${PYTHON_BIN}" "${RECORDER_DRAIN_HELPER}" \
    --sim-manifest "${OUTPUT_DIR}/sim_step_frames/manifest.jsonl" \
    --video-frames-csv "${OUTPUT_DIR}/debug/video_frames.csv" \
    --timeout-sec "${RECORDER_DRAIN_TIMEOUT_S}" \
    --poll-sec "${RECORDER_DRAIN_POLL_S}" \
    --progress-sec "${RECORDER_DRAIN_PROGRESS_S}" \
    --stall-timeout-sec "${RECORDER_DRAIN_STALL_TIMEOUT_S}" \
    --recorder-pid "${RECORDER_PID}" \
    --step-sync-capture-every "${STEP_SYNC_CAPTURE_EVERY}" \
    --offline-raw-recording \
    --raw-step-manifest "${OUTPUT_DIR}/debug/raw/step_boundaries.jsonl" \
    || RECORDER_DRAIN_STATUS=$?
fi
if [[ -n "${RECORDER_PID}" ]]; then
  cleanup_process "${RECORDER_PID}" "${RECORDER_SHUTDOWN_GRACE_S}"
  RECORDER_PID=""
fi
FINAL_RECORDER_DRAIN_STATUS=0
if [[ "${ENABLE_RECORDING}" == true ]] && [[ "${SKIP_DEBUG_RECORDER}" != true ]]; then
  "${PYTHON_BIN}" "${RECORDER_DRAIN_HELPER}" \
    --sim-manifest "${OUTPUT_DIR}/sim_step_frames/manifest.jsonl" \
    --video-frames-csv "${OUTPUT_DIR}/debug/video_frames.csv" \
    --timeout-sec 0 \
    --step-sync-capture-every "${STEP_SYNC_CAPTURE_EVERY}" \
    --offline-raw-recording \
    --raw-step-manifest "${OUTPUT_DIR}/debug/raw/step_boundaries.jsonl" \
    --recorder-summary "${OUTPUT_DIR}/debug/summary.json" \
    || FINAL_RECORDER_DRAIN_STATUS=$?
fi
VIDEO_STATE_ALIGNMENT=exact
if (( FINAL_RECORDER_DRAIN_STATUS != 0 )); then
  printf '%s\n' "Recorder finalization failed (live=${RECORDER_DRAIN_STATUS}, final=${FINAL_RECORDER_DRAIN_STATUS})." >&2
  exit 4
fi
if (( RECORDER_DRAIN_STATUS != 0 )); then
  VIDEO_STATE_ALIGNMENT=latest
  printf '%s\n' "Recorder stopped with incomplete live state panels; using latest-state offline fallback with exact simulator camera frames." >&2
fi

if [[ "${ENABLE_RECORDING}" == true ]] && [[ "${SKIP_OFFLINE_VIDEO}" != true ]] && [[ "${SKIP_DEBUG_RECORDER}" != true ]]; then
  OFFLINE_VIDEO_START=$(python -c 'import time; print(time.perf_counter())')
  OFFLINE_VIDEO_INSET_ARGS=()
  if [[ "${VIDEO_SEMANTIC_XY_OVERVIEW_INSET}" == true ]]; then
    OFFLINE_VIDEO_INSET_ARGS=(--semantic-xy-overview-inset)
  fi
  python "${VIDEO_BUILDER}" \
    --scene-dir "${OUTPUT_DIR}" \
    --debug-dir "${OUTPUT_DIR}/debug" \
    --fps "${VIDEO_FPS}" \
    --state-alignment "${VIDEO_STATE_ALIGNMENT}" \
    --output-stem overview_6panel \
    "${OFFLINE_VIDEO_INSET_ARGS[@]}" \
    >"${OUTPUT_DIR}/offline_video.log" 2>&1
  OFFLINE_VIDEO_ELAPSED_SEC=$(python - "${OFFLINE_VIDEO_START}" <<'PY'
import sys
import time
print(max(0.0, time.perf_counter() - float(sys.argv[1])))
PY
  )
  if [[ ! -f "${OUTPUT_DIR}/videos/overview_6panel.mp4" ]] && \
     [[ -f "${OUTPUT_DIR}/debug/videos/overview_6panel.mp4" ]]; then
    mv "${OUTPUT_DIR}/debug/videos/overview_6panel.mp4" "${OUTPUT_DIR}/videos/overview_6panel.mp4"
  fi
else
  OFFLINE_VIDEO_ELAPSED_SEC=0.0
fi
print -r -- "${OFFLINE_VIDEO_ELAPSED_SEC}" >"${OUTPUT_DIR}/offline_video_elapsed_sec.txt"

if [[ "${SKIP_COVERAGE}" != true ]] && [[ "${SKIP_DEBUG_RECORDER}" != true ]]; then
  ANALYSIS_START=$(python -c 'import time; print(time.perf_counter())')
  python "${SCRIPT_DIR}/evaluate_exploration_coverage.py" \
    --run-dir "${OUTPUT_DIR}/debug" \
    --robot rby1 \
    --scene-dataset procthor-10k \
    --data-split train \
    --house-ind "${HOUSE_IND}" \
    --gt-agent-radius-m 0.10 \
    >"${OUTPUT_DIR}/coverage.log" 2>&1 || true
  ANALYSIS_ELAPSED_SEC=$(python - "${ANALYSIS_START}" <<'PY'
import sys
import time
print(max(0.0, time.perf_counter() - float(sys.argv[1])))
PY
  )
else
  ANALYSIS_ELAPSED_SEC=0.0
fi
print -r -- "${ANALYSIS_ELAPSED_SEC}" >"${OUTPUT_DIR}/analysis_elapsed_sec.txt"

python - "${OUTPUT_DIR}" "${METHOD}" "${ROUTE_ID}" "${TASK_HORIZON}" "${HOUSE_IND}" "${POINTCLOUD_STRIDE}" "${MAPPING_SCAN_SOURCE}" <<'PY'
import json
import re
from pathlib import Path
import statistics
import sys

output_dir = Path(sys.argv[1])
method = sys.argv[2]
route_id = sys.argv[3]
task_horizon = int(sys.argv[4])
house_ind = int(sys.argv[5])
pointcloud_stride = int(sys.argv[6])
mapping_scan_source = sys.argv[7]
def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

debug_summary = read_json(output_dir / "debug" / "summary.json")
offline_video_summary = read_json(output_dir / "offline_video_summary.json")
sim_frames = int(
    offline_video_summary.get(
        "aligned_sim_frame_count",
        offline_video_summary.get("sim_frame_count", debug_summary.get("step_sync_count", 0)),
    )
    or 0
)
if sim_frames <= 0:
    manifest_path = output_dir / "sim_step_frames" / "manifest.jsonl"
    try:
        sim_frames = sum(1 for line in manifest_path.open(encoding="utf-8") if line.strip())
    except OSError:
        sim_frames = 0
video_frames = int(
    offline_video_summary.get(
        "output_frame_count", debug_summary.get("first_person_video_frame_count", 0)
    )
    or 0
)
video_path = output_dir / "videos" / "overview_6panel.mp4"
video = {
    "output_frame_count": video_frames,
    "video": str(video_path) if video_path.exists() else "",
}
coverage = read_json(output_dir / "debug" / "exploration_coverage.json")
semantic_summary = read_json(output_dir / "debug" / "summary.json").get("semantic_summary", {})
debug_summary = read_json(output_dir / "debug" / "summary.json")
force = read_json(output_dir / "force_interaction_events.json")
completion = read_json(output_dir / "completion_status.json")
if sim_frames <= 0:
    sim_frames = int(completion.get("completed_steps", 0) or 0)
timing_pattern = re.compile(
    r"SimLoop timing over (?P<count>\d+) steps: policy=(?P<policy>[0-9.]+)ms, "
    r"task=(?P<task>[0-9.]+)ms \(physics=(?P<physics>[0-9.]+)ms sensors=(?P<sensors>[0-9.]+)ms\), "
    r"loop=(?P<loop>[0-9.]+)ms, simulated_dt=(?P<dt>[0-9.]+)s"
)
timing_windows = []
roslaunch_log = output_dir / "roslaunch.log"
if roslaunch_log.exists():
    for line in roslaunch_log.read_text(errors="replace").splitlines():
        match = timing_pattern.search(line)
        if match:
            row = {key: float(value) for key, value in match.groupdict().items()}
            row["count"] = int(row["count"])
            timing_windows.append(row)
step_timing = {
    "timing_window_count": len(timing_windows),
    "policy_ms_avg": sum(row["policy"] for row in timing_windows) / len(timing_windows) if timing_windows else None,
    "task_ms_avg": sum(row["task"] for row in timing_windows) / len(timing_windows) if timing_windows else None,
    "physics_ms_avg": sum(row["physics"] for row in timing_windows) / len(timing_windows) if timing_windows else None,
    "sensors_ms_avg": sum(row["sensors"] for row in timing_windows) / len(timing_windows) if timing_windows else None,
    "loop_ms_avg": sum(row["loop"] for row in timing_windows) / len(timing_windows) if timing_windows else None,
    "simulated_dt_s": timing_windows[-1]["dt"] if timing_windows else None,
}
interaction_results = [event.get("result") or {} for event in force.get("events", [])]


def is_physical_container_interaction(result):
    candidate_id = str(result.get("candidate_id") or "")
    sequence_type = str(result.get("sequence_type") or "").casefold()
    interaction_mode = str(result.get("interaction_mode") or "").casefold()
    return (
        candidate_id.startswith("interaction:container_")
        or sequence_type.startswith("drawer_")
        or interaction_mode.startswith("drawer_")
    )


def public_interaction_outcome(result):
    """Keep batch summaries explicit without copying force/private payloads."""

    return {
        "candidate_id": str(result.get("candidate_id") or ""),
        "target_id": str(result.get("node_id") or result.get("object_id") or ""),
        "interaction_mode": str(result.get("interaction_mode") or ""),
        "sequence_type": str(result.get("sequence_type") or ""),
        "status": str(result.get("status") or ""),
        "success": bool(result.get("success", False)),
        "pre_state": str(result.get("pre_state") or ""),
        "post_state": str(result.get("post_state") or ""),
        "task_steps_consumed": int(result.get("task_steps_consumed", 0) or 0),
        "failure_reason": str(result.get("failure_reason") or ""),
    }


physical_container_results = [
    result for result in interaction_results if is_physical_container_interaction(result)
]
drawer_scan_results = [
    result
    for result in physical_container_results
    if str(result.get("sequence_type") or "").casefold() == "drawer_scan"
]
debug_events = []
events_path = output_dir / "debug" / "events.jsonl"
if events_path.exists():
    for line in events_path.read_text().splitlines():
        try:
            debug_events.append(json.loads(line))
        except ValueError:
            pass
decision_rows = [event for event in debug_events if event.get("type") == "semantic_decision_selected"]
feedback_rows = [event for event in debug_events if event.get("type") == "semantic_decision_feedback"]
target_container_candidate_ids = {
    str((event.get("payload") or {}).get("candidate_id") or "")
    for event in decision_rows
    if (event.get("payload") or {}).get("behavior_type") == "INTERACT"
    and bool(((event.get("payload") or {}).get("metadata") or {}).get("target_match"))
}
terminal_feedback = [
    event.get("payload") or {}
    for event in feedback_rows
    if (event.get("payload") or {}).get("status") in {"SUCCEEDED", "FAILED", "CANCELED", "REJECTED"}
]
target_navigation_succeeded = any(
    payload.get("status") == "SUCCEEDED"
    and payload.get("behavior_type") == "NAVIGATE"
    and str(payload.get("candidate_id") or "").startswith("target:")
    for payload in terminal_feedback
)

def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

mllm_rows = []
mllm_path = output_dir / "mllm_metrics.jsonl"
if mllm_path.exists():
    for line in mllm_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            mllm_rows.append(row)
mllm_by_role = {}
for role in sorted({str(row.get("role") or "unknown") for row in mllm_rows}):
    rows = [row for row in mllm_rows if str(row.get("role") or "unknown") == role]
    latencies = [float(row.get("latency_s", 0.0) or 0.0) for row in rows]
    valid = [row for row in rows if not str(row.get("error") or "")]
    mllm_by_role[role] = {
        "request_count": len(rows),
        "valid_response_rate": len(valid) / max(1, len(rows)),
        "mean_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_s": percentile(latencies, 0.50),
        "p95_latency_s": percentile(latencies, 0.95),
        "mean_prompt_tokens": statistics.mean(float(row.get("prompt_tokens", 0) or 0) for row in rows) if rows else 0.0,
        "mean_completion_tokens": statistics.mean(float(row.get("completion_tokens", 0) or 0) for row in rows) if rows else 0.0,
        "mean_reasoning_tokens": statistics.mean(float(row.get("reasoning_tokens", 0) or 0) for row in rows) if rows else 0.0,
        "errors": [str(row.get("error") or "") for row in rows if str(row.get("error") or "")],
    }
result = {
    "method": method,
    "recording_enabled": bool(video_path.exists()),
    "route_id": route_id,
    "house_ind": house_ind,
    "task_horizon": task_horizon,
    "pointcloud_stride": pointcloud_stride,
    "mapping_scan_source": mapping_scan_source,
    "completed_early": 0 < sim_frames < task_horizon,
    "completion_requested": bool(completion.get("requested", False)),
    "completion_reason": completion.get("reason", ""),
    "completion_status": completion,
    "sim_step_frames": sim_frames,
    "video_frames": video.get("output_frame_count", 0),
    "video": video.get("video", ""),
    "coverage_ratio": coverage.get("exploration_coverage_ratio"),
    "mapped_free_coverage_ratio": coverage.get("mapped_free_coverage_ratio"),
    "interaction_count": len(interaction_results),
    "physical_interaction_success_count": sum(
        bool(result.get("success", False)) for result in interaction_results
    ),
    "physical_interaction_failure_count": sum(
        not bool(result.get("success", False)) for result in interaction_results
    ),
    "physical_container_interaction_count": len(physical_container_results),
    "physical_container_interaction_success_count": sum(
        bool(result.get("success", False)) for result in physical_container_results
    ),
    "physical_container_interaction_failure_count": sum(
        not bool(result.get("success", False)) for result in physical_container_results
    ),
    "drawer_scan_interaction_count": len(drawer_scan_results),
    "drawer_scan_success_count": sum(
        bool(result.get("success", False)) for result in drawer_scan_results
    ),
    "physical_container_interaction_outcomes": [
        public_interaction_outcome(result) for result in physical_container_results
    ],
    "interaction_roots": [event.get("object_id", "") for event in interaction_results],
    "interaction_steps": [event.get("step") for event in interaction_results],
    "contains_edge_count": semantic_summary.get("contains_edge_count", 0),
    "container_with_children_count": semantic_summary.get("container_with_children_count", 0),
    "semantic_node_counts": semantic_summary.get("node_counts", {}),
    "decision_count": len(decision_rows),
    "selected_behaviors": [
        (event.get("payload") or {}).get("behavior_type", "") for event in decision_rows
    ],
    "successful_behavior_count": sum(
        (event.get("payload") or {}).get("status") == "SUCCEEDED" for event in feedback_rows
    ),
    "target_goal_success": bool(
        completion.get("target_goal_succeeded", False)
    ) or target_navigation_succeeded,
    "target_selection": read_json(output_dir / "target_selection.json"),
    "target_container_interaction_success": any(
        payload.get("status") == "SUCCEEDED"
        and str(payload.get("candidate_id") or "") in target_container_candidate_ids
        for payload in terminal_feedback
    ),
    "target_object_visible_navigation_success": bool(
        target_navigation_succeeded
        and any(
            (event.get("payload") or {}).get("status") == "SUCCEEDED"
            and (event.get("payload") or {}).get("behavior_type") == "NAVIGATE"
            and str((event.get("payload") or {}).get("candidate_id") or "").startswith("target:")
            for event in feedback_rows
        )
    ),
    "overall_success": bool(
        completion.get("target_goal_succeeded", False)
    ) or target_navigation_succeeded,
    "offline_video_elapsed_sec": float(
        (output_dir / "offline_video_elapsed_sec.txt").read_text().strip()
    ) if (output_dir / "offline_video_elapsed_sec.txt").exists() else None,
    "offline_analysis_elapsed_sec": float(
        (output_dir / "analysis_elapsed_sec.txt").read_text().strip()
    ) if (output_dir / "analysis_elapsed_sec.txt").exists() else None,
    "step_timing": step_timing,
    "valid_step_video": sim_frames > 0 and video.get("output_frame_count") == sim_frames,
    "exact_step_video": bool(
        sim_frames > 0
        and video.get("output_frame_count") == sim_frames
        and debug_summary.get("first_person_video_trigger") == "step_sync"
        and int(debug_summary.get("step_sync_count", 0) or 0) >= sim_frames
        and int(debug_summary.get("video_frame_jobs_dropped", 0) or 0) == 0
        and int(offline_video_summary.get("exact_step_match_count", 0) or 0) == sim_frames
        and int(debug_summary.get("step_sync_image_reuse_count", 0) or 0) == 0
        and int(debug_summary.get("step_sync_placeholder_count", 0) or 0) == 0
    ),
    "step_sync_image_match_count": debug_summary.get("step_sync_image_match_count", 0),
    "step_sync_image_reuse_count": debug_summary.get("step_sync_image_reuse_count", 0),
    "step_sync_placeholder_count": debug_summary.get("step_sync_placeholder_count", 0),
    "mllm_metrics_path": str(mllm_path) if mllm_path.exists() else "",
    "mllm_request_count": len(mllm_rows),
    "mllm_by_role": mllm_by_role,
}
(output_dir / "semantic_exploration_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(result, ensure_ascii=False))
PY

if [[ "${CLEAN_INTERMEDIATE}" == true ]]; then
  rm -rf \
    "${OUTPUT_DIR}/debug/videos/composite_frames" \
    "${OUTPUT_DIR}/debug/stall_snapshots" \
    "${OUTPUT_DIR}/ros_home/log"
fi

print -- "House ${HOUSE_IND} semantic exploration complete: ${OUTPUT_DIR}"
