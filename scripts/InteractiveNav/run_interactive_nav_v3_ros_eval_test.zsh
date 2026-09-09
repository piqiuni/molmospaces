#!/usr/bin/env bash
# Run one frozen V3 ROS object-goal episode.
#
# Usage:
#   bash scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_test.zsh \
#     <run-output-dir> <episode-index>
#
# The script intentionally owns one ROS master per episode.  In normal mode it
# also owns a recorder; FAST_EVAL=true is evaluator-only and deliberately has
# neither a recorder nor a recorder acknowledgement barrier.

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RUN_DIR=${1:?"usage: $0 <run-output-dir> <episode-index>"}
EPISODE_INDEX=${2:?"usage: $0 <run-output-dir> <episode-index>"}
if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${PWD}/${RUN_DIR}"
fi
if [[ ! "${EPISODE_INDEX}" =~ ^[0-9]+$ ]]; then
  printf '%s\n' "episode-index must be a non-negative integer: ${EPISODE_INDEX}" >&2
  exit 2
fi

BENCHMARK=${BENCHMARK:-${REPO_ROOT}/scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_val_release_v1_1/benchmark/benchmark.json}
# POLICY is the evaluator's ROS/restricted-GT adapter.  METHOD names the
# semantic method loaded into that adapter and is deliberately fixed to MLLM
# for all V3 benchmark runs through this entry point.
METHOD=${METHOD:-full_mllm_object_goal}
POLICY=${POLICY:-ros_object_goal_rule}
MAX_STEPS=${MAX_STEPS:-2000}
STEP_BUDGET_MODE=${STEP_BUDGET_MODE:-dynamic}
MIN_STEPS=${MIN_STEPS:-300}
DYNAMIC_PATH_FREE_M=${DYNAMIC_PATH_FREE_M:-3.0}
DYNAMIC_STEPS_PER_PATH_M=${DYNAMIC_STEPS_PER_PATH_M:-25.0}
DYNAMIC_CHANNEL_INTERACTION_STEPS=${DYNAMIC_CHANNEL_INTERACTION_STEPS:-150}
DYNAMIC_CONTAINER_INTERACTION_STEPS=${DYNAMIC_CONTAINER_INTERACTION_STEPS:-200}
DYNAMIC_CONTAINER_JOINT_STEPS=${DYNAMIC_CONTAINER_JOINT_STEPS:-40}
DYNAMIC_STEP_QUANTUM=${DYNAMIC_STEP_QUANTUM:-50}
VIDEO_FPS=${VIDEO_FPS:-5}
# This is a diagnostic runner, not the large-scale benchmark launcher: keep
# one path/costmap/semantic composite for every evaluator step.
VIDEO_STEP_SAMPLE_EVERY=${VIDEO_STEP_SAMPLE_EVERY:-1}
VIDEO_PANEL_WIDTH_PX=${VIDEO_PANEL_WIDTH_PX:-480}
# A 1500-step V3 episode needs a large frozen-snapshot queue when six-panel
# rendering is slower than simulation.  If this exceptional capacity is still
# exhausted, discard the oldest pending render job so fresh RGB/plan snapshots
# remain paired instead of blocking callbacks into placeholder frames.
VIDEO_FRAME_JOB_QUEUE_SIZE=${VIDEO_FRAME_JOB_QUEUE_SIZE:-2048}
VIDEO_FRAME_QUEUE_OVERFLOW=${VIDEO_FRAME_QUEUE_OVERFLOW:-drop_oldest}
STEP_SYNC_QUEUE_SIZE=${STEP_SYNC_QUEUE_SIZE:-4096}
STEP_SYNC_IMAGE_CACHE_SIZE=${STEP_SYNC_IMAGE_CACHE_SIZE:-256}
STEP_SYNC_IMAGE_FALLBACK_MAX_AGE_SEC=${STEP_SYNC_IMAGE_FALLBACK_MAX_AGE_SEC:-12}
VIDEO_HISTORY_SIZE=${VIDEO_HISTORY_SIZE:-16}
# Frozen OCC/local proxies use this cap.  The recorder intentionally retains
# the global costmap at native grid resolution as lossless PNG so its
# inflation bands remain inspectable in the six-panel diagnostic video.
VIDEO_SNAPSHOT_GRID_MAX_DIM=${VIDEO_SNAPSHOT_GRID_MAX_DIM:-512}
VIDEO_SNAPSHOT_JPEG_QUALITY=${VIDEO_SNAPSHOT_JPEG_QUALITY:-90}
VIDEO_SNAPSHOT_CATEGORICAL_FORMAT=${VIDEO_SNAPSHOT_CATEGORICAL_FORMAT:-png}
VIDEO_OCC_CROP_MARGIN_M=${VIDEO_OCC_CROP_MARGIN_M:-2.5}
# Keep the full-known-map coverage inset enabled for the maintained offline
# six-panel artifact.  It can be disabled for a compact legacy replay with
# VIDEO_SEMANTIC_XY_OVERVIEW_INSET=false.
VIDEO_SEMANTIC_XY_OVERVIEW_INSET=${VIDEO_SEMANTIC_XY_OVERVIEW_INSET:-false}
ARTIFACT_WRITE_QUEUE_SIZE=${ARTIFACT_WRITE_QUEUE_SIZE:-256}
# Full per-step composites may take substantially longer than the simulator;
# let the recorder finish them before teardown.
RECORDER_DRAIN_TIMEOUT_S=${RECORDER_DRAIN_TIMEOUT_S:-${RECORDER_DRAIN_WAIT_S:-5400}}
RECORDER_DRAIN_POLL_S=${RECORDER_DRAIN_POLL_S:-0.5}
RECORDER_DRAIN_PROGRESS_S=${RECORDER_DRAIN_PROGRESS_S:-10}
RECORDER_SHUTDOWN_GRACE_S=${RECORDER_SHUTDOWN_GRACE_S:-600}
RECORD_HEAD_CAMERA=${RECORD_HEAD_CAMERA:-false}
FAST_EVAL=${FAST_EVAL:-false}
# Load the complete ProcTHOR scene by default, then project the final ROS map
# onto its GT navigable area.  TOPDOWN_ROS_ONLY=true is an explicit lightweight
# fallback; it cannot report whole-scene GT coverage because it lacks the GT
# denominator.
TOPDOWN_ROS_ONLY=${TOPDOWN_ROS_ONLY:-false}
if [[ -z "${TOPDOWN_REQUIRE_FULL_SCENE+x}" ]]; then
  if [[ "${TOPDOWN_ROS_ONLY}" == true ]]; then
    TOPDOWN_REQUIRE_FULL_SCENE=false
  else
    TOPDOWN_REQUIRE_FULL_SCENE=true
  fi
fi
ROS_MASTER_URI=${ROS_MASTER_URI:-http://127.0.0.1:11311}
RUN_ROS_MASTER_URI=${ROS_MASTER_URI}
# This entry point is a Bash script (despite its historical .zsh suffix), so
# source the Bash ROS environment by default.  Sourcing setup.zsh under Bash
# can fail before the evaluator starts (e.g. zsh's `cd -q` syntax).
ROS_SETUP=${ROS_SETUP:-${REPO_ROOT}/Interactive-Nav-SG-nav/devel/setup.bash}
SEMANTIC_MODEL_ENV_FILE=${SEMANTIC_MODEL_ENV_FILE:-${REPO_ROOT}/.env}
# Keep V3's Module-1 cap explicit as well: portal visual-evidence JSON can
# exceed the historical 256-token budget when the model pretty-prints fields.
# This launch argument reaches only the object-attribute (M1) lane; M2/M3 use
# object_goal_v3_full_mllm.yaml and room MLLM keeps its own mapping config cap.
SEMANTIC_ATTRIBUTE_MAX_OUTPUT_TOKENS=${SEMANTIC_ATTRIBUTE_MAX_OUTPUT_TOKENS:-384}
# Match the ordinary full-MLLM interaction profile.  A caller may still lower
# this explicitly for a dedicated throughput experiment.
SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S=${SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S:-30.0}
# Keep the bridge freshness/synchronisation contract aligned with the ordinary
# interaction runner.  These values control public observation delivery only;
# restricted perception and evaluator scoring remain separate V3 concerns.
ROS_ACTION_TIMEOUT_S=${ROS_ACTION_TIMEOUT_S:-0.2}
ROS_STEP_READY_BARRIER_ENABLED=${ROS_STEP_READY_BARRIER_ENABLED:-true}
ROS_STEP_READY_TOPIC=${ROS_STEP_READY_TOPIC:-/semantic_decision/step_ready}
ROS_STEP_READY_WARMUP_SKIP_FRAMES=${ROS_STEP_READY_WARMUP_SKIP_FRAMES:-0}
ROS_STEP_READY_TIMEOUT_S=${ROS_STEP_READY_TIMEOUT_S:-2.0}
ROS_STEP_READY_BOOTSTRAP_TIMEOUT_S=${ROS_STEP_READY_BOOTSTRAP_TIMEOUT_S:-10.0}
# Non-applied bridge refreshes must not make a fixed applied-step evaluation
# unbounded.  1.5x admits the observed normal async slack while bounding an
# intermittent-command/no-progress loop before the outer scene timeout.
ROS_COMMAND_STARVATION_TIMEOUT_S=${ROS_COMMAND_STARVATION_TIMEOUT_S:-60.0}
ROS_OBSERVATION_TURN_MULTIPLIER=${ROS_OBSERVATION_TURN_MULTIPLIER:-1.5}
SEMANTIC_DECISION_OVERRIDE=${SEMANTIC_DECISION_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/object_goal_v3_full_mllm.yaml}
SEMANTIC_MAPPING_OVERRIDE=${SEMANTIC_MAPPING_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/full_mllm_mapping.yaml}
EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_controlled_explore.yaml}
NAV_CONFIG_OVERRIDE=${NAV_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_interaction_nav.yaml}
RECORDER=${RECORDER:-${REPO_ROOT}/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts/record_explore_debug.py}
RECORDER_DRAIN_HELPER=${RECORDER_DRAIN_HELPER:-${SCRIPT_DIR}/wait_for_recorder_drain.py}
VIDEO_BUILDER=${VIDEO_BUILDER:-${SCRIPT_DIR}/build_semantic_video_offline.py}
STEP_FRAME_QUEUE_SIZE=${STEP_FRAME_QUEUE_SIZE:-4}
STEP_CAPTURE_ACK_TIMEOUT_S=${STEP_CAPTURE_ACK_TIMEOUT_S:-2.0}
RECORDER_DRAIN_STALL_TIMEOUT_S=${RECORDER_DRAIN_STALL_TIMEOUT_S:-300}
# Keep defaults on the large /home volume even when this runner is invoked
# directly rather than through the batch wrapper.
SHARED_MPLCONFIGDIR=${MPLCONFIGDIR:-/home/ldl/.cache/molmospaces/matplotlib-${UID}}
RUNTIME_TMPDIR=${TMPDIR:-/home/ldl/tmp/molmospaces-v3-${UID}}
RUNTIME_XDG_CACHE_HOME=${XDG_CACHE_HOME:-/home/ldl/.cache}

for required_path in "${BENCHMARK}" "${ROS_SETUP}" "${SEMANTIC_MODEL_ENV_FILE}" \
  "${SEMANTIC_DECISION_OVERRIDE}" "${SEMANTIC_MAPPING_OVERRIDE}" \
  "${EXPLORE_PY_CONFIG_OVERRIDE}" "${NAV_CONFIG_OVERRIDE}"; do
  if [[ ! -f "${required_path}" ]]; then
    printf '%s\n' "Missing required file: ${required_path}" >&2
    exit 2
  fi
done
if [[ "${FAST_EVAL}" != true ]]; then
  for required_path in "${RECORDER}" "${RECORDER_DRAIN_HELPER}" "${VIDEO_BUILDER}"; do
    if [[ ! -f "${required_path}" ]]; then
      printf '%s\n' "Missing required recorder support file: ${required_path}" >&2
      exit 2
    fi
  done
fi

if [[ "${METHOD}" != "full_mllm_object_goal" ]]; then
  printf '%s\n' "V3 benchmark wrapper requires METHOD=full_mllm_object_goal, got: ${METHOD}" >&2
  exit 2
fi
if [[ "${POLICY}" != "ros_object_goal_rule" ]]; then
  printf '%s\n' "V3 full-MLLM uses the ros_object_goal_rule evaluator adapter, got POLICY=${POLICY}" >&2
  exit 2
fi
if [[ ! "${VIDEO_STEP_SAMPLE_EVERY}" =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' "VIDEO_STEP_SAMPLE_EVERY must be a positive integer: ${VIDEO_STEP_SAMPLE_EVERY}" >&2
  exit 2
fi
if [[ "${FAST_EVAL}" != true && "${VIDEO_STEP_SAMPLE_EVERY}" -ne 1 ]]; then
  printf '%s\n' "Exact offline V3 six-panel recording requires VIDEO_STEP_SAMPLE_EVERY=1, got: ${VIDEO_STEP_SAMPLE_EVERY}" >&2
  exit 2
fi
for required_mllm_setting in \
  'module1: "dynamic_mllm"' \
  'module2: "mllm_score"' \
  'module3: "mllm_skill_verified"'; do
  if ! grep -Fq -- "${required_mllm_setting}" "${SEMANTIC_DECISION_OVERRIDE}"; then
    printf '%s\n' "V3 semantic override is not the required full-MLLM method: missing ${required_mllm_setting}" >&2
    exit 2
  fi
done
printf '%s\n' "[v3-eval] method=${METHOD} policy_adapter=${POLICY}"
printf '%s\n' "[v3-eval] step_budget_mode=${STEP_BUDGET_MODE} min_steps=${MIN_STEPS} max_steps=${MAX_STEPS}"
printf '%s\n' "[v3-eval] m1_attribute_max_output_tokens=${SEMANTIC_ATTRIBUTE_MAX_OUTPUT_TOKENS} request_timeout_s=${SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S} ros_action_timeout_s=${ROS_ACTION_TIMEOUT_S} step_ready=${ROS_STEP_READY_BARRIER_ENABLED} ros_command_starvation_timeout_s=${ROS_COMMAND_STARVATION_TIMEOUT_S} ros_observation_turn_multiplier=${ROS_OBSERVATION_TURN_MULTIPLIER}"
if [[ "${FAST_EVAL}" == true ]]; then
  printf '%s\n' "[v3-eval] fast_eval=true recorder_enabled=false step_capture_ack_barrier=false"
else
  printf '%s\n' "[v3-eval] fast_eval=false recorder_enabled=true step_capture_ack_barrier=true"
fi
printf '%s\n' "[v3-eval] video_fps=${VIDEO_FPS} video_step_sample_every=${VIDEO_STEP_SAMPLE_EVERY} render_queue=${VIDEO_FRAME_JOB_QUEUE_SIZE} overflow=${VIDEO_FRAME_QUEUE_OVERFLOW} occ_local_proxy=${VIDEO_SNAPSHOT_GRID_MAX_DIM}px/${VIDEO_SNAPSHOT_CATEGORICAL_FORMAT} global_costmap=native/png crop_margin=${VIDEO_OCC_CROP_MARGIN_M}m semantic_xy_overview_inset=${VIDEO_SEMANTIC_XY_OVERVIEW_INSET}"

mkdir -p "${RUN_DIR}" "${RUN_DIR}/ros_home/log" "${SHARED_MPLCONFIGDIR}" "${RUNTIME_TMPDIR}" "${RUNTIME_XDG_CACHE_HOME}"
if [[ "${FAST_EVAL}" != true ]]; then
  mkdir -p "${RUN_DIR}/debug" "${RUN_DIR}/sim_step_frames"
fi
if [[ -e "${RUN_DIR}/eval" ]]; then
  printf '%s\n' "Refusing to overwrite existing evaluator output: ${RUN_DIR}/eval" >&2
  exit 2
fi

export ROS_MASTER_URI
export ROS_IP=${ROS_IP:-127.0.0.1}
export ROS_HOSTNAME=${ROS_HOSTNAME:-127.0.0.1}
export ROS_HOME="${RUN_DIR}/ros_home"
export ROS_LOG_DIR="${RUN_DIR}/ros_home/log"
export MPLCONFIGDIR="${SHARED_MPLCONFIGDIR}"
export TMPDIR="${RUNTIME_TMPDIR}"
export XDG_CACHE_HOME="${RUNTIME_XDG_CACHE_HOME}"
export SEMANTIC_DECISION_ENV_FILE="${SEMANTIC_MODEL_ENV_FILE}"
export SEMANTIC_MODEL_METRICS_PATH="${RUN_DIR}/mllm_metrics.jsonl"
export PYTHONUNBUFFERED=1

set +u
CONDA_SH=${CONDA_SH:-/home/ldl/miniconda3/etc/profile.d/conda.sh}
if [[ ! -f "${CONDA_SH}" ]]; then
  printf '%s\n' "Missing conda initialization script: ${CONDA_SH}" >&2
  exit 2
fi
source "${CONDA_SH}"
# Use an absolute default.  The old default "mlspaces/bin/python" was a
# relative path when CONDA_ENV was a name and could select the wrong Python.
CONDA_ENV=${CONDA_ENV:-/home/ldl/conda_envs/mlspaces}
conda activate "${CONDA_ENV}"
source "${ROS_SETUP}"
ROS_SOURCE_DIR=${ROS_SOURCE_DIR:-$(cd -- "$(dirname -- "${ROS_SETUP}")/../src" && pwd)}
if [[ ! -d "${ROS_SOURCE_DIR}" ]]; then
  printf '%s\n' "Missing ROS source directory: ${ROS_SOURCE_DIR}" >&2
  exit 2
fi
ACTIVE_CONDA_PREFIX=${CONDA_PREFIX:-}
if [[ -z "${ACTIVE_CONDA_PREFIX}" || ! -d "${ACTIVE_CONDA_PREFIX}" ]]; then
  printf '%s\n' "Conda activation did not provide an absolute CONDA_PREFIX: ${ACTIVE_CONDA_PREFIX}" >&2
  exit 2
fi
PYTHON_BIN=${PYTHON_BIN:-${ACTIVE_CONDA_PREFIX}/bin/python}
if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf '%s\n' "Missing MolmoSpaces Python executable: ${PYTHON_BIN}" >&2
  exit 2
fi
MLSPACES_SITE_PACKAGES="$(${PYTHON_BIN} -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="${MLSPACES_SITE_PACKAGES}:${PYTHONPATH:-}"
export PATH="${ACTIVE_CONDA_PREFIX}/bin:${PATH}"
set -u
# ROS setup files may restore a default master URI; keep this episode's
# explicitly isolated master after sourcing.
export ROS_MASTER_URI="${RUN_ROS_MASTER_URI}"
export ROS_PACKAGE_PATH="${ROS_SOURCE_DIR}:${ROS_PACKAGE_PATH#*:}"
export PYTHONPATH="${ROS_SOURCE_DIR}/semantic_mapping_py_pkg/scripts:${ROS_SOURCE_DIR}/semantic_decision_py_pkg/scripts:${ROS_SOURCE_DIR}/semantic_mllm_py_pkg/scripts:${ROS_SOURCE_DIR}/explore_py_pkg/scripts:${PYTHONPATH:-}"

cleanup_process() {
  local pid="${1:-}"
  local grace_s="${2:-20}"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return
  fi
  kill -INT "${pid}" 2>/dev/null || true
  local attempts=$(( grace_s * 2 ))
  local attempt=1
  while (( attempt <= attempts )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      return
    fi
    sleep 0.5
    attempt=$(( attempt + 1 ))
  done
  kill -TERM "${pid}" 2>/dev/null || true
  sleep 1
  kill -KILL "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

cleanup_process_group() {
  local pid="${1:-}"
  local grace_s="${2:-20}"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return
  fi
  local pgid
  pgid=$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)
  if [[ "${pgid}" =~ ^[0-9]+$ ]] && [[ "${pgid}" -gt 1 ]] && [[ "${pgid}" != "${BASHPID}" ]]; then
    kill -INT -- "-${pgid}" 2>/dev/null || true
  fi
  cleanup_process "${pid}" "${grace_s}"
  if [[ "${pgid}" =~ ^[0-9]+$ ]] && [[ "${pgid}" -gt 1 ]] && [[ "${pgid}" != "${BASHPID}" ]]; then
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  fi
}

ROSCORE_PID=""
ROSLAUNCH_PID=""
RECORDER_PID=""
cleanup() {
  cleanup_process "${RECORDER_PID:-}" 30
  cleanup_process_group "${ROSLAUNCH_PID:-}" 20
  cleanup_process_group "${ROSCORE_PID:-}" 10
}
trap cleanup EXIT INT TERM

MASTER_PORT=${ROS_MASTER_URI##*:}
MASTER_PORT=${MASTER_PORT%%/*}
if [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ ]]; then
  printf '%s\n' "ROS_MASTER_URI must include a numeric port: ${ROS_MASTER_URI}" >&2
  exit 2
fi
if timeout 1s rosparam list >/dev/null 2>&1; then
  printf '%s\n' "Refusing to reuse an existing ROS master: ${ROS_MASTER_URI}" >&2
  exit 2
fi

roscore -p "${MASTER_PORT}" >"${RUN_DIR}/roscore.log" 2>&1 &
ROSCORE_PID=$!
MASTER_READY=false
for _attempt in {1..120}; do
  if ! kill -0 "${ROSCORE_PID}" 2>/dev/null; then
    wait "${ROSCORE_PID}" 2>/dev/null || true
    printf '%s\n' "roscore exited before becoming ready; port ${MASTER_PORT} may be occupied" >&2
    exit 3
  fi
  if timeout 1s rosparam list >/dev/null 2>&1; then
    sleep 0.1
    if kill -0 "${ROSCORE_PID}" 2>/dev/null; then
      MASTER_READY=true
      break
    fi
  fi
  sleep 0.25
done
if [[ "${MASTER_READY}" != true ]]; then
  printf '%s\n' "ROS master did not become ready: ${ROS_MASTER_URI}" >&2
  exit 3
fi

roslaunch "${ROS_SOURCE_DIR}/nav_pkg/launch/molmospaces_nav_system.launch" \
  start_sim:=false \
  start_mapping:=true \
  mapping_mode:=odom_locked \
  start_nav:=true \
  start_explore:=false \
  start_explore_py:=true \
  start_semantic_mapping:=true \
  semantic_source:=realtime_gt \
  publish_realtime_gt:=false \
  start_semantic_decision:=true \
  semantic_attribute_inference:=true \
  semantic_attribute_model_name:= \
  semantic_attribute_max_output_tokens:="${SEMANTIC_ATTRIBUTE_MAX_OUTPUT_TOKENS}" \
  semantic_attribute_request_timeout_s:="${SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S}" \
  semantic_decision_config_override_file:="${SEMANTIC_DECISION_OVERRIDE}" \
  semantic_config_override_file:="${SEMANTIC_MAPPING_OVERRIDE}" \
  explore_py_config_override_file:="${EXPLORE_PY_CONFIG_OVERRIDE}" \
  nav_config_override_file:="${NAV_CONFIG_OVERRIDE}" \
  >"${RUN_DIR}/roslaunch.log" 2>&1 &
ROSLAUNCH_PID=$!

if [[ "${FAST_EVAL}" != true ]]; then
  # Start before the evaluator so every public observation/step-sync is captured.
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${RECORDER}" \
    --output-dir "${RUN_DIR}/debug" \
    --occupancy-grid-topic /semantic_mapping/planning_occ_map \
    --raw-occupancy-grid-topic /struct_mapping/occ_map \
    --image-topic /molmo_spaces/head_camera/image \
    --video-step-sync-topic /molmo_spaces/step_sync \
    --step-sync-queue-size "${STEP_SYNC_QUEUE_SIZE}" \
    --step-sync-capture-every "${VIDEO_STEP_SAMPLE_EVERY}" \
    --step-sync-image-cache-size "${STEP_SYNC_IMAGE_CACHE_SIZE}" \
    --step-sync-image-fallback-max-age-sec "${STEP_SYNC_IMAGE_FALLBACK_MAX_AGE_SEC}" \
    --video-snapshot-grid-max-dim "${VIDEO_SNAPSHOT_GRID_MAX_DIM}" \
    --video-snapshot-jpeg-quality "${VIDEO_SNAPSHOT_JPEG_QUALITY}" \
    --video-snapshot-categorical-format "${VIDEO_SNAPSHOT_CATEGORICAL_FORMAT}" \
    --video-occ-crop-margin-m "${VIDEO_OCC_CROP_MARGIN_M}" \
    --first-person-video-capture-mode step \
    --semantic-video \
    --first-person-video-with-map \
    --first-person-video-fps "${VIDEO_FPS}" \
    --first-person-video-width-px "${VIDEO_PANEL_WIDTH_PX}" \
    --video-frame-job-queue-size "${VIDEO_FRAME_JOB_QUEUE_SIZE}" \
    --video-frame-queue-overflow "${VIDEO_FRAME_QUEUE_OVERFLOW}" \
    --video-history-size "${VIDEO_HISTORY_SIZE}" \
    --artifact-write-queue-size "${ARTIFACT_WRITE_QUEUE_SIZE}" \
    --step-capture-ack-topic /molmo_spaces/step_capture_ack \
    --no-runtime-video-encode \
    --offline-video-only \
    --no-video-save-panel-frames \
    --no-video-save-composite-frames \
    --no-first-person-video-h264 \
    --interaction-result-topic /semantic_mapping/interaction_result \
    --no-external-video \
    >"${RUN_DIR}/recorder.log" 2>&1 &
  RECORDER_PID=$!
  sleep 1
fi
if ! kill -0 "${ROSLAUNCH_PID}" 2>/dev/null; then
  wait "${ROSLAUNCH_PID}" 2>/dev/null || true
  printf '%s\n' "roslaunch exited before the evaluator started; see ${RUN_DIR}/roslaunch.log" >&2
  exit 3
fi

EVAL_ARGS=(
  "${REPO_ROOT}/scripts/InteractiveNav/evaluate_interactive_nav_v3.py"
  --benchmark "${BENCHMARK}"
  --output-dir "${RUN_DIR}/eval"
  --policy "${POLICY}"
  --workers 1
  --episode-indices "${EPISODE_INDEX}"
  --max-steps "${MAX_STEPS}"
  --step-budget-mode "${STEP_BUDGET_MODE}"
  --min-steps "${MIN_STEPS}"
  --dynamic-path-free-m "${DYNAMIC_PATH_FREE_M}"
  --dynamic-steps-per-path-m "${DYNAMIC_STEPS_PER_PATH_M}"
  --dynamic-channel-interaction-steps "${DYNAMIC_CHANNEL_INTERACTION_STEPS}"
  --dynamic-container-interaction-steps "${DYNAMIC_CONTAINER_INTERACTION_STEPS}"
  --dynamic-container-joint-steps "${DYNAMIC_CONTAINER_JOINT_STEPS}"
  --dynamic-step-quantum "${DYNAMIC_STEP_QUANTUM}"
  --ros-action-timeout-s "${ROS_ACTION_TIMEOUT_S}"
  --ros-step-ready-topic "${ROS_STEP_READY_TOPIC}"
  --ros-step-ready-warmup-skip-frames "${ROS_STEP_READY_WARMUP_SKIP_FRAMES}"
  --ros-step-ready-timeout-s "${ROS_STEP_READY_TIMEOUT_S}"
  --ros-step-ready-bootstrap-timeout-s "${ROS_STEP_READY_BOOTSTRAP_TIMEOUT_S}"
  --ros-command-starvation-timeout-s "${ROS_COMMAND_STARVATION_TIMEOUT_S}"
  --ros-observation-turn-multiplier "${ROS_OBSERVATION_TURN_MULTIPLIER}"
  --no-ros-require-move-base-active
  --ros-map-warmup-skip-frames 0
  --video-fps "${VIDEO_FPS}"
  --progress-every 1
)
if [[ "${ROS_STEP_READY_BARRIER_ENABLED}" == true ]]; then
  EVAL_ARGS+=(--ros-step-ready-barrier-enabled)
else
  EVAL_ARGS+=(--no-ros-step-ready-barrier-enabled)
fi
if [[ "${FAST_EVAL}" != true ]]; then
  # Frame persistence and the acknowledgement barrier are inseparable in the
  # recordable protocol.  Do not pass either in FAST_EVAL: no recorder exists
  # to acknowledge, and a queue/writer would only add avoidable CPU and I/O.
  EVAL_ARGS+=(
    --ros-step-frame-dir "${RUN_DIR}/sim_step_frames"
    --ros-step-frame-queue-size "${STEP_FRAME_QUEUE_SIZE}"
    --ros-step-capture-ack-topic /molmo_spaces/step_capture_ack
    --ros-step-capture-ack-barrier-enabled
    --ros-step-capture-ack-timeout-s "${STEP_CAPTURE_ACK_TIMEOUT_S}"
  )
fi
if [[ "${FAST_EVAL}" != true && "${RECORD_HEAD_CAMERA}" == true ]]; then
  EVAL_ARGS+=(--record-video)
fi
set +e
MUJOCO_GL=egl "${PYTHON_BIN}" "${EVAL_ARGS[@]}" >"${RUN_DIR}/eval.log" 2>&1
EVAL_EXIT=$?
set -e

if [[ "${FAST_EVAL}" == true ]]; then
  cleanup_process_group "${ROSLAUNCH_PID}" 20
  ROSLAUNCH_PID=""
  cleanup_process_group "${ROSCORE_PID}" 10
  ROSCORE_PID=""
fi
EPISODE_RESULTS=()
EPISODE_INDEX_PADDED=$(printf '%04d' "${EPISODE_INDEX}")
EPISODE_RESULTS+=("${RUN_DIR}"/eval/episodes/${EPISODE_INDEX_PADDED}_*/episode_result.json)
if (( ${#EPISODE_RESULTS[@]} != 1 )); then
  printf '%s\n' "Expected one completed episode result for index ${EPISODE_INDEX}; found ${#EPISODE_RESULTS[@]}" >&2
  exit 4
fi
EPISODE_RESULT=${EPISODE_RESULTS[0]}
EPISODE_DIR=$(dirname -- "${EPISODE_RESULT}")
if [[ "${FAST_EVAL}" == true ]]; then
  "${PYTHON_BIN}" - "${EPISODE_RESULT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    document = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    raise SystemExit(f"fast-eval result is unreadable: {path}: {exc}")
result = document.get("result") if isinstance(document, dict) else None
if not isinstance(result, dict):
    result = document if isinstance(document, dict) else {}
if document.get("status") != "complete" or result.get("status") != "complete":
    raise SystemExit(
        f"fast-eval result is not complete: document={document.get('status')!r} "
        f"result={result.get('status')!r}"
    )
PY
  printf '%s\n' "[v3-ros-eval-fast] result=${EPISODE_RESULT}"
  exit "${EVAL_EXIT}"
fi

# The evaluator may finish while bridge PNG writes or recorder raw receipts are
# still queued.  The bridge manifest is the authoritative source of recordable
# sensor steps: an interaction can complete between two policy calls, so an
# evaluator decision count is not necessarily a camera-frame count.
RECORDER_DRAIN_STATUS=0
"${PYTHON_BIN}" "${RECORDER_DRAIN_HELPER}" \
  --sim-manifest "${RUN_DIR}/sim_step_frames/manifest.jsonl" \
  --video-frames-csv "${RUN_DIR}/debug/video_frames.csv" \
  --timeout-sec "${RECORDER_DRAIN_TIMEOUT_S}" \
  --poll-sec "${RECORDER_DRAIN_POLL_S}" \
  --progress-sec "${RECORDER_DRAIN_PROGRESS_S}" \
  --stall-timeout-sec "${RECORDER_DRAIN_STALL_TIMEOUT_S}" \
  --recorder-pid "${RECORDER_PID}" \
  --step-sync-capture-every "${VIDEO_STEP_SAMPLE_EVERY}" \
  --offline-raw-recording \
  --raw-step-manifest "${RUN_DIR}/debug/raw/step_boundaries.jsonl" \
  || RECORDER_DRAIN_STATUS=$?

cleanup_process "${RECORDER_PID}" "${RECORDER_SHUTDOWN_GRACE_S}"
RECORDER_PID=""
cleanup_process_group "${ROSLAUNCH_PID}" 20
ROSLAUNCH_PID=""
cleanup_process_group "${ROSCORE_PID}" 10
ROSCORE_PID=""
# Re-check after recorder shutdown because its final join may complete the last
# raw receipt even if the live drain reached its timeout boundary.
FINAL_DRAIN_STATUS=0
"${PYTHON_BIN}" "${RECORDER_DRAIN_HELPER}" \
  --sim-manifest "${RUN_DIR}/sim_step_frames/manifest.jsonl" \
  --video-frames-csv "${RUN_DIR}/debug/video_frames.csv" \
  --timeout-sec 0 \
  --step-sync-capture-every "${VIDEO_STEP_SAMPLE_EVERY}" \
  --offline-raw-recording \
  --raw-step-manifest "${RUN_DIR}/debug/raw/step_boundaries.jsonl" \
  --recorder-summary "${RUN_DIR}/debug/summary.json" \
  || FINAL_DRAIN_STATUS=$?
if (( FINAL_DRAIN_STATUS != 0 )); then
  printf '%s\n' "Recorder did not capture every completed evaluator step (live_drain_status=${RECORDER_DRAIN_STATUS})." >&2
  exit 4
fi

VIDEO_BUILDER_ARGS=(
  "${VIDEO_BUILDER}"
  --scene-dir "${RUN_DIR}"
  --debug-dir "${RUN_DIR}/debug"
  --fps "${VIDEO_FPS}"
  --state-alignment exact
  --output-stem overview_6panel
)
if [[ "${VIDEO_SEMANTIC_XY_OVERVIEW_INSET}" == true ]]; then
  VIDEO_BUILDER_ARGS+=(--semantic-xy-overview-inset)
fi
"${PYTHON_BIN}" "${VIDEO_BUILDER_ARGS[@]}" \
  >"${RUN_DIR}/offline_video.log" 2>&1

SIX_PANEL_PATH="${RUN_DIR}/videos/overview_6panel.mp4"
"${PYTHON_BIN}" - "${RUN_DIR}/sim_step_frames/manifest.jsonl" "${RUN_DIR}/offline_video_summary.json" "${SIX_PANEL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, summary_path, video_path = map(Path, sys.argv[1:])
expected = sum(
    bool(line.strip()) and line.endswith("\n")
    for line in manifest_path.read_text(encoding="utf-8").splitlines(keepends=True)
)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
actual = int(summary.get("output_frame_count", -1))
exact = int(summary.get("exact_step_match_count", -1))
missing = list(summary.get("missing_sim_step_indexes") or []) + list(summary.get("missing_raw_step_indexes") or [])
if expected <= 0 or actual != expected or exact != expected or missing:
    raise SystemExit(
        "offline V3 video alignment failed: "
        f"expected={expected} output={actual} exact={exact} missing={missing[:8]}"
    )
if not video_path.is_file() or video_path.stat().st_size <= 0:
    raise SystemExit(f"offline V3 video is missing or empty: {video_path}")
PY

TOPDOWN_PATH="${EPISODE_DIR}/episode_topdown.png"
TOPDOWN_ARGS=(
  --episode-result "${EPISODE_RESULT}"
  --benchmark "${BENCHMARK}"
  --debug-dir "${RUN_DIR}/debug"
  --private-context "${EPISODE_DIR}/episode_visualization.json"
  --output "${TOPDOWN_PATH}"
)
if [[ "${TOPDOWN_ROS_ONLY}" == true ]]; then
  TOPDOWN_ARGS+=(--ros-only)
fi
if [[ "${TOPDOWN_REQUIRE_FULL_SCENE}" == true ]]; then
  TOPDOWN_ARGS+=(--require-full-scene)
fi
MUJOCO_GL=egl "${PYTHON_BIN}" "${REPO_ROOT}/scripts/InteractiveNav/render_interactive_nav_v3_topdown.py" \
  "${TOPDOWN_ARGS[@]}" \
  >"${RUN_DIR}/topdown.log" 2>&1

for required_artifact in "${RUN_DIR}/debug/final_occ_map.yaml" "${RUN_DIR}/debug/trajectory.csv" \
  "${RUN_DIR}/offline_video_summary.json" "${SIX_PANEL_PATH}" "${TOPDOWN_PATH}"; do
  if [[ ! -s "${required_artifact}" ]]; then
    printf '%s\n' "Required visual artifact is missing or empty: ${required_artifact}" >&2
    exit 4
  fi
done

printf '%s\n' "[v3-ros-eval] six-panel=${SIX_PANEL_PATH}"
printf '%s\n' "[v3-ros-eval] topdown=${TOPDOWN_PATH}"
printf '%s\n' "[v3-ros-eval] result=${EPISODE_RESULT}"
exit "${EVAL_EXIT}"
