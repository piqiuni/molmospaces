#!/usr/bin/env zsh
# Run one frozen V3 ROS object-goal episode with the required visual artifacts.
#
# Usage:
#   zsh scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_test.zsh \
#     <run-output-dir> <episode-index>
#
# The script intentionally owns one ROS master and one recorder per episode.
# A recorder directory cannot be attributed safely to multiple sequential V3
# episodes, so this is a single-episode test entry point rather than a batch
# evaluator wrapper.

set -euo pipefail
setopt null_glob

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
RUN_DIR=${1:?"usage: $0 <run-output-dir> <episode-index>"}
EPISODE_INDEX=${2:?"usage: $0 <run-output-dir> <episode-index>"}
if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${PWD}/${RUN_DIR}"
fi
if [[ ! "${EPISODE_INDEX}" =~ '^[0-9]+$' ]]; then
  print -u2 -- "episode-index must be a non-negative integer: ${EPISODE_INDEX}"
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
# The simulator publishes one step marker at 10 Hz.  Sampling every two
# markers keeps the required six-panel video at its configured 5 fps without
# allowing expensive composite rendering to stall the RGB/step-sync pairing.
VIDEO_STEP_SAMPLE_EVERY=${VIDEO_STEP_SAMPLE_EVERY:-2}
VIDEO_PANEL_WIDTH_PX=${VIDEO_PANEL_WIDTH_PX:-480}
# Absorb short render bursts without making an unbounded back-pressure buffer.
VIDEO_FRAME_JOB_QUEUE_SIZE=${VIDEO_FRAME_JOB_QUEUE_SIZE:-48}
RECORDER_DRAIN_TIMEOUT_S=${RECORDER_DRAIN_TIMEOUT_S:-${RECORDER_DRAIN_WAIT_S:-300}}
RECORDER_DRAIN_POLL_S=${RECORDER_DRAIN_POLL_S:-0.5}
RECORDER_DRAIN_PROGRESS_S=${RECORDER_DRAIN_PROGRESS_S:-10}
RECORDER_SHUTDOWN_GRACE_S=${RECORDER_SHUTDOWN_GRACE_S:-120}
RECORD_HEAD_CAMERA=${RECORD_HEAD_CAMERA:-false}
ROS_MASTER_URI=${ROS_MASTER_URI:-http://127.0.0.1:11311}
ROS_SETUP=${ROS_SETUP:-${REPO_ROOT}/Interactive-Nav-SG-nav/devel/setup.zsh}
SEMANTIC_MODEL_ENV_FILE=${SEMANTIC_MODEL_ENV_FILE:-${REPO_ROOT}/.env}
SEMANTIC_DECISION_OVERRIDE=${SEMANTIC_DECISION_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/object_goal_v3_full_mllm.yaml}
SEMANTIC_MAPPING_OVERRIDE=${SEMANTIC_MAPPING_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/full_mllm_mapping.yaml}
EXPLORE_PY_CONFIG_OVERRIDE=${EXPLORE_PY_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_controlled_explore.yaml}
NAV_CONFIG_OVERRIDE=${NAV_CONFIG_OVERRIDE:-${SCRIPT_DIR}/configs/semantic_decision/semantic_interaction_nav.yaml}
RECORDER=${RECORDER:-${REPO_ROOT}/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts/record_explore_debug.py}
RECORDER_DRAIN_HELPER=${RECORDER_DRAIN_HELPER:-${SCRIPT_DIR}/wait_for_recorder_drain.py}

for required_path in "${BENCHMARK}" "${ROS_SETUP}" "${SEMANTIC_MODEL_ENV_FILE}" \
  "${SEMANTIC_DECISION_OVERRIDE}" "${SEMANTIC_MAPPING_OVERRIDE}" \
  "${EXPLORE_PY_CONFIG_OVERRIDE}" "${NAV_CONFIG_OVERRIDE}" "${RECORDER}" \
  "${RECORDER_DRAIN_HELPER}"; do
  if [[ ! -f "${required_path}" ]]; then
    print -u2 -- "Missing required file: ${required_path}"
    exit 2
  fi
done

if [[ "${METHOD}" != "full_mllm_object_goal" ]]; then
  print -u2 -- "V3 benchmark wrapper requires METHOD=full_mllm_object_goal, got: ${METHOD}"
  exit 2
fi
if [[ "${POLICY}" != "ros_object_goal_rule" ]]; then
  print -u2 -- "V3 full-MLLM uses the ros_object_goal_rule evaluator adapter, got POLICY=${POLICY}"
  exit 2
fi
if [[ ! "${VIDEO_STEP_SAMPLE_EVERY}" =~ '^[1-9][0-9]*$' ]]; then
  print -u2 -- "VIDEO_STEP_SAMPLE_EVERY must be a positive integer: ${VIDEO_STEP_SAMPLE_EVERY}"
  exit 2
fi
for required_mllm_setting in \
  'module1: "dynamic_mllm"' \
  'module2: "mllm_score"' \
  'module3: "mllm_skill_verified"'; do
  if ! grep -Fq -- "${required_mllm_setting}" "${SEMANTIC_DECISION_OVERRIDE}"; then
    print -u2 -- "V3 semantic override is not the required full-MLLM method: missing ${required_mllm_setting}"
    exit 2
  fi
done
print -r -- "[v3-eval] method=${METHOD} policy_adapter=${POLICY}"
print -r -- "[v3-eval] step_budget_mode=${STEP_BUDGET_MODE} min_steps=${MIN_STEPS} max_steps=${MAX_STEPS}"
print -r -- "[v3-eval] video_fps=${VIDEO_FPS} video_step_sample_every=${VIDEO_STEP_SAMPLE_EVERY}"

mkdir -p "${RUN_DIR}" "${RUN_DIR}/debug" "${RUN_DIR}/ros_home/log"
if [[ -e "${RUN_DIR}/eval" ]]; then
  print -u2 -- "Refusing to overwrite existing evaluator output: ${RUN_DIR}/eval"
  exit 2
fi

export ROS_MASTER_URI
export ROS_IP=${ROS_IP:-127.0.0.1}
export ROS_HOSTNAME=${ROS_HOSTNAME:-127.0.0.1}
export ROS_HOME="${RUN_DIR}/ros_home"
export ROS_LOG_DIR="${RUN_DIR}/ros_home/log"
export SEMANTIC_DECISION_ENV_FILE="${SEMANTIC_MODEL_ENV_FILE}"
export SEMANTIC_MODEL_METRICS_PATH="${RUN_DIR}/mllm_metrics.jsonl"
export PYTHONUNBUFFERED=1

set +u
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate mlspaces
set -u
source "${ROS_SETUP}"
export ROS_PACKAGE_PATH="${REPO_ROOT}/Interactive-Nav-SG-nav/src:/opt/ros/noetic/share"
export PYTHONPATH="${REPO_ROOT}/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts:${REPO_ROOT}/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts:${REPO_ROOT}/Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts:${REPO_ROOT}/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts:${PYTHONPATH:-}"

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

ROSCORE_PID=""
ROSLAUNCH_PID=""
RECORDER_PID=""
cleanup() {
  cleanup_process "${RECORDER_PID:-}" 30
  cleanup_process "${ROSLAUNCH_PID:-}" 20
  cleanup_process "${ROSCORE_PID:-}" 10
}
trap cleanup EXIT INT TERM

MASTER_PORT=${ROS_MASTER_URI##*:}
MASTER_PORT=${MASTER_PORT%%/*}
if [[ ! "${MASTER_PORT}" =~ '^[0-9]+$' ]]; then
  print -u2 -- "ROS_MASTER_URI must include a numeric port: ${ROS_MASTER_URI}"
  exit 2
fi

roscore -p "${MASTER_PORT}" >"${RUN_DIR}/roscore.log" 2>&1 &
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
  print -u2 -- "ROS master did not become ready: ${ROS_MASTER_URI}"
  exit 3
fi

roslaunch nav_pkg molmospaces_nav_system.launch \
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
  semantic_decision_config_override_file:="${SEMANTIC_DECISION_OVERRIDE}" \
  semantic_config_override_file:="${SEMANTIC_MAPPING_OVERRIDE}" \
  explore_py_config_override_file:="${EXPLORE_PY_CONFIG_OVERRIDE}" \
  nav_config_override_file:="${NAV_CONFIG_OVERRIDE}" \
  >"${RUN_DIR}/roslaunch.log" 2>&1 &
ROSLAUNCH_PID=$!

# Start before the evaluator so every public observation/step-sync is captured.
PYTHONUNBUFFERED=1 python -u "${RECORDER}" \
  --output-dir "${RUN_DIR}/debug" \
  --occupancy-grid-topic /semantic_mapping/planning_occ_map \
  --raw-occupancy-grid-topic /struct_mapping/occ_map \
  --image-topic /molmo_spaces/head_camera/image \
  --video-step-sync-topic /molmo_spaces/step_sync \
  --step-sync-capture-every "${VIDEO_STEP_SAMPLE_EVERY}" \
  --first-person-video-capture-mode step \
  --semantic-video \
  --first-person-video-with-map \
  --first-person-video-fps "${VIDEO_FPS}" \
  --first-person-video-width-px "${VIDEO_PANEL_WIDTH_PX}" \
  --video-frame-job-queue-size "${VIDEO_FRAME_JOB_QUEUE_SIZE}" \
  --video-frame-queue-overflow block \
  --video-history-size 16 \
  --artifact-write-queue-size 4 \
  --runtime-video-encode \
  --no-video-save-panel-frames \
  --no-video-save-composite-frames \
  --interaction-result-topic /semantic_mapping/interaction_result \
  --no-external-video \
  >"${RUN_DIR}/recorder.log" 2>&1 &
RECORDER_PID=$!
sleep 1

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
  --ros-action-timeout-s 1.0
  --no-ros-require-move-base-active
  --ros-map-warmup-skip-frames 0
  --video-fps "${VIDEO_FPS}"
  --progress-every 1
)
if [[ "${RECORD_HEAD_CAMERA}" == true ]]; then
  EVAL_ARGS+=(--record-video)
fi
set +e
MUJOCO_GL=egl python "${EVAL_ARGS[@]}" >"${RUN_DIR}/eval.log" 2>&1
EVAL_EXIT=$?
set -e

EPISODE_RESULTS=("${RUN_DIR}"/eval/episodes/${EPISODE_INDEX}_*/episode_result.json)
if (( ${#EPISODE_RESULTS} != 1 )); then
  print -u2 -- "Expected one completed episode result for index ${EPISODE_INDEX}; found ${#EPISODE_RESULTS}"
  exit 4
fi
EPISODE_RESULT=${EPISODE_RESULTS[1]}
EPISODE_DIR=${EPISODE_RESULT:h}

# The evaluator may finish while step-sync callbacks or six-panel renders are
# still queued.  Drain against the evaluator's exact completed-step count,
# rather than sleeping for a fixed interval that is too short under 3 workers.
RECORDER_DRAIN_STATUS=0
python "${RECORDER_DRAIN_HELPER}" \
  --episode-result "${EPISODE_RESULT}" \
  --video-frames-csv "${RUN_DIR}/debug/video_frames.csv" \
  --timeout-sec "${RECORDER_DRAIN_TIMEOUT_S}" \
  --poll-sec "${RECORDER_DRAIN_POLL_S}" \
  --progress-sec "${RECORDER_DRAIN_PROGRESS_S}" \
  --recorder-pid "${RECORDER_PID}" \
  --step-sync-capture-every "${VIDEO_STEP_SAMPLE_EVERY}" \
  || RECORDER_DRAIN_STATUS=$?

cleanup_process "${RECORDER_PID}" "${RECORDER_SHUTDOWN_GRACE_S}"
RECORDER_PID=""
cleanup_process "${ROSLAUNCH_PID}" 20
ROSLAUNCH_PID=""
cleanup_process "${ROSCORE_PID}" 10
ROSCORE_PID=""

TOPDOWN_PATH="${EPISODE_DIR}/episode_topdown.png"

MUJOCO_GL=egl python "${REPO_ROOT}/scripts/InteractiveNav/render_interactive_nav_v3_topdown.py" \
  --episode-result "${EPISODE_RESULT}" \
  --benchmark "${BENCHMARK}" \
  --debug-dir "${RUN_DIR}/debug" \
  --private-context "${EPISODE_DIR}/episode_visualization.json" \
  --output "${TOPDOWN_PATH}" \
  >"${RUN_DIR}/topdown.log" 2>&1

SIX_PANEL_PATH="${RUN_DIR}/debug/videos/overview_6panel.mp4"
for required_artifact in "${RUN_DIR}/debug/final_occ_map.yaml" "${RUN_DIR}/debug/trajectory.csv" \
  "${SIX_PANEL_PATH}" "${TOPDOWN_PATH}"; do
  if [[ ! -s "${required_artifact}" ]]; then
    print -u2 -- "Required visual artifact is missing or empty: ${required_artifact}"
    exit 4
  fi
done

# Re-check after recorder shutdown because its final join may complete the last
# already-enqueued frame even if the live drain reached its timeout boundary.
FINAL_DRAIN_STATUS=0
python "${RECORDER_DRAIN_HELPER}" \
  --episode-result "${EPISODE_RESULT}" \
  --video-frames-csv "${RUN_DIR}/debug/video_frames.csv" \
  --timeout-sec 0 \
  --step-sync-capture-every "${VIDEO_STEP_SAMPLE_EVERY}" \
  --recorder-summary "${RUN_DIR}/debug/summary.json" \
  || FINAL_DRAIN_STATUS=$?
if (( FINAL_DRAIN_STATUS != 0 )); then
  print -u2 -- "Recorder did not capture every completed evaluator step (live_drain_status=${RECORDER_DRAIN_STATUS})."
  exit 4
fi

print -r -- "[v3-ros-eval] six-panel=${SIX_PANEL_PATH}"
print -r -- "[v3-ros-eval] topdown=${TOPDOWN_PATH}"
print -r -- "[v3-ros-eval] result=${EPISODE_RESULT}"
exit "${EVAL_EXIT}"
