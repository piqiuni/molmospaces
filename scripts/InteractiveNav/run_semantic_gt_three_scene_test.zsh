#!/usr/bin/env zsh
set -e

ROOT_DIR="${0:A:h:h:h}"
OUTPUT_ROOT="${1:-${ROOT_DIR}/outputs/semantic_gt_three_scene_$(date +%Y%m%d_%H%M%S)}"
RECORD_SEC="${RECORD_SEC:-600}"
TASK_HORIZON="${TASK_HORIZON:-5000}"
SIM_WAIT_TIMEOUT_SEC="${SIM_WAIT_TIMEOUT_SEC:-900}"
RECORDER_DRAIN_TIMEOUT_SEC="${RECORDER_DRAIN_TIMEOUT_SEC:-900}"
FIRST_STEP_TIMEOUT_SEC="${FIRST_STEP_TIMEOUT_SEC:-600}"
GT_STEP_INTERVAL="${GT_STEP_INTERVAL:-3}"
GT_MAX_DISTANCE_M="${GT_MAX_DISTANCE_M:-4.0}"
GT_MIN_VISIBLE_PIXELS="${GT_MIN_VISIBLE_PIXELS:-16}"
ENABLE_SEMANTIC="${ENABLE_SEMANTIC:-true}"
ACTION_TIMEOUT_SEC="${ACTION_TIMEOUT_SEC:-0.0}"
ROS_PORT_BASE="${ROS_PORT_BASE:-11400}"
HOUSE_LIST="${HOUSE_INDS:-4 7 10}"
HOUSES=(${=HOUSE_LIST})

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate mlspaces
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts:${PYTHONPATH:-}"
export ROS_PACKAGE_PATH="${ROOT_DIR}/Interactive-Nav-SG-nav/src:${ROS_PACKAGE_PATH:-}"

mkdir -p "${OUTPUT_ROOT}"
print -r -- "${OUTPUT_ROOT}" > /tmp/semantic_gt_three_scene_output_root

cleanup_process() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -INT "${pid}" 2>/dev/null || true
    for _attempt in {1..10}; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" 2>/dev/null || true
        return
      fi
      sleep 0.5
    done
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 1
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  fi
}

finish_recorder() {
  local pid="${1:-}"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return
  fi
  kill -INT "${pid}" 2>/dev/null || true
  for _attempt in {1..120}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      return
    fi
    sleep 1
  done
  kill -TERM "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

for house_ind in "${HOUSES[@]}"; do
  scene_dir="${OUTPUT_ROOT}/house_${house_ind}"
  mkdir -p "${scene_dir}"
  ros_port=$((ROS_PORT_BASE + house_ind))
  export ROS_MASTER_URI="http://127.0.0.1:${ros_port}"
  export ROS_HOSTNAME="127.0.0.1"
  export ROS_HOME="/tmp/codex_ros_semantic_${ros_port}_house_${house_ind}"
  export ROS_LOG_DIR="${scene_dir}/ros_logs"
  mkdir -p "${ROS_HOME}" "${ROS_LOG_DIR}"
  print -r -- "${ROS_MASTER_URI}" > "${scene_dir}/ros_master_uri.txt"

  stack_pid=""
  semantic_pid=""
  recorder_pid=""
  sim_pid=""
  trap 'cleanup_process "${sim_pid}"; cleanup_process "${recorder_pid}"; cleanup_process "${semantic_pid}"; cleanup_process "${stack_pid}"' INT TERM EXIT

  roslaunch -p "${ros_port}" "${ROOT_DIR}/Interactive-Nav-SG-nav/src/nav_pkg/launch/molmospaces_nav_system.launch" \
    start_sim:=false \
    start_semantic_mapping:=false \
    start_explore_py:=true \
    start_explore:=false \
    explore_py_config_file:="${ROOT_DIR}/Interactive-Nav-SG-nav/src/explore_py_pkg/config/explore_py.yaml" \
    > "${scene_dir}/navigation_stack.log" 2>&1 &
  stack_pid=$!

  stack_ready="false"
  for _attempt in {1..60}; do
    if rosnode list 2>/dev/null | grep -qx /move_base; then
      stack_ready="true"
      break
    fi
    sleep 1
  done
  if [[ "${stack_ready}" != "true" ]]; then
    print -u2 -- "Navigation stack did not become ready for house ${house_ind}"
    exit 3
  fi

  if [[ "${ENABLE_SEMANTIC}" == "true" ]]; then
    rosparam load "${ROOT_DIR}/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/config/default.yaml" /semantic_mapping_py
    python3 "${ROOT_DIR}/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/semantic_mapping_node.py" \
      > "${scene_dir}/semantic_mapping.log" 2>&1 &
    semantic_pid=$!
  fi

  recorder_args=(
    --output-dir "${scene_dir}" \
    --first-person-video-fps 15 \
    --first-person-video-capture-mode step \
    --image-queue-size 1 \
    --no-video-save-panel-frames \
    --first-person-video-width-px 640 \
    --no-external-video
  )
  if [[ "${ENABLE_SEMANTIC}" == "true" ]]; then
    recorder_args+=(--semantic-video)
  fi
  python3 "${ROOT_DIR}/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts/record_explore_debug.py" "${recorder_args[@]}" \
    > "${scene_dir}/recorder.log" 2>&1 &
  recorder_pid=$!

  sleep 2
  inflation_radius="$(rosparam get /move_base/local_costmap/inflation_layer/inflation_radius 2>/dev/null || true)"
  print -r -- "${inflation_radius}" > "${scene_dir}/local_inflation_radius.txt"
  if [[ "${inflation_radius}" != "0.15" ]]; then
    print -u2 -- "Unexpected local inflation radius for house ${house_ind}: ${inflation_radius}"
    exit 2
  fi

  sim_args=(
    --robot rby1 \
    --scene_dataset procthor-10k \
    --data_split train \
    --house_ind "${house_ind}" \
    --target_types Chair \
    --exploration_only true \
    --task_horizon "${TASK_HORIZON}" \
    --randomize_camera false \
    --publish_debug_front_camera false \
    --observation_queue_size 1 \
    --action_timeout_s "${ACTION_TIMEOUT_SEC}" \
    --max_consecutive_action_timeouts 0 \
    --step_frame_dir "${scene_dir}/sim_step_frames"
  )
  if [[ "${ENABLE_SEMANTIC}" == "true" ]]; then
    sim_args+=(
      --publish_realtime_gt true
      --realtime_gt_step_interval "${GT_STEP_INTERVAL}"
      --realtime_gt_max_distance_m "${GT_MAX_DISTANCE_M}"
      --realtime_gt_min_visible_pixels "${GT_MIN_VISIBLE_PIXELS}"
    )
  else
    sim_args+=(--publish_realtime_gt false)
  fi
  python3 "${ROOT_DIR}/scripts/InteractiveNav/run_nav_ros_sim.py" "${sim_args[@]}" \
    > "${scene_dir}/simulation.log" 2>&1 &
  sim_pid=$!

  first_step_waited_sec=0
  while [[ ! -s "${scene_dir}/sim_step_frames/manifest.jsonl" ]]; do
    if ! kill -0 "${sim_pid}" 2>/dev/null; then
      if ! wait "${sim_pid}"; then
        print -u2 -- "Simulator exited before first step for house ${house_ind}"
      else
        print -u2 -- "Simulator ended without producing a first step for house ${house_ind}"
      fi
      sim_pid=""
      exit 4
    fi
    if (( first_step_waited_sec >= FIRST_STEP_TIMEOUT_SEC )); then
      print -u2 -- "Timed out waiting ${FIRST_STEP_TIMEOUT_SEC}s for first simulator step in house ${house_ind}"
      exit 4
    fi
    sleep 1
    first_step_waited_sec=$((first_step_waited_sec + 1))
  done
  if [[ "${ENABLE_SEMANTIC}" == "true" ]]; then
    if ! timeout 60s rostopic echo -n 1 /semantic_mapping/gt_observations \
      > "${scene_dir}/first_gt_observation.txt" 2>&1; then
      print -u2 -- "Timed out waiting for realtime GT observations in house ${house_ind}"
      exit 4
    fi
  fi
  print -r -- "$(date --iso-8601=seconds)" > "${scene_dir}/effective_recording_start.txt"

  sim_waited_sec=0
  while kill -0 "${sim_pid}" 2>/dev/null; do
    if (( sim_waited_sec >= SIM_WAIT_TIMEOUT_SEC )); then
      print -u2 -- "Simulator did not finish within ${SIM_WAIT_TIMEOUT_SEC}s for house ${house_ind}"
      exit 5
    fi
    sleep 1
    sim_waited_sec=$((sim_waited_sec + 1))
  done
  if ! wait "${sim_pid}"; then
    print -u2 -- "Simulator failed for house ${house_ind}"
    exit 6
  fi
  sim_pid=""

  sim_step_frames=0
  if [[ -f "${scene_dir}/sim_step_frames/manifest.jsonl" ]]; then
    sim_step_frames=$(wc -l < "${scene_dir}/sim_step_frames/manifest.jsonl")
  fi
  if (( sim_step_frames != TASK_HORIZON )); then
    print -u2 -- "Simulator saved ${sim_step_frames}/${TASK_HORIZON} step frames for house ${house_ind}"
    exit 7
  fi
  recorder_waited_sec=0
  recorded_frames=0
  stable_seconds=0
  while (( recorder_waited_sec < RECORDER_DRAIN_TIMEOUT_SEC && stable_seconds < 10 )); do
    current_frames=0
    if [[ -f "${scene_dir}/video_frames.csv" ]]; then
      current_frames=$(( $(wc -l < "${scene_dir}/video_frames.csv") - 1 ))
    fi
    if (( current_frames == recorded_frames && current_frames > 0 )); then
      stable_seconds=$((stable_seconds + 1))
    else
      stable_seconds=0
      recorded_frames=${current_frames}
    fi
    sleep 1
    recorder_waited_sec=$((recorder_waited_sec + 1))
  done
  finish_recorder "${recorder_pid}"
  recorder_pid=""
  if [[ "${ENABLE_SEMANTIC}" == "true" ]]; then
    output_stem="overview_6panel"
  else
    output_stem="overview_4panel"
  fi
  python3 "${ROOT_DIR}/scripts/InteractiveNav/build_semantic_video_offline.py" \
    --scene-dir "${scene_dir}" \
    --fps 15 \
    --output-stem "${output_stem}" \
    > "${scene_dir}/offline_video_build.log" 2>&1
  offline_frames=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_frame_count"])' "${scene_dir}/offline_video_summary.json")
  print -r -- "${offline_frames}" > "${scene_dir}/recorded_step_frames.txt"
  if (( offline_frames != TASK_HORIZON )); then
    print -u2 -- "Offline video built ${offline_frames}/${TASK_HORIZON} frames for house ${house_ind}"
    exit 8
  fi
  cleanup_process "${semantic_pid}"
  semantic_pid=""
  cleanup_process "${stack_pid}"
  stack_pid=""
  trap - INT TERM EXIT
done

print -r -- "Exploration debug test complete (semantic=${ENABLE_SEMANTIC}): ${OUTPUT_ROOT}"
