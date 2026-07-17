#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
ROUTE_CONFIG=${ROUTE_CONFIG:-${SCRIPT_DIR}/configs/semantic_decision/house7_force_routes.yaml}
ROUTE_ID=${2:-${ROUTE_ID:-house7_force_route_01}}
METHOD=${METHOD:-interactive_rule}
OUTPUT_DIR=${1:-${REPO_ROOT}/outputs/house7_${METHOD}_${ROUTE_ID}_$(date +%Y%m%d_%H%M%S)}
ROS_SETUP=${ROS_SETUP:-${REPO_ROOT}/Interactive-Nav-SG-nav/devel/setup.zsh}
ROS_MASTER_URI=${ROS_MASTER_URI:-http://127.0.0.1:11501}
TASK_HORIZON=${TASK_HORIZON:-1000}
VIDEO_FPS=${VIDEO_FPS:-15}
VIDEO_PANEL_WIDTH_PX=${VIDEO_PANEL_WIDTH_PX:-640}
GT_STEP_INTERVAL=${GT_STEP_INTERVAL:-3}
GT_MAX_DISTANCE_M=${GT_MAX_DISTANCE_M:-6.0}
GT_MIN_VISIBLE_PIXELS=${GT_MIN_VISIBLE_PIXELS:-16}
SIM_TIMEOUT_S=${SIM_TIMEOUT_S:-1200}
RECORDER_DRAIN_TIMEOUT_S=${RECORDER_DRAIN_TIMEOUT_S:-240}
ROUTE_NAV_CONFIG=${ROUTE_NAV_CONFIG:-${SCRIPT_DIR}/configs/semantic_decision/house7_force_route_nav.yaml}
SEMANTIC_DECISION_CONFIG=${SEMANTIC_DECISION_CONFIG:-${REPO_ROOT}/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/config/default.yaml}

case "${METHOD}" in
  frontier_only)
    START_SEMANTIC_DECISION=false
    ;;
  interactive_rule)
    START_SEMANTIC_DECISION=true
    ;;
  *)
    print -u2 -- "Unsupported METHOD=${METHOD}; use frontier_only or interactive_rule"
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_DIR}/sim" "${OUTPUT_DIR}/debug" "${OUTPUT_DIR}/ros_home/log"
export ROS_MASTER_URI
export ROS_IP=${ROS_IP:-127.0.0.1}
export ROS_HOSTNAME=${ROS_HOSTNAME:-127.0.0.1}
export ROS_HOME="${OUTPUT_DIR}/ros_home"
export ROS_LOG_DIR="${OUTPUT_DIR}/ros_home/log"

set +u
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate mlspaces
set -u
source "${ROS_SETUP}"
export ROS_PACKAGE_PATH="${REPO_ROOT}/Interactive-Nav-SG-nav/src:/opt/ros/noetic/share"

ROUTE_FIELDS=$(python -c 'import sys,yaml; p=yaml.safe_load(open(sys.argv[1])); r=next(x for x in p["routes"] if x["route_id"]==sys.argv[2]); print("{}\t{}".format(r["seed"], ",".join(str(v) for v in r["start_xyyaw"])))' "${ROUTE_CONFIG}" "${ROUTE_ID}")
IFS=$'\t' read -r ROUTE_SEED ROBOT_XYYAW <<< "${ROUTE_FIELDS}"

cleanup_process() {
  local pid="${1:-}"
  local grace_s="${2:-20}"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return
  fi
  kill -INT "${pid}" 2>/dev/null || true
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
  kill -TERM "${pid}" 2>/dev/null || true
  sleep 1
  kill -KILL "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  cleanup_process "${LAUNCH_PID:-}" 20
  cleanup_process "${RECORDER_PID:-}" 180
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

PYTHONUNBUFFERED=1 python -u "${REPO_ROOT}/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts/record_explore_debug.py" \
  --output-dir "${OUTPUT_DIR}/debug" \
  --video-step-sync-topic /molmo_spaces/step_sync \
  --first-person-video-capture-mode step \
  --semantic-video \
  --first-person-video-with-map \
  --first-person-video-fps "${VIDEO_FPS}" \
  --first-person-video-width-px "${VIDEO_PANEL_WIDTH_PX}" \
  --no-external-video \
  --no-video-save-panel-frames \
  --no-first-person-video-h264 \
  >"${OUTPUT_DIR}/recorder.log" 2>&1 &
RECORDER_PID=$!
sleep 1

SIM_EXTRA_ARGS="--seed ${ROUTE_SEED} --fixed_robot_xyyaw ${ROBOT_XYYAW} --enable_force_interaction true --force_interaction_log_path ${OUTPUT_DIR}/force_interaction_events.json --realtime_gt_step_interval ${GT_STEP_INTERVAL} --realtime_gt_min_visible_pixels ${GT_MIN_VISIBLE_PIXELS} --realtime_gt_max_distance_m ${GT_MAX_DISTANCE_M} --action_timeout_s 0.5 --map_warmup_skip_frames 3 --observation_queue_size 0 --step_frame_dir ${OUTPUT_DIR}/sim_step_frames --step_log_every_n_steps 50 --sim_timing_log_every_n_steps 50"

roslaunch "${REPO_ROOT}/Interactive-Nav-SG-nav/src/nav_pkg/launch/molmospaces_nav_system.launch" \
  start_sim:=true \
  start_mapping:=true \
  mapping_mode:=odom_locked \
  start_semantic_mapping:=true \
  semantic_source:=realtime_gt \
  publish_realtime_gt:=true \
  start_nav:=true \
  start_explore:=false \
  start_explore_py:=true \
  start_semantic_decision:="${START_SEMANTIC_DECISION}" \
  semantic_decision_config_file:="${SEMANTIC_DECISION_CONFIG}" \
  nav_config_override_file:="${ROUTE_NAV_CONFIG}" \
  global_planner_allow_unknown:=true \
  local_costmap_inflation_radius:=0.20 \
  exploration_only:=true \
  randomize_camera:=false \
  publish_debug_front_camera:=true \
  robot:=rby1 \
  scene_dataset:=procthor-10k \
  data_split:=train \
  house_ind:=7 \
  house_inds:=7 \
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
  print -u2 -- "Navigation launch exited with ${LAUNCH_EXIT}"
  exit "${LAUNCH_EXIT}"
fi

SIM_FRAME_MANIFEST="${OUTPUT_DIR}/sim_step_frames/manifest.jsonl"
RECORDER_WAITED=0
LAST_RECORDER_FRAMES=-1
STABLE_RECORDER_SECONDS=0
while (( RECORDER_WAITED < RECORDER_DRAIN_TIMEOUT_S )); do
  SIM_FRAMES=0
  RECORDER_FRAMES=0
  [[ -f "${SIM_FRAME_MANIFEST}" ]] && SIM_FRAMES=$(wc -l < "${SIM_FRAME_MANIFEST}")
  [[ -f "${OUTPUT_DIR}/debug/video_frames.csv" ]] && RECORDER_FRAMES=$(( $(wc -l < "${OUTPUT_DIR}/debug/video_frames.csv") - 1 ))
  if (( SIM_FRAMES >= TASK_HORIZON && RECORDER_FRAMES >= TASK_HORIZON )); then
    break
  fi
  if (( RECORDER_FRAMES == LAST_RECORDER_FRAMES && RECORDER_FRAMES > 0 )); then
    STABLE_RECORDER_SECONDS=$((STABLE_RECORDER_SECONDS + 1))
  else
    STABLE_RECORDER_SECONDS=0
  fi
  LAST_RECORDER_FRAMES=${RECORDER_FRAMES}
  if (( STABLE_RECORDER_SECONDS >= 10 )); then
    break
  fi
  sleep 1
  RECORDER_WAITED=$((RECORDER_WAITED + 1))
done
cleanup_process "${RECORDER_PID}" 180
RECORDER_PID=""

python "${SCRIPT_DIR}/build_semantic_video_offline.py" \
  --scene-dir "${OUTPUT_DIR}" \
  --debug-dir "${OUTPUT_DIR}/debug" \
  --fps "${VIDEO_FPS}" \
  --output-stem overview_6panel \
  >"${OUTPUT_DIR}/offline_video.log" 2>&1

python "${SCRIPT_DIR}/evaluate_exploration_coverage.py" \
  --run-dir "${OUTPUT_DIR}/debug" \
  --robot rby1 \
  --scene-dataset procthor-10k \
  --data-split train \
  --house-ind 7 \
  --gt-agent-radius-m 0.10 \
  >"${OUTPUT_DIR}/coverage.log" 2>&1 || true

python - "${OUTPUT_DIR}" "${METHOD}" "${ROUTE_ID}" "${TASK_HORIZON}" <<'PY'
import json
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
method = sys.argv[2]
route_id = sys.argv[3]
task_horizon = int(sys.argv[4])
def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

manifest = output_dir / "sim_step_frames" / "manifest.jsonl"
sim_frames = len(manifest.read_text().splitlines()) if manifest.exists() else 0
video = read_json(output_dir / "offline_video_summary.json")
coverage = read_json(output_dir / "debug" / "exploration_coverage.json")
force = read_json(output_dir / "force_interaction_events.json")
interaction_results = [event.get("result") or {} for event in force.get("events", [])]
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
result = {
    "method": method,
    "route_id": route_id,
    "task_horizon": task_horizon,
    "sim_step_frames": sim_frames,
    "video_frames": video.get("output_frame_count", 0),
    "video": video.get("video", ""),
    "coverage_ratio": coverage.get("exploration_coverage_ratio"),
    "mapped_free_coverage_ratio": coverage.get("mapped_free_coverage_ratio"),
    "interaction_count": len(interaction_results),
    "interaction_roots": [event.get("source_object_name", "") for event in interaction_results],
    "interaction_steps": [event.get("step") for event in interaction_results],
    "decision_count": len(decision_rows),
    "selected_behaviors": [
        (event.get("payload") or {}).get("behavior_type", "") for event in decision_rows
    ],
    "successful_behavior_count": sum(
        (event.get("payload") or {}).get("status") == "SUCCEEDED" for event in feedback_rows
    ),
    "valid_step_video": sim_frames == task_horizon and video.get("output_frame_count") == task_horizon,
}
(output_dir / "semantic_exploration_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(result, ensure_ascii=False))
PY

print -- "House 7 semantic exploration complete: ${OUTPUT_DIR}"
