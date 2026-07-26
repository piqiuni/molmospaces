#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
ROUTE_CONFIG=${ROUTE_CONFIG:-${SCRIPT_DIR}/configs/semantic_decision/house7_force_routes.yaml}
VIDEO_BUILDER=${VIDEO_BUILDER:-${SCRIPT_DIR}/build_semantic_video_offline.py}
ROUTE_ID=${2:-${ROUTE_ID:-house7_force_route_01}}
HOUSE_IND=${HOUSE_IND:-7}
USE_FIXED_ROUTE=${USE_FIXED_ROUTE:-true}
SCENE_SEED=${SCENE_SEED:-${HOUSE_IND}}
METHOD=${METHOD:-interactive_rule}
OUTPUT_DIR=${1:-${REPO_ROOT}/outputs/house7_${METHOD}_${ROUTE_ID}_$(date +%Y%m%d_%H%M%S)}
ROS_SETUP=${ROS_SETUP:-${REPO_ROOT}/Interactive-Nav-SG-nav/devel/setup.zsh}
ROS_MASTER_URI=${ROS_MASTER_URI:-http://127.0.0.1:11501}
TASK_HORIZON=${TASK_HORIZON:-1000}
VIDEO_FPS=${VIDEO_FPS:-15}
if [[ -z "${VIDEO_PANEL_WIDTH_PX:-}" ]]; then
  if [[ "${USE_FIXED_ROUTE}" == true ]]; then
    VIDEO_PANEL_WIDTH_PX=640
  else
    VIDEO_PANEL_WIDTH_PX=480
  fi
fi
VIDEO_FRAME_JOB_QUEUE_SIZE=${VIDEO_FRAME_JOB_QUEUE_SIZE:-4}
ARTIFACT_WRITE_QUEUE_SIZE=${ARTIFACT_WRITE_QUEUE_SIZE:-4}
VIDEO_HISTORY_SIZE=${VIDEO_HISTORY_SIZE:-16}
IMAGE_QUEUE_SIZE=${IMAGE_QUEUE_SIZE:-4}
VIDEO_ENCODER_PRESET=${VIDEO_ENCODER_PRESET:-ultrafast}
GT_STEP_INTERVAL=${GT_STEP_INTERVAL:-3}
GT_MAX_DISTANCE_M=${GT_MAX_DISTANCE_M:-6.0}
GT_MIN_VISIBLE_PIXELS=${GT_MIN_VISIBLE_PIXELS:-16}
GT_MIN_VISIBLE_FRACTION=${GT_MIN_VISIBLE_FRACTION:-0.20}
GT_REQUIRED_CONSECUTIVE_OBSERVATIONS=${GT_REQUIRED_CONSECUTIVE_OBSERVATIONS:-2}
GT_ROI_X_MIN_RATIO=${GT_ROI_X_MIN_RATIO:-0.10}
GT_ROI_X_MAX_RATIO=${GT_ROI_X_MAX_RATIO:-0.90}
GT_MIN_FORWARD_COSINE=${GT_MIN_FORWARD_COSINE:-0.15}
LOCAL_COSTMAP_INFLATION_RADIUS=${LOCAL_COSTMAP_INFLATION_RADIUS:-0.30}
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
if [[ -z "${INTERACTION_EXECUTION_MODE:-}" ]]; then
  if [[ -n "${DRAWER_EXECUTION_MODE:-}" ]]; then
    INTERACTION_EXECUTION_MODE=${DRAWER_EXECUTION_MODE}
  elif [[ "${ENABLE_RECORDING}" == true ]]; then
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
DRAWER_OBSERVATION_STEPS=${DRAWER_OBSERVATION_STEPS:-1}

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
  *)
    print -u2 -- "Unsupported METHOD=${METHOD}; use semantic_interaction_exploration, semantic_interaction_object_goal, frontier_only, interactive_rule, container_exploration, object_goal_rule, object_goal_model_mock, or object_goal_runtime"
    exit 2
    ;;
esac

COMPLETION_POST_HOLD_STEPS=${COMPLETION_POST_HOLD_STEPS:-0}

mkdir -p "${OUTPUT_DIR}/sim" "${OUTPUT_DIR}/ros_home/log"
if [[ "${ENABLE_RECORDING}" == true ]]; then
  mkdir -p "${OUTPUT_DIR}/videos"
fi
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

FIXED_ROUTE_ARGS=""
if [[ "${USE_FIXED_ROUTE}" == true ]]; then
  ROUTE_FIELDS=$(python -c 'import sys,yaml; p=yaml.safe_load(open(sys.argv[1])); r=next(x for x in p["routes"] if x["route_id"]==sys.argv[2]); print("{}\t{}".format(r["seed"], ",".join(str(v) for v in r["start_xyyaw"])))' "${ROUTE_CONFIG}" "${ROUTE_ID}")
  IFS=$'\t' read -r SCENE_SEED ROBOT_XYYAW <<< "${ROUTE_FIELDS}"
  FIXED_ROUTE_ARGS="--fixed_robot_xyyaw ${ROBOT_XYYAW}"
fi

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

if [[ "${ENABLE_RECORDING}" == true ]]; then
  PYTHONUNBUFFERED=1 python -u "${REPO_ROOT}/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts/record_explore_debug.py" \
    --output-dir "${OUTPUT_DIR}/debug" \
    --occupancy-grid-topic /semantic_mapping/planning_occ_map \
    --first-person-video-capture-mode step \
    --semantic-video \
    --first-person-video-with-map \
    --first-person-video-fps "${VIDEO_FPS}" \
    --first-person-video-width-px "${VIDEO_PANEL_WIDTH_PX}" \
    --video-frame-job-queue-size "${VIDEO_FRAME_JOB_QUEUE_SIZE}" \
    --artifact-write-queue-size "${ARTIFACT_WRITE_QUEUE_SIZE}" \
    --video-history-size "${VIDEO_HISTORY_SIZE}" \
    --image-queue-size "${IMAGE_QUEUE_SIZE}" \
    --video-global-panel-scale 1.8 \
    --runtime-video-encode \
    --first-person-video-h264-preset "${VIDEO_ENCODER_PRESET}" \
    --no-external-video \
    --no-video-save-panel-frames \
    --no-first-person-video-h264 \
    >"${OUTPUT_DIR}/recorder.log" 2>&1 &
else
  PYTHONUNBUFFERED=1 python -u "${REPO_ROOT}/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts/record_explore_debug.py" \
    --output-dir "${OUTPUT_DIR}/debug" \
    --occupancy-grid-topic /semantic_mapping/planning_occ_map \
    --no-first-person-video \
    --no-first-person-video-with-map \
    --no-semantic-video \
    --no-external-video \
    >"${OUTPUT_DIR}/recorder.log" 2>&1 &
fi
RECORDER_PID=$!
sleep 1

RUNTIME_TARGET_MODE=none
if [[ "${METHOD}" == object_goal_runtime || "${METHOD}" == semantic_interaction_object_goal ]]; then
  RUNTIME_TARGET_MODE=random_far_container_object
fi
if [[ "${ENABLE_RECORDING}" == true ]]; then
  SIM_CAPTURE_ARGS="--observation_queue_size 0 --step_frame_dir ${OUTPUT_DIR}/sim_step_frames --step_frame_queue_size 4"
else
  SIM_CAPTURE_ARGS="--observation_queue_size 1"
fi
SIM_EXTRA_ARGS="--seed ${SCENE_SEED} ${FIXED_ROUTE_ARGS} --initial_door_state ${INITIAL_DOOR_STATE} --enable_force_interaction true --force_interaction_close_all_containers_on_prepare ${FORCE_CLOSE_CONTAINERS} --force_interaction_log_path ${OUTPUT_DIR}/force_interaction_events.json --force_interaction_execution_mode ${INTERACTION_EXECUTION_MODE} --force_interaction_transition_steps ${INTERACTION_TRANSITION_STEPS} --force_interaction_drawer_execution_mode ${DRAWER_EXECUTION_MODE} --force_interaction_drawer_transition_steps ${DRAWER_TRANSITION_STEPS} --force_interaction_drawer_observation_steps ${DRAWER_OBSERVATION_STEPS} --realtime_gt_step_interval ${GT_STEP_INTERVAL} --realtime_gt_min_visible_pixels ${GT_MIN_VISIBLE_PIXELS} --realtime_gt_min_visible_fraction ${GT_MIN_VISIBLE_FRACTION} --realtime_gt_required_consecutive_observations ${GT_REQUIRED_CONSECUTIVE_OBSERVATIONS} --realtime_gt_max_distance_m ${GT_MAX_DISTANCE_M} --action_timeout_s 0.5 --map_warmup_skip_frames 3 ${SIM_CAPTURE_ARGS} --require_move_base_active_for_cmd_vel false --no-retain_task_history --runtime_target_selection_mode ${RUNTIME_TARGET_MODE} --runtime_target_selection_top_k 3 --runtime_target_selection_path ${OUTPUT_DIR}/target_selection.json --completion_mode ${COMPLETION_MODE} --completion_confirmations ${COMPLETION_CONFIRMATIONS} --completion_post_hold_steps ${COMPLETION_POST_HOLD_STEPS} --completion_status_path ${OUTPUT_DIR}/completion_status.json --step_log_every_n_steps 50 --sim_timing_log_every_n_steps 50"

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
  explore_py_config_override_file:="${EXPLORE_PY_CONFIG_OVERRIDE}" \
  start_semantic_decision:="${START_SEMANTIC_DECISION}" \
  semantic_decision_config_file:="${SEMANTIC_DECISION_CONFIG}" \
  semantic_decision_config_override_file:="${SEMANTIC_DECISION_OVERRIDE}" \
  nav_config_override_file:="${ROUTE_NAV_CONFIG}" \
  local_costmap_inflation_radius:="${LOCAL_COSTMAP_INFLATION_RADIUS}" \
  exploration_only:=true \
  randomize_camera:=false \
  publish_debug_front_camera:=false \
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
  print -u2 -- "Navigation launch exited with ${LAUNCH_EXIT}"
  exit "${LAUNCH_EXIT}"
fi

cleanup_process "${RECORDER_PID}" 20
RECORDER_PID=""

if [[ "${ENABLE_RECORDING}" == true ]]; then
  OFFLINE_VIDEO_START=$(python -c 'import time; print(time.perf_counter())')
  python "${VIDEO_BUILDER}" \
    --scene-dir "${OUTPUT_DIR}" \
    --debug-dir "${OUTPUT_DIR}/debug" \
    --fps "${VIDEO_FPS}" \
    --state-alignment latest \
    --output-stem overview_6panel \
    >"${OUTPUT_DIR}/offline_video.log" 2>&1
  OFFLINE_VIDEO_ELAPSED_SEC=$(python - "${OFFLINE_VIDEO_START}" <<'PY'
import sys
import time
print(max(0.0, time.perf_counter() - float(sys.argv[1])))
PY
  )
else
  OFFLINE_VIDEO_ELAPSED_SEC=0.0
fi
print -r -- "${OFFLINE_VIDEO_ELAPSED_SEC}" >"${OUTPUT_DIR}/offline_video_elapsed_sec.txt"

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
print -r -- "${ANALYSIS_ELAPSED_SEC}" >"${OUTPUT_DIR}/analysis_elapsed_sec.txt"

python - "${OUTPUT_DIR}" "${METHOD}" "${ROUTE_ID}" "${TASK_HORIZON}" "${HOUSE_IND}" <<'PY'
import json
import re
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
method = sys.argv[2]
route_id = sys.argv[3]
task_horizon = int(sys.argv[4])
house_ind = int(sys.argv[5])
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
result = {
    "method": method,
    "recording_enabled": bool(video_path.exists()),
    "route_id": route_id,
    "house_ind": house_ind,
    "task_horizon": task_horizon,
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
    "interaction_roots": [event.get("source_object_name", "") for event in interaction_results],
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
        bool(event.get("result", {}).get("success"))
        and event.get("result", {}).get("source_object_name")
        == read_json(output_dir / "target_selection.json").get("container_name")
        for event in force.get("events", [])
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
    "step_sync_image_match_count": debug_summary.get("step_sync_image_match_count", 0),
    "step_sync_image_reuse_count": debug_summary.get("step_sync_image_reuse_count", 0),
    "step_sync_placeholder_count": debug_summary.get("step_sync_placeholder_count", 0),
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
