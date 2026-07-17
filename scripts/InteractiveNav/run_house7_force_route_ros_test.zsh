#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
ROUTE_CONFIG=${ROUTE_CONFIG:-${SCRIPT_DIR}/configs/semantic_decision/house7_force_routes.yaml}
ROUTE_NAV_CONFIG=${ROUTE_NAV_CONFIG:-${SCRIPT_DIR}/configs/semantic_decision/house7_force_route_nav.yaml}
ROUTE_ID=${2:-${ROUTE_ID:-house7_force_route_01}}
OUTPUT_DIR=${1:-${REPO_ROOT}/outputs/${ROUTE_ID}_$(date +%Y%m%d_%H%M%S)}
ROS_SETUP=${ROS_SETUP:-${REPO_ROOT}/Interactive-Nav-SG-nav/devel/setup.zsh}
ROS_MASTER_URI=${ROS_MASTER_URI:-http://127.0.0.1:11431}
TASK_HORIZON=${TASK_HORIZON:-1000}
ROUTE_READY_TIMEOUT_S=${ROUTE_READY_TIMEOUT_S:-60}
ROUTE_NAVIGATION_TIMEOUT_S=${ROUTE_NAVIGATION_TIMEOUT_S:-180}
ROUTE_INTERACTION_TIMEOUT_S=${ROUTE_INTERACTION_TIMEOUT_S:-30}
ROUTE_GRAPH_TIMEOUT_S=${ROUTE_GRAPH_TIMEOUT_S:-30}

mkdir -p "${OUTPUT_DIR}/sim" "${OUTPUT_DIR}/ros_home/log"
export ROS_MASTER_URI
export ROS_IP=${ROS_IP:-127.0.0.1}
export ROS_HOME="${OUTPUT_DIR}/ros_home"
export ROS_LOG_DIR="${OUTPUT_DIR}/ros_home/log"

set +u
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate mlspaces
set -u
source "${ROS_SETUP}"
export ROS_PACKAGE_PATH="${REPO_ROOT}/Interactive-Nav-SG-nav/src:/opt/ros/noetic/share"
rospack profile >/dev/null

ROUTE_FIELDS=$(python -c 'import sys,yaml; p=yaml.safe_load(open(sys.argv[1])); r=next(x for x in p["routes"] if x["route_id"]==sys.argv[2]); print("{}\t{}".format(r["seed"], ",".join(str(v) for v in r["start_xyyaw"])))' "${ROUTE_CONFIG}" "${ROUTE_ID}")
IFS=$'\t' read -r ROUTE_SEED ROBOT_XYYAW <<< "${ROUTE_FIELDS}"

cleanup() {
  if [[ -n ${LAUNCH_PID:-} ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -INT "${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
  if [[ -n ${ROSCORE_PID:-} ]] && kill -0 "${ROSCORE_PID}" 2>/dev/null; then
    kill -INT "${ROSCORE_PID}" 2>/dev/null || true
    wait "${ROSCORE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

MASTER_PORT=${ROS_MASTER_URI##*:}
MASTER_PORT=${MASTER_PORT%%/*}
roscore -p "${MASTER_PORT}" >"${OUTPUT_DIR}/roscore.log" 2>&1 &
ROSCORE_PID=$!
MASTER_READY=false
for _index in {1..120}; do
  if timeout 1s rosparam list >/dev/null 2>&1; then
    MASTER_READY=true
    break
  fi
  sleep 0.25
done
if [[ ${MASTER_READY} != true ]]; then
  print -u2 "ROS master did not become ready. See ${OUTPUT_DIR}/roscore.log"
  exit 1
fi

SIM_EXTRA_ARGS="--seed ${ROUTE_SEED} --fixed_robot_xyyaw ${ROBOT_XYYAW} --enable_force_interaction true --force_interaction_log_path ${OUTPUT_DIR}/force_interaction_events.json --realtime_gt_step_interval 1 --realtime_gt_min_visible_pixels 4 --realtime_gt_max_distance_m 6.0 --map_warmup_skip_frames 3 --observation_queue_size 0 --step_frame_dir ${OUTPUT_DIR}/step_frames --step_log_every_n_steps 50 --sim_timing_log_every_n_steps 50"

roslaunch "${REPO_ROOT}/Interactive-Nav-SG-nav/src/nav_pkg/launch/molmospaces_nav_system.launch" \
  start_sim:=true \
  start_mapping:=true \
  mapping_mode:=odom_locked \
  start_semantic_mapping:=true \
  semantic_source:=realtime_gt \
  publish_realtime_gt:=true \
  start_nav:=true \
  start_explore:=false \
  start_explore_py:=false \
  nav_config_override_file:="${ROUTE_NAV_CONFIG}" \
  global_planner_allow_unknown:=true \
  manual_control:=true \
  exploration_only:=true \
  randomize_camera:=false \
  publish_debug_front_camera:=true \
  robot:=rby1 \
  scene_dataset:=procthor-10k \
  data_split:=train \
  house_ind:=7 \
  house_inds:=7 \
  task_horizon:="${TASK_HORIZON}" \
  scene_timeout_s:=300 \
  output_dir:="${OUTPUT_DIR}/sim" \
  sim_extra_args:="${SIM_EXTRA_ARGS}" \
  >"${OUTPUT_DIR}/roslaunch.log" 2>&1 &
LAUNCH_PID=$!

PYTHONUNBUFFERED=1 python -u "${SCRIPT_DIR}/run_house7_force_route.py" \
  --route-config "${ROUTE_CONFIG}" \
  --route-id "${ROUTE_ID}" \
  --output "${OUTPUT_DIR}/route_result.json" \
  --ready-timeout-s "${ROUTE_READY_TIMEOUT_S}" \
  --navigation-timeout-s "${ROUTE_NAVIGATION_TIMEOUT_S}" \
  --interaction-timeout-s "${ROUTE_INTERACTION_TIMEOUT_S}" \
  --graph-timeout-s "${ROUTE_GRAPH_TIMEOUT_S}"

sleep 3
print "House 7 force route completed: ${OUTPUT_DIR}"
