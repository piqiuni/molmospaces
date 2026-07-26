#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${SCRIPT_DIR:h:h}
OUTPUT_DIR=${1:-${REPO_ROOT}/outputs/semantic_door_occ_house7_ros_$(date +%Y%m%d_%H%M%S)}
TARGET_ROOT=${TARGET_ROOT:-doorway_a7d89b1f5e3818edcbcff3ec135e6ac7_1_0_4}
ROBOT_XYYAW=${ROBOT_XYYAW:-3.324263487680909,2.556999047254367,1.5707963267948966}
OPEN_STEP=${OPEN_STEP:-180}
POSE_SEQUENCE=${POSE_SEQUENCE:-}
EXPECTED_PHASES=${EXPECTED_PHASES:-5}
ROS_SETUP=${ROS_SETUP:-${REPO_ROOT}/Interactive-Nav-SG-nav/devel/setup.zsh}

mkdir -p "${OUTPUT_DIR}/sim" "${OUTPUT_DIR}/occ" "${OUTPUT_DIR}/ros_home/log"
export ROS_HOME="${OUTPUT_DIR}/ros_home"
export ROS_LOG_DIR="${OUTPUT_DIR}/ros_home/log"
export ROS_MASTER_URI=${ROS_MASTER_URI:-http://127.0.0.1:11329}
export ROS_IP=${ROS_IP:-127.0.0.1}

set +u
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate mlspaces
set -u
source "${ROS_SETUP}"
export ROS_PACKAGE_PATH="${REPO_ROOT}/Interactive-Nav-SG-nav/src:/opt/ros/noetic/share"
rospack profile >/dev/null

LAUNCH_FILE="${REPO_ROOT}/Interactive-Nav-SG-nav/src/nav_pkg/launch/molmospaces_nav_system.launch"
DOOR_TEST_ARGS="--door_occ_test_root_name ${TARGET_ROOT} --door_occ_test_transition_path ${OUTPUT_DIR}/door_occ_test_transitions.json"
RECORDER_ARGS=(--timeout-s 150)
if [[ -n ${POSE_SEQUENCE} ]]; then
  DOOR_TEST_ARGS="${DOOR_TEST_ARGS} --door_occ_test_pose_sequence ${POSE_SEQUENCE}"
  RECORDER_ARGS+=(
    --phase-file "${OUTPUT_DIR}/door_occ_test_transitions.json"
    --expected-phases "${EXPECTED_PHASES}"
  )
else
  DOOR_TEST_ARGS="${DOOR_TEST_ARGS} --door_occ_test_open_step ${OPEN_STEP}"
fi
SIM_EXTRA_ARGS="--fixed_robot_xyyaw ${ROBOT_XYYAW} --immediate_noop_after_publish --map_warmup_skip_frames 3 --realtime_gt_step_interval 1 --realtime_gt_min_visible_pixels 4 ${DOOR_TEST_ARGS} --step_log_every_n_steps 50 --sim_timing_log_every_n_steps 50"

cleanup() {
  if [[ -n ${LAUNCH_PID:-} ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -INT "${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

roslaunch "${LAUNCH_FILE}" \
  start_sim:=true \
  start_mapping:=true \
  mapping_mode:=odom_locked \
  start_semantic_mapping:=true \
  semantic_source:=realtime_gt \
  publish_realtime_gt:=true \
  start_nav:=true \
  start_explore:=false \
  start_explore_py:=false \
  manual_control:=true \
  exploration_only:=true \
  randomize_camera:=false \
  robot:=rby1 \
  scene_dataset:=procthor-10k \
  data_split:=train \
  house_ind:=7 \
  house_inds:=7 \
  task_horizon:=600 \
  scene_timeout_s:=160 \
  output_dir:="${OUTPUT_DIR}/sim" \
  sim_extra_args:="${SIM_EXTRA_ARGS}" \
  >"${OUTPUT_DIR}/roslaunch.log" 2>&1 &
LAUNCH_PID=$!

MASTER_READY=false
for _index in {1..120}; do
  if rosnode list >/dev/null 2>&1; then
    MASTER_READY=true
    break
  fi
  sleep 0.5
done
if [[ ${MASTER_READY} != true ]]; then
  print -u2 "ROS master did not become ready. See ${OUTPUT_DIR}/roslaunch.log"
  exit 1
fi

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python "${SCRIPT_DIR}/record_semantic_door_occ_transition.py" \
  --target-root "${TARGET_ROOT}" \
  --output-dir "${OUTPUT_DIR}/occ" \
  "${RECORDER_ARGS[@]}"

print "House 7 semantic door OCC ROS test completed: ${OUTPUT_DIR}"
