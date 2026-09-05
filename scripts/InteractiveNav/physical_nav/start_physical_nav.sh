#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # Keep ROS' generated Python/message paths when this script is run from a
  # clean shell. The Go2 process never sources ROS; this is policy-host only.
  source /opt/ros/noetic/setup.bash
fi
CATKIN_SETUP="${ROOT_DIR}/../../../Interactive-Nav-SG-nav/devel/setup.bash"
if [[ -f "${CATKIN_SETUP}" ]]; then
  # This supplies the compiled struct_mapping executables and generated ROS
  # package paths.  It is optional so protocol/web tests work from a clean
  # checkout without a catkin build.
  source "${CATKIN_SETUP}"
fi
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/../../../Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts:${ROOT_DIR}/../../../Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts:${ROOT_DIR}/../../../Interactive-Nav-SG-nav/src/explore_py_pkg/scripts:${ROOT_DIR}/../../../Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts:${ROOT_DIR}/ros_compat:${PYTHONPATH:-}"
export ROS_PACKAGE_PATH="${ROOT_DIR}/../../../Interactive-Nav-SG-nav/src:${ROS_PACKAGE_PATH:-}"
# Several ROS Python message packages import NumPy.  On this 64-core host the
# default BLAS thread fan-out can leave small policy nodes spinning during
# import for minutes while YOLO/mapping are already busy.  These nodes do no
# matrix-heavy CPU work; keep one BLAS/OpenMP thread per process.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

WEB_HOST="${PHYSICAL_NAV_WEB_HOST:-0.0.0.0}"
GATEWAY_PYTHON="${PHYSICAL_NAV_GATEWAY_PYTHON:-/home/user/miniconda3/envs/mlspaces/bin/python3}"
WEB_PORT="${PHYSICAL_NAV_WEB_PORT:-8765}"
WS_HOST="${PHYSICAL_NAV_WS_HOST:-0.0.0.0}"
WS_PORT="${PHYSICAL_NAV_WS_PORT:-12334}"
QWEN_URL="${PHYSICAL_NAV_QWEN_URL:-http://127.0.0.1:18080/v1}"
QWEN_MODEL="${PHYSICAL_NAV_QWEN_MODEL:-qwen3.6-35b-a3b-fp8}"
QWEN_AUTO_INTERVAL="${PHYSICAL_NAV_QWEN_AUTO_INTERVAL:-0}"
RECORD_DIR="${PHYSICAL_NAV_RECORD_DIR:-/home/user/ldl/recordings/go2_physical}"
RECORD_MODE="${PHYSICAL_NAV_RECORD_MODE:-raw_plus_panels}"
RECORD_QUEUE_SIZE="${PHYSICAL_NAV_RECORD_QUEUE_SIZE:-4096}"
RECORD_ON_START="${PHYSICAL_NAV_RECORD_ON_START:-0}"
INTERACTION_PROFILE="${PHYSICAL_NAV_INTERACTION_PROFILE:-physical_human}"
ENABLE_M1="${PHYSICAL_NAV_ENABLE_M1:-true}"
ENABLE_INTERACTION_POLICY="${PHYSICAL_NAV_ENABLE_INTERACTION_POLICY:-true}"
SEMANTIC_OVERRIDE_CONFIG="${PHYSICAL_NAV_SEMANTIC_OVERRIDE_CONFIG:-${ROOT_DIR}/config/semantic_shadow_override.yaml}"
export SEMANTIC_MODEL_MODE="${SEMANTIC_MODEL_MODE:-http}"
export SEMANTIC_MODEL_ENDPOINT="${SEMANTIC_MODEL_ENDPOINT:-http://127.0.0.1:18080/v1}"
export SEMANTIC_MODEL_NAME="${SEMANTIC_MODEL_NAME:-${QWEN_MODEL}}"
export SEMANTIC_MODEL_PROTOCOL="${SEMANTIC_MODEL_PROTOCOL:-openai_chat}"
export SEMANTIC_MODEL_METRICS_PATH="${SEMANTIC_MODEL_METRICS_PATH:-/tmp/physical_nav_mllm.jsonl}"
export SEMANTIC_MODEL_TRACE_URL="${SEMANTIC_MODEL_TRACE_URL:-http://127.0.0.1:${WEB_PORT}/api/mllm-event}"
# Measured Go2 standing-pose calibration: camera is about 3 cm forward of
# the base centre (38 cm from a 70 cm rear-to-front body) and 0.62 m above
# the base. The base-to-ground offset is therefore about 0.43 m (1.05 m
# camera height), which is not part of this base-frame extrinsic.
# Normal standing base_link height is 0.305 m; camera ground height is 1.285 m,
# so the camera is 0.980 m above base_link. Camera is pitched 13 degrees down.
CAMERA_X="${PHYSICAL_NAV_CAMERA_X:-0.03}"; CAMERA_Y="${PHYSICAL_NAV_CAMERA_Y:-0}"; CAMERA_Z="${PHYSICAL_NAV_CAMERA_Z:-0.98}"; CAMERA_ROLL="${PHYSICAL_NAV_CAMERA_ROLL:-0}"; CAMERA_PITCH="${PHYSICAL_NAV_CAMERA_PITCH:-0.2268928}"; CAMERA_YAW="${PHYSICAL_NAV_CAMERA_YAW:-0}"
RUNTIME_DIR="${PHYSICAL_NAV_RUNTIME_DIR:-/tmp/molmospaces-physical-nav-${UID}}"
LOG_DIR="${PHYSICAL_NAV_LOG_DIR:-${RUNTIME_DIR}/logs}"
GATEWAY_PID_FILE="${PHYSICAL_NAV_GATEWAY_PID_FILE:-${RUNTIME_DIR}/gateway.pid}"
mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"

declare -a RECORD_ARGS=(--record-dir "${RECORD_DIR}" --record-mode "${RECORD_MODE}" --record-queue-size "${RECORD_QUEUE_SIZE}")
if [[ "${RECORD_ON_START}" == "1" ]]; then
  RECORD_ARGS+=(--record-on-start)
fi

declare -a SUPERVISED_PIDS=()
declare -A PROCESS_NAMES=()
EXIT_REASON="supervisor exited"
CLEANUP_STARTED=0

log_supervisor() {
  local message="$(date -Is) $*"
  echo "${message}"
  echo "${message}" >>"${LOG_DIR}/supervisor.log"
}

register_process() {
  local name="$1" pid="$2"
  SUPERVISED_PIDS+=("${pid}")
  PROCESS_NAMES["${pid}"]="${name}"
  log_supervisor "started ${name} pid=${pid}"
}

cleanup() {
  local status=$?
  if (( CLEANUP_STARTED )); then return; fi
  CLEANUP_STARTED=1
  trap - EXIT INT TERM HUP
  log_supervisor "stopping stack: ${EXIT_REASON} (status=${status})"
  for pid in "${SUPERVISED_PIDS[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for _ in {1..100}; do
    local alive=0
    for pid in "${SUPERVISED_PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then alive=1; break; fi
    done
    (( alive == 0 )) && break
    sleep .1
  done
  for pid in "${SUPERVISED_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      log_supervisor "forcing stop of ${PROCESS_NAMES[$pid]:-process} pid=${pid}"
      kill -KILL "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done
  printf '%s\t%s\t%s\n' "$(date -Is)" "${status}" "${EXIT_REASON}" >"${RUNTIME_DIR}/last_exit.tsv"
  if [[ -f "${RUNTIME_DIR}/supervisor.pid" ]] &&
     [[ "$(<"${RUNTIME_DIR}/supervisor.pid")" == "$$" ]]; then
    rm -f "${RUNTIME_DIR}/supervisor.pid"
  fi
}

handle_signal() {
  EXIT_REASON="received signal $1"
  exit "$2"
}

trap cleanup EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP

if command -v roscore >/dev/null 2>&1 && [[ "${PHYSICAL_NAV_START_ROSCORE:-1}" == "1" ]]; then
  if ! (exec 3<>/dev/tcp/127.0.0.1/11311) 2>/dev/null; then
    roscore >>"${LOG_DIR}/roscore.log" 2>&1 &
    register_process roscore "$!"
    for _ in {1..20}; do
      if (exec 3<>/dev/tcp/127.0.0.1/11311) 2>/dev/null; then break; fi
      sleep .25
    done
  fi
fi

if [[ "${PHYSICAL_NAV_START_QWEN_TUNNEL:-0}" == "1" ]]; then
  python3 "${ROOT_DIR}/qwen_ssh_tunnel.py" \
    --ssh-port "${PHYSICAL_NAV_QWEN_SSH_PORT:-41051}" \
    --user "${PHYSICAL_NAV_QWEN_USER:-root}" --host "${PHYSICAL_NAV_QWEN_HOST:-115.190.90.101}" \
    --local-port "${PHYSICAL_NAV_QWEN_LOCAL_PORT:-18080}" --remote-port "${PHYSICAL_NAV_QWEN_REMOTE_PORT:-8000}" \
    >>"${LOG_DIR}/qwen_tunnel.log" 2>&1 &
  register_process qwen_tunnel "$!"
fi

start_or_reuse_gateway() {
  local gateway_pid=""
  if [[ -f "${GATEWAY_PID_FILE}" ]]; then gateway_pid="$(<"${GATEWAY_PID_FILE}")"; fi
  if [[ -z "${gateway_pid}" ]]; then
    gateway_pid="$(pgrep -f '[p]hysical_six_panel_server.py' | head -n1 || true)"
  fi
  if [[ "${gateway_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${gateway_pid}" 2>/dev/null &&
     [[ "$(tr '\0' ' ' <"/proc/${gateway_pid}/cmdline" 2>/dev/null || true)" == *physical_six_panel_server.py* ]]; then
    GATEWAY_PID="${gateway_pid}"
    log_supervisor "reusing persistent gateway pid=${GATEWAY_PID}"
    return 0
  fi
  rm -f "${GATEWAY_PID_FILE}"
  # The gateway is deliberately not registered as a critical child. It owns
  # the persistent web/control plane and must survive navigation watchdog
  # failures and ROS stack restarts.
  # Put the persistent visualization/control gateway in its own session so a
  # navigation supervisor restart cannot terminate it via the supervisor PGID.
  nohup setsid "${GATEWAY_PYTHON}" "${ROOT_DIR}/physical_six_panel_server.py" \
  --ws-host "${WS_HOST}" --ws-port "${WS_PORT}" \
  --http-host "${WEB_HOST}" --http-port "${WEB_PORT}" \
  --qwen-url "${QWEN_URL}" --qwen-model "${QWEN_MODEL}" --qwen-auto-interval "${QWEN_AUTO_INTERVAL}" \
  "${RECORD_ARGS[@]}" \
  --camera-x "${CAMERA_X}" --camera-y "${CAMERA_Y}" --camera-z "${CAMERA_Z}" --camera-roll "${CAMERA_ROLL}" --camera-pitch "${CAMERA_PITCH}" --camera-yaw "${CAMERA_YAW}" \
    >>"${LOG_DIR}/gateway.log" 2>&1 </dev/null &
  GATEWAY_PID=$!
  printf '%s\n' "${GATEWAY_PID}" >"${GATEWAY_PID_FILE}"
  log_supervisor "started persistent gateway pid=${GATEWAY_PID}"
}

# Start/reuse the gateway first so downstream workers never enter a long retry
# loop before their only input endpoint exists. It is intentionally persistent.
start_or_reuse_gateway

GATEWAY_READY=0
for _ in {1..100}; do
  if ! kill -0 "${GATEWAY_PID}" 2>/dev/null; then break; fi
  if (exec 3<>/dev/tcp/127.0.0.1/"${WEB_PORT}") 2>/dev/null; then
    exec 3>&- 3<&-
    GATEWAY_READY=1
    break
  fi
  sleep .1
done
if (( GATEWAY_READY == 0 )); then
  EXIT_REASON="gateway failed to become ready; see ${LOG_DIR}/gateway.log"
  exit 1
fi

START_YOLO="${PHYSICAL_NAV_START_YOLO_WORKER:-1}"
if [[ "${START_YOLO}" == "1" ]]; then
  # YOLOE/Ultralytics is installed in the mlspaces environment; using the
  # system interpreter silently starts the ROS stack but drops perception.
  ALGORITHM_PYTHON="${PHYSICAL_NAV_ALGORITHM_PYTHON:-}"
  if [[ -z "${ALGORITHM_PYTHON}" ]]; then
    if [[ -x /home/user/miniconda3/envs/mlspaces/bin/python3 ]]; then
      ALGORITHM_PYTHON=/home/user/miniconda3/envs/mlspaces/bin/python3
    else
      ALGORITHM_PYTHON=python3
    fi
  fi
  "${ALGORITHM_PYTHON}" "${ROOT_DIR}/physical_yoloe_bridge.py" \
    --web-url "http://127.0.0.1:${WEB_PORT}" \
    --model-path "${PHYSICAL_NAV_MODEL_PATH:-/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26l-seg-pf.pt}" \
    --detector-config "${PHYSICAL_NAV_DETECTOR_CONFIG:-${ROOT_DIR}/config/physical_nav.yaml}" \
    --device "${PHYSICAL_NAV_YOLO_DEVICE:-cuda:0}" --rate "${PHYSICAL_NAV_YOLO_RATE:-10}" \
    --camera-x "${CAMERA_X}" --camera-y "${CAMERA_Y}" --camera-z "${CAMERA_Z}" \
    --camera-roll "${CAMERA_ROLL}" --camera-pitch "${CAMERA_PITCH}" --camera-yaw "${CAMERA_YAW}" \
    >>"${LOG_DIR}/yoloe.log" 2>&1 &
  register_process yoloe "$!"
fi

if [[ "${PHYSICAL_NAV_SKIP_ROS:-0}" == "1" ]]; then
  log_supervisor "ROS launch disabled"
elif command -v roscore >/dev/null 2>&1 && command -v roslaunch >/dev/null 2>&1; then
  roslaunch "${ROOT_DIR}/launch/physical_nav_readonly.launch" \
    config_file:="${ROOT_DIR}/config/physical_nav.yaml" \
    semantic_override_config:="${SEMANTIC_OVERRIDE_CONFIG}" \
    move_base_override_config:="${ROOT_DIR}/config/physical_move_base_override.yaml" \
    model_path:="${PHYSICAL_NAV_MODEL_PATH:-/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26l-seg-pf.pt}" \
    camera_x:="${CAMERA_X}" camera_y:="${CAMERA_Y}" camera_z:="${CAMERA_Z}" \
    camera_roll:="${CAMERA_ROLL}" camera_pitch:="${CAMERA_PITCH}" camera_yaw:="${CAMERA_YAW}" \
    interaction_profile:="${INTERACTION_PROFILE}" enable_m1:="${ENABLE_M1}" \
    enable_interaction_policy:="${ENABLE_INTERACTION_POLICY}" \
    ros_python:="${PHYSICAL_NAV_ROS_PYTHON:-/home/user/miniconda3/envs/mlspaces/bin/python3}" \
    system_ros_python:="${PHYSICAL_NAV_SYSTEM_ROS_PYTHON:-${ROOT_DIR}/physical_ros_python.sh}" \
    >"${PHYSICAL_NAV_ROSLAUNCH_STDOUT:-${LOG_DIR}/roslaunch.log}" 2>&1 &
  register_process roslaunch "$!"
else
  log_supervisor "ROS1 tools not found; running gateway and detector only"
fi

if [[ "${PHYSICAL_NAV_WATCHDOG_ENABLED:-1}" == "1" ]]; then
  WATCHDOG_ARGS=(
    --url "http://127.0.0.1:${WEB_PORT}/api/health"
    --interval-s "${PHYSICAL_NAV_WATCHDOG_INTERVAL_S:-2}"
    --timeout-s "${PHYSICAL_NAV_WATCHDOG_TIMEOUT_S:-1}"
    --startup-grace-s "${PHYSICAL_NAV_WATCHDOG_STARTUP_GRACE_S:-60}"
    --frame-stale-s "${PHYSICAL_NAV_WATCHDOG_FRAME_STALE_S:-15}"
    --perception-stale-s "${PHYSICAL_NAV_WATCHDOG_PERCEPTION_STALE_S:-15}"
    --failure-limit "${PHYSICAL_NAV_WATCHDOG_FAILURE_LIMIT:-5}"
    --keep-running
  )
  [[ "${START_YOLO}" == "1" ]] && WATCHDOG_ARGS+=(--require-perception)
  python3 "${ROOT_DIR}/physical_nav_watchdog.py" "${WATCHDOG_ARGS[@]}" \
    >>"${LOG_DIR}/watchdog.log" 2>&1 &
  register_process watchdog "$!"
fi

log_supervisor "Physical gateway: http://$(hostname -I | awk '{print $1}'):${WEB_PORT}/"
log_supervisor "component logs: ${LOG_DIR}"

# Any critical child exit is an all-stack failure.  This prevents ROS from
# silently consuming a frozen last frame after the gateway or detector dies.
set +e
EXITED_PID=""
wait -n -p EXITED_PID "${SUPERVISED_PIDS[@]}"
EXIT_CODE=$?
set -e
EXITED_NAME="unknown"
if [[ -n "${EXITED_PID}" ]]; then
  EXITED_NAME="${PROCESS_NAMES[$EXITED_PID]:-unknown}"
fi
EXIT_REASON="critical process ${EXITED_NAME} pid=${EXITED_PID} exited with status ${EXIT_CODE}"
log_supervisor "${EXIT_REASON}"
(( EXIT_CODE == 0 )) && EXIT_CODE=1
exit "${EXIT_CODE}"
