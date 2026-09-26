#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # Keep ROS' generated Python/message paths when this script is run from a
  # clean shell. The Go2 process never sources ROS; this is policy-host only.
  # Catkin profile scripts reference ROS_DISTRO; under set -u a clean
  # non-interactive shell would abort before the stack starts.
  export ROS_DISTRO="${ROS_DISTRO:-noetic}"
  export ROS_VERSION="${ROS_VERSION:-1}"
  export ROS_PYTHON_VERSION="${ROS_PYTHON_VERSION:-3}"
  set +u
  source /opt/ros/noetic/setup.bash
  set -u
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

# Single-machine ROS: bind XML-RPC/TCPROS to loopback only (avoid 0.0.0.0).
# Override via environment if multi-machine ROS is required.
export ROS_IP="${ROS_IP:-127.0.0.1}"
export ROS_HOSTNAME="${ROS_HOSTNAME:-localhost}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"

# Missing Conda ABI dependencies otherwise fail only inside subscriber threads:
# nodes remain alive while every ROS input silently stops connecting.
if [[ "${PHYSICAL_NAV_SKIP_ROS:-0}" != 1 ]] && command -v roslaunch >/dev/null 2>&1; then
  if ! "${PHYSICAL_NAV_ALGORITHM_PYTHON:-/home/user/miniconda3/envs/mlspaces/bin/python3}" \
      -c 'import netifaces; import rosgraph.network; rosgraph.network.get_local_addresses()'; then
    echo "ROS Python preflight failed; install requirements_policy.txt in PHYSICAL_NAV_ALGORITHM_PYTHON's environment" >&2
    exit 1
  fi
fi

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
START_WEB="${PHYSICAL_NAV_START_WEB:-1}"
INTERACTION_PROFILE="${PHYSICAL_NAV_INTERACTION_PROFILE:-physical_human}"
ENABLE_M1="${PHYSICAL_NAV_ENABLE_M1:-true}"
ENABLE_INTERACTION_POLICY="${PHYSICAL_NAV_ENABLE_INTERACTION_POLICY:-true}"
SEMANTIC_OVERRIDE_CONFIG="${PHYSICAL_NAV_SEMANTIC_OVERRIDE_CONFIG:-${ROOT_DIR}/config/semantic_shadow_override.yaml}"
export SEMANTIC_MODEL_MODE="${SEMANTIC_MODEL_MODE:-http}"
export SEMANTIC_MODEL_ENDPOINT="${SEMANTIC_MODEL_ENDPOINT:-http://127.0.0.1:18080/v1}"
export SEMANTIC_MODEL_NAME="${SEMANTIC_MODEL_NAME:-${QWEN_MODEL}}"
export SEMANTIC_MODEL_PROTOCOL="${SEMANTIC_MODEL_PROTOCOL:-openai_chat}"
export SEMANTIC_MODEL_METRICS_PATH="${SEMANTIC_MODEL_METRICS_PATH:-/tmp/physical_nav_mllm.jsonl}"
if [[ "${START_WEB}" == "1" ]]; then
  export SEMANTIC_MODEL_TRACE_URL="${SEMANTIC_MODEL_TRACE_URL:-http://127.0.0.1:${WEB_PORT}/api/mllm-event}"
else
  export SEMANTIC_MODEL_TRACE_URL="${SEMANTIC_MODEL_TRACE_URL:-}"
fi
# Measured Go2 standing-pose calibration: camera is about 3 cm forward of
# the base centre (38 cm from a 70 cm rear-to-front body) and 0.62 m above
# the base. The base-to-ground offset is therefore about 0.43 m (1.05 m
# camera height), which is not part of this base-frame extrinsic.
# Normal standing base_link height is 0.305 m; camera ground height is 1.190 m,
# so the camera is 0.885 m above base_link. Camera is pitched 9 degrees down.
CAMERA_X="${PHYSICAL_NAV_CAMERA_X:-0.03}"; CAMERA_Y="${PHYSICAL_NAV_CAMERA_Y:-0}"; CAMERA_Z="${PHYSICAL_NAV_CAMERA_Z:-0.885}"; CAMERA_ROLL="${PHYSICAL_NAV_CAMERA_ROLL:-0}"; CAMERA_PITCH="${PHYSICAL_NAV_CAMERA_PITCH:-0.157079633}"; CAMERA_YAW="${PHYSICAL_NAV_CAMERA_YAW:-0}"
CAMERA_IMU="${PHYSICAL_NAV_CAMERA_IMU:-0}"
RUNTIME_DIR="${PHYSICAL_NAV_RUNTIME_DIR:-/tmp/molmospaces-physical-nav-${UID}}"
LOG_DIR="${PHYSICAL_NAV_LOG_DIR:-${RUNTIME_DIR}/logs}"
GATEWAY_PID_FILE="${PHYSICAL_NAV_GATEWAY_PID_FILE:-${RUNTIME_DIR}/gateway.pid}"
GATEWAY_FINGERPRINT_FILE="${PHYSICAL_NAV_GATEWAY_FINGERPRINT_FILE:-${RUNTIME_DIR}/gateway.fingerprint}"
YOLO_FINGERPRINT_FILE="${PHYSICAL_NAV_YOLO_FINGERPRINT_FILE:-${RUNTIME_DIR}/yoloe.fingerprint}"
mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"

# A second supervisor can otherwise start a second roslaunch tree during a
# restart race.  ROS then lets the newer executor evict the older one (and the
# older roslaunch may immediately respawn it), leaving neither executor stable.
# Keep the lock open for the lifetime of this supervisor, not just startup.
STACK_LOCK_FILE="${RUNTIME_DIR}/stack.lock"
exec 8>"${STACK_LOCK_FILE}"
if ! flock -n 8; then
  echo "physical navigation supervisor already owns ${RUNTIME_DIR}" >&2
  exit 1
fi

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
    --local-port "${PHYSICAL_NAV_QWEN_LOCAL_PORT:-18080}" --remote-port "${PHYSICAL_NAV_QWEN_REMOTE_PORT:-8100}" \
    >>"${LOG_DIR}/qwen_tunnel.log" 2>&1 &
  register_process qwen_tunnel "$!"
fi

start_or_reuse_gateway() {
  local gateway_pid="" expected_fingerprint="" recorded_fingerprint=""
  # The web gateway is deliberately persistent across ROS-worker restarts,
  # but persistence must not turn source/config changes into a silent no-op.
  # Include every module that participates in the live state/TF/render path as
  # well as its command-line configuration.  A missing fingerprint is treated
  # as stale, which upgrades gateways created by older versions of this script.
  expected_fingerprint="$(
    {
      printf 'python=%s\n' "${GATEWAY_PYTHON}"
      printf 'web=%s:%s ws=%s:%s qwen=%s model=%s interval=%s\n' \
        "${WEB_HOST}" "${WEB_PORT}" "${WS_HOST}" "${WS_PORT}" \
        "${QWEN_URL}" "${QWEN_MODEL}" "${QWEN_AUTO_INTERVAL}"
      printf 'record=%s mode=%s queue=%s on_start=%s camera=%s,%s,%s,%s,%s,%s\n' \
        "${RECORD_DIR}" "${RECORD_MODE}" "${RECORD_QUEUE_SIZE}" \
        "${RECORD_ON_START}" "${CAMERA_X}" "${CAMERA_Y}" "${CAMERA_Z}" \
        "${CAMERA_ROLL}" "${CAMERA_PITCH}" "${CAMERA_YAW}"
      # The canonical renderer lives one directory above physical_nav;
      # keeping it in the fingerprint is required because panel geometry
      # and heading fixes are imported at gateway startup.
      for source_file in \
        "${ROOT_DIR}/physical_six_panel_server.py" \
        "${ROOT_DIR}/runtime_state.py" \
        "${ROOT_DIR}/physical_protocol.py" \
        "${ROOT_DIR}/../offline_semantic_renderer.py" \
        "${ROOT_DIR}/showcase_pages.py" \
        "${ROOT_DIR}/physical_ros_gateway.py"; do
        if command -v sha256sum >/dev/null 2>&1; then
          sha256sum "${source_file}"
        else
          shasum -a 256 "${source_file}"
        fi
      done
    } | if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi
  )"
  if [[ -f "${GATEWAY_PID_FILE}" ]]; then gateway_pid="$(<"${GATEWAY_PID_FILE}")"; fi
  if [[ -z "${gateway_pid}" ]]; then
    gateway_pid="$(pgrep -f '[p]hysical_six_panel_server.py' | head -n1 || true)"
  fi
  if [[ "${gateway_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${gateway_pid}" 2>/dev/null &&
     [[ "$(tr '\0' ' ' <"/proc/${gateway_pid}/cmdline" 2>/dev/null || true)" == *physical_six_panel_server.py* ]]; then
    [[ -f "${GATEWAY_FINGERPRINT_FILE}" ]] && recorded_fingerprint="$(<"${GATEWAY_FINGERPRINT_FILE}")"
    if [[ "${PHYSICAL_NAV_FORCE_GATEWAY_RESTART:-0}" != "1" &&
          -n "${recorded_fingerprint}" && "${recorded_fingerprint}" == "${expected_fingerprint}" ]]; then
      GATEWAY_PID="${gateway_pid}"
      log_supervisor "reusing persistent gateway pid=${GATEWAY_PID}"
      return 0
    fi
    if [[ "${PHYSICAL_NAV_FORCE_GATEWAY_RESTART:-0}" == "1" ]]; then
      log_supervisor "restarting gateway pid=${gateway_pid} (explicit restart resets navigation step)"
    else
      log_supervisor "restarting stale persistent gateway pid=${gateway_pid} (source/config fingerprint changed)"
    fi
    kill -TERM "${gateway_pid}" 2>/dev/null || true
    for _ in {1..50}; do
      if ! kill -0 "${gateway_pid}" 2>/dev/null; then break; fi
      sleep .1
    done
    if kill -0 "${gateway_pid}" 2>/dev/null; then
      log_supervisor "forcing stop of stale gateway pid=${gateway_pid}"
      kill -KILL "${gateway_pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${GATEWAY_PID_FILE}" "${GATEWAY_FINGERPRINT_FILE}"
  # The gateway is deliberately not registered as a critical child. It owns
  # the persistent web/control plane and must survive navigation watchdog
  # failures and ROS stack restarts.
  # Put the persistent visualization/control gateway in its own session so a
  # navigation supervisor restart cannot terminate it via the supervisor PGID.
  # Close FD 8 explicitly: it is the supervisor's stack.lock descriptor.  If
  # the persistent gateway inherits it, the gateway keeps the lock held after
  # the supervisor exits and every later start reports "already owns".
  nohup setsid "${GATEWAY_PYTHON}" "${ROOT_DIR}/physical_six_panel_server.py" \
  --ws-host "${WS_HOST}" --ws-port "${WS_PORT}" \
  --http-host "${WEB_HOST}" --http-port "${WEB_PORT}" \
  --qwen-url "${QWEN_URL}" --qwen-model "${QWEN_MODEL}" --qwen-auto-interval "${QWEN_AUTO_INTERVAL}" \
  "${RECORD_ARGS[@]}" \
  --camera-x "${CAMERA_X}" --camera-y "${CAMERA_Y}" --camera-z "${CAMERA_Z}" --camera-roll "${CAMERA_ROLL}" --camera-pitch "${CAMERA_PITCH}" --camera-yaw "${CAMERA_YAW}" \
    >>"${LOG_DIR}/gateway.log" 2>&1 </dev/null 8>&- &
  GATEWAY_PID=$!
  printf '%s\n' "${GATEWAY_PID}" >"${GATEWAY_PID_FILE}"
  printf '%s\n' "${expected_fingerprint}" >"${GATEWAY_FINGERPRINT_FILE}"
  log_supervisor "started persistent gateway pid=${GATEWAY_PID}"
}

# The dashboard is optional. Its HTTP readiness must not gate ROS/YOLO startup.
if [[ "${START_WEB}" == "1" ]]; then
  start_or_reuse_gateway || log_supervisor "optional dashboard startup failed; continuing ROS pipeline"
else
  log_supervisor "web dashboard disabled; ROS sensor path remains active"
  GATEWAY_PID=""
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
  YOLO_MODEL_PATH="${PHYSICAL_NAV_MODEL_PATH:-/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26l-seg-pf.pt}"
  YOLO_CONFIG_PATH="${PHYSICAL_NAV_DETECTOR_CONFIG:-${ROOT_DIR}/config/physical_nav.yaml}"
  # Validate the shared launcher before replacing an existing worker.
  bash "${ROOT_DIR}/run_yolo.sh" --dry-run >/dev/null
  YOLO_GPU="${PHYSICAL_NAV_YOLO_GPU:-0}"
  YOLO_RATE="${PHYSICAL_NAV_YOLO_RATE:-10}"
  YOLO_EXPECTED_FINGERPRINT="$({
    printf 'python=%s gpu=%s rate=%s camera=%s,%s,%s,%s,%s,%s\n' \
      "${ALGORITHM_PYTHON}" "${YOLO_GPU}" "${YOLO_RATE}" \
      "${CAMERA_X}" "${CAMERA_Y}" \
      "${CAMERA_Z}" "${CAMERA_ROLL}" "${CAMERA_PITCH}" "${CAMERA_YAW}"
    printf 'model=%s ' "${YOLO_MODEL_PATH}"
    if [[ -f "${YOLO_MODEL_PATH}" ]]; then
      stat -c '%Y:%s' "${YOLO_MODEL_PATH}" 2>/dev/null || wc -c <"${YOLO_MODEL_PATH}"
    else
      printf 'missing\n'
    fi
    printf 'config=%s\n' "${YOLO_CONFIG_PATH}"
    if [[ -f "${YOLO_CONFIG_PATH}" ]]; then
      if command -v sha256sum >/dev/null 2>&1; then sha256sum "${YOLO_CONFIG_PATH}"; else shasum -a 256 "${YOLO_CONFIG_PATH}"; fi
    else
      printf 'missing-config\n'
    fi
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "${ROOT_DIR}/physical_yoloe_bridge.py"; else shasum -a 256 "${ROOT_DIR}/physical_yoloe_bridge.py"; fi
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "${ROOT_DIR}/run_yolo.sh"; else shasum -a 256 "${ROOT_DIR}/run_yolo.sh"; fi
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "${ROOT_DIR}/capture_geometry.py"; else shasum -a 256 "${ROOT_DIR}/capture_geometry.py"; fi
  } | if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi)"
  # Reusing one detector is important: two workers would consume the same
  # latest-only ROS image stream and publish competing detections.  As with
  # the persistent web gateway, do not silently keep an old worker after a
  # source/config change: the next explicit restart replaces it.
  EXISTING_YOLO="$(pgrep -f '[p]hysical_yoloe_bridge.py' | head -n1 || true)"
  YOLO_REUSED=0
  if [[ "${EXISTING_YOLO}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${EXISTING_YOLO}" 2>/dev/null &&
     [[ "$(tr '\0' ' ' <"/proc/${EXISTING_YOLO}/cmdline" 2>/dev/null || true)" == *physical_yoloe_bridge.py* ]]; then
    YOLO_RECORDED_FINGERPRINT=""
    [[ -f "${YOLO_FINGERPRINT_FILE}" ]] && YOLO_RECORDED_FINGERPRINT="$(<"${YOLO_FINGERPRINT_FILE}")"
    if [[ -n "${YOLO_RECORDED_FINGERPRINT}" && "${YOLO_RECORDED_FINGERPRINT}" == "${YOLO_EXPECTED_FINGERPRINT}" ]]; then
      log_supervisor "reusing persistent yoloe pid=${EXISTING_YOLO}"
      YOLO_REUSED=1
    else
      log_supervisor "restarting stale persistent yoloe pid=${EXISTING_YOLO} (source/config fingerprint changed)"
      kill -TERM "${EXISTING_YOLO}" 2>/dev/null || true
      for _ in {1..50}; do
        if ! kill -0 "${EXISTING_YOLO}" 2>/dev/null; then break; fi
        sleep .1
      done
      if kill -0 "${EXISTING_YOLO}" 2>/dev/null; then
        log_supervisor "forcing stop of stale yoloe pid=${EXISTING_YOLO}"
        kill -KILL "${EXISTING_YOLO}" 2>/dev/null || true
      fi
      rm -f "${YOLO_FINGERPRINT_FILE}"
    fi
  fi
  if (( YOLO_REUSED == 0 )); then
    bash "${ROOT_DIR}/run_yolo.sh" \
      >>"${LOG_DIR}/yoloe.log" 2>&1 &
    register_process yoloe "$!"
    printf '%s\n' "${YOLO_EXPECTED_FINGERPRINT}" >"${YOLO_FINGERPRINT_FILE}"
  fi
fi

ROS_STARTED=0
if [[ "${PHYSICAL_NAV_SKIP_ROS:-0}" == "1" ]]; then
  log_supervisor "ROS launch disabled"
elif command -v roscore >/dev/null 2>&1 && command -v roslaunch >/dev/null 2>&1; then
  roslaunch physical_nav physical_nav_readonly.launch \
    config_file:="${ROOT_DIR}/config/physical_nav.yaml" \
    semantic_override_config:="${SEMANTIC_OVERRIDE_CONFIG}" \
    move_base_override_config:="${ROOT_DIR}/config/physical_move_base_override.yaml" \
    model_path:="${PHYSICAL_NAV_MODEL_PATH:-/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26l-seg-pf.pt}" \
    camera_x:="${CAMERA_X}" camera_y:="${CAMERA_Y}" camera_z:="${CAMERA_Z}" \
    camera_roll:="${CAMERA_ROLL}" camera_pitch:="${CAMERA_PITCH}" camera_yaw:="${CAMERA_YAW}" \
    sensor_ws_url:="ws://127.0.0.1:${PHYSICAL_NAV_SENSOR_WS_PORT:-12335}" \
    web_url:="http://127.0.0.1:${WEB_PORT}" web_state_enabled:="${START_WEB}" \
    interaction_profile:="${INTERACTION_PROFILE}" enable_m1:="${ENABLE_M1}" \
    enable_interaction_policy:="${ENABLE_INTERACTION_POLICY}" \
    ros_python:="${PHYSICAL_NAV_ROS_PYTHON:-/home/user/miniconda3/envs/mlspaces/bin/python3}" \
    system_ros_python:="${PHYSICAL_NAV_SYSTEM_ROS_PYTHON:-${ROOT_DIR}/physical_ros_python.sh}" \
    >"${PHYSICAL_NAV_ROSLAUNCH_STDOUT:-${LOG_DIR}/roslaunch.log}" 2>&1 &
  register_process roslaunch "$!"
  ROS_STARTED=1
else
  log_supervisor "ROS1 tools not found; running gateway and detector only"
fi

if [[ "${PHYSICAL_NAV_WATCHDOG_ENABLED:-1}" == "1" && "${ROS_STARTED}" == "1" ]]; then
  WATCHDOG_ARGS=(
    --source ros
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
elif [[ "${PHYSICAL_NAV_WATCHDOG_ENABLED:-1}" == "1" ]]; then
  log_supervisor "ROS health watchdog disabled because ROS was not started"
fi

if [[ "${START_WEB}" == "1" ]]; then
  log_supervisor "Physical gateway: http://$(hostname -I | awk '{print $1}'):${WEB_PORT}/"
fi
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
