#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for the physical, read-only Go2 pipeline.  This file
# runs on the policy machine.  It starts the two SSH transports (the Go2
# tunnel is started through SSH on the dog), then starts the local gateway,
# YOLOE and ROS stack through physical_nav_service.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE="${ROOT_DIR}/physical_nav_service.sh"
POLICY_CONTROL_SCRIPT="${ROOT_DIR}/../uni_control/policy_control_server.py"
POLICY_CONTROL_PYTHON="${PHYSICAL_NAV_POLICY_CONTROL_PYTHON:-/usr/bin/python3}"
POLICY_CONTROL_PYTHONPATH="${PHYSICAL_NAV_POLICY_CONTROL_PYTHONPATH:-/opt/ros/noetic/lib/python3/dist-packages:/home/user/miniconda3/envs/mlspaces/lib/python3.11/site-packages}"
STATE_DIR="${PHYSICAL_NAV_ALL_RUNTIME_DIR:-/tmp/molmospaces-physical-nav-all-${UID}}"
LOG_DIR="${PHYSICAL_NAV_ALL_LOG_DIR:-${STATE_DIR}/logs}"
mkdir -p "${STATE_DIR}" "${LOG_DIR}"

GO2_SSH_TARGET="${PHYSICAL_NAV_GO2_SSH_TARGET:-unitree}"
GO2_TUNNEL_TARGET="${PHYSICAL_NAV_GO2_TUNNEL_TARGET:-zgca_gpu}"
GO2_BRIDGE_PATH="${PHYSICAL_NAV_GO2_BRIDGE_PATH:-/home/unitree/physical_nav/go2_readonly_sensor_bridge.py}"
GO2_BRIDGE_LOG="${PHYSICAL_NAV_GO2_BRIDGE_LOG:-/home/unitree/physical_nav/go2_readonly_sensor_bridge.log}"
GO2_INTERFACE="${PHYSICAL_NAV_GO2_INTERFACE:-eth0}"
GO2_FPS="${PHYSICAL_NAV_GO2_FPS:-10}"
GO2_COLOR_WIDTH="${PHYSICAL_NAV_GO2_COLOR_WIDTH:-1280}"
GO2_COLOR_HEIGHT="${PHYSICAL_NAV_GO2_COLOR_HEIGHT:-720}"
GO2_COLOR_FPS="${PHYSICAL_NAV_GO2_COLOR_FPS:-10}"
GO2_DEPTH_WIDTH="${PHYSICAL_NAV_GO2_DEPTH_WIDTH:-848}"
GO2_DEPTH_HEIGHT="${PHYSICAL_NAV_GO2_DEPTH_HEIGHT:-480}"
GO2_DEPTH_FPS="${PHYSICAL_NAV_GO2_DEPTH_FPS:-10}"
GO2_ALIGN_TO="${PHYSICAL_NAV_GO2_ALIGN_TO:-depth}"
GO2_CAMERA_IMU="${PHYSICAL_NAV_GO2_CAMERA_IMU:-0}"
GO2_TELEMETRY_PERIOD="${PHYSICAL_NAV_GO2_TELEMETRY_PERIOD:-0.05}"
# Both machines are on the same experiment LAN. Direct WebSocket transport
# avoids SSH channel head-of-line buffering and lets a reconnect discard an
# incomplete stale frame. Set this to ws://127.0.0.1:12334 to restore tunneling.
GO2_SENSOR_URL="${PHYSICAL_NAV_GO2_SENSOR_URL:-ws://10.100.5.3:12334}"
QWEN_TUNNEL_SCRIPT="${ROOT_DIR}/qwen_ssh_tunnel.py"
QWEN_LOG="${LOG_DIR}/qwen_tunnel.log"

GO2_TUNNEL_PID_FILE="${STATE_DIR}/go2_tunnel.pid"
GO2_TUNNEL_OWNED_FILE="${STATE_DIR}/go2_tunnel.owned"
GO2_BRIDGE_PID_FILE="${STATE_DIR}/go2_bridge.pid"
GO2_BRIDGE_OWNED_FILE="${STATE_DIR}/go2_bridge.owned"
QWEN_PID_FILE="${STATE_DIR}/qwen_tunnel.pid"
POLICY_PID_FILE="${STATE_DIR}/policy_control.pid"
MOTION_REMOTE_PID_FILE="${PHYSICAL_NAV_MOTION_REMOTE_PID_FILE:-/home/unitree/uni_control/go2_control.pid}"
MOTION_REMOTE_READY_FILE="${PHYSICAL_NAV_MOTION_REMOTE_READY_FILE:-/home/unitree/uni_control/go2_control.ready}"
MOTION_OWNED_FILE="${STATE_DIR}/motion_control.owned"
MOTION_STATUS_FILE="${STATE_DIR}/motion_control.status"
MOTION_MAX_VX="${PHYSICAL_NAV_MOTION_MAX_VX:-0.6}"
MOTION_MAX_WZ="${PHYSICAL_NAV_MOTION_MAX_WZ:-1.3}"
MOTION_ENABLED=0

usage() {
  cat <<EOF
Usage: $(basename "$0") {start|start_control|stop|restart|status|logs} [enable_motion] [obj_goal]

Starts the Go2 read-only sensor bridge, Go2 SSH tunnel, Qwen SSH tunnel and
the local physical navigation stack. Motion control is started only when the
optional second argument is exactly 'enable_motion'. If an object goal is
supplied, it is injected into the semantic decision chain after ROS starts;
the algorithm then finds the object and generates its navigation subgoal.
EOF
}

pid_alive() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]] && kill -0 "$1" 2>/dev/null
}

wait_local_port() {
  local port="$1" tries="${2:-100}"
  for _ in $(seq 1 "${tries}"); do
    if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
      exec 3>&- 3<&-
      return 0
    fi
    sleep .1
  done
  return 1
}

wait_ros_command_subscriber() {
  local topic="${PHYSICAL_NAV_MOTION_CMD_VEL_TOPIC:-/physical_nav/actuated_cmd_vel}"
  local timeout="${PHYSICAL_NAV_MOTION_READY_TIMEOUT_S:-30}"
  command -v rostopic >/dev/null 2>&1 || return 0
  for _ in $(seq 1 $((timeout * 4))); do
    # Do not start the remote actuator until the local mux/safety chain has a
    # live subscriber.  Without this barrier the bridge can become ready first
    # and the first command is lost, making startup appear randomly delayed.
    if rostopic info "${topic}" 2>/dev/null | grep -q '^ Subscribers:'; then
      return 0
    fi
    sleep .25
  done
  echo "warning: no ROS subscriber on ${topic} after ${timeout}s; continuing" >&2
  return 0
}

start_qwen_tunnel() {
  if [[ -f "${QWEN_PID_FILE}" ]] && pid_alive "$(<"${QWEN_PID_FILE}")"; then
    echo "Qwen tunnel already running (pid=$(<"${QWEN_PID_FILE}"))"
    return 0
  fi
  rm -f "${QWEN_PID_FILE}"
  nohup python3 "${QWEN_TUNNEL_SCRIPT}" \
    --ssh-port "${PHYSICAL_NAV_QWEN_SSH_PORT:-41051}" \
    --user "${PHYSICAL_NAV_QWEN_USER:-root}" \
    --host "${PHYSICAL_NAV_QWEN_HOST:-115.190.90.101}" \
    --local-port "${PHYSICAL_NAV_QWEN_LOCAL_PORT:-18080}" \
    --remote-port "${PHYSICAL_NAV_QWEN_REMOTE_PORT:-8000}" \
    >>"${QWEN_LOG}" 2>&1 </dev/null &
  echo $! >"${QWEN_PID_FILE}"
  echo "Qwen tunnel started (pid=$(<"${QWEN_PID_FILE}"))"
}

start_go2_components() {
  # The tunnel and bridge run on the Go2.  Reuse an already-running matching
  # bridge, but record ownership so stop never kills a process we did not
  # create.
  local result tunnel_pid bridge_pid
  result="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "${GO2_SSH_TARGET}" bash -s -- \
    "${GO2_TUNNEL_TARGET}" "${GO2_BRIDGE_PATH}" "${GO2_BRIDGE_LOG}" \
    "${GO2_INTERFACE}" "${GO2_FPS}" "${GO2_TELEMETRY_PERIOD}" \
    "${GO2_SENSOR_URL}" "${GO2_COLOR_WIDTH}" "${GO2_COLOR_HEIGHT}" "${GO2_COLOR_FPS}" \
    "${GO2_DEPTH_WIDTH}" "${GO2_DEPTH_HEIGHT}" "${GO2_DEPTH_FPS}" "${GO2_ALIGN_TO}" "${GO2_CAMERA_IMU}" <<'REMOTE'
set -u
tunnel_pid="0"
tunnel_owned=0
if [ "$7" = "ws://127.0.0.1:12334" ]; then
  tunnel_pid="$(pgrep -f '[s]sh -N -T .*127\.0\.0\.1:12334.*127\.0\.0\.1:12334' | head -n1 || true)"
  if [ -z "${tunnel_pid}" ]; then
    nohup setsid ssh -N -T \
      -o BatchMode=yes -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -L 127.0.0.1:12334:127.0.0.1:12334 "$1" \
      >>/home/unitree/physical_nav/go2_sensor_tunnel.log 2>&1 </dev/null &
    tunnel_pid=$!
    tunnel_owned=1
  fi
fi
bridge_pid="$(pgrep -f "[p]ython3? .*${2}" | head -n1 || true)"
bridge_owned=0
  if [ -z "${bridge_pid}" ]; then
  nohup setsid python3 "$2" \
    --url "$7" --interface "$4" --fps "$5" \
    --color-width "$8" --color-height "$9" --color-fps "${10}" \
    --depth-width "${11}" --depth-height "${12}" --depth-fps "${13}" --align-to "${14}" \
    --telemetry-period "$6" \
    $(if [ "${15}" = 1 ]; then echo --enable-camera-imu; fi) \
    >>"$3" 2>&1 </dev/null &
  bridge_pid=$!
  bridge_owned=1
fi
printf '%s\t%s\t%s\t%s\n' "$tunnel_pid" "$tunnel_owned" "$bridge_pid" "$bridge_owned"
REMOTE
  )"
  read -r tunnel_pid tunnel_owned bridge_pid bridge_owned <<<"${result}"
  printf '%s\n' "${tunnel_pid}" >"${GO2_TUNNEL_PID_FILE}"
  printf '%s\n' "${bridge_pid}" >"${GO2_BRIDGE_PID_FILE}"
  if [[ "${tunnel_owned}" == 1 ]]; then : >"${GO2_TUNNEL_OWNED_FILE}"; else rm -f "${GO2_TUNNEL_OWNED_FILE}"; fi
  if [[ "${bridge_owned}" == 1 ]]; then : >"${GO2_BRIDGE_OWNED_FILE}"; else rm -f "${GO2_BRIDGE_OWNED_FILE}"; fi
  if [[ "${tunnel_pid}" == 0 ]]; then
    echo "Go2 sensor transport direct: ${GO2_SENSOR_URL}"
  else
    echo "Go2 SSH tunnel pid=${tunnel_pid}$( [[ "${tunnel_owned}" == 1 ]] && echo ' (started)' || echo ' (reused)' )"
  fi
  echo "Go2 sensor bridge pid=${bridge_pid}$( [[ "${bridge_owned}" == 1 ]] && echo ' (started)' || echo ' (reused)' )"
}

start_motion_control() {
  # Fail closed before opening the local policy server: an untracked legacy
  # bridge may still own port 12333. Never let the read-only/speech launch
  # connect to a process that was started with --enable-motion.
  local existing_mode
  existing_mode="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "${GO2_SSH_TARGET}" \
    "mkdir -p \"$(dirname "${MOTION_REMOTE_PID_FILE}")\"; \
     existing_pid=\$(pgrep -f '[s]tart_go2_control.py' | head -n1 || true); \
     if [ -n \"\${existing_pid}\" ] && kill -0 \"\${existing_pid}\" 2>/dev/null; then \
       existing_args=\$(tr '\\0' ' ' < /proc/\${existing_pid}/cmdline); \
       echo \"\${existing_pid}\" > \"${MOTION_REMOTE_PID_FILE}\"; \
       if printf '%s' \"\${existing_args}\" | grep -q -- '--enable-motion'; then \
         if printf '%s' \"\${existing_args}\" | grep -q -- '--bridge-arg=${MOTION_MAX_VX}' \
            && printf '%s' \"\${existing_args}\" | grep -q -- '--bridge-arg=${MOTION_MAX_WZ}'; then \
           echo motion; \
         else \
           echo motion_mismatch; \
         fi; \
       else echo speech_only; fi; \
     else echo none; fi")"
  if (( ! MOTION_ENABLED )) && [[ "${existing_mode}" == motion* ]]; then
    echo "refusing read-only launch: an existing Go2 bridge has --enable-motion" >&2
    return 1
  fi
  if (( MOTION_ENABLED )) && [[ "${existing_mode}" == "speech_only" || "${existing_mode}" == "motion_mismatch" ]]; then
    if [[ "${existing_mode}" == "motion_mismatch" ]]; then
      echo "restarting existing Go2 bridge to apply velocity limits vx=${MOTION_MAX_VX}, wz=${MOTION_MAX_WZ}"
    else
      echo "switching existing Go2 bridge from speech-only to motion-enabled"
    fi
    # start_go2_control cannot change the mode of a live child. Stop the old
    # launcher/bridge first; the bridge's shutdown path sends an immediate
    # zero velocity before the motion-enabled replacement is created below.
    ssh -o BatchMode=yes -o ConnectTimeout=5 "${GO2_SSH_TARGET}" bash -s -- \
      "${MOTION_REMOTE_PID_FILE}" "${MOTION_REMOTE_READY_FILE}" <<'REMOTE'
set -u
pid_file="$1"
ready_file="$2"
launcher_pid="$(cat "${pid_file}" 2>/dev/null || true)"
bridge_pid="$(cat "${ready_file}" 2>/dev/null || true)"
if [[ "${launcher_pid}" =~ ^[1-9][0-9]*$ ]]; then
  kill -TERM "${launcher_pid}" 2>/dev/null || true
fi
for _ in {1..40}; do
  launcher_alive=0; bridge_alive=0
  [[ "${launcher_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${launcher_pid}" 2>/dev/null && launcher_alive=1
  [[ "${bridge_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${bridge_pid}" 2>/dev/null && bridge_alive=1
  (( launcher_alive == 0 && bridge_alive == 0 )) && break
  sleep .25
done
for pid in "${bridge_pid}" "${launcher_pid}"; do
  if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -KILL "${pid}" 2>/dev/null || true
  fi
done
rm -f "${pid_file}" "${ready_file}"
REMOTE
    existing_mode="none"
  fi
  if [[ -f "${POLICY_PID_FILE}" ]] && pid_alive "$(<"${POLICY_PID_FILE}")"; then
    echo "policy control server already running (pid=$(<"${POLICY_PID_FILE}"))"
  else
    rm -f "${POLICY_PID_FILE}"
    nohup setsid env PYTHONPATH="${POLICY_CONTROL_PYTHONPATH}" "${POLICY_CONTROL_PYTHON}" "${POLICY_CONTROL_SCRIPT}" \
      --control-mode continuous --source ros \
      --continuous-ttl-ms "${PHYSICAL_NAV_MOTION_TTL_MS:-500}" \
      --ros-command-refresh-hz "${PHYSICAL_NAV_MOTION_REFRESH_HZ:-20}" \
      --ros-command-stale-after-s "${PHYSICAL_NAV_MOTION_STALE_AFTER_S:-0.50}" \
      --cmd-vel-topic "${PHYSICAL_NAV_MOTION_CMD_VEL_TOPIC:-/physical_nav/actuated_cmd_vel}" \
      --speech-request-topic "${PHYSICAL_NAV_SPEECH_REQUEST_TOPIC:-/physical_nav/speech_request}" \
      --speech-status-topic "${PHYSICAL_NAV_SPEECH_STATUS_TOPIC:-/physical_nav/speech_status}" \
      >>"${LOG_DIR}/policy_control.log" 2>&1 </dev/null &
    echo $! >"${POLICY_PID_FILE}"
    echo "policy control server started (pid=$(<"${POLICY_PID_FILE}"))"
  fi
  if ! wait_local_port "${PHYSICAL_NAV_MOTION_POLICY_PORT:-12333}" 50; then
    echo "policy control server did not open port 12333" >&2
    return 1
  fi

  # Speech and velocity share this bridge, but --enable-motion remains the
  # explicit safety boundary. The default starts auxiliary/speech transport
  # without initializing the Go2 velocity client.
  local result motion_flag="" motion_mode="speech_only"
  if (( MOTION_ENABLED )); then
    motion_flag="--enable-motion"
    motion_mode="enable_motion"
  fi
  if ! result="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "${GO2_SSH_TARGET}" bash -s -- \
    "${MOTION_REMOTE_PID_FILE}" "${MOTION_REMOTE_READY_FILE}" "${motion_mode}" \
    "${PHYSICAL_NAV_MOTION_STARTUP_TIMEOUT_S:-30}" "${MOTION_MAX_VX}" "${MOTION_MAX_WZ}" <<'REMOTE'
set -u
pid_file="$1"
ready_file="$2"
motion_mode="$3"
timeout_s="$4"
max_vx="$5"
max_wz="$6"
motion_flag=""
if [ "${motion_mode}" = "enable_motion" ]; then
  motion_flag="--enable-motion"
fi
mkdir -p "$(dirname "${pid_file}")"

launcher_pid=""
started=0
if [ -f "${pid_file}" ]; then
  launcher_pid="$(cat "${pid_file}")"
fi
if ! [[ "${launcher_pid}" =~ ^[1-9][0-9]*$ ]] || ! kill -0 "${launcher_pid}" 2>/dev/null; then
  rm -f "${pid_file}" "${ready_file}"
  nohup setsid python3 /home/unitree/uni_control/start_go2_control.py ${motion_flag} --no-restart \
    --bridge-ready-file "${ready_file}" \
    --bridge-arg=--max-vx --bridge-arg="${max_vx}" \
    --bridge-arg=--max-wz --bridge-arg="${max_wz}" \
    > /home/unitree/uni_control/go2_control.log 2>&1 </dev/null &
  launcher_pid=$!
  printf '%s\n' "${launcher_pid}" >"${pid_file}"
  started=1
fi

for ((attempt=0; attempt < timeout_s * 4; attempt++)); do
  if ! kill -0 "${launcher_pid}" 2>/dev/null; then
    echo "Go2 control launcher exited before becoming ready" >&2
    tail -n 30 /home/unitree/uni_control/go2_control.log >&2 || true
    rm -f "${pid_file}" "${ready_file}"
    exit 1
  fi
  if [ -f "${ready_file}" ]; then
    bridge_pid="$(cat "${ready_file}")"
    if [[ "${bridge_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${bridge_pid}" 2>/dev/null; then
      bridge_args="$(tr '\0' ' ' <"/proc/${bridge_pid}/cmdline" 2>/dev/null || true)"
      if [[ "${bridge_args}" == *go2_control_bridge.py* ]]; then
        if (( started )); then
          echo "auxiliary_control_started launcher_pid=${launcher_pid} bridge_pid=${bridge_pid}"
        else
          echo "auxiliary_control_already_running launcher_pid=${launcher_pid} bridge_pid=${bridge_pid}"
        fi
        exit 0
      fi
    fi
  fi
  sleep .25
done

echo "Go2 control bridge did not become ready within ${timeout_s}s" >&2
tail -n 30 /home/unitree/uni_control/go2_control.log >&2 || true
kill -TERM "${launcher_pid}" 2>/dev/null || true
rm -f "${pid_file}" "${ready_file}"
exit 1
REMOTE
  )"; then
    echo "Go2 auxiliary control startup failed" >&2
    stop_motion_control
    return 1
  fi
  echo "${result}"
  # Keep a small local status record even when the remote bridge was reused.
  # The dashboard uses this together with the policy TCP connection, instead
  # of displaying a permanently hard-coded read-only badge.
  printf '%s\n' "${motion_mode}" >"${MOTION_STATUS_FILE}"
  if [[ "${result}" == *auxiliary_control_started* ]]; then : >"${MOTION_OWNED_FILE}"; else rm -f "${MOTION_OWNED_FILE}"; fi
  local connected=0 policy_port="${PHYSICAL_NAV_MOTION_POLICY_PORT:-12333}"
  for _ in {1..40}; do
    if ss -Htn state established "( sport = :${policy_port} )" | grep -q .; then
      connected=1
      break
    fi
    sleep .25
  done
  if (( ! connected )); then
    echo "Go2 bridge became ready but did not connect to policy port ${policy_port}" >&2
    stop_motion_control
    return 1
  fi
  if (( MOTION_ENABLED )); then
    echo "Go2 motion control is enabled"
  else
    echo "Go2 speech bridge is enabled (motion disabled)"
  fi
}

start_all() {
  if [[ "${PHYSICAL_NAV_START_QWEN_TUNNEL:-1}" == 1 ]]; then
    start_qwen_tunnel
  fi
  # Keep a generous grace period: the Go2 bridge may need to initialise the
  # D435i before the first frame reaches the watchdog.
  PHYSICAL_NAV_START_QWEN_TUNNEL=0 \
  PHYSICAL_NAV_START_ROSCORE="${PHYSICAL_NAV_START_ROSCORE:-1}" \
  PHYSICAL_NAV_START_YOLO_WORKER="${PHYSICAL_NAV_START_YOLO_WORKER:-1}" \
  PHYSICAL_NAV_ALGORITHM_PYTHON="${PHYSICAL_NAV_ALGORITHM_PYTHON:-/home/user/miniconda3/envs/mlspaces/bin/python3}" \
  PHYSICAL_NAV_YOLO_DEVICE="${PHYSICAL_NAV_YOLO_DEVICE:-cuda:0}" \
  PHYSICAL_NAV_YOLO_RATE="${PHYSICAL_NAV_YOLO_RATE:-10}" \
  PHYSICAL_NAV_WATCHDOG_STARTUP_GRACE_S="${PHYSICAL_NAV_WATCHDOG_STARTUP_GRACE_S:-180}" \
    bash "${SERVICE}" start
  wait_local_port "${PHYSICAL_NAV_WS_PORT:-12334}" 100
  wait_ros_command_subscriber
  start_go2_components
  publish_object_goal
  start_motion_control
  echo "dashboard: http://$(hostname -I | awk '{print $1}'):${PHYSICAL_NAV_WEB_PORT:-8765}/"
}

publish_object_goal() {
  [[ -n "${PHYSICAL_NAV_OBJECT_GOAL:-}" ]] || return 0
  local escaped
  escaped="${PHYSICAL_NAV_OBJECT_GOAL//\\/\\\\}"
  escaped="${escaped//\"/\\\"}"
  echo "publishing semantic object goal: ${PHYSICAL_NAV_OBJECT_GOAL}"
  rostopic pub -1 /semantic_decision/target std_msgs/String \
    "{data: '{\"enabled\":true,\"object_labels\":[\"${escaped}\"],\"target_name\":\"${escaped}\",\"mode\":\"object_goal\"}'}" \
    >/dev/null
}

stop_motion_control() {
  # Stop the policy source first; the Go2 bridge then zeros velocity on
  # WebSocket disconnect before its process is terminated.
  if [[ -f "${POLICY_PID_FILE}" ]]; then
    local pid="$(<"${POLICY_PID_FILE}")"
    if pid_alive "${pid}"; then kill -TERM "${pid}" 2>/dev/null || true; fi
    rm -f "${POLICY_PID_FILE}"
    echo "stopped policy control server (pid=${pid})"
  fi
  rm -f "${MOTION_STATUS_FILE}"
  # An explicit `restart enable_motion` is a request for a clean actuator
  # handshake, including a bridge that was reused from an earlier run.  A
  # plain read-only stop still leaves bridges we do not own untouched.
  if [[ ! -f "${MOTION_OWNED_FILE}" && "${MOTION_ENABLED}" != 1 ]]; then return 0; fi
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${GO2_SSH_TARGET}" \
    "if [ -f \"${MOTION_REMOTE_PID_FILE}\" ]; then \
       pid=\$(cat \"${MOTION_REMOTE_PID_FILE}\"); \
       kill -TERM \"\$pid\" 2>/dev/null || true; \
       rm -f \"${MOTION_REMOTE_PID_FILE}\" \"${MOTION_REMOTE_READY_FILE}\"; \
       echo stopped_go2_motion_control; \
     fi" >/dev/null 2>&1 || true
  rm -f "${MOTION_OWNED_FILE}"
}

stop_remote_owned() {
  local pid_file="$1" owned_file="$2" kind="$3" pid
  [[ -f "${owned_file}" && -f "${pid_file}" ]] || return 0
  pid="$(<"${pid_file}")"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${GO2_SSH_TARGET}" \
    "if kill -0 '${pid}' 2>/dev/null; then kill -TERM '${pid}' 2>/dev/null || true; fi" \
    >/dev/null 2>&1 || true
  echo "stopped Go2 ${kind} (pid=${pid})"
  rm -f "${owned_file}"
}

stop_all() {
  stop_motion_control
  bash "${SERVICE}" stop || true
  stop_remote_owned "${GO2_BRIDGE_PID_FILE}" "${GO2_BRIDGE_OWNED_FILE}" "sensor bridge"
  stop_remote_owned "${GO2_TUNNEL_PID_FILE}" "${GO2_TUNNEL_OWNED_FILE}" "SSH tunnel"
  if [[ -f "${QWEN_PID_FILE}" ]]; then
    local pid="$(<"${QWEN_PID_FILE}")"
    if pid_alive "${pid}"; then kill -TERM "${pid}" 2>/dev/null || true; fi
    rm -f "${QWEN_PID_FILE}"
    echo "stopped Qwen tunnel (pid=${pid})"
  fi
  rm -f "${GO2_TUNNEL_PID_FILE}" "${GO2_BRIDGE_PID_FILE}"
}

status_all() {
  bash "${SERVICE}" status || true
  if [[ -f "${QWEN_PID_FILE}" ]] && pid_alive "$(<"${QWEN_PID_FILE}")"; then
    echo "Qwen tunnel: running (pid=$(<"${QWEN_PID_FILE}"))"
  else
    echo "Qwen tunnel: stopped"
  fi
  if [[ -f "${POLICY_PID_FILE}" ]] && pid_alive "$(<"${POLICY_PID_FILE}")"; then
    echo "policy control server: running (pid=$(<"${POLICY_PID_FILE}"))"
  else
    echo "policy control server: stopped"
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${GO2_SSH_TARGET}" \
    "if [ -f \"${MOTION_REMOTE_PID_FILE}\" ] && [ -f \"${MOTION_REMOTE_READY_FILE}\" ] && \
        kill -0 \"\$(cat \"${MOTION_REMOTE_PID_FILE}\")\" 2>/dev/null && \
        kill -0 \"\$(cat \"${MOTION_REMOTE_READY_FILE}\")\" 2>/dev/null; then \
       pid=\$(cat \"${MOTION_REMOTE_PID_FILE}\"); \
       bridge_pid=\$(cat \"${MOTION_REMOTE_READY_FILE}\"); \
       args=\$(tr '\\0' ' ' < /proc/\${bridge_pid}/cmdline); \
       if printf '%s' \"\${args}\" | grep -q -- '--enable-motion'; then \
         echo 'Go2 auxiliary bridge: running (motion enabled)'; \
       else \
         echo 'Go2 auxiliary bridge: running (speech only, motion disabled)'; \
       fi; \
     else echo 'Go2 auxiliary bridge: stopped'; fi" \
    2>/dev/null || echo "Go2 auxiliary bridge: unavailable"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${GO2_SSH_TARGET}" \
    "ps -eo pid,etime,args | grep -E 'go2_readonly_sensor_bridge|ssh -N -T' | grep -v grep || true" \
    2>/dev/null || echo "Go2 SSH status: unavailable"
}

show_logs() {
  bash "${SERVICE}" logs
}

ACTION="${1:-}"
if [[ "${2:-}" == "enable_motion" ]]; then
  MOTION_ENABLED=1
elif [[ -n "${2:-}" ]]; then
  echo "unknown option: ${2}" >&2
  usage >&2
  exit 2
fi

if [[ -n "${3:-}" || -n "${4:-}" ]]; then
  if [[ "${2:-}" != "enable_motion" || -z "${3:-}" || -n "${4:-}" ]]; then
    echo "object goal requires: [enable_motion] obj_goal" >&2
    usage >&2
    exit 2
  fi
  PHYSICAL_NAV_OBJECT_GOAL="${3}"
  export PHYSICAL_NAV_OBJECT_GOAL
fi

case "${ACTION}" in
  start) start_all ;;
  # Start or repair only the policy-side WebSocket server and the Go2
  # speech/motion bridge. This leaves the running perception/navigation stack
  # untouched and is useful after either transport exits independently.
  start_control) start_motion_control ;;
  stop) stop_all ;;
  restart) stop_all; start_all ;;
  status) status_all ;;
  logs) show_logs ;;
  *) usage; exit 2 ;;
esac
