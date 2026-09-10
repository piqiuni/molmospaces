#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_SCRIPT="${ROOT_DIR}/start_physical_nav.sh"
RUNTIME_DIR="${PHYSICAL_NAV_RUNTIME_DIR:-/tmp/molmospaces-physical-nav-${UID}}"
LOG_DIR="${PHYSICAL_NAV_LOG_DIR:-${RUNTIME_DIR}/logs}"
PID_FILE="${RUNTIME_DIR}/supervisor.pid"
GATEWAY_FINGERPRINT_FILE="${PHYSICAL_NAV_GATEWAY_FINGERPRINT_FILE:-${RUNTIME_DIR}/gateway.fingerprint}"
SERVICE_LOG="${LOG_DIR}/service.log"

usage() {
  cat <<EOF
Usage: $(basename "$0") {start|stop|restart|status|logs|web-stop}

The service runs start_physical_nav.sh in a detached session. Configuration is
passed through PHYSICAL_NAV_* environment variables at start time.
EOF
}

read_pid() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(<"${PID_FILE}")"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "${pid}"
}

is_expected_process() {
  local pid="$1" cmdline
  kill -0 "${pid}" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  [[ "${cmdline}" == *"${START_SCRIPT}"* ]]
}

has_isolated_process_group() {
  local pid="$1" pgid
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  [[ "${pgid}" == "${pid}" ]]
}

running_pid() {
  local pid
  pid="$(read_pid)" || return 1
  is_expected_process "${pid}" || return 1
  printf '%s\n' "${pid}"
}

is_pipeline_residual() {
  local cmdline="$1"
  case "${cmdline}" in
    *"start_physical_nav.sh"*|\
    *"${ROOT_DIR}/physical_yoloe_bridge.py"*|\
    *"${ROOT_DIR}/physical_nav_watchdog.py"*|\
    *"physical_nav_readonly.launch"*|\
    *"physical_sensor_ros_bridge.py"*|\
    *"__name:=semantic_mapping_py"*|\
    *"__name:=interaction_attribute_inference"*|\
    *"__name:=slam_gmapping"*|\
    *"__name:=voronoi_mapping"*|\
    *"__name:=physical_sensor_ros_bridge"*|\
    *"__name:=physical_ros_gateway"*|\
    *"__name:=physical_nav_consistency"*|\
    *"__name:=physical_velocity_safety"*|\
    *"__name:=physical_interaction_policy"*|\
    *"__name:=relay_node"*|\
    *"__name:=move_base"*|\
    *"__name:=explore_py"*|\
    *"__name:=semantic_candidate_node"*|\
    *"__name:=semantic_rule_decision_node"*|\
    *"__name:=semantic_behavior_executor"*) return 0 ;;
    *) return 1 ;;
  esac
}

pipeline_residual_pids() {
  local pid cmdline
  for pid_dir in /proc/[0-9]*; do
    pid="${pid_dir##*/}"
    [[ "${pid}" != "$$" && -r "${pid_dir}/cmdline" ]] || continue
    cmdline="$(tr '\0' ' ' <"${pid_dir}/cmdline" 2>/dev/null || true)"
    [[ -n "${cmdline}" ]] || continue
    if is_pipeline_residual "${cmdline}"; then
      printf '%s\n' "${pid}"
    fi
  done
}

cleanup_pipeline_residuals() {
  local -a pids=()
  local pid
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] && pids+=("${pid}")
  done < <(pipeline_residual_pids)
  (( ${#pids[@]} > 0 )) || return 0

  echo "cleaning residual physical-navigation processes: ${pids[*]}"
  for pid in "${pids[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for _ in {1..50}; do
    local alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then alive=1; break; fi
    done
    (( alive == 0 )) && break
    sleep .1
  done
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      echo "forcing residual process to stop (pid=${pid})" >&2
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
}

start_service() {
  local pid
  if pid="$(running_pid)"; then
    echo "physical navigation is already running (pid=${pid})"
    return 0
  fi
  command -v setsid >/dev/null 2>&1 || {
    echo "setsid is required to detach the physical-navigation service" >&2
    return 1
  }
  mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"
  # A supervisor started directly (or whose pid file was overwritten after a
  # watchdog failure) is still a live navigation stack. Remove those workers
  # before launching another copy; otherwise duplicate ROS node names make
  # roslaunch instances repeatedly evict and respawn each other's controllers.
  cleanup_pipeline_residuals
  rm -f "${PID_FILE}"
  printf '\n%s starting physical navigation\n' "$(date -Is)" >>"${SERVICE_LOG}"
  PHYSICAL_NAV_RUNTIME_DIR="${RUNTIME_DIR}" \
  PHYSICAL_NAV_LOG_DIR="${LOG_DIR}" \
    nohup setsid bash "${START_SCRIPT}" >>"${SERVICE_LOG}" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "${pid}" >"${PID_FILE}"
  sleep .5
  if ! is_expected_process "${pid}" || ! has_isolated_process_group "${pid}"; then
    echo "physical navigation failed during startup; inspect ${SERVICE_LOG}" >&2
    tail -n 30 "${SERVICE_LOG}" >&2 || true
    rm -f "${PID_FILE}"
    return 1
  fi
  echo "physical navigation started (pid=${pid})"
  echo "dashboard: http://127.0.0.1:${PHYSICAL_NAV_WEB_PORT:-8765}/"
  echo "logs: ${LOG_DIR}"
}

stop_service() {
  local pid
  if ! pid="$(running_pid)"; then
    echo "physical navigation is not running"
    cleanup_pipeline_residuals
    rm -f "${PID_FILE}"
    return 0
  fi
  echo "stopping physical navigation (pid=${pid})"
  # The supervisor is a setsid session leader, so the negative PID addresses
  # only this service's process group. The supervisor then performs ordered
  # child cleanup and records the exit reason.
  if has_isolated_process_group "${pid}"; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
  else
    kill -TERM "${pid}" 2>/dev/null || true
  fi
  for _ in {1..150}; do
    if ! is_expected_process "${pid}"; then
      rm -f "${PID_FILE}"
      cleanup_pipeline_residuals
      echo "physical navigation stopped"
      return 0
    fi
    sleep .1
  done
  echo "supervisor did not exit within 15 seconds; forcing its process group to stop" >&2
  if has_isolated_process_group "${pid}"; then
    kill -KILL -- "-${pid}" 2>/dev/null || true
  else
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
  cleanup_pipeline_residuals
}

status_service() {
  local pid
  local gateway_pid=""
  if [[ -f "${RUNTIME_DIR}/gateway.pid" ]]; then gateway_pid="$(<"${RUNTIME_DIR}/gateway.pid")"; fi
  if [[ "${gateway_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${gateway_pid}" 2>/dev/null &&
     [[ "$(tr '\0' ' ' <"/proc/${gateway_pid}/cmdline" 2>/dev/null || true)" == *physical_six_panel_server.py* ]]; then
    echo "web gateway is running (pid=${gateway_pid}, port=${PHYSICAL_NAV_WEB_PORT:-8765})"
  else
    echo "web gateway is stopped"
  fi
  if ! pid="$(running_pid)"; then
    echo "physical navigation is stopped"
    [[ -f "${RUNTIME_DIR}/last_exit.tsv" ]] && {
      printf 'last exit: '
      tr '\t' ' ' <"${RUNTIME_DIR}/last_exit.tsv"
    }
    return 1
  fi
  echo "physical navigation is running (pid=${pid})"
  if command -v curl >/dev/null 2>&1; then
    curl --silent --show-error --fail --max-time 1 \
      "http://127.0.0.1:${PHYSICAL_NAV_WEB_PORT:-8765}/api/health" || true
    echo
  fi
}

stop_web_gateway() {
  local pid=""
  [[ -f "${RUNTIME_DIR}/gateway.pid" ]] && pid="$(<"${RUNTIME_DIR}/gateway.pid")"
  if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null &&
     [[ "$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)" == *physical_six_panel_server.py* ]]; then
    kill -TERM "${pid}" 2>/dev/null || true
    rm -f "${RUNTIME_DIR}/gateway.pid" "${GATEWAY_FINGERPRINT_FILE}"
    echo "web gateway stopped (pid=${pid})"
  else
    rm -f "${RUNTIME_DIR}/gateway.pid" "${GATEWAY_FINGERPRINT_FILE}"
    echo "web gateway is not running"
  fi
}

show_logs() {
  mkdir -p "${LOG_DIR}"
  local files=()
  for name in service supervisor gateway yoloe watchdog qwen_tunnel roscore; do
    [[ -f "${LOG_DIR}/${name}.log" ]] && files+=("${LOG_DIR}/${name}.log")
  done
  if (( ${#files[@]} == 0 )); then
    echo "no physical-navigation logs under ${LOG_DIR}"
    return 0
  fi
  tail -n 80 -F "${files[@]}"
}

case "${1:-}" in
  start) start_service ;;
  stop) stop_service ;;
  restart) stop_service; start_service ;;
  status) status_service ;;
  logs) show_logs ;;
  web-stop) stop_web_gateway ;;
  *) usage; exit 2 ;;
esac
