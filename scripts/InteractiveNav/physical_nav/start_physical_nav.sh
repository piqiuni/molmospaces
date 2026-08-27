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

WEB_HOST="${PHYSICAL_NAV_WEB_HOST:-0.0.0.0}"
WEB_PORT="${PHYSICAL_NAV_WEB_PORT:-8765}"
WS_HOST="${PHYSICAL_NAV_WS_HOST:-0.0.0.0}"
WS_PORT="${PHYSICAL_NAV_WS_PORT:-12334}"
QWEN_URL="${PHYSICAL_NAV_QWEN_URL:-http://127.0.0.1:18080/v1}"
QWEN_MODEL="${PHYSICAL_NAV_QWEN_MODEL:-qwen3.6-35b-a3b-fp8}"
QWEN_AUTO_INTERVAL="${PHYSICAL_NAV_QWEN_AUTO_INTERVAL:-0}"
export SEMANTIC_MODEL_MODE="${SEMANTIC_MODEL_MODE:-http}"
export SEMANTIC_MODEL_ENDPOINT="${SEMANTIC_MODEL_ENDPOINT:-http://127.0.0.1:18080/v1}"
export SEMANTIC_MODEL_NAME="${SEMANTIC_MODEL_NAME:-${QWEN_MODEL}}"
export SEMANTIC_MODEL_PROTOCOL="${SEMANTIC_MODEL_PROTOCOL:-openai_chat}"
export SEMANTIC_MODEL_METRICS_PATH="${SEMANTIC_MODEL_METRICS_PATH:-/tmp/physical_nav_mllm.jsonl}"
export SEMANTIC_MODEL_TRACE_URL="${SEMANTIC_MODEL_TRACE_URL:-http://127.0.0.1:${WEB_PORT}/api/mllm-event}"
# Measured Go2 standing-pose calibration: camera is about 3 cm forward of
# the base centre (38 cm from a 70 cm rear-to-front body) and 0.75 m above
# the base. The base-to-ground offset is therefore about 0.43 m (1.18 m
# camera height), which is not part of this base-frame extrinsic.
CAMERA_X="${PHYSICAL_NAV_CAMERA_X:-0.03}"; CAMERA_Y="${PHYSICAL_NAV_CAMERA_Y:-0}"; CAMERA_Z="${PHYSICAL_NAV_CAMERA_Z:-0.75}"; CAMERA_ROLL="${PHYSICAL_NAV_CAMERA_ROLL:-0}"; CAMERA_PITCH="${PHYSICAL_NAV_CAMERA_PITCH:-0}"; CAMERA_YAW="${PHYSICAL_NAV_CAMERA_YAW:-0}"
QWEN_TUNNEL_PID=""
ROSCORE_PID=""
YOLO_PID=""

if command -v roscore >/dev/null 2>&1 && [[ "${PHYSICAL_NAV_START_ROSCORE:-1}" == "1" ]]; then
  if ! (exec 3<>/dev/tcp/127.0.0.1/11311) 2>/dev/null; then
    roscore >/tmp/physical_nav_roscore.log 2>&1 & ROSCORE_PID=$!
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
    --local-port "${PHYSICAL_NAV_QWEN_LOCAL_PORT:-18080}" --remote-port "${PHYSICAL_NAV_QWEN_REMOTE_PORT:-8000}" &
  QWEN_TUNNEL_PID=$!
fi

if [[ "${PHYSICAL_NAV_START_YOLO_WORKER:-1}" == "1" ]]; then
  "${PHYSICAL_NAV_ALGORITHM_PYTHON:-python3}" "${ROOT_DIR}/physical_yoloe_bridge.py" \
    --web-url "http://127.0.0.1:${WEB_PORT}" \
    --model-path "${PHYSICAL_NAV_MODEL_PATH:-/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26l-seg-pf.pt}" \
    --device "${PHYSICAL_NAV_YOLO_DEVICE:-cuda:0}" --rate "${PHYSICAL_NAV_YOLO_RATE:-10}" \
    --camera-x "${CAMERA_X}" --camera-y "${CAMERA_Y}" --camera-z "${CAMERA_Z}" \
    --camera-roll "${CAMERA_ROLL}" --camera-pitch "${CAMERA_PITCH}" --camera-yaw "${CAMERA_YAW}" &
  YOLO_PID=$!
fi

python3 "${ROOT_DIR}/physical_six_panel_server.py" \
  --ws-host "${WS_HOST}" --ws-port "${WS_PORT}" \
  --http-host "${WEB_HOST}" --http-port "${WEB_PORT}" \
  --qwen-url "${QWEN_URL}" --qwen-model "${QWEN_MODEL}" --qwen-auto-interval "${QWEN_AUTO_INTERVAL}" \
  --camera-x "${CAMERA_X}" --camera-y "${CAMERA_Y}" --camera-z "${CAMERA_Z}" --camera-roll "${CAMERA_ROLL}" --camera-pitch "${CAMERA_PITCH}" --camera-yaw "${CAMERA_YAW}" &
GATEWAY_PID=$!
trap 'kill "${GATEWAY_PID}" 2>/dev/null || true; [[ -z "${QWEN_TUNNEL_PID}" ]] || kill "${QWEN_TUNNEL_PID}" 2>/dev/null || true; [[ -z "${ROSCORE_PID}" ]] || kill "${ROSCORE_PID}" 2>/dev/null || true; [[ -z "${YOLO_PID}" ]] || kill "${YOLO_PID}" 2>/dev/null || true' EXIT INT TERM

echo "Physical gateway: http://$(hostname -I | awk '{print $1}'):${WEB_PORT}/"
if [[ "${PHYSICAL_NAV_SKIP_ROS:-0}" == "1" ]]; then
  echo "ROS launch disabled; gateway remains available for protocol/web tests."
  wait "${GATEWAY_PID}"
elif command -v roscore >/dev/null 2>&1 && command -v roslaunch >/dev/null 2>&1; then
  roslaunch "${ROOT_DIR}/launch/physical_nav_readonly.launch" \
    config_file:="${ROOT_DIR}/config/physical_nav.yaml" \
    model_path:="${PHYSICAL_NAV_MODEL_PATH:-/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26l-seg-pf.pt}" \
    camera_x:="${CAMERA_X}" camera_y:="${CAMERA_Y}" camera_z:="${CAMERA_Z}" \
    camera_roll:="${CAMERA_ROLL}" camera_pitch:="${CAMERA_PITCH}" camera_yaw:="${CAMERA_YAW}" \
    ros_python:="${PHYSICAL_NAV_ROS_PYTHON:-/home/user/miniconda3/envs/mlspaces/bin/python3}"
else
  echo "ROS1 tools not found; gateway remains available for protocol/web tests."
  wait "${GATEWAY_PID}"
fi
