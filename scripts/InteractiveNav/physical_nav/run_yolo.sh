#!/usr/bin/env bash
# Shared foreground entry point for standalone and supervised perception.
set -eo pipefail
YOLO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YOLO_REPO="$(cd "${YOLO_ROOT}/../../.." && pwd)"
YOLO_GPU="${PHYSICAL_NAV_YOLO_GPU:-0}"
DRY_RUN=0
EXTRA_ARGS=()
while (( $# )); do
  case "$1" in
    --gpu)
      [[ $# -ge 2 ]] || { echo '--gpu requires a GPU index or cpu' >&2; exit 2; }
      YOLO_GPU="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h)
      echo 'Usage: bash run_yolo.sh [--gpu INDEX|cpu] [--dry-run] [YOLO bridge options]'
      echo 'Default GPU: PHYSICAL_NAV_YOLO_GPU (otherwise 0). --dry-run does not load ROS or CUDA.'
      exit 0 ;;
    --device|--device=*) echo 'Use --gpu instead of --device.' >&2; exit 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
if [[ -n "${PHYSICAL_NAV_YOLO_DEVICE:-}" || -n "${PHYSICAL_NAV_YOLO_CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'Remove PHYSICAL_NAV_YOLO_DEVICE / PHYSICAL_NAV_YOLO_CUDA_VISIBLE_DEVICES; use PHYSICAL_NAV_YOLO_GPU or --gpu instead.' >&2
  exit 2
fi
if [[ "${YOLO_GPU}" == cpu ]]; then
  YOLO_DEVICE=cpu
elif [[ "${YOLO_GPU}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  YOLO_DEVICE="cuda:${YOLO_GPU}"
else
  echo "Invalid GPU '${YOLO_GPU}': expected an index or cpu." >&2
  exit 2
fi
# One selection authority: --device uses the unrestricted CUDA device list.
# Do not let a parent shell's mask silently remap GPU indices.
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  echo 'YOLO: ignoring inherited CUDA_VISIBLE_DEVICES; selection is controlled by --gpu.' >&2
fi
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${PHYSICAL_NAV_YOLO_OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${PHYSICAL_NAV_YOLO_OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${PHYSICAL_NAV_YOLO_MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${PHYSICAL_NAV_YOLO_NUMEXPR_NUM_THREADS:-1}"
YOLO_PYTHON="${PHYSICAL_NAV_ALGORITHM_PYTHON:-/home/user/miniconda3/envs/mlspaces/bin/python3}"
if [[ ! -x "${YOLO_PYTHON}" && -z "${PHYSICAL_NAV_ALGORITHM_PYTHON:-}" ]]; then
  YOLO_PYTHON=python3
fi
YOLO_WEB_URL=""
if [[ "${PHYSICAL_NAV_START_WEB:-1}" == 1 ]]; then
  YOLO_WEB_URL="http://127.0.0.1:${PHYSICAL_NAV_WEB_PORT:-8765}"
fi
YOLO_COMMAND=("${YOLO_PYTHON}" "${YOLO_ROOT}/physical_yoloe_bridge.py"
  --web-url "${YOLO_WEB_URL}"
  --model-path "${PHYSICAL_NAV_MODEL_PATH:-${YOLO_REPO}/detection_models/yoloe/weights/yoloe-26l-seg-pf.pt}"
  --detector-config "${PHYSICAL_NAV_DETECTOR_CONFIG:-${YOLO_ROOT}/config/physical_nav.yaml}"
  --device "${YOLO_DEVICE}" --rate "${PHYSICAL_NAV_YOLO_RATE:-10}"
  --camera-x "${PHYSICAL_NAV_CAMERA_X:-0.03}" --camera-y "${PHYSICAL_NAV_CAMERA_Y:-0}"
  --camera-z "${PHYSICAL_NAV_CAMERA_Z:-0.98}" --camera-roll "${PHYSICAL_NAV_CAMERA_ROLL:-0}"
  --camera-pitch "${PHYSICAL_NAV_CAMERA_PITCH:-0.1396263}" --camera-yaw "${PHYSICAL_NAV_CAMERA_YAW:-0}"
  "${EXTRA_ARGS[@]}")
echo "YOLO: gpu=${YOLO_GPU}, device=${YOLO_DEVICE}, rate=${PHYSICAL_NAV_YOLO_RATE:-10} Hz" >&2
if (( DRY_RUN )); then
  printf '%q ' "${YOLO_COMMAND[@]}"; printf '\n'
  exit 0
fi
if [[ -f /opt/ros/noetic/setup.bash ]]; then
  source /opt/ros/noetic/setup.bash
fi
if [[ -f "${YOLO_REPO}/Interactive-Nav-SG-nav/devel/setup.bash" ]]; then
  source "${YOLO_REPO}/Interactive-Nav-SG-nav/devel/setup.bash"
fi
export PYTHONPATH="${YOLO_ROOT}:${YOLO_REPO}/Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts:${YOLO_REPO}/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts:${YOLO_REPO}/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts:${YOLO_REPO}/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts:${YOLO_ROOT}/ros_compat:${PYTHONPATH:-}"
export ROS_PACKAGE_PATH="${YOLO_REPO}/Interactive-Nav-SG-nav/src:${ROS_PACKAGE_PATH:-}"
exec "${YOLO_COMMAND[@]}"
