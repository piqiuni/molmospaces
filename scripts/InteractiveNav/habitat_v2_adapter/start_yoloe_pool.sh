#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/ldl/molmospaces-exp-setting
ADAPTER_DIR="$PROJECT_ROOT/scripts/InteractiveNav/habitat_v2_adapter"
CONDA_ROOT=/home/ldl/miniconda3
ENV_PREFIX=/home/ldl/conda_envs/ros-noetic
MODEL_PATH="${YOLO_MODEL_PATH:-${YOLOE_MODEL_PATH:-/home/ldl/.cache/habitat-detector-replay/weights/yolo26x-seg.pt}}"
MODEL_MODE="${YOLO_MODEL_MODE:-yolo_closed}"
PROMPT_LIST="${YOLO_PROMPT_LIST:-chair,bed,potted plant,toilet,tv,couch}"
ASSET_DIR="${YOLO_ASSET_DIR:-/home/ldl/.cache/habitat-detector-replay/weights}"
CONFIDENCE_THRESHOLD="${YOLO_CONFIDENCE_THRESHOLD:-0.35}"
CLASS_MAPPING="$PROJECT_ROOT/scripts/InteractiveNav/configs/habitat_objectnav_v2/objectnav_v2_yoloe_class_mapping.json"
SERVICE_PORT="${YOLOE_SERVICE_PORT:-${YOLOE_GATEWAY_PORT:-12219}}"
GPU_ID="${YOLOE_GPU_ID:-0}"
RUNTIME_ROOT="${YOLOE_RUNTIME_ROOT:-/home/ldl/tmp/habitat-yoloe/${SERVICE_PORT}}"

mkdir -p "$RUNTIME_ROOT/logs" /home/ldl/.cache/habitat-yoloe-pool /home/ldl/tmp/habitat-yoloe-pool
export TMPDIR=/home/ldl/tmp/habitat-yoloe-pool
export XDG_CACHE_HOME=/home/ldl/.cache/habitat-yoloe-pool
export TORCH_HOME=/home/ldl/.cache/habitat-yoloe-pool/torch
export HF_HOME=/home/ldl/.cache/habitat-yoloe-pool/huggingface

source "$CONDA_ROOT/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_PREFIX"
source "$PROJECT_ROOT/Interactive-Nav-SG-nav/devel/setup.bash"
set -u
export PYTHONPATH="$PROJECT_ROOT/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts:${PYTHONPATH:-}"
if [[ -d "$ASSET_DIR" ]]; then
  cd "$ASSET_DIR"
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "YOLOE checkpoint not found: $MODEL_PATH" >&2
  exit 2
fi

python - "$SERVICE_PORT" <<'PY'
import socket, sys
port = int(sys.argv[1])
ports = [port]
occupied = []
for port in ports:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            occupied.append(port)
if occupied:
    raise SystemExit(f"YOLO worker/gateway ports already in use: {occupied}")
PY

declare -a PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill -- "-$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$GPU_ID" setsid python "$ADAPTER_DIR/yoloe_worker_server.py" \
    --port "$SERVICE_PORT" \
    --replica-id 0 \
    --model-path "$MODEL_PATH" \
    --model-mode "$MODEL_MODE" \
    --prompt-list "$PROMPT_LIST" \
    --confidence-threshold "$CONFIDENCE_THRESHOLD" \
    --class-mapping "$CLASS_MAPPING" \
    --device cuda:0 \
    >"$RUNTIME_ROOT/logs/yolo.log" 2>&1 &
PIDS+=("$!")

for ((attempt=0; attempt<180; attempt++)); do
  if curl --silent --max-time 1 "http://127.0.0.1:${SERVICE_PORT}/health" | grep -q '"ready":true'; then
    ready=1
    break
  fi
  sleep 1
done
[[ "${ready:-0}" -eq 1 ]] || { echo "YOLOE service did not become ready" >&2; exit 3; }

echo "YOLOE single service ready: http://127.0.0.1:${SERVICE_PORT}/detect"
echo "model_mode=${MODEL_MODE} model_path=${MODEL_PATH}"
echo "confidence_threshold=${CONFIDENCE_THRESHOLD} prompts=${PROMPT_LIST}"
echo "replicas=1 gateway=disabled physical_gpu=${GPU_ID}"
wait "${PIDS[0]}"
