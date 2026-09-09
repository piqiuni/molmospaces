#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/ldl/molmospaces-exp-setting
ADAPTER_DIR="$PROJECT_ROOT/scripts/InteractiveNav/habitat_v2_adapter"
CONDA_ROOT=/home/ldl/miniconda3
ENV_PREFIX=/home/ldl/conda_envs/ros-noetic
MODEL_PATH="$PROJECT_ROOT/detection_models/yoloe/weights/yoloe-26x-seg-pf.pt"
CLASS_MAPPING="$PROJECT_ROOT/scripts/InteractiveNav/configs/habitat_objectnav_v2/objectnav_v2_yoloe_class_mapping.json"
GATEWAY_PORT="${YOLOE_GATEWAY_PORT:-12219}"
WORKER_BASE_PORT="${YOLOE_WORKER_BASE_PORT:-12220}"
REPLICA_COUNT="${YOLOE_REPLICA_COUNT:-5}"
GPU_ID="${YOLOE_GPU_ID:-0}"
RUNTIME_ROOT="${YOLOE_RUNTIME_ROOT:-/home/ldl/tmp/habitat-yoloe-pool/gateway-${GATEWAY_PORT}}"

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

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "YOLOE checkpoint not found: $MODEL_PATH" >&2
  exit 2
fi

declare -a PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill -- "-$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

declare -a GATEWAY_ARGS=()
for ((replica=0; replica<REPLICA_COUNT; replica++)); do
  port=$((WORKER_BASE_PORT + replica))
  CUDA_VISIBLE_DEVICES="$GPU_ID" setsid python "$ADAPTER_DIR/yoloe_worker_server.py" \
    --port "$port" \
    --replica-id "$replica" \
    --model-path "$MODEL_PATH" \
    --class-mapping "$CLASS_MAPPING" \
    --device cuda:0 \
    >"$RUNTIME_ROOT/logs/replica-${replica}.log" 2>&1 &
  PIDS+=("$!")
  GATEWAY_ARGS+=(--worker "http://127.0.0.1:${port}")
done

for ((attempt=0; attempt<180; attempt++)); do
  ready=0
  for ((replica=0; replica<REPLICA_COUNT; replica++)); do
    port=$((WORKER_BASE_PORT + replica))
    if curl --silent --max-time 1 "http://127.0.0.1:${port}/health" | grep -q '"ready":true'; then
      ready=$((ready + 1))
    fi
  done
  [[ "$ready" -eq "$REPLICA_COUNT" ]] && break
  sleep 1
done
[[ "${ready:-0}" -eq "$REPLICA_COUNT" ]] || { echo "YOLOE replicas did not become ready" >&2; exit 3; }

setsid python "$ADAPTER_DIR/yoloe_gateway.py" \
  --port "$GATEWAY_PORT" \
  "${GATEWAY_ARGS[@]}" \
  >"$RUNTIME_ROOT/logs/gateway.log" 2>&1 &
PIDS+=("$!")

for _ in $(seq 1 30); do
  health="$(curl --silent --max-time 2 "http://127.0.0.1:${GATEWAY_PORT}/health" || true)"
  [[ "$health" == *'"ready":true'* ]] && break
  sleep 1
done
[[ "${health:-}" == *'"ready":true'* ]] || { echo "YOLOE gateway did not become ready" >&2; exit 4; }

echo "YOLOE pool ready: http://127.0.0.1:${GATEWAY_PORT}/detect"
echo "replicas=${REPLICA_COUNT} ingress_concurrency_limit=none physical_gpu=${GPU_ID}"
wait "${PIDS[-1]}"
