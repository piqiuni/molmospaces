#!/usr/bin/env bash
set -euo pipefail

# QwenService's private _serve hook. This never starts/stops the shared local service.
[[ "${1:-}" == _serve && $# == 4 ]] || {
  echo "Usage: $0 _serve GPU_ID PORT LABEL" >&2
  exit 2
}
deploy_root=/home/ldl/qwen36-fp8
cache_root="${QWEN36_RUNTIME_DIR:?QWEN36_RUNTIME_DIR must be task-owned}/${4}"
[[ "${cache_root}" == /home/ldl/* ]] || exit 2
mkdir -p "${cache_root}"/{tmp,xdg,torchinductor,triton,vllm}
export CUDA_VISIBLE_DEVICES="$2"
# ZeroMQ appends a UUID to TMPDIR; Unix-domain socket paths must fit 107 bytes.
# Keep only temporary IPC files here; compilation caches stay under the run.
mkdir -p /home/ldl/tmp
TMPDIR=$(mktemp -d /home/ldl/tmp/m2q-XXXXXXXX)
export TMPDIR
export XDG_CACHE_HOME="${cache_root}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export TRITON_CACHE_DIR="${cache_root}/triton"
export VLLM_CACHE_ROOT="${cache_root}/vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=1

# Match the existing local server, except remote TP=1 vs local TP=2.
# In particular, do not silently enable MTP speculative decoding on only one lane.
exec "${deploy_root}/venv/bin/vllm" serve \
  "${deploy_root}/model/Qwen3.6-35B-A3B-FP8" \
  --served-model-name qwen3.6-35b-a3b-fp8 \
  --host 127.0.0.1 --port "$3" \
  --tensor-parallel-size 1 --data-parallel-size 1 \
  --max-model-len 16384 --max-num-seqs 32 \
  --gpu-memory-utilization 0.60 --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image": 1, "video": 0}'
