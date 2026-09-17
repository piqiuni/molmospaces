#!/usr/bin/env bash
set -euo pipefail

# One ml.pni2.7xlarge instance: two Qwen replicas plus the configured ROS/MuJoCo workers.
REPO_ROOT="${REPO_ROOT:-/home/ldl/molmospaces-exp-setting}"
TASK_ID="${MLP_TASK_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/home/ldl/outputs/interactive-nav/custom-task-${TASK_ID}}"
EVALUATION_OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-${RUN_ROOT}/evaluation}"
BENCHMARK_LAUNCHER_CONFIG="${BENCHMARK_LAUNCHER_CONFIG:-scripts/InteractiveNav/configs/evaluation/benchmark_batch_custom_task_10w10s100.json}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-10}"
SHORT_TASK_ID="${TASK_ID##*-}"
STATE_DIR="${RUN_ROOT}/task-state"
QWEN_ROOT=/home/ldl/qwen36-fp8
QWEN_SERVICE_MODE="${QWEN_SERVICE_MODE:-legacy}"
QWEN_MANAGE_SCRIPT="${QWEN_MANAGE_SCRIPT:-${QWEN_ROOT}/manage_qwen36_mtp3.sh}"
PYTHON310_INCLUDE=/home/ldl/.cache/python3.10-dev/usr/include/python3.10
PYTHON310_MULTIARCH_INCLUDE=/home/ldl/.cache/python3.10-dev/usr/include
EGL_RUNTIME_LIB=/home/ldl/.cache/egl-runtime/usr/lib/x86_64-linux-gnu
EGL_VENDOR_CONFIG="${REPO_ROOT}/scripts/InteractiveNav/configs/custom_task/10_nvidia.json"

mkdir -p "${STATE_DIR}" "${RUN_ROOT}/cache" "${RUN_ROOT}/ros" \
  "/home/ldl/tmp/inav-${SHORT_TASK_ID}-q0" \
  "/home/ldl/tmp/inav-${SHORT_TASK_ID}-q1" \
  "/home/ldl/tmp/inav-${SHORT_TASK_ID}-eval"
export HF_HOME=/home/ldl/.cache/huggingface
export TORCH_HOME=/home/ldl/.cache/torch
export TRITON_CACHE_DIR=/home/ldl/.cache/triton-qwen-py310
export ROS_HOME="${RUN_ROOT}/ros"
export MLSPACES_CACHE_DIR=/home/ldl/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/home/ldl/molmospaces/assets
export NLTK_DATA=/home/ldl/nltk_data

[[ -f "${PYTHON310_INCLUDE}/Python.h" ]] || {
  echo "Missing ${PYTHON310_INCLUDE}/Python.h" >&2
  exit 2
}
export CPATH="${PYTHON310_INCLUDE}:${PYTHON310_MULTIARCH_INCLUDE}${CPATH:+:${CPATH}}"
export C_INCLUDE_PATH="${PYTHON310_INCLUDE}:${PYTHON310_MULTIARCH_INCLUDE}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"

[[ -f "${EGL_RUNTIME_LIB}/libEGL.so.1" && -f "${EGL_RUNTIME_LIB}/libGLdispatch.so.0" ]] || {
  echo "Missing vendored EGL/GLVND runtime under ${EGL_RUNTIME_LIB}" >&2
  exit 2
}
export LD_LIBRARY_PATH="${EGL_RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${EGL_VENDOR_CONFIG}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

MUJOCO_EGL_DEVICE_ID=0 /home/ldl/conda_envs/mlspaces/bin/python - <<'PY'
import mujoco

context = mujoco.GLContext(1, 1)
context.make_current()
context.free()
print("MuJoCo EGL preflight passed")
PY

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( gpu_count < 2 )); then
  echo "Expected a two-GPU instance, found ${gpu_count} visible GPU(s)" >&2
  exit 2
fi

pids=()
managed_qwen_started=false
cleanup() {
  trap - EXIT INT TERM
  if [[ "${managed_qwen_started}" == true ]]; then
    "${QWEN_MANAGE_SCRIPT}" stop >/dev/null 2>&1 || true
  fi
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

start_qwen() {
  local gpu="$1" port="$2" name="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  TMPDIR="/home/ldl/tmp/inav-${SHORT_TASK_ID}-q${gpu}" \
  XDG_CACHE_HOME="${RUN_ROOT}/cache/qwen-${gpu}" \
  "${QWEN_ROOT}/venv/bin/vllm" serve \
    "${QWEN_ROOT}/model/Qwen3.6-35B-A3B-FP8" \
    --served-model-name qwen3.6-35b-a3b-fp8 \
    --host 127.0.0.1 --port "${port}" --tensor-parallel-size 1 \
    --max-model-len 10240 --max-num-seqs 24 --gpu-memory-utilization 0.5 \
    --disable-custom-all-reduce --no-enable-prefix-caching \
    >"${RUN_ROOT}/${name}.log" 2>&1 &
  pids+=("$!")
}

if [[ "${QWEN_SERVICE_MODE}" == managed ]]; then
  export QWEN36_RUNTIME_DIR="/home/ldl/tmp/inav-${SHORT_TASK_ID}-qwen-mtp3"
  export QWEN36_LOG_DIR="${RUN_ROOT}/qwen-service"
  export QWEN36_MTP_TOKENS="${QWEN36_MTP_TOKENS:-3}"
  export QWEN36_GPU_MEMORY_UTILIZATION="${QWEN36_GPU_MEMORY_UTILIZATION:-0.6}"
  managed_qwen_started=true
  "${QWEN_MANAGE_SCRIPT}" start
  "${QWEN_MANAGE_SCRIPT}" status | tee "${STATE_DIR}/qwen.status"
else
  start_qwen 0 8000 qwen-gpu0
  start_qwen 1 8001 qwen-gpu1

  for port in 8000 8001; do
    ready=false
    for _ in $(seq 1 240); do
      if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
        ready=true
        break
      fi
      for pid in "${pids[@]}"; do
        kill -0 "${pid}" 2>/dev/null || {
          echo "A Qwen process exited before readiness; inspect ${RUN_ROOT}/qwen-gpu*.log" >&2
          exit 1
        }
      done
      sleep 5
    done
    [[ "${ready}" == true ]] || { echo "Qwen port ${port} readiness timed out" >&2; exit 1; }
  done

  "${QWEN_ROOT}/venv/bin/python" "${QWEN_ROOT}/bench/lb.py" \
    --listen-port 8010 --backends 8000,8001 >"${RUN_ROOT}/qwen-lb.log" 2>&1 &
  pids+=("$!")
  for _ in $(seq 1 60); do
    curl -fsS http://127.0.0.1:8010/v1/models >/dev/null 2>&1 && break
    kill -0 "${pids[2]}" 2>/dev/null || { echo "Qwen load balancer exited" >&2; exit 1; }
    sleep 2
  done
  curl -fsS http://127.0.0.1:8010/v1/models >/dev/null 2>&1 || {
    echo "Qwen load balancer readiness timed out" >&2
    exit 1
  }
fi
touch "${STATE_DIR}/qwen.ready"

cd "${REPO_ROOT}"
export TMPDIR="/home/ldl/tmp/inav-${SHORT_TASK_ID}-eval"
export XDG_CACHE_HOME="${RUN_ROOT}/cache/evaluation"
export INTERACTIVE_NAV_IMPORT_DIAGNOSTICS=1
set +e
/home/ldl/conda_envs/mlspaces/bin/python -u \
  scripts/InteractiveNav/run_benchmark_eval.py \
  --config "${BENCHMARK_LAUNCHER_CONFIG}" \
  --output-dir "${EVALUATION_OUTPUT_DIR}"
eval_exit_code=$?
set -e
if (( eval_exit_code == 0 )); then
  set +e
  /home/ldl/conda_envs/mlspaces/bin/python - \
    "${EVALUATION_OUTPUT_DIR}/summary.json" "${EXPECTED_EPISODES}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected = int(sys.argv[2])
summary = json.loads(summary_path.read_text())
episodes = summary.get("episodes", [])
completed = [row for row in episodes if row.get("completed") is True]
if len(episodes) != expected or len(completed) != expected:
    raise SystemExit(
        f"Expected {expected} completed episodes, "
        f"got reported={len(episodes)} completed={len(completed)}"
    )
PY
  eval_exit_code=$?
  set -e
fi
printf '%s\n' "${eval_exit_code}" >"${STATE_DIR}/evaluation.exit_code"
touch "${STATE_DIR}/evaluation.done"
exit "${eval_exit_code}"
