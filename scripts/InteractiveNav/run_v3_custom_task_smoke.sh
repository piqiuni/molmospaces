#!/usr/bin/env bash
set -euo pipefail

# Batch-owned Qwen replica(s) plus the configured ROS/MuJoCo workers.
REPO_ROOT="${REPO_ROOT:-/home/ldl/molmospaces-exp-setting}"
TASK_ID="${MLP_TASK_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/home/ldl/outputs/interactive-nav/custom-task-${TASK_ID}}"
EVALUATION_OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-${RUN_ROOT}/evaluation}"
BENCHMARK_LAUNCHER_CONFIG="${BENCHMARK_LAUNCHER_CONFIG:-scripts/InteractiveNav/configs/evaluation/benchmark_batch_custom_task_10w10s100.json}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-10}"
SHORT_TASK_ID="${TASK_ID##*-}"
STATE_DIR="${RUN_ROOT}/task-state"
QWEN_ROOT="${QWEN_ROOT:-/home/ldl/qwen36-fp8}"
export QWEN_ROOT
PYTHON310_INCLUDE=/home/ldl/.cache/python3.10-dev/usr/include/python3.10
PYTHON310_MULTIARCH_INCLUDE=/home/ldl/.cache/python3.10-dev/usr/include
EGL_RUNTIME_LIB=/home/ldl/.cache/egl-runtime/usr/lib/x86_64-linux-gnu
EGL_VENDOR_CONFIG="${REPO_ROOT}/scripts/InteractiveNav/configs/custom_task/10_nvidia.json"

cd "${REPO_ROOT}"
if [[ "${CUSTOM_TASK_DRY_RUN:-false}" == true ]]; then
  exec /home/ldl/conda_envs/mlspaces/bin/python -u \
    scripts/InteractiveNav/run_benchmark_eval.py \
    --config "${BENCHMARK_LAUNCHER_CONFIG}" --expected-episodes "${EXPECTED_EPISODES}" \
    --output-dir "${EVALUATION_OUTPUT_DIR}" --start-qwen --dry-run
fi

mkdir -p "${STATE_DIR}" "${RUN_ROOT}/cache" "${RUN_ROOT}/ros" \
  "/home/ldl/tmp/inav-${SHORT_TASK_ID}-q0" \
  "/home/ldl/tmp/inav-${SHORT_TASK_ID}-q1" \
  "/home/ldl/tmp/inav-${SHORT_TASK_ID}-eval"
export TMPDIR="/home/ldl/tmp/inav-${SHORT_TASK_ID}-eval"
export XDG_CACHE_HOME="${RUN_ROOT}/cache/evaluation"
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

# Platform environment setup above; evaluation and service ownership live in Python.
cd "${REPO_ROOT}"
if [[ -n "${M2_EXPERIMENT_MANIFEST:-}" ]]; then
  exec /home/ldl/conda_envs/mlspaces/bin/python -u \
    scripts/InteractiveNav/run_benchmark_eval.py \
    --experiment-manifest "${M2_EXPERIMENT_MANIFEST}" \
    --experiment-lane "${M2_EXPERIMENT_LANE:-2}" --start-qwen
fi
exec /home/ldl/conda_envs/mlspaces/bin/python -u \
  scripts/InteractiveNav/run_benchmark_eval.py \
  --config "${BENCHMARK_LAUNCHER_CONFIG}" \
  --expected-episodes "${EXPECTED_EPISODES}" \
  --output-dir "${EVALUATION_OUTPUT_DIR}" --start-qwen
