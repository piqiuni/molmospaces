#!/usr/bin/env bash
# Start a full native nav_to_obj benchmark with 25 long-lived workers. The
# lease manager keeps worker slots occupied as episodes finish, and starts
# them in waves to avoid a ROS launch storm.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

RUN_ROOT=${1:?"Usage: $0 <run-root>"}
BATCH_PYTHON=${BATCH_PYTHON:-python3}
BENCHMARK_DIR=${BENCHMARK_DIR:-${REPO_ROOT}/assets/benchmarks/molmospaces-bench-v2/procthor-10k/NavToObjDataGenConfig/NavToObjProcthor10kBench_20260112_json_benchmark}
WORKERS=${WORKERS:-25}
# Leave empty for the full benchmark. Set SCENE_COUNT only for a bounded
# distinct-house smoke subset.
SCENE_COUNT=${SCENE_COUNT:-}
BASE_ROS_MASTER_PORT=${BASE_ROS_MASTER_PORT:-19600}
START_WAVE_SIZE=${START_WAVE_SIZE:-5}
START_WAVE_INTERVAL_SECONDS=${START_WAVE_INTERVAL_SECONDS:-20}
TASK_HORIZON_STEPS=${TASK_HORIZON_STEPS:-500}
CUDA_VISIBLE_DEVICES_LIST=${CUDA_VISIBLE_DEVICES_LIST:-}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3,4,5,6,7}

if [[ -z "${CUDA_VISIBLE_DEVICES_LIST}" ]]; then
  IFS=, read -r -a gpu_devices <<<"${GPU_DEVICES}"
  if [[ "${#gpu_devices[@]}" -eq 0 ]]; then
    echo "GPU_DEVICES must name at least one GPU" >&2
    exit 2
  fi
  gpu_bindings=()
  for ((worker_index = 0; worker_index < WORKERS; worker_index++)); do
    gpu_bindings+=("${gpu_devices[worker_index % ${#gpu_devices[@]}]}")
  done
  CUDA_VISIBLE_DEVICES_LIST="$(IFS=,; echo "${gpu_bindings[*]}")"
fi

if [[ "$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES_LIST}")" -ne "${WORKERS}" ]]; then
  echo "CUDA_VISIBLE_DEVICES_LIST must contain exactly WORKERS (${WORKERS}) bindings" >&2
  exit 2
fi

worker_env_args=(
  --worker-env "ENABLE_RECORDING=${ENABLE_RECORDING:-false}"
  --worker-env "NATIVE_NAV_RECORD_VIDEOS=${NATIVE_NAV_RECORD_VIDEOS:-false}"
  --worker-env "SKIP_OFFLINE_VIDEO=${SKIP_OFFLINE_VIDEO:-true}"
  --worker-env "NATIVE_NAV_DYNAMIC_HORIZON=${NATIVE_NAV_DYNAMIC_HORIZON:-true}"
  --worker-env "NATIVE_NAV_DYNAMIC_HORIZON_BASE_STEPS=${NATIVE_NAV_DYNAMIC_HORIZON_BASE_STEPS:-240}"
  --worker-env "NATIVE_NAV_DYNAMIC_HORIZON_MIN_STEPS=${NATIVE_NAV_DYNAMIC_HORIZON_MIN_STEPS:-420}"
  --worker-env "NATIVE_NAV_DYNAMIC_HORIZON_STEPS_PER_METER=${NATIVE_NAV_DYNAMIC_HORIZON_STEPS_PER_METER:-45}"
)
for env_key in CONDA_ENV CONDA_SH HOME MLSPACES_PYTHON MLSPACES_ASSETS_DIR MLSPACES_CACHE_DIR MUJOCO_GL PYOPENGL_PLATFORM ROS_SETUP USER_CACHE_ROOT; do
  if [[ -n "${!env_key:-}" ]]; then
    worker_env_args+=(--worker-env "${env_key}=${!env_key}")
  fi
done

init_args=(
  --benchmark-dir "${BENCHMARK_DIR}"
  --run-root "${RUN_ROOT}"
  --base-ros-master-port "${BASE_ROS_MASTER_PORT}"
  --task-horizon-steps "${TASK_HORIZON_STEPS}"
  --filter-missing-scene-objects
  --episode-timeout-seconds "${EPISODE_TIMEOUT_SECONDS:-1200}"
  --max-attempts-per-episode 1
  "${worker_env_args[@]}"
)
if [[ -n "${SCENE_COUNT}" ]]; then
  init_args+=(--scene-count "${SCENE_COUNT}")
fi

"${BATCH_PYTHON}" "${SCRIPT_DIR}/nav_to_obj_batch_manager.py" init \
  "${init_args[@]}"

exec "${BATCH_PYTHON}" "${SCRIPT_DIR}/nav_to_obj_batch_manager.py" run \
  --run-root "${RUN_ROOT}" \
  --workers "${WORKERS}" \
  --worker-id-prefix "nav25" \
  --worker-start-wave-size "${START_WAVE_SIZE}" \
  --worker-start-interval-seconds "${START_WAVE_INTERVAL_SECONDS}" \
  --cuda-visible-devices-list "${CUDA_VISIBLE_DEVICES_LIST}"
