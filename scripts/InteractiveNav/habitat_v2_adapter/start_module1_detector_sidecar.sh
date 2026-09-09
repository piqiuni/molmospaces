#!/usr/bin/env bash
# Start the detector-only Habitat bridge.  It contains no Module-2, Module-3,
# ROS executor, action, or interaction endpoint.  YOLOv7 itself must already be
# available on the local loopback endpoint; this script downloads nothing.
set -euo pipefail

adapter_dir="/home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/habitat_v2_adapter"
python_bin="/home/ldl/conda_envs/habitat-challenge-2023/bin/python"
task_tmp="/home/ldl/tmp/habitat-objectnav-module1-sidecar"
task_cache="/home/ldl/.cache/habitat-objectnav-module1-sidecar"

mkdir -p "${task_tmp}" "${task_cache}"
export TMPDIR="${task_tmp}"
export XDG_CACHE_HOME="${task_cache}"
export PYTHONPYCACHEPREFIX="${task_cache}/pycache"

exec "${python_bin}" "${adapter_dir}/module1_detector_sidecar.py" "$@"
