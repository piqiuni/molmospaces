#!/usr/bin/env bash
# Launch the already-installed VLFM PointNav worker outside the Challenge Python
# environment.  It loads only the local pointnav_weights.pth checkpoint; it does
# not download a model or alter the original InteractiveNav source tree.
set -euo pipefail

adapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
vlfm_root="/home/ldl/molmospaces-exp-compare/external_methods/vlfm"
vlfm_python="/home/ldl/molmospaces-exp-compare/external_methods/.envs/mlspaces_vlfm/bin/python"
task_cache="/home/ldl/.cache/habitat_objectnav_pointnav"

export TMPDIR="/home/ldl/tmp/habitat_objectnav_pointnav"
export XDG_CACHE_HOME="$task_cache"
export HF_HOME="$task_cache/huggingface"
export TORCH_HOME="$task_cache/torch"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$vlfm_root${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$TMPDIR" "$task_cache"
exec "$vlfm_python" "$adapter_dir/pointnav_server.py" "$@"
