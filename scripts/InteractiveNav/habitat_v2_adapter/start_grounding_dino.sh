#!/usr/bin/env bash
# Launch the already-installed VLFM GroundingDINO worker outside the Habitat
# Challenge Python environment.  No model download or original-code edit occurs.
set -euo pipefail

adapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
vlfm_root="/home/ldl/molmospaces-exp-compare/external_methods/vlfm"
vlfm_python="/home/ldl/molmospaces-exp-compare/external_methods/.envs/mlspaces_vlfm/bin/python"
task_cache="/home/ldl/.cache/habitat_objectnav_grounding_dino"

export TMPDIR="/home/ldl/tmp/habitat_objectnav_grounding_dino"
export XDG_CACHE_HOME="$task_cache"
export HF_HOME="$task_cache/huggingface"
export TRANSFORMERS_CACHE="$task_cache/transformers"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$vlfm_root:$vlfm_root/GroundingDINO${PYTHONPATH:+:$PYTHONPATH}"
export GROUNDING_DINO_CONFIG="$vlfm_root/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
export GROUNDING_DINO_WEIGHTS="$vlfm_root/data/groundingdino_swint_ogc.pth"
export GROUNDING_DINO_TEXT_ENCODER="/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/data/models/bert-base-uncased"
export GROUNDING_DINO_CONFIG_ONLY_BERT=1

mkdir -p "$TMPDIR" "$task_cache"
exec "$vlfm_python" "$adapter_dir/grounding_dino_server.py" "$@"
