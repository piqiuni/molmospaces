#!/usr/bin/env bash
set -euo pipefail

cd /home/user/ldl/molmospaces-exp-setting
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate mlspaces

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MLSPACES_CACHE_DIR=/tmp/container_probe_resources_20260702a
export MLSPACES_ASSETS_DIR=/tmp/container_probe_assets_20260702a
export MPLCONFIGDIR=/tmp/matplotlib

python scripts/InteractiveNav/container_scene_probe.py \
  --output_dir scripts/InteractiveNav/output/container_scene_probe_first100_scenes \
  scan-container-target-overlap \
  --max_episodes 2000 \
  --house_inds $(seq 0 99)
