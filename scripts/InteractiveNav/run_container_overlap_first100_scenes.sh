#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"
if ! command -v conda >/dev/null 2>&1; then
  source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
fi
conda activate "${CONDA_ENV:-mlspaces}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MLSPACES_CACHE_DIR=${MLSPACES_CACHE_DIR:-/tmp/container_probe_resources}
export MLSPACES_ASSETS_DIR=${MLSPACES_ASSETS_DIR:-/tmp/container_probe_assets}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

python scripts/InteractiveNav/container_scene_probe.py \
  --output_dir scripts/InteractiveNav/output/container_scene_probe_first100_scenes \
  scan-container-target-overlap \
  --max_episodes 2000 \
  --house_inds $(seq 0 99)
