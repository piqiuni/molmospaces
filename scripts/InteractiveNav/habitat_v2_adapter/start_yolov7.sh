#!/usr/bin/env bash
# Start the pre-existing VLFM YOLOv7 COCO detector without writing its TorchScript
# trace into the external VLFM checkout.  All generated state stays under /home/ldl.
set -euo pipefail

vlfm_root="/home/ldl/molmospaces-exp-compare/external_methods/vlfm"
vlfm_python="/home/ldl/molmospaces-exp-compare/external_methods/.envs/mlspaces_vlfm/bin/python"
task_cache="/home/ldl/.cache/habitat-objectnav-yolov7"
trace_dir="${task_cache}/trace"
task_tmp="/home/ldl/tmp/habitat-objectnav-yolov7"

mkdir -p "${trace_dir}" "${task_tmp}"
# vlfm.vlm.yolov7 uses a CWD-relative `yolov7/` import and writes
# `traced_model.pt` to CWD.  The symlink makes both behaviours resolve safely.
if [[ ! -e "${trace_dir}/yolov7" ]]; then
  ln -s "${vlfm_root}/yolov7" "${trace_dir}/yolov7"
fi

export TMPDIR="${task_tmp}"
export XDG_CACHE_HOME="${task_cache}"
export TORCH_HOME="${task_cache}/torch"
export PYTHONPATH="${vlfm_root}${PYTHONPATH:+:${PYTHONPATH}}"
export YOLOV7_WEIGHTS="${vlfm_root}/data/yolov7-e6e.pt"

cd "${trace_dir}"
if [[ "$#" -eq 0 ]]; then
  set -- --port 12184
fi
exec "${vlfm_python}" -m vlfm.vlm.yolov7 "$@"
