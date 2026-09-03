#!/usr/bin/env bash
set -euo pipefail

# This wrapper deliberately runs ROS Noetic nodes with Ubuntu's Python 3.10.
# Do not expose binary extension modules from the Conda 3.11 environment to
# that interpreter (Pillow's _imaging is one example).  Keep source-only
# project paths inherited from the launcher, while dropping foreign
# site-packages entries.
clean_pythonpath=""
IFS=':' read -r -a pythonpath_entries <<<"${PYTHONPATH:-}"
for entry in "${pythonpath_entries[@]}"; do
  [[ -n "${entry}" ]] || continue
  case "${entry}" in
    */miniconda3/envs/*/lib/python*/site-packages|*/conda/envs/*/lib/python*/site-packages)
      continue
      ;;
  esac
  clean_pythonpath+="${clean_pythonpath:+:}${entry}"
done
export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages${clean_pythonpath:+:${clean_pythonpath}}"
exec /usr/bin/python3 "$@"
