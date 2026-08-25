#!/usr/bin/env bash
# Download the access-controlled HM3D-Sem v0.2 validation assets required by
# ObjectNav-v2.  Run this locally after accepting Matterport's dataset terms.
# Credentials are requested by the terminal and are never written to this repo.
set -euo pipefail

task_prefix=/home/ldl/conda_envs/habitat-challenge-2023
task_data=/home/ldl/habitat-objectnav/data
task_tmp=/home/ldl/tmp/habitat-objectnav
task_cache=/home/ldl/.cache/habitat-objectnav

mkdir -p "$task_data" "$task_tmp" "$task_cache"
read -r -p "Matterport username: " mp_username
read -r -s -p "Matterport password: " mp_password
printf '\n'

# The datasets downloader comes from Habitat-Sim.  Keep credentials out of the
# process command line.  Curl's netrc mechanism needs a temporary protected
# file; it is mode 600 and removed on every normal exit or interruption.  The
# downloader obtains precisely the habitat, config, semantic annotation, and
# semantic config parts of the official HM3D validation group.
umask 077
netrc_file=$(mktemp "$task_tmp/hm3d-netrc.XXXXXX")
trap 'rm -f "$netrc_file"' EXIT HUP INT TERM
printf 'machine api.matterport.com login %s password %s\n' "$mp_username" "$mp_password" > "$netrc_file"
unset mp_username mp_password

export task_hm3d_data="$task_data"
export task_hm3d_netrc="$netrc_file"
TMPDIR="$task_tmp" XDG_CACHE_HOME="$task_cache" \
  "$task_prefix/bin/python" - <<'PY'
import os

from habitat_sim.utils import datasets_download

data_path = os.path.abspath(os.environ["task_hm3d_data"]) + "/"
datasets_download.initialize_test_data_sources(data_path)
for uid in datasets_download.data_groups["hm3d_val_v0.2"]:
    # Avoid the downloader's ``--user username:password`` argument.  Curl reads
    # the protected temporary netrc file instead, while the official source and
    # extraction logic remain unchanged.
    source = datasets_download.data_sources[uid]
    source["requires_auth"] = False
    source["download_pre_args"] = (
        source.get("download_pre_args", "")
        + " --netrc-file "
        + os.environ["task_hm3d_netrc"]
    )
    datasets_download.download_and_place(
        uid,
        data_path,
        replace=False,
    )
PY
unset task_hm3d_data task_hm3d_netrc

# Challenge-2023 expects this path.  The downloader creates scene_datasets/hm3d
# as a link to the versioned data; provide a stable v0.2 alias if absent.
if [[ ! -e "$task_data/scene_datasets/hm3d_v0.2" ]]; then
  ln -s hm3d "$task_data/scene_datasets/hm3d_v0.2"
fi

printf 'HM3D validation assets installed under %s\n' "$task_data/scene_datasets/hm3d_v0.2"
