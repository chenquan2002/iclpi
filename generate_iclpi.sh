#!/bin/bash
set -e
export XDG_CACHE_HOME=/data/RoboTwin/data/lerobot_data

data_dir=${1}
repo_id=${2}
shift 2


uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin_iclpi.py \
  --raw_dir "$data_dir" \
  --repo_id "$repo_id" \
  "$@"



