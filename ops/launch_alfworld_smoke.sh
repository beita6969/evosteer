#!/bin/bash
# ALFWorld smoke run of EvoSteer on ONE GPU (executor + actor share the card).
#   ops/launch_alfworld_smoke.sh <output_dir>
# Prerequisites: scripts/provision_alfworld.sh has been run for venv313_alfworld,
# and the frozen executor is up:  ops/launch_executor_9b.sh 0 31000
# (on a single card set its --mem-fraction-static to 0.45 so the actor fits).
set -euo pipefail
OUT="${1:?output dir}"
ROOT=${EVOSTEER_ROOT:-/workspace/evosteer/SKILLEV-new-main}
cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVOSTEER_EXECUTOR_URL=${EVOSTEER_EXECUTOR_URL:-http://127.0.0.1:31000}
export EVOSTEER_ROLLOUT_CONCURRENCY=${EVOSTEER_ROLLOUT_CONCURRENCY:-8}
export ALFWORLD_DATA=${ALFWORLD_DATA:?set ALFWORLD_DATA}
export EVOSTEER_ALFWORLD_SPLIT=${EVOSTEER_ALFWORLD_SPLIT:-train}
export EVOSTEER_ALFWORLD_COUNT=${EVOSTEER_ALFWORLD_COUNT:-8}
export ALFWORLD_MAX_STEPS=${ALFWORLD_MAX_STEPS:-50} ALFWORLD_STEPS_PER_NODE=${ALFWORLD_STEPS_PER_NODE:-20}
export PYTHONPATH="$ROOT/src"
PY=${EVOSTEER_PYTHON:-/workspace/evosteer/venv313_alfworld/bin/python}
exec "$PY" -u -m skillev.evosteer_cli train \
  --config "$ROOT/configs/paper_runs/evosteer_9b_alfworld_smoke_config.json" \
  --task-factory skillev.experiments.curve_alfworld:task_factory \
  --output "$OUT"
