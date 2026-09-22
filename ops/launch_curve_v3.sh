#!/bin/bash
# EvoSteer 27B curve run v3: fixed FP8 loading, FP8 input gradients, activation
# checkpointing, SGLang executor (GPU1, port 31000), fixed graders; training curves only (no held-out eval), dense gradients, lr 2e-5, 12 tasks per step.
set -euo pipefail
cd /workspace/evosteer/SKILLEV-new-main
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=/workspace/evosteer_runs/27b_curve_v3_${STAMP}
export HF_HUB_OFFLINE=1 TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVOSTEER_EXECUTOR_URL=http://127.0.0.1:31000 EVOSTEER_AUTHOR_URL=http://127.0.0.1:31000
export EVOSTEER_ROLLOUT_CONCURRENCY=32
export EVOSTEER_CURVE_DATA_ROOT=/workspace/evosteer/SKILLEV-new-main/data/curve_benchmarks
export PYTHONPATH=/workspace/evosteer/SKILLEV-new-main/src
echo "$OUT" > /workspace/evosteer_runs/active_27b_curve_v3.out
exec /workspace/evosteer/venv313/bin/python -u -m skillev.evosteer_cli train \
  --config /workspace/evosteer_27b_curve_v3_config.json \
  --task-factory skillev.experiments.curve_benchmarks:task_factory \
  --output "$OUT"
