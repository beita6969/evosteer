#!/bin/bash
# EvoSteer 27B curve run v6 (paper benchmarks, full method with skill admission): fixed FP8 loading, FP8 input gradients, activation
# checkpointing, SGLang executor (GPU1, port 31000), fixed graders; training curves only (no held-out eval), dense gradients, lr 2e-5, 12 tasks per step.
set -euo pipefail
cd /workspace/evosteer/SKILLEV-new-main
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=/workspace/evosteer_runs/27b_paper_v6_${STAMP}
export HF_HUB_OFFLINE=1 TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVOSTEER_EXECUTOR_URL=http://127.0.0.1:31000 EVOSTEER_AUTHOR_URL=http://127.0.0.1:31000
export EVOSTEER_ROLLOUT_CONCURRENCY=32
export EVOSTEER_CURVE_DATA=/workspace/evosteer/SKILLEV-new-main/data/paper_benchmarks/paper_benchmarks.jsonl
# Skill author: hosted model on the user's gateway; key stays in a private container-local file.
export EVOSTEER_AUTHOR_API_BASE=${EVOSTEER_AUTHOR_API_BASE:-https://llm-gateway.example.org/v1} EVOSTEER_AUTHOR_API_MODEL=lab-gpt-5.6-sol EVOSTEER_AUTHOR_API_EFFORT=low
export EVOSTEER_AUTHOR_API_KEY_FILE=/root/.config/evosteer/author_api_key SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export PYTHONPATH=/workspace/evosteer/SKILLEV-new-main/src
echo "$OUT" > /workspace/evosteer_runs/active_27b_paper_v6.out
exec /workspace/evosteer/venv313/bin/python -u -m skillev.evosteer_cli train \
  --config /workspace/evosteer_27b_paper_v6_config.json \
  --task-factory skillev.experiments.curve_benchmarks:task_factory \
  --output "$OUT"
