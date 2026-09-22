#!/bin/bash
# EvoSteer Qwen3.5-9B curve run v10: v7 + skill warmup (no skills in batches 1-8; author writes one skill per family at batch 9 from a pooled evidence buffer) + sync-free update.
# Fixes since v6: verifier sees the draft, truncation reported to features/author, controller view
# renders each output once, fitted value head, author sees output endings, interface-only
# orchestrator prompt. Disclosed deviations: training pool = 83 tasks with 0<p_hat<1 in an 8-sample
# solver-only pilot; batch 24; lr 2e-5; executor output cap 8192 / input 20480; total_token_cap 98304.
set -euo pipefail
cd /workspace/evosteer/SKILLEV-new-main
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=/workspace/evosteer_runs/9b_paper_v10_${STAMP}
export HF_HUB_OFFLINE=1 TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVOSTEER_EXECUTOR_URL=http://127.0.0.1:31000 EVOSTEER_AUTHOR_URL=http://127.0.0.1:31000
export EVOSTEER_ROLLOUT_CONCURRENCY=64
export EVOSTEER_CURVE_DATA=/workspace/evosteer/SKILLEV-new-main/data/paper_benchmarks/paper_benchmarks_9b_mixed.jsonl
# Skill author: hosted model on the user's gateway; key stays in a private container-local file.
export EVOSTEER_AUTHOR_API_BASE=${EVOSTEER_AUTHOR_API_BASE:-https://llm-gateway.example.org/v1} EVOSTEER_AUTHOR_API_MODEL=lab-gpt-5.6-sol EVOSTEER_AUTHOR_API_EFFORT=medium
export EVOSTEER_AUTHOR_API_KEY_FILE=/root/.config/evosteer/author_api_key SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
# Asynchronous prefetch: rollouts on a replica (GPU1) overlap the update (GPU0).
export EVOSTEER_PIPELINE=1 EVOSTEER_SAMPLER_DEVICE=cuda:1
# Shorter GIL hand-offs between the update thread and the rollout loop (measured ~2x faster update).
export EVOSTEER_GIL_SWITCH_INTERVAL=0.0005
export PYTHONPATH=/workspace/evosteer/SKILLEV-new-main/src
echo "$OUT" > /workspace/evosteer_runs/active_9b_paper_v10.out
exec /workspace/evosteer/venv313/bin/python -u -m skillev.evosteer_cli train \
  --config /workspace/evosteer_9b_paper_v10_config.json \
  --task-factory skillev.experiments.curve_benchmarks:task_factory \
  --output "$OUT"
