#!/bin/bash
# EvoSteer Qwen3.5-9B curve run v13.
# Deviations from the paper main text, all disclosed:
#   - training pool = 145 tasks with 0<p_hat<1 in an 8-sample solver-only pilot
#   - allow_repeated_pair_tasks=true (a task may contribute paired evidence more than once)
#   - validation_interval=30: the alpha-spending LOOK is taken every 30 batches instead of on
#     every new discordant pair. Evidence still accrues every batch; only the look schedule is
#     batched, and it is pre-declared, so alpha_spent still sums to <= alpha.
#   - actor_beta1=0.95 (AdamW beta1 on the actor group only; heads stay at 0.9)
#   - solver instruction carries a commit clause ("if you are running long, give your
#     best current answer"); executor output cap stays at 8192
#   - executor input cap 20480, total_token_cap 98304
#   - grader: \boxed{} now outranks the GSM8K '####' marker in math answer extraction
#   - answer_present is reported honestly for plan / fact-list outputs

set -euo pipefail
RESUME="${1:-}"
cd /workspace/evosteer/SKILLEV-new-main
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=/workspace/evosteer_runs/9b_paper_v13_ext_${STAMP}
export HF_HUB_OFFLINE=1 TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVOSTEER_EXECUTOR_URL=http://127.0.0.1:31000 EVOSTEER_AUTHOR_URL=http://127.0.0.1:31000
export EVOSTEER_ROLLOUT_CONCURRENCY=64
export EVOSTEER_CURVE_DATA=/workspace/evosteer/SKILLEV-new-main/data/paper_benchmarks/paper_benchmarks_9b_mixed_large.jsonl
# Skill author: hosted model on the user's gateway; key stays in a private container-local file.
export EVOSTEER_AUTHOR_API_BASE=${EVOSTEER_AUTHOR_API_BASE:-https://llm-gateway.example.org/v1} EVOSTEER_AUTHOR_API_MODEL=lab-gpt-5.6-sol EVOSTEER_AUTHOR_API_EFFORT=medium
export EVOSTEER_AUTHOR_API_KEY_FILE=/root/.config/evosteer/author_api_key SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export EVOSTEER_PIPELINE=1 EVOSTEER_SAMPLER_DEVICE=cuda:1
export EVOSTEER_GIL_SWITCH_INTERVAL=0.0005
export PYTHONPATH=/workspace/evosteer/SKILLEV-new-main/src
# ---- resume wiring (added for the 250-step extension) ----
PREV=$(cat /workspace/evosteer_runs/active_9b_paper_v13.out)
# newest checkpoint that actually has all four files (the last one can be mid-write)
CKPT=""
for d in $(ls -d "$PREV"/checkpoints/batch-* | sort -r); do
  if [ -f "$d/cli-context.json" ] && [ -f "$d/state.json" ] && [ -f "$d/state.sha256" ] && [ -f "$d/trainable.pt" ]; then
    CKPT="$d"; break
  fi
  echo "[v13-ext] skipping incomplete checkpoint $d"
done
[ -n "$CKPT" ] || { echo "no complete checkpoint to resume from under $PREV" >&2; exit 1; }
echo "[v13-ext] resuming from $CKPT (target 250 steps)"
echo "$PREV" > /workspace/evosteer_runs/active_9b_paper_v13_prev.out
grep -qxF "$PREV" /workspace/evosteer_runs/chain_v13.txt 2>/dev/null || echo "$PREV" >> /workspace/evosteer_runs/chain_v13.txt
echo "$OUT" > /workspace/evosteer_runs/active_9b_paper_v13.out
exec /workspace/evosteer/venv313/bin/python -u -m skillev.evosteer_cli train \
  --config /workspace/evosteer_9b_paper_v13_config.json \
  --task-factory skillev.experiments.curve_benchmarks:task_factory \
  --resume "$CKPT" \
  --output "$OUT"
