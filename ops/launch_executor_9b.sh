#!/bin/bash
# Frozen Qwen3.5-9B executor server for EvoSteer. usage: launch_executor_9b.sh <gpu> <port>
# mem-fraction leaves ~40 GB of the card for the actor or its rollout replica
# (another project's server also holds ~51 GB on each card).
S=/workspace/projects/skillev-glm-session-20260918/runtime/sglang-0.5.15/bin/python
GPU="${1:-1}"; PORT="${2:-31000}"
cd /workspace/evosteer
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 SGLANG_ENABLE_JIT_DEEPGEMM=0 SGL_ENABLE_JIT_DEEPGEMM=0
exec $S -u -m sglang.launch_server --model-path /workspace/models/Qwen3.5-9B-ms \
  --served-model-name qwen9b-exec --host 127.0.0.1 --port $PORT --context-length 32768 \
  --tp-size 1 --mem-fraction-static ${MEM_FRACTION:-0.70} --max-total-tokens 400000 --max-mamba-cache-size 320 \
  --max-running-requests 64 --chunked-prefill-size 8192 --random-seed 0
