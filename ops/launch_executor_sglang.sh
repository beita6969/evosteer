#!/bin/bash
# Frozen executor/author server for EvoSteer (GPU1, port 31000). KV pool and
# mamba state cache are capped so a rollout replica of the actor fits beside it.
S=/workspace/projects/skillev-glm-session-20260918/runtime/sglang-0.5.15/bin/python
cd /workspace/evosteer
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 SGLANG_ENABLE_JIT_DEEPGEMM=0 SGL_ENABLE_JIT_DEEPGEMM=0
exec $S -u -m sglang.launch_server --model-path /workspace/models/Qwen3.8-27B-FP8 \
  --served-model-name qwen27b-exec --host 127.0.0.1 --port 31000 --context-length 32768 \
  --tp-size 1 --mem-fraction-static 0.80 --max-total-tokens 160000 --max-mamba-cache-size 64 \
  --max-running-requests 32 --chunked-prefill-size 8192 --random-seed 0
