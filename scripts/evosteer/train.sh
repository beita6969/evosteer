#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
RESUME="${1:-}"
CONFIG=${EVOSTEER_CONFIG:-$EVOSTEER_HOME/configs/evosteer/evosteer_9b_paper.json}
OUT=${EVOSTEER_OUT:-$EVOSTEER_RUNS/evosteer_$(date +%Y%m%d-%H%M%S)}
GPU=${EVOSTEER_TRAIN_GPU:-0}

missing=()
for name in EVOSTEER_AUTHOR_API_BASE EVOSTEER_AUTHOR_API_KEY_FILE EVOSTEER_AUTHOR_API_MODEL; do
  [ -n "${!name:-}" ] || missing+=("$name")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "The skill author model is not configured. Set: ${missing[*]}" >&2
  echo "  EVOSTEER_AUTHOR_API_BASE      base URL of an OpenAI Responses-compatible endpoint" >&2
  echo "  EVOSTEER_AUTHOR_API_KEY_FILE  file that contains the API key" >&2
  echo "  EVOSTEER_AUTHOR_API_MODEL     model name of the skill author" >&2
  exit 2
fi
[ -s "$EVOSTEER_AUTHOR_API_KEY_FILE" ] || { echo "EVOSTEER_AUTHOR_API_KEY_FILE is empty or missing" >&2; exit 2; }
export EVOSTEER_AUTHOR_API_EFFORT=${EVOSTEER_AUTHOR_API_EFFORT:-high}
export EVOSTEER_AUTHOR_URL=$EVOSTEER_EXECUTOR_URL

export EVOSTEER_CURVE_DATA=${EVOSTEER_CURVE_DATA:-$EVOSTEER_DATA/evosteer/train/train_pool.jsonl}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$EVOSTEER_DATA/alfworld}
export ALFWORLD_MAX_STEPS=50 ALFWORLD_STEPS_PER_NODE=25
export EVOSTEER_PIPELINE=${EVOSTEER_PIPELINE:-1}
if [ "$EVOSTEER_PIPELINE" = 1 ]; then export EVOSTEER_SAMPLER_DEVICE=cuda:0; else unset EVOSTEER_SAMPLER_DEVICE; fi
export EVOSTEER_ROLLOUT_CONCURRENCY=${EVOSTEER_ROLLOUT_CONCURRENCY:-64} EVOSTEER_GIL_SWITCH_INTERVAL=0.0005
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$EVOSTEER_HOME/src"
cd "$EVOSTEER_HOME"

for url in ${EVOSTEER_EXECUTOR_URL//,/ }; do
  curl -sf -m 10 "$url/health" > /dev/null || { echo "executor replica not healthy at $url (scripts/evosteer/serve_executor.sh start)" >&2; exit 3; }
done
curl -sf -m 20 "$EVOSTEER_RETRIEVAL_URL/health" > /dev/null || { echo "retrieval service not healthy at $EVOSTEER_RETRIEVAL_URL (scripts/evosteer/serve_retrieval.sh start)" >&2; exit 3; }
if [ -z "${EVOSTEER_RETRIEVAL_IDENTITY:-}" ]; then
  EVOSTEER_RETRIEVAL_IDENTITY=$("$EVOSTEER_TRAIN_PY" - "$CONFIG" <<'PY'
import json, sys
from skillev.orchestration.retrieval_client import expected_identity_for
print(expected_identity_for(json.load(open(sys.argv[1]))["application"]["text_tools"]))
PY
  )
  export EVOSTEER_RETRIEVAL_IDENTITY
fi

"$EVOSTEER_TRAIN_PY" - "$EVOSTEER_CURVE_DATA" <<'PY' || { echo "a python tool program can read $EVOSTEER_CURVE_DATA (answers); refusing to run" >&2; exit 4; }
import sys
from skillev.orchestration.text_tools import PythonSandbox
sandbox = PythonSandbox()
result = sandbox.execute(f"print(open({sys.argv[1]!r}).read(20))")
print("python tool sandbox:", sandbox.isolation, sandbox.fs or "-", "task file read:", result.status)
sys.exit(0 if result.status == "failed" and "PermissionError" in result.text else 1)
PY

mkdir -p "$(dirname "$OUT")"
exec "$EVOSTEER_TRAIN_PY" -u -m skillev.evosteer_cli train \
  --config "$CONFIG" \
  --task-factory skillev.experiments.curve_tools:task_factory \
  --output "$OUT" ${RESUME:+--resume "$RESUME"}
