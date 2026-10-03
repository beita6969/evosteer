#!/bin/bash
source "$(dirname "$0")/env.sh"
PORT=${EVOSTEER_EXECUTOR_PORT:-31000}
GPU=${EVOSTEER_EXECUTOR_GPU:-0}
FRACTION=${MEM_FRACTION:-0.60}
MODEL=${EVOSTEER_EXECUTOR_MODEL:-$EVOSTEER_MODELS/Qwen3.5-9B}
LOG=$EVOSTEER_LOGS/executor_$PORT.log
PIDF=$EVOSTEER_LOGS/executor_$PORT.pid
case "${1:-status}" in
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "executor already running: $(cat "$PIDF")"; exit 0; fi
    CUDA_VISIBLE_DEVICES=$GPU SGLANG_ENABLE_JIT_DEEPGEMM=0 setsid nohup "$EVOSTEER_SGLANG_PY" -u -m sglang.launch_server \
      --model-path "$MODEL" --served-model-name qwen9b-exec --host 127.0.0.1 --port "$PORT" \
      --context-length 32768 --tp-size 1 --mem-fraction-static "$FRACTION" --max-total-tokens 400000 \
      --max-mamba-cache-size 320 --max-running-requests 64 --chunked-prefill-size 8192 --random-seed 0 \
      > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"; echo "started $(cat "$PIDF") on port $PORT; log $LOG";;
  stop)
    P=$(cat "$PIDF" 2>/dev/null); [ -n "$P" ] || { echo "no pid file"; exit 0; }
    tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null | grep -q "sglang.launch_server --model-path $MODEL" && { kill "$P"; echo "stopped $P"; } || echo "pid $P is not this executor"
    rm -f "$PIDF";;
  status)
    curl -s -m 5 "http://127.0.0.1:$PORT/health" && echo " healthy" || echo "not healthy";;
  *) echo "usage: $0 start|stop|status" >&2; exit 2;;
esac
