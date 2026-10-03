#!/bin/bash
source "$(dirname "$0")/env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)
WIKI=${EVOSTEER_WIKI18:-$EVOSTEER_DATA/retrieval/wiki18}
E5=${EVOSTEER_E5:-$EVOSTEER_MODELS/e5-base-v2}
PORT=${EVOSTEER_RETRIEVAL_PORT:-18010}
LOG=$EVOSTEER_LOGS/retrieval.log
PIDF=$EVOSTEER_LOGS/retrieval.pid
case "${1:-status}" in
  prepare) "$EVOSTEER_TRAIN_PY" -u "$HERE/retrieval_service.py" prepare "$WIKI";;
  embed) "$EVOSTEER_TRAIN_PY" -u "$HERE/retrieval_service.py" embed "$WIKI" "$E5" --device "${EMBED_DEVICE:-cuda:0}";;
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "retrieval already running"; exit 0; fi
    DEVICE=${RETRIEVAL_DEVICE:-cpu}
    setsid nohup "$EVOSTEER_TRAIN_PY" -u "$HERE/retrieval_service.py" serve "$WIKI" "$E5" \
      --port "$PORT" --host 127.0.0.1 --device "$DEVICE" --threads "${RETRIEVAL_THREADS:-32}" > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"; echo "started $(cat "$PIDF") on port $PORT";;
  stop)
    P=$(cat "$PIDF" 2>/dev/null)
    tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null | grep -q "retrieval_service.py serve" && kill "$P" && echo "stopped $P"
    rm -f "$PIDF";;
  status) curl -s -m 5 "http://127.0.0.1:$PORT/health"; echo;;
  *) echo "usage: $0 prepare|embed|start|stop|status" >&2; exit 2;;
esac
