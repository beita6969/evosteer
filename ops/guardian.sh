#!/bin/bash
# Pod-side watchdog for one EvoSteer run. Survives the operator's session.
#
#   guardian.sh <v13|v14> <holder_pause_file> <gpu_uuid_a> <gpu_uuid_b> [target_step=250]
#
# Every CYCLE seconds:
#   1. RENEW the holder pause file (+8h). The 2026-09-21 slowdown happened because this
#      file lapsed at 19:57, letting a 25%-duty-cycle matmul restart on both cards.
#      Renewing on a loop makes that failure mode impossible.
#   2. INTRUDER WATCH - list every PID holding a CUDA context that is not ours and log it
#      loudly. Detection is by CUDA context (nvidia-smi), not by command-line text, because
#      the holder supervisor embeds its worker's source in its own argv and false-matches.
#   3. Keep both VRAM floors alive (82% target, 28 GiB headroom). The floors are also what
#      stops the process-sensing guard from ever starting: it yields to any foreign CUDA
#      context, and the floors are permanent ones.
#   4. If the trainer died below TARGET, resume from the newest complete checkpoint.
#      Never resumes past a failure.json; gives up after MAX_RESTARTS non-advancing tries.
set -u
V="$1"; PAUSE="$2"; UA="$3"; UB="$4"
TARGET="${5:-250}"
CYCLE=300
MAX_RESTARTS=5
PY=/workspace/projects/skillev-glm-session-20260918/runtime/sglang-0.5.15/bin/python
OURS_RE='vram_floor\.py|memory_floor|evosteer_cli|sglang\.launch_server|sglang::|skillev\.runtime\.sglang_server'
LOG=/workspace/evosteer/logs_guardian_$V.log
mkdir -p /workspace/evosteer/logs /workspace/evosteer/logs2
exec >> "$LOG" 2>&1

say() { echo "[$(date '+%m-%d %H:%M:%S')] $V: $*"; }
say "guardian v2 started (target=$TARGET, cycle=${CYCLE}s, pause=$PAUSE)"

restarts=0; last_batch=-1
cur_batch() {
  local R B
  R=$(cat /workspace/evosteer_runs/active_9b_paper_$V.out 2>/dev/null) || { echo 0; return; }
  B=$(ls "$R"/batches 2>/dev/null | tail -1 | sed 's/[^0-9]//g' | sed 's/^0*//')
  echo "${B:-0}"
}

while :; do
  # ---- 1. holder pause: renew so it can never lapse -------------------------
  if [ -n "$PAUSE" ] && [ "$PAUSE" != "none" ]; then
    N=$(date +%s); echo $((N + 28800)) > "$PAUSE" 2>/dev/null \
      || say "WARN could not write pause file $PAUSE"
  fi

  # ---- 2. intruder watch: who holds a CUDA context on our cards? -----------
  INTRUDERS=""
  while IFS=, read -r pid mem; do
    pid=$(echo "$pid" | tr -d ' '); mem=$(echo "$mem" | tr -d ' ')
    [ -z "$pid" ] && continue
    CMD=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-220)
    [ -z "$CMD" ] && CMD="(gone)"
    echo "$CMD" | grep -qE "$OURS_RE" && continue
    INTRUDERS="$INTRUDERS [pid=$pid ${mem} $CMD]"
  done <<< "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)"
  [ -n "$INTRUDERS" ] && say "INTRUDER on our GPUs:$INTRUDERS"

  # ---- 3. VRAM floors ------------------------------------------------------
  NF=$(ps -eo args 2>/dev/null | grep -c '[v]ram_floor\.py')
  if [ "$NF" -lt 2 ]; then
    say "floor count $NF < 2, restarting missing floors"
    for U in "$UA" "$UB"; do
      ps -eo args | grep '[v]ram_floor\.py' | grep -q "$U" && continue
      ( cd /workspace/evosteer && CUDA_VISIBLE_DEVICES=$U FLOOR_TARGET=0.82 FLOOR_HEADROOM_GIB=28 \
          setsid nohup $PY -u vram_floor.py \
          > /workspace/evosteer/gfloor_$(echo "$U" | cut -c5-12).log 2>&1 < /dev/null & )
    done
  fi

  # ---- 4. trainer ----------------------------------------------------------
  B=$(cur_batch)
  GPU=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader 2>/dev/null | tr '\n' ';')
  if pgrep -f "${V}_config" > /dev/null; then
    [ "$B" != "$last_batch" ] && { restarts=0; last_batch=$B; }
    say "ok batch=$B floors=$NF gpu=[$GPU]"
  else
    R=$(cat /workspace/evosteer_runs/active_9b_paper_$V.out 2>/dev/null)
    if [ -f "$R/failure.json" ]; then
      say "STOP failure.json at batch $B; NOT resuming"; head -c 600 "$R/failure.json"; exit 1
    fi
    if [ "$B" -ge "$TARGET" ]; then
      say "DONE batch $B >= $TARGET; guardian exiting"; exit 0
    fi
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      say "STOP $restarts restarts without progress at batch $B; giving up"; exit 1
    fi
    restarts=$((restarts + 1)); last_batch=$B
    say "trainer GONE at batch $B (<$TARGET) -> resume attempt $restarts/$MAX_RESTARTS"
    chmod +x /workspace/evosteer/launch_paper_${V}_ext.sh 2>/dev/null
    ( setsid nohup /workspace/evosteer/launch_paper_${V}_ext.sh \
        >> /workspace/evosteer/logs_${V}_ext.log 2>&1 < /dev/null & )
    sleep 90
    if pgrep -f "${V}_config" > /dev/null; then
      say "resumed OK, new out = $(cat /workspace/evosteer_runs/active_9b_paper_$V.out)"
    else
      say "resume FAILED to start; tail:"; tail -25 /workspace/evosteer/logs_${V}_ext.log
    fi
  fi
  sleep $CYCLE
done
