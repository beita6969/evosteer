#!/bin/bash
# Stop a run cleanly once it has COMPLETED a given batch.
#   stopat.sh <v13|v14> <target_step>
# Waits for the target batch's checkpoint to be complete (all four files), then SIGTERMs
# the trainer. Never kills mid-write: the batch json and the checkpoint are both verified.
set -u
V="$1"; T="$2"
LOG=/workspace/evosteer/logs_stopat_$V.log
exec >> "$LOG" 2>&1
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $V: $*"; }
say "stopat armed: will stop after batch $T completes"
while :; do
  R=$(cat /workspace/evosteer_runs/active_9b_paper_$V.out 2>/dev/null)
  B=$(ls "$R"/batches 2>/dev/null | tail -1 | sed 's/[^0-9]//g' | sed 's/^0*//'); B=${B:-0}
  if [ "$B" -ge "$T" ]; then
    C=$(printf '%s/checkpoints/batch-%06d' "$R" "$T")
    if [ -f "$C/cli-context.json" ] && [ -f "$C/state.json" ] && [ -f "$C/state.sha256" ] && [ -f "$C/trainable.pt" ]; then
      say "batch $B done and checkpoint batch-$(printf '%06d' $T) is complete -> stopping trainer"
      P=$(pgrep -f "${V}_config" | head -1)
      if [ -n "$P" ]; then
        kill -TERM "$P"; sleep 25
        pgrep -f "${V}_config" > /dev/null && { say "SIGTERM ignored, sending KILL"; pkill -9 -f "${V}_config"; }
      fi
      sleep 5
      pgrep -f "${V}_config" > /dev/null && say "WARN trainer still alive" || say "trainer stopped. final batch = $B"
      exit 0
    fi
    say "batch $B >= $T but checkpoint not yet complete, waiting"
  fi
  sleep 60
done
