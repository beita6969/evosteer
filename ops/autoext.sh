#!/bin/bash
# Watch one run; when it finishes its 150-step leg cleanly, resume it to 250.
# Only ever fires ONCE, and only on a clean finish at >= FIRST_TARGET with no failure.json.
# A crash (process gone below the target, or a failure.json) is reported, never auto-restarted.
V="$1"                       # v13 | v14
FIRST_TARGET="${2:-150}"
LOG=/workspace/evosteer/logs_autoext_$V.log
exec >> "$LOG" 2>&1
echo "=== autoext $V started $(date), first target $FIRST_TARGET ==="
while :; do
  if pgrep -f "${V}_config" > /dev/null; then sleep 120; continue; fi
  # trainer is gone - decide why
  R=$(cat /workspace/evosteer_runs/active_9b_paper_$V.out 2>/dev/null)
  LAST=$(ls "$R"/batches 2>/dev/null | tail -1 | sed 's/[^0-9]//g' | sed 's/^0*//')
  LAST=${LAST:-0}
  if [ -f "$R/failure.json" ]; then
    echo "$(date) $V: failure.json present at batch $LAST - NOT resuming"; cat "$R/failure.json"; exit 1
  fi
  if [ "$LAST" -lt "$FIRST_TARGET" ]; then
    echo "$(date) $V: trainer gone at batch $LAST < $FIRST_TARGET - looks like a crash, NOT resuming"; exit 1
  fi
  echo "$(date) $V: clean finish at batch $LAST -> resuming to 250"
  chmod +x /workspace/evosteer/launch_paper_${V}_ext.sh
  ( setsid nohup /workspace/evosteer/launch_paper_${V}_ext.sh \
      > /workspace/evosteer/logs_${V}_ext.log 2>&1 < /dev/null & )
  sleep 60
  if pgrep -f "${V}_config" > /dev/null; then
    echo "$(date) $V: extension RUNNING, new out = $(cat /workspace/evosteer_runs/active_9b_paper_$V.out)"
  else
    echo "$(date) $V: extension FAILED TO START - tail of its log:"; tail -40 /workspace/evosteer/logs_${V}_ext.log
  fi
  exit 0
done
