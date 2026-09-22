"""Build a --resume launch script for <v>, extending the run to the config's new step target.

Derives everything from the original launch script so the environment is byte-identical;
only OUT and the exec line change. The previous run dir is remembered so analysis can
concatenate both halves.
"""
import json, os, re, sys

V = sys.argv[1]                       # v13 | v14
SRC = f'/workspace/evosteer/launch_paper_{V}.sh'
DST = f'/workspace/evosteer/launch_paper_{V}_ext.sh'
CFG = f'/workspace/evosteer_9b_paper_{V}_config.json'

steps = json.load(open(CFG))['steps']
s = open(SRC).read()

# OUT gets an _ext suffix so the resumed half writes to its own directory.
s2, n = re.subn(rf'^OUT=/workspace/evosteer_runs/9b_paper_{V}_\$\{{STAMP\}}$',
                f'OUT=/workspace/evosteer_runs/9b_paper_{V}_ext_${{STAMP}}', s, flags=re.M)
assert n == 1, f'OUT line not matched ({n})'

# Resolve the checkpoint to resume from, at launch time, from the CURRENT active run.
preamble = f'''
# ---- resume wiring (added for the {steps}-step extension) ----
PREV=$(cat /workspace/evosteer_runs/active_9b_paper_{V}.out)
# newest checkpoint that actually has all four files (the last one can be mid-write)
CKPT=""
for d in $(ls -d "$PREV"/checkpoints/batch-* | sort -r); do
  if [ -f "$d/cli-context.json" ] && [ -f "$d/state.json" ] && [ -f "$d/state.sha256" ] && [ -f "$d/trainable.pt" ]; then
    CKPT="$d"; break
  fi
  echo "[{V}-ext] skipping incomplete checkpoint $d"
done
[ -n "$CKPT" ] || {{ echo "no complete checkpoint to resume from under $PREV" >&2; exit 1; }}
echo "[{V}-ext] resuming from $CKPT (target {steps} steps)"
echo "$PREV" > /workspace/evosteer_runs/active_9b_paper_{V}_prev.out
grep -qxF "$PREV" /workspace/evosteer_runs/chain_{V}.txt 2>/dev/null || echo "$PREV" >> /workspace/evosteer_runs/chain_{V}.txt
'''
marker = f'echo "$OUT" > /workspace/evosteer_runs/active_9b_paper_{V}.out'
assert s2.count(marker) == 1
s2 = s2.replace(marker, preamble.strip() + '\n' + marker)

# Append --resume to the exec.
old_exec = f'''  --config /workspace/evosteer_9b_paper_{V}_config.json \\
  --task-factory skillev.experiments.curve_benchmarks:task_factory \\
  --output "$OUT"'''
new_exec = f'''  --config /workspace/evosteer_9b_paper_{V}_config.json \\
  --task-factory skillev.experiments.curve_benchmarks:task_factory \\
  --resume "$CKPT" \\
  --output "$OUT"'''
assert s2.count(old_exec) == 1, 'exec block not matched'
s2 = s2.replace(old_exec, new_exec)

open(DST, 'w').write(s2)
os.chmod(DST, 0o755)
print(f'wrote {DST}  (steps target = {steps})')
