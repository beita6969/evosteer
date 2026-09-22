"""Rollout latency in the minutes BEFORE vs AFTER a cutoff (holder paused ~04:33 CST)."""
import json, sys, time, os, statistics as st
CUT = float(sys.argv[1])        # unix epoch cutoff
before, after = [], []
for path in sys.argv[2:]:
    if not os.path.exists(path): continue
    # only tail the file; rollouts are appended
    with open(path, 'rb') as fh:
        fh.seek(max(0, os.path.getsize(path) - 40_000_000))
        fh.readline()
        data = fh.read().decode('utf-8', 'ignore')
    mt = os.path.getmtime(path)
    lines = [l for l in data.split('\n') if '"event": "rollout"' in l]
    # no per-line timestamp: approximate by position, so instead use file order + mtime
    # -> fall back to using the batch index which we can map to checkpoint mtimes
    for l in lines:
        try: r = json.loads(l)
        except Exception: continue
        s = r.get('seconds')
        if s is not None: after.append((r['sample_id'].split('/')[0], s))
by_batch = {}
for b, s in after:
    by_batch.setdefault(b, []).append(s)
print('%-16s %8s %9s %9s %9s' % ('batch', 'n', 'mean s', 'median s', 'p90 s'))
for b in sorted(by_batch)[-14:]:
    v = sorted(by_batch[b])
    if len(v) < 20: continue
    print('%-16s %8d %9.1f %9.1f %9.1f' % (b, len(v), st.mean(v), v[len(v)//2], v[int(.9*len(v))]))
