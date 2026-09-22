"""Per-rollout wall seconds over time: is each executor call slower, or are we just queueing?"""
import json, sys, statistics as st
from collections import defaultdict
buckets = defaultdict(list)
for path in sys.argv[1:]:
    try: fh = open(path, errors='ignore')
    except OSError: continue
    for line in fh:
        if '"event": "rollout"' not in line: continue
        try: r = json.loads(line)
        except Exception: continue
        sid = r.get('sample_id', '')
        if not sid.startswith('batch-'): continue
        try: n = int(sid.split('/')[0].split('-')[1])
        except Exception: continue
        s = r.get('seconds')
        if s is None: continue
        buckets[n // 15 * 15].append((s, len(r.get('actions') or ())))
print('%-12s %8s %10s %10s %10s %8s' % ('steps','rollouts','mean s','median s','p90 s','acts'))
for b in sorted(buckets):
    v = [x[0] for x in buckets[b]]; a = [x[1] for x in buckets[b]]
    if len(v) < 30: continue
    v.sort()
    print('%-12s %8d %10.1f %10.1f %10.1f %8.2f' % (
        '%d-%d' % (b, b+14), len(v), st.mean(v), v[len(v)//2], v[int(.9*len(v))], st.mean(a)))
