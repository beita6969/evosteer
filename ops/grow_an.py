"""Does orchestrator prompt size grow with training step? (explains the sec/step climb?)"""
import json, glob, sys, os, statistics as st
from collections import defaultdict
dirs = [l.strip() for l in open(f'/workspace/evosteer_runs/chain_{sys.argv[1]}.txt') if l.strip()]
agg = defaultdict(lambda: defaultdict(float)); cnt = defaultdict(int)
mt = {}
for R in dirs:
    for f in sorted(glob.glob(R + '/trajectories/*.jsonl')):
        idx = int(f.rsplit('-', 1)[1].split('.')[0])
        b = idx // 15 * 15
        try: mt[idx] = os.path.getmtime(R + '/batches/batch-%06d.json' % idx)
        except OSError: pass
        for line in open(f):
            r = json.loads(line)
            u = r.get('usage_json') or {}
            if isinstance(u, str):
                try: u = json.loads(u)
                except Exception: u = {}
            tin = u.get('input_tokens')
            if tin is None: continue
            ts = r.get('terminal_state_json') or {}
            if isinstance(ts, str):
                try: ts = json.loads(ts)
                except Exception: ts = {}
            nu = ts.get('node_usage') or {}
            ein = nu.get('input_tokens', 0) or 0 if isinstance(nu, dict) else 0
            calls = u.get('model_calls', 0) or 0
            a = agg[b]
            a['orch_in'] += tin - ein; a['calls'] += calls; a['exec_in'] += ein
            a['dec'] += len(r.get('decisions') or ()) or calls
            cnt[b] += 1
print('%-10s %7s %11s %10s %12s %10s' % ('steps','episodes','orch_in/ep','calls/ep','orch_in/call','sec/step'))
prev=None
for b in sorted(agg):
    c = cnt[b]
    if c < 100: continue
    secs = [mt[k]-mt[k-1] for k in range(b, b+15) if k in mt and k-1 in mt and 0 < mt[k]-mt[k-1] < 6000]
    a = agg[b]
    print('%-10s %7d %11.0f %10.2f %12.0f %10s' % (
        '%d-%d' % (b, b+14), c, a['orch_in']/c, a['calls']/c,
        a['orch_in']/max(a['calls'],1), '%.0f' % st.mean(secs) if secs else '-'))
