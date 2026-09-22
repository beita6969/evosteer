"""Separate 'grows with step index' from 'grows with validated skill count'."""
import json, glob, os, sys, statistics as st
V = sys.argv[1]
dirs = [l.strip() for l in open(f'/workspace/evosteer_runs/chain_{V}.txt')]
cur = open(f'/workspace/evosteer_runs/active_9b_paper_{V}.out').read().strip()
if cur not in dirs: dirs.append(cur)
rows = {}
for R in dirs:
    for f in sorted(glob.glob(R + '/batches/*.json')):
        j = json.load(open(f)); s = j.get('skill_status') or {}
        rows[j['batch_index']] = (sum(1 for x in s.values() if x == 'validated'),
                                  os.path.getmtime(f), len(s))
ks = sorted(rows)
pts = [(k, rows[k][0], rows[k][1]-rows[k-1][1], rows[k][2]) for k in ks if k-1 in rows]
# drop the resume boundary and the pod-1 holder window
pts = [p for p in pts if p[2] < 4000 and not (V=='v13' and 145 <= p[0+0] <= 0)]
print('validated -> sec/step  (unconditional)')
byv = {}
for k, v, s, n in pts: byv.setdefault(v, []).append(s)
for v in sorted(byv): print('   val=%d  n=%3d  mean %6.0f s' % (v, len(byv[v]), st.mean(byv[v])))
print()
print('WITHIN each validated level, does it still climb with step index?')
for v in sorted(byv):
    sel = [(k, s) for k, vv, s, n in pts if vv == v]
    if len(sel) < 8: continue
    xs = [k for k, _ in sel]; ys = [s for _, s in sel]
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    b = sxy/sxx
    r = [y-(my + b*(x-mx)) for x, y in zip(xs, ys)]
    se = (sum(t*t for t in r)/(len(xs)-2)/sxx)**0.5
    print('   val=%d  n=%3d  slope %+6.2f s/step  t=%+5.1f  (span %d-%d)' % (
        v, len(xs), b, b/se, min(xs), max(xs)))
print()
print('OVERALL slope vs step index, all points:')
xs = [k for k, _, _, _ in pts]; ys = [s for _, _, s, _ in pts]
mx, my = st.mean(xs), st.mean(ys)
sxx = sum((x-mx)**2 for x in xs); b = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sxx
print('   %+.2f s per step  -> over 250 steps that is %+.0f s' % (b, b*250))
