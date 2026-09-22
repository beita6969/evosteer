"""Per-step wall time vs the work actually done, across the whole chained run."""
import json, glob, os, sys, statistics as st
V = sys.argv[1]
dirs = [l.strip() for l in open(f'/workspace/evosteer_runs/chain_{V}.txt')]
cur = open(f'/workspace/evosteer_runs/active_9b_paper_{V}.out').read().strip()
if cur not in dirs: dirs.append(cur)
rows = {}
for R in dirs:
    for f in sorted(glob.glob(R + '/batches/*.json')):
        j = json.load(open(f)); u = j.get('usage') or {}
        st_ = j.get('skill_status') or {}
        rows[j['batch_index']] = dict(
            pairs=j.get('pair_count') or 0,
            val=sum(1 for v in st_.values() if v == 'validated'),
            cand=sum(1 for v in st_.values() if v == 'candidate'),
            tin=u.get('input_tokens', 0), tout=u.get('output_tokens', 0),
            calls=u.get('model_calls', 0), turns=u.get('agent_turns', 0),
            traj=j.get("trajectories") if isinstance(j.get("trajectories"), int) else len(j.get("trajectories") or []),
            mt=os.path.getmtime(f))
ks = sorted(rows)
print('%-10s %4s %7s %6s %6s %10s %9s %8s %9s %9s' % (
    'steps','n','sec/step','pairs','val','in_tok','out_tok','calls','in/call','sec/call'))
for a in range(0, 260, 15):
    sel = [k for k in ks if a <= k < a+15]
    if len(sel) < 3: continue
    secs = [rows[k]['mt']-rows[k-1]['mt'] for k in sel if k-1 in rows]
    if not secs: continue
    g = lambda f: st.mean(rows[k][f] for k in sel)
    print('%-10s %4d %7.0f %6.1f %6.1f %10.0f %9.0f %8.0f %9.0f %9.2f' % (
        '%d-%d'%(a,a+14), len(sel), st.mean(secs), g('pairs'), g('val'),
        g('tin'), g('tout'), g('calls'), g('tin')/max(g('calls'),1),
        st.mean(secs)/max(g('calls'),1)))
