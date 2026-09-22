import json, glob, os, sys, statistics as st
V = sys.argv[1]
dirs = []
p = f'/workspace/evosteer_runs/active_9b_paper_{V}_prev.out'
if os.path.exists(p): dirs.append(open(p).read().strip())
dirs.append(open(f'/workspace/evosteer_runs/active_9b_paper_{V}.out').read().strip())
rows = {}
for R in dirs:
    for f in sorted(glob.glob(R + '/batches/*.json')):
        j = json.load(open(f)); u = j.get('usage') or {}
        rows[j['batch_index']] = (j.get('pair_count') or 0, len(j.get('skill_status') or {}),
                                  u.get('input_tokens', 0), u.get('output_tokens', 0),
                                  u.get('model_calls', 0), os.path.getmtime(f))
ks = sorted(rows)
print('%-12s %4s %6s %7s %10s %9s %8s %9s' % ('steps','n','pairs','skills','in_tok','out_tok','calls','sec/step'))
for a in range(0, 260, 15):
    sel = [k for k in ks if a <= k < a+15]
    if len(sel) < 3: continue
    v = [rows[k] for k in sel]
    secs = [rows[k][5] - rows[k-1][5] for k in sel if k-1 in rows]
    print('%-12s %4d %6.1f %7.1f %10.0f %9.0f %8.1f %9.0f' % (
        '%d-%d' % (a, a+14), len(sel), st.mean(x[0] for x in v), st.mean(x[1] for x in v),
        st.mean(x[2] for x in v), st.mean(x[3] for x in v), st.mean(x[4] for x in v),
        st.mean(secs) if secs else float('nan')))
