"""Per-step rollout duration distribution: is the batch gated by the slowest episode?"""
import json, sys, glob, os, statistics as st
from collections import defaultdict
per = defaultdict(list)          # batch -> [(seconds, family, nactions, source)]
for path in sys.argv[1:]:
    if not os.path.exists(path): continue
    for line in open(path, errors='ignore'):
        if '"event": "rollout"' not in line: continue
        try: r = json.loads(line)
        except Exception: continue
        sid = r.get('sample_id','')
        if not sid.startswith('batch-'): continue
        try: n = int(sid.split('/')[0].split('-')[1])
        except Exception: continue
        s = r.get('seconds')
        if s is None: continue
        per[n].append((s, r.get('family'), len(r.get('actions') or ()), r.get('source')))
print('%6s %6s %8s %8s %8s %8s %8s %9s' % ('step','n','sum_s','p50','p90','max','mean','max/p50'))
rows=[]
for n in sorted(per):
    v=sorted(x[0] for x in per[n])
    if len(v)<20: continue
    p50=v[len(v)//2]; p90=v[int(.9*len(v))]; mx=v[-1]
    rows.append((n,len(v),sum(v),p50,p90,mx,st.mean(v)))
for r in rows[-24:]:
    print('%6d %6d %8.0f %8.1f %8.1f %8.1f %8.1f %9.1f' % (r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[5]/max(r[3],1e-9)))
# serial-vs-parallel: if perfectly parallel, step_time ~= max; if serial, ~= sum
print()
print('LAST 10 STEPS: mean sum_s=%.0f  mean max=%.0f  (step wall from pace ~= see pace.py)'
      % (st.mean(r[2] for r in rows[-10:]), st.mean(r[5] for r in rows[-10:])))
# family breakdown of the tail, unconditional
fam=defaultdict(list)
for n in sorted(per)[-15:]:
    for s,f,a,src in per[n]: fam[f].append(s)
print()
print('%-12s %7s %8s %8s %8s %8s' % ('family','n','mean','p50','p90','max'))
for f in sorted(fam, key=lambda k:-st.mean(fam[k])):
    v=sorted(fam[f])
    print('%-12s %7d %8.1f %8.1f %8.1f %8.1f' % (f,len(v),st.mean(v),v[len(v)//2],v[int(.9*len(v))],v[-1]))
