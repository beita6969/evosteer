"""Did the orchestrator's decision prompt grow with the skill library?"""
import json, glob, os, sys, statistics as st
V = sys.argv[1]
dirs = []
p = f'/workspace/evosteer_runs/active_9b_paper_{V}_prev.out'
if os.path.exists(p): dirs.append(open(p).read().strip())
dirs.append(open(f'/workspace/evosteer_runs/active_9b_paper_{V}.out').read().strip())
files = sorted(g for R in dirs for g in glob.glob(R + '/trajectories/*.jsonl'))
# sample every Nth step file to keep this cheap
step_of = lambda f: int(f.rsplit('-', 1)[1].split('.')[0])
sel = [f for f in files if step_of(f) % 12 == 1]
print('%-8s %9s %12s %12s %10s %9s' % ('step','decisions','prompt mean','prompt p90','dec/ep','skills'))
skills = {}
for R in dirs:
    for b in sorted(glob.glob(R + '/batches/*.json')):
        j = json.load(open(b)); skills[j['batch_index']] = len(j.get('skill_status') or {})
for f in sel:
    n = step_of(f); lens = []; eps = 0
    for line in open(f):
        r = json.loads(line)
        d = r.get('decisions') or []
        if not d: continue
        eps += 1
        for x in d:
            pi = x.get('prompt_ids')
            if isinstance(pi, list): lens.append(len(pi))
    if len(lens) < 20: continue
    lens.sort()
    print('%-8d %9d %12.0f %12.0f %10.2f %9s' % (
        n, len(lens), st.mean(lens), lens[int(.9*len(lens))], len(lens)/max(eps,1),
        skills.get(n, '?')))
