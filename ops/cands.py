"""Per-candidate paired effect with task-clustered CI, across every run."""
import json, glob, sys, random, statistics as st
from collections import defaultdict
random.seed(7)
tr, ct, meta = {}, {}, {}
for d in sorted(glob.glob(sys.argv[1])):
    run = d.rsplit("/", 1)[1]
    for f in sorted(glob.glob(d + "/trajectories/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            pid, src, rw = r.get("pair_id"), r.get("source"), r.get("reward")
            if not pid or rw is None: continue
            t = r.get("task") or {}
            cid = r.get("candidate_id")
            if src == "paired_treatment":
                tr[pid] = rw
                if cid: meta[pid] = (cid, t.get("family"), t.get("task_id"), run)
            elif src == "paired_control": ct[pid] = rw
by = defaultdict(lambda: defaultdict(list))
info = {}
for pid in set(tr) & set(ct):
    if pid not in meta: continue
    cid, fam, tid, run = meta[pid]
    by[cid][tid].append(tr[pid] - ct[pid])
    info[cid] = (fam, run)
out = []
for cid, tm in by.items():
    tasks = list(tm)
    if len(tasks) < 3: continue
    obs = st.mean([st.mean(tm[t]) for t in tasks])
    B, means = 4000, []
    for _ in range(B):
        s = [tm[tasks[random.randrange(len(tasks))]] for _ in tasks]
        means.append(st.mean([st.mean(x) for x in s]))
    means.sort()
    npair = sum(len(v) for v in tm.values())
    W = sum(1 for v in tm.values() if st.mean(v) > 0)
    L = sum(1 for v in tm.values() if st.mean(v) < 0)
    out.append(dict(cid=cid[:22], family=info[cid][0], run=info[cid][1], effect=round(obs, 4),
                    lo=round(means[int(.025*B)], 4), hi=round(means[int(.975*B)], 4),
                    pairs=npair, tasks=len(tasks), W=W, L=L))
out.sort(key=lambda r: r["effect"])
json.dump(out, open(sys.argv[2], "w"))
print(f"candidates with >=3 tasks: {len(out)}")
for r in out:
    star = "*" if (r["lo"] > 0 or r["hi"] < 0) else " "
    print(f"  {star} {r['run'][:22]:22s} {r['family']:10s} {r['effect']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]  {r['pairs']:4d}p/{r['tasks']:3d}t  {r['W']}W/{r['L']}L")
