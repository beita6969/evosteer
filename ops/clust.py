"""Paired skill effect with TASK-CLUSTERED uncertainty (cluster bootstrap + one-vote-per-task)."""
import json, glob, sys, random, statistics as st
from collections import defaultdict
random.seed(12345)
tr, ct, fam = {}, {}, {}
for d in sorted(glob.glob(sys.argv[1])):
    for f in sorted(glob.glob(d + "/trajectories/*.jsonl")):
        idx = int(f.rsplit("-", 1)[1].split(".")[0])
        if idx > int(sys.argv[2]): continue          # fixed common step
        for line in open(f):
            r = json.loads(line)
            pid, src, rw = r.get("pair_id"), r.get("source"), r.get("reward")
            if not pid or rw is None: continue
            t = r.get("task") or {}
            fam[pid] = (t.get("family"), t.get("task_id"))
            if src == "paired_treatment": tr[pid] = rw
            elif src == "paired_control": ct[pid] = rw
by_task = defaultdict(list); by_fam_task = defaultdict(lambda: defaultdict(list))
for pid in set(tr) & set(ct):
    f, tid = fam[pid]
    by_task[tid].append(tr[pid] - ct[pid]); by_fam_task[f][tid].append(tr[pid] - ct[pid])

def cluster_boot(task_map, B=20000):
    tasks = list(task_map)
    if len(tasks) < 2: return None, None, None
    obs = st.mean([st.mean(task_map[t]) for t in tasks])
    ge = 0; means = []
    for _ in range(B):
        samp = [task_map[tasks[random.randrange(len(tasks))]] for _ in tasks]
        m = st.mean([st.mean(s) for s in samp]); means.append(m)
        if m <= 0: ge += 1
    means.sort()
    return obs, (means[int(.025*B)], means[int(.975*B)]), (ge + 1) / (B + 1)

print(f"=== {sys.argv[3]}  (pairs up to step {sys.argv[2]}) ===")
print(f"{'family':12s} {'tasks':>6s} {'pairs':>6s} {'effect':>9s} {'clustered 95% CI':>24s} {'p_clu':>8s} {'1vote W/L':>10s}")
for f in sorted(by_fam_task):
    tm = by_fam_task[f]
    npairs = sum(len(v) for v in tm.values())
    obs, ci, p = cluster_boot(tm)
    votes = [st.mean(v) for v in tm.values()]
    W = sum(1 for v in votes if v > 0); L = sum(1 for v in votes if v < 0)
    print(f"{f:12s} {len(tm):6d} {npairs:6d} {obs:+9.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]   {p:8.4f} {W:4d}/{L:<4d}")
obs, ci, p = cluster_boot(by_task)
npairs = sum(len(v) for v in by_task.values())
votes = [st.mean(v) for v in by_task.values()]
W = sum(1 for v in votes if v > 0); L = sum(1 for v in votes if v < 0)
naive = [x for v in by_task.values() for x in v]
nse = st.stdev(naive) / len(naive) ** 0.5
print(f"{'POOLED':12s} {len(by_task):6d} {npairs:6d} {obs:+9.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]   {p:8.4f} {W:4d}/{L:<4d}")
print(f"  naive (treats every pair as independent): {st.mean(naive):+.4f} +- {nse:.4f}   t={st.mean(naive)/nse:+.2f}   n={len(naive)}")
