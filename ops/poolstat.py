import json, glob, sys, statistics as st
from collections import defaultdict
rho = defaultdict(list); cur = defaultdict(list); fam = {}
for d in sorted(glob.glob(sys.argv[1])):
    for f in sorted(glob.glob(d + "/trajectories/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            t = r.get("task"); tid = t.get("task_id") if isinstance(t, dict) else t
            if not tid or r.get("reward") is None: continue
            fam[tid] = tid.split("/")[0]
            if r.get("source") == "natural_reference": rho[tid].append(r["reward"])
            elif r.get("source") == "current": cur[tid].append(r["reward"])
ns = sorted(len(v) for v in rho.values())
print("tasks with rho evidence:", len(rho), "| rho obs per task: min", ns[0], "median", ns[len(ns)//2], "max", ns[-1])
lo, hi = float(sys.argv[2]), float(sys.argv[3])
keep = [t for t, v in rho.items() if len(v) >= 8 and lo <= sum(v)/len(v) <= hi]
byfam = defaultdict(int)
for t in keep: byfam[fam[t]] += 1
allfam = defaultdict(int)
for t in rho: allfam[fam[t]] += 1
print(f"mid-band [{lo},{hi}] with >=8 rho obs: {len(keep)} of {len(rho)} tasks")
print("  kept per family:", dict(byfam))
print("  total per family:", dict(allfam))
# informative fraction: fraction of 2-rollout current cells with nonzero variance, on kept vs all
def infofrac(ts):
    v = [sum(cur[t])/len(cur[t]) for t in ts if len(cur[t]) >= 4]
    return round(st.mean([4*p*(1-p) for p in v]), 3) if v else None
print("  expected within-task variance factor 4p(1-p): kept", infofrac(keep), "| all", infofrac(list(rho)))
