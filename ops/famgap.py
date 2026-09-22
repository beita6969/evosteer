"""Per-family: pi vs rho on the SAME tasks in the same step (isolates policy from orchestration)."""
import json, glob, sys, statistics as st
from collections import defaultdict
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 10
cur, ref = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
for d in sorted(glob.glob(sys.argv[1])):
    for f in sorted(glob.glob(d + "/trajectories/*.jsonl")):
        idx = int(f.rsplit("-", 1)[1].split(".")[0])
        if idx < lo: continue
        for line in open(f):
            r = json.loads(line)
            t = r.get("task") or {}
            fam, tid, rw = t.get("family"), t.get("task_id"), r.get("reward")
            if rw is None or not fam: continue
            key = (idx, tid)
            if r.get("source") == "current": cur[fam][key].append(rw)
            elif r.get("source") == "natural_reference": ref[fam][key].append(rw)
print(f"{'family':12s} {'n_pairs':>8s} {'pi':>7s} {'rho':>7s} {'pi-rho':>8s} {'se':>7s} {'t':>6s}")
allд = []
for fam in sorted(cur):
    keys = set(cur[fam]) & set(ref[fam])
    d = [st.mean(cur[fam][k]) - st.mean(ref[fam][k]) for k in keys]
    if not d: continue
    allд += d
    se = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0
    p = st.mean([st.mean(cur[fam][k]) for k in keys]); q = st.mean([st.mean(ref[fam][k]) for k in keys])
    print(f"{fam:12s} {len(d):8d} {p:7.3f} {q:7.3f} {st.mean(d):+8.4f} {se:7.4f} {(st.mean(d)/se if se else 0):+6.2f}")
se = st.stdev(allд) / len(allд) ** 0.5
print(f"{'POOLED':12s} {len(allд):8d} {'':7s} {'':7s} {st.mean(allд):+8.4f} {se:7.4f} {st.mean(allд)/se:+6.2f}")
