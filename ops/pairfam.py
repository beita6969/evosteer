"""Paired treatment(with skill) vs control(no skill) per family, with the sign-test tally."""
import json, glob, sys, statistics as st
from collections import defaultdict
tr, ct = defaultdict(dict), defaultdict(dict)
fam_of = {}
for d in sorted(glob.glob(sys.argv[1])):
    for f in sorted(glob.glob(d + "/trajectories/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            pid, src, rw = r.get("pair_id"), r.get("source"), r.get("reward")
            if not pid or rw is None: continue
            fam_of[pid] = (r.get("task") or {}).get("family")
            if src == "paired_treatment": tr[fam_of[pid]][pid] = rw
            elif src == "paired_control": ct[fam_of[pid]][pid] = rw
print(f"{'family':12s} {'n':>5s} {'W':>4s} {'L':>4s} {'T':>4s} {'effect':>9s} {'se':>7s}")
tot = []
for fam in sorted(tr):
    keys = set(tr[fam]) & set(ct[fam])
    d = [tr[fam][k] - ct[fam][k] for k in keys]
    if not d: continue
    tot += d
    W = sum(1 for x in d if x > 0); L = sum(1 for x in d if x < 0); T = len(d) - W - L
    se = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0
    print(f"{fam:12s} {len(d):5d} {W:4d} {L:4d} {T:4d} {st.mean(d):+9.4f} {se:7.4f}")
W = sum(1 for x in tot if x > 0); L = sum(1 for x in tot if x < 0)
se = st.stdev(tot) / len(tot) ** 0.5
print(f"{'POOLED':12s} {len(tot):5d} {W:4d} {L:4d} {len(tot)-W-L:4d} {st.mean(tot):+9.4f} {se:7.4f}")
