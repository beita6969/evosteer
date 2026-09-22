"""Per-step mean reward of every rollout arm, written as JSON for plotting.

current           : pi_theta, full menu
natural_reference : rho, full menu
paired_treatment  : rho, forced first agent WITH the candidate skill
paired_control    : rho, forced first agent WITHOUT it (the no-skill reference)
"""
import json, glob, sys, collections
out = {}
for run in sys.argv[2:]:
    for path in sorted(glob.glob(run + "/trajectories/batch-*.jsonl")):
        step = int(path.rsplit("-", 1)[1][:6])
        arms = collections.defaultdict(list)
        for line in open(path):
            t = json.loads(line)
            arms[t["source"]].append(t["reward"])
        out[step] = {k: sum(v) / len(v) for k, v in arms.items()}
json.dump([{"step": s, **out[s]} for s in sorted(out)], open(sys.argv[1], "w"))
print("steps", min(out), "-", max(out), "arms", sorted({k for v in out.values() for k in v}))
