"""Full method (pi_theta, skills available) vs a no-skill reference, on the same tasks per step.

no-skill reference = paired_control arm (forced first agent, candidate slot empty, rho continues).
Warmup steps have no pairs, so the series starts when skills appear.
"""
import json, glob, sys, collections, math
run = sys.argv[1]
per = collections.defaultdict(lambda: collections.defaultdict(list))
for path in sorted(glob.glob(run + "/trajectories/batch-*.jsonl")):
    step = int(path.rsplit("-", 1)[1][:6])
    for line in open(path):
        t = json.loads(line)
        per[step][t["source"]].append((t["task"]["task_id"], t["reward"]))
diffs_all = []
print("step   pi_theta   no-skill ref   gap (same tasks)")
for step in sorted(per):
    cur = collections.defaultdict(list)
    for task, reward in per[step].get("current", []):
        cur[task].append(reward)
    ctl = collections.defaultdict(list)
    for task, reward in per[step].get("paired_control", []):
        ctl[task].append(reward)
    common = sorted(set(cur) & set(ctl))
    if not common:
        continue
    d = [sum(cur[k]) / len(cur[k]) - sum(ctl[k]) / len(ctl[k]) for k in common]
    diffs_all += d
    print(f"{step:4d}   {sum(sum(cur[k])/len(cur[k]) for k in common)/len(common):8.3f}   "
          f"{sum(sum(ctl[k])/len(ctl[k]) for k in common)/len(common):12.3f}   {sum(d)/len(d):+.3f} (n={len(d)})")
if diffs_all:
    m = sum(diffs_all) / len(diffs_all)
    se = math.sqrt(sum((x - m) ** 2 for x in diffs_all) / max(1, len(diffs_all) - 1)) / math.sqrt(len(diffs_all))
    print(f"pooled pi_theta minus no-skill reference: {m:+.4f} ± {se:.4f} over {len(diffs_all)} task-steps")
