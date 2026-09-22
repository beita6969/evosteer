import json, glob, collections
rows = []
for d in ("/workspace/evosteer_runs/27b_paper_v6_20260919-200820", "/workspace/evosteer_runs/27b_paper_v6p_20260919-211824"):
    for f in sorted(glob.glob(d + "/trajectories/*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
def uses(t): return any(json.loads(x["action_json"]).get("skill_id") for x in t["decisions"])
# 1) paired trials: treatment vs control reward by family
pairs = collections.defaultdict(list)
by_pair = collections.defaultdict(dict)
for t in rows:
    if t["source"] in ("paired_treatment", "paired_control"):
        by_pair[(t["batch_id"], t["pair_id"])][t["source"]] = t
for (b, pid), arms in by_pair.items():
    if len(arms) == 2:
        fam = arms["paired_treatment"]["task"]["family"]
        pairs[fam].append((arms["paired_treatment"]["reward"], arms["paired_control"]["reward"]))
print("PAIRED TRIALS (skill vs no skill, same task & first role):")
for fam, v in sorted(pairs.items()):
    w = sum(a > b for a, b in v); l = sum(a < b for a, b in v); t = len(v) - w - l
    print(f"  {fam:10s} n={len(v):2d} mean skill {sum(a for a,_ in v)/len(v):.2f} vs no-skill {sum(b for _,b in v)/len(v):.2f} | W{w} L{l} T{t}")
# 2) natural rollouts: within-task comparison of episodes with vs without a skill
eff = collections.defaultdict(list)
tasks = collections.defaultdict(list)
for t in rows:
    if t["source"] in ("current", "natural_reference"):
        tasks[(t["batch_id"], t["task"]["task_id"])].append(t)
for (b, tid), ts in tasks.items():
    w = [t["reward"] for t in ts if uses(t)]; wo = [t["reward"] for t in ts if not uses(t)]
    if w and wo: eff[ts[0]["task"]["family"]].append(sum(w)/len(w) - sum(wo)/len(wo))
print("NATURAL ROLLOUTS, within-task (with skill − without):")
for fam, v in sorted(eff.items()): print(f"  {fam:10s} tasks={len(v):2d} mean diff {sum(v)/len(v):+.3f}")
