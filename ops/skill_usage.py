import json, glob, sys, collections
run = sys.argv[1]
rows = collections.defaultdict(lambda: {"n": 0, "skill": 0, "r_skill": [], "r_plain": []})
paired = {"treat": [], "control": []}
for path in sorted(glob.glob(run + "/trajectories/batch-*.jsonl")):
    step = int(path.rsplit("-", 1)[1][:6])
    for line in open(path):
        t = json.loads(line)
        acts = [json.loads(d["action_json"]) for d in t["decisions"]]
        bound = any(a.get("skill_id") for a in acts)
        if t["source"] == "current":
            row = rows[step]; row["n"] += 1; row["skill"] += bound
            (row["r_skill"] if bound else row["r_plain"]).append(t["reward"])
        elif t["source"] == "paired_treatment": paired["treat"].append((t["task"]["task_id"], step, t["reward"]))
        elif t["source"] == "paired_control": paired["control"].append((t["task"]["task_id"], step, t["reward"]))
print("step  n  bound  reward(bound)  reward(no skill)")
for step in sorted(rows):
    r = rows[step]
    m = lambda v: f"{sum(v)/len(v):.3f}({len(v)})" if v else "   -   "
    print(f"{step:4d} {r['n']:3d}  {r['skill']/r['n']:.2f}   {m(r['r_skill']):>14s} {m(r['r_plain']):>14s}")
t = {(a, b): c for a, b, c in paired["treat"]}; c = {(a, b): v for a, b, v in paired["control"]}
common = sorted(set(t) & set(c))
if common:
    d = [t[k] - c[k] for k in common]
    wins = sum(1 for x in d if x > 0); losses = sum(1 for x in d if x < 0)
    print(f"paired trials: n={len(d)} mean(skill - no skill) = {sum(d)/len(d):+.4f}  wins {wins} losses {losses} ties {len(d)-wins-losses}")
