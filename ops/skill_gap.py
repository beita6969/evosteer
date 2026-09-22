"""Per-step reward of the arms that DO and DO NOT get the skill, on the same tasks.

paired_treatment: forced first ADD_AGENT(c1, sigma), continued by rho  -> "with skill"
paired_control  : forced first ADD_AGENT(c1, empty), continued by rho  -> "without skill"
Also prints pi_theta (natural, with the whole menu available) for reference.
"""
import json, glob, sys, collections, math
run = sys.argv[1]
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for path in sorted(glob.glob(run + "/trajectories/batch-*.jsonl")):
    step = int(path.rsplit("-", 1)[1][:6])
    for line in open(path):
        t = json.loads(line)
        rows[step][t["source"]].append((t["task"]["task_id"], t["reward"]))
print("step  pi_theta  rho_natural  with_skill  without_skill  paired_diff(same task)")
cum = []
for step in sorted(rows):
    r = rows[step]
    mean = lambda k: (sum(v for _, v in r[k]) / len(r[k])) if r.get(k) else float("nan")
    treat = {k: v for k, v in r.get("paired_treatment", [])}
    control = {k: v for k, v in r.get("paired_control", [])}
    common = sorted(set(treat) & set(control))
    d = [treat[k] - control[k] for k in common]
    cum += d
    print(f"{step:4d}  {mean('current'):8.3f}  {mean('natural_reference'):11.3f}  "
          f"{mean('paired_treatment'):10.3f}  {mean('paired_control'):13.3f}  "
          f"{(sum(d)/len(d)) if d else float('nan'):+.3f} (n={len(d)})")
if cum:
    m = sum(cum) / len(cum)
    se = math.sqrt(sum((x - m) ** 2 for x in cum) / max(1, len(cum) - 1)) / math.sqrt(len(cum))
    w = sum(1 for x in cum if x > 0); l = sum(1 for x in cum if x < 0)
    p = sum(math.comb(w + l, i) for i in range(w, w + l + 1)) / 2 ** (w + l) if w + l else 1.0
    print(f"pooled with-skill minus without-skill: {m:+.4f} ± {se:.4f}  W={w} L={l} T={len(cum)-w-l} sign p={p:.5f}")
