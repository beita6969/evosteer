import json, glob, collections
rows = []
for d in ("/workspace/evosteer_runs/27b_paper_v6_20260919-200820", "/workspace/evosteer_runs/27b_paper_v6p_20260919-211824"):
    for f in sorted(glob.glob(d + "/trajectories/*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
tasks = collections.defaultdict(list)
for t in rows:
    if t["source"] in ("current", "natural_reference"):
        tasks[(t["batch_id"], t["task"]["family"], t["task"]["task_id"])].append(t["reward"])
fam = collections.defaultdict(lambda: {"n": 0, "mean": [], "solved": 0, "failed": 0, "mixed": 0, "trunc": 0})
for (b, f, tid), rs in tasks.items():
    x = fam[f]; x["n"] += 1; x["mean"].append(sum(rs) / len(rs))
    if min(rs) >= 0.99: x["solved"] += 1
    elif max(rs) <= 0.01: x["failed"] += 1
    else: x["mixed"] += 1
aime_trunc = sum(1 for t in rows if t["task"]["family"] == "aime_2026" and t["source"] in ("current", "natural_reference") and "boxed" not in t["output"])
aime_all = sum(1 for t in rows if t["task"]["family"] == "aime_2026" and t["source"] in ("current", "natural_reference"))
print(f"{'family':10s} tasks  mean   all-4-solved  all-4-failed  mixed(signal)")
for f, x in sorted(fam.items()):
    print(f"{f:10s} {x['n']:5d}  {sum(x['mean'])/x['n']:.2f}   {x['solved']:4d} ({x['solved']/x['n']:.0%})   {x['failed']:4d} ({x['failed']/x['n']:.0%})   {x['mixed']:4d} ({x['mixed']/x['n']:.0%})")
print(f"AIME outputs with no \\boxed answer (truncated/unfinished): {aime_trunc}/{aime_all}")
