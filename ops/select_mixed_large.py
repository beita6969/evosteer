"""Training pool for the 9B run: tasks the frozen 9B solver gets right on some but not all of 8 samples.

Disclosed deviation: difficulty filtering of the TRAINING pool from an offline solver-only
pilot (pilot_9b.json); groups that are always right or always wrong carry no learning signal.
"""
import json, collections
pilot = json.load(open("/workspace/evosteer/pilot_9b_large.json"))
by = collections.defaultdict(list)
for r in pilot["results"]:
    by[r["task_id"]].append(r["reward"])
p = {t: sum(v) / len(v) for t, v in by.items()}
keep = {t for t, x in p.items() if 0 < x < 1}
src = "/workspace/evosteer/SKILLEV-new-main/data/paper_benchmarks/paper_benchmarks_large.jsonl"
dst = "/workspace/evosteer/SKILLEV-new-main/data/paper_benchmarks/paper_benchmarks_9b_mixed_large.jsonl"
rows = [json.loads(l) for l in open(src) if l.strip()]
out = [r for r in rows if r.get("split") == "train" and r["task_id"] in keep]
with open(dst, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
json.dump({"pilot_samples": pilot["samples"], "pilot_cap": pilot["cap"], "rule": "0 < p_hat < 1",
           "p_hat": {t: p[t] for t in sorted(keep)},
           "task_sources": {r["task_id"]: r["kind"] for r in out}},
          open("/workspace/evosteer/pool_9b_mixed_large.json", "w"), indent=1)
c = collections.Counter(r["kind"] for r in out)
print(len(out), dict(c), "mean p_hat", round(sum(p[t] for t in keep) / len(keep), 3))
