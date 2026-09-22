import json, sys
run, fams, n = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
batch = sys.argv[4] if len(sys.argv) > 4 else "batch-000001"
data = {json.loads(l)["task_id"]: json.loads(l) for l in open("/workspace/evosteer/SKILLEV-new-main/data/paper_benchmarks/paper_benchmarks.jsonl")}
rows = [json.loads(l) for l in open(f"{run}/trajectories/{batch}.jsonl")]
for fam in fams:
    for t in [t for t in rows if t["task"]["family"] == fam][:n]:
        g = data[t["task"]["task_id"]]
        gold = g.get("answers") or g.get("answer")
        kinds = [json.loads(d["action_json"])["kind"] for d in t["decisions"]]
        print(f"[{fam}] {t['source'][:4]} r={t['reward']} gold={gold} actions={kinds}")
        print(f"   out={t['output'][-260:]!r}")
