import json, glob, collections
R = open("/workspace/evosteer_runs/active_9b_paper_v8.out").read().strip()
by_out = collections.defaultdict(list); by_first = collections.defaultdict(list); per_batch = collections.defaultdict(lambda: collections.Counter())
for path in sorted(glob.glob(R + "/trajectories/batch-*.jsonl")):
    b = path.rsplit("-", 1)[1][:6]
    for line in open(path):
        t = json.loads(line)
        if t["source"] not in ("current", "natural_reference"): continue
        acts = [json.loads(d["action_json"]) for d in t["decisions"]]
        roles = {a["node_id"]: a.get("role_id") for a in acts if a["kind"] == "ADD_AGENT"}
        outs = [a["node_id"] for a in acts if a["kind"] == "SET_OUTPUT"]
        out_role = roles.get(outs[-1]) if outs else None
        first = next((a.get("role_id") for a in acts if a["kind"] == "ADD_AGENT"), None)
        by_out[out_role].append(t["reward"]); by_first[first].append(t["reward"])
        per_batch[(b, t["source"])][out_role] += 1
print("reward by OUTPUT node role:")
for k, v in sorted(by_out.items(), key=lambda x: -len(x[1])): print(f"  {str(k):10s} n={len(v):4d} mean={sum(v)/len(v):.3f}")
print("reward by FIRST agent role:")
for k, v in sorted(by_first.items(), key=lambda x: -len(x[1])): print(f"  {str(k):10s} n={len(v):4d} mean={sum(v)/len(v):.3f}")
print("output-role share of pi (current) per batch:")
for (b, s), c in sorted(per_batch.items()):
    if s == "current":
        n = sum(c.values()); print(" ", b, {k: round(v / n, 2) for k, v in c.most_common()})
