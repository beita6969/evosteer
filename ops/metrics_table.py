import json, sys
R = open("/workspace/evosteer_runs/active_9b_paper_v8.out").read().strip()
rows = [json.loads(l) for l in open(R + "/metrics.jsonl")]
keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
for k in keys:
    vals = []
    for r in rows:
        v = r.get(k)
        vals.append(f"{v:>9.4f}" if isinstance(v, float) else f"{str(v):>9s}")
    print(f"{k:32s}", " ".join(vals))
