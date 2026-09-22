import json, glob
d = "/workspace/evosteer_runs/27b_paper_v6p_20260919-211824"
ck = sorted(glob.glob(d + "/checkpoints/batch-*"))[-1]
s = json.load(open(ck + "/state.json"))
for sk in s["admission"]["state"]["skills"]:
    body = json.loads(sk["entry"]["body"])
    print(f"\n### {sk['entry']['family']} [{sk['status']}] {body['name']}")
    print("  trigger:", str(body["trigger"])[:220])
    print("  plan:", json.dumps(body["plan"], ensure_ascii=False)[:500])
    print("  constraint:", str(body["constraint"])[:220])
rows = [json.loads(l) for f in sorted(glob.glob(d + "/trajectories/*.jsonl")) for l in open(f)]
arms = {}
for t in rows:
    if t["source"].startswith("paired"): arms.setdefault(t["pair_id"], {})[t["source"]] = t
shown = 0
for pid, a in arms.items():
    if len(a) == 2 and a["paired_treatment"]["reward"] < a["paired_control"]["reward"] and shown < 2:
        shown += 1
        for src in ("paired_treatment", "paired_control"):
            t = a[src]
            print(f"\n--- LOSS {t['task']['family']} {src} r={t['reward']} actions={[json.loads(x['action_json'])['kind'] + ('+skill' if json.loads(x['action_json']).get('skill_id') else '') for x in t['decisions']]}")
            print("    output tail:", repr(t["output"][-400:]))
