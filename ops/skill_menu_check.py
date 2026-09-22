import json, sys, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/workspace/models/Qwen3.8-27B-FP8")
run, batch = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(f"{run}/trajectories/{batch}.jsonl")]
legal = collections.Counter(); chosen = collections.Counter(); menus_with_skill = collections.Counter(); decisions = collections.Counter()
for t in rows:
    if t["source"] not in ("current", "natural_reference"): continue
    for d in t["decisions"]:
        acts = [json.loads(tok.decode(p[:-1])) for p in d["legal_token_paths"]]
        decisions[t["source"]] += 1
        has = [a for a in acts if a.get("skill_id") or a["kind"] == "BIND_SKILL"]
        if has: menus_with_skill[t["source"]] += 1
        for a in acts: legal[(t["source"], a["kind"], bool(a.get("skill_id")))] += 1
        c = json.loads(d["action_json"]); chosen[(t["source"], c["kind"], bool(c.get("skill_id")))] += 1
print("decisions:", dict(decisions), "| decisions whose menu offers a skill:", dict(menus_with_skill))
print("legal options (source, kind, carries skill):"); [print("  ", k, v) for k, v in sorted(legal.items())]
print("chosen:"); [print("  ", k, v) for k, v in sorted(chosen.items())]
st = json.loads(rows[0]["decisions"][0]["state_json"])
print("state keys:", list(st)[:30])
print("skill info in state:", {k: str(v)[:300] for k, v in st.items() if "skill" in k.lower()})
