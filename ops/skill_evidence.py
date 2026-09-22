"""Paired A/B evidence per candidate skill, from the latest checkpoint's admission ledger."""
import json, glob
R = open("/workspace/evosteer_runs/active_9b_paper_v8.out").read().strip()
ckpt = sorted(glob.glob(R + "/checkpoints/batch-*"))[-1]
state = json.load(open(ckpt + "/state.json"))
def find(node, key, out):
    if isinstance(node, dict):
        if key in node and isinstance(node[key], list):
            out.extend(node[key])
        for v in node.values():
            find(v, key, out)
    elif isinstance(node, list):
        for v in node:
            find(v, key, out)
comparisons, skills = [], []
find(state, "comparisons", comparisons)
find(state, "skills", skills)
family = {}
for s in skills:
    entry = s.get("entry", s)
    if isinstance(entry, dict) and entry.get("skill_id"):
        family[entry["skill_id"]] = (entry.get("family"), s.get("status", "?"))
print(ckpt.rsplit("/", 1)[1], "comparisons:", len(comparisons))
seen = set()
for c in comparisons:
    key = c.get("comparison_id")
    if key in seen:
        continue
    seen.add(key)
    sid = c.get("skill_id", "?")
    fam, status = family.get(sid, ("?", "?"))
    w, l, t = c.get("wins", 0), c.get("losses", 0), c.get("ties", 0)
    print(f"  {str(fam):10s} {sid[-6:]} status={status:9s} W={w:2d} L={l:2d} T={t:2d} closed={c.get('closed')}")
