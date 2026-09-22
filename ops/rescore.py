"""Offline: what is the '\boxed{} outranks ####' grader fix worth on data we already have?"""
import json, glob, sys, re
sys.path.insert(0, "/workspace/evosteer/SKILLEV-new-main/src")
from decimal import Decimal, InvalidOperation
from skillev.experiments.curve_benchmarks import _last_number, _NUMBER, _MATH_CUE

gold_by_prompt = {}
for mf in glob.glob("/workspace/evosteer/SKILLEV-new-main/data/paper_benchmarks/paper_benchmarks_9b_mixed*.jsonl"):
    for line in open(mf):
        rec = json.loads(line)
        gold_by_prompt[rec["prompt"].strip()] = (rec.get("kind"), rec.get("answer"))

def predict(output, boxed_first):
    pats = (r"\\boxed\{([^{}]+)\}", r"####\s*([^\n]+)") if boxed_first else (r"####\s*([^\n]+)", r"\\boxed\{([^{}]+)\}")
    for pattern in pats:
        spans = re.findall(pattern, output)
        if spans:
            v = _last_number(spans[-1])
            if v is not None: return v
    cues = list(_MATH_CUE.finditer(output))
    if cues:
        m = _NUMBER.search(output[cues[-1].end():][:200].replace("{,}", ","))
        if m:
            try: return Decimal(m.group(0).replace(",", ""))
            except InvalidOperation: pass
    return _last_number(output)

tot, miss, hashed = {}, 0, 0
for d in sorted(glob.glob(sys.argv[1])):
    for f in sorted(glob.glob(d + "/trajectories/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            t = r.get("task")
            if not isinstance(t, dict) or t.get("family") != "aime_2026": continue
            hit = gold_by_prompt.get((t.get("prompt") or "").strip())
            if not hit: miss += 1; continue
            try: g = Decimal(str(hit[1]).strip())
            except InvalidOperation: continue
            o = r.get("output") or ""
            if "####" in o: hashed += 1
            src = r.get("source") or "?"
            old = float(predict(o, False) == g); new = float(predict(o, True) == g)
            a = tot.setdefault(src, [0, 0.0, 0.0, 0, 0])
            a[0] += 1; a[1] += old; a[2] += new; a[3] += (new > old); a[4] += (new < old)
print(f"unmatched prompts: {miss} | outputs containing '####': {hashed}")
print(f"{'arm':20s} {'n':>6s} {'old':>7s} {'new':>7s} {'delta':>8s} {'rescued':>8s} {'broken':>7s}")
for src, (n, o, nw, resc, brk) in sorted(tot.items()):
    print(f"{src:20s} {n:6d} {o/n:7.4f} {nw/n:7.4f} {(nw-o)/n:+8.4f} {resc:8d} {brk:7d}")
