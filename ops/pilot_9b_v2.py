"""Offline difficulty pilot (v2: matched to the v13/v14 executor settings): solver-only success rate of the frozen 9B executor per task.

Uses the same executor prompt shape as FrozenTextExecutor (first solver node, no skills,
no messages) and the same graders. Output: per-task samples, success rate, output tokens.
usage: pilot_9b.py <out.json> [samples] [max_new_tokens]
"""
import json, os, sys, time, hashlib
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from skillev.contracts.canonical import canonical_json
from skillev.policy.sglang_text import SGLangFrozenText
from skillev.experiments import curve_benchmarks as cb

OUT = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
SOLVER = os.environ.get(
    "EVOSTEER_PILOT_SOLVER",
    "Solve the public task. Give the final answer in the form the task requests. "
    "If you are running long, stop exploring and give your best current answer "
    "in the requested form immediately.",
)
tok = AutoTokenizer.from_pretrained("/workspace/models/Qwen3.5-9B-ms")
model = SGLangFrozenText("http://127.0.0.1:31000", tok, reference_id="pilot-9b", context_window=32768)
manifest = sys.argv[4] if len(sys.argv) > 4 else "data/paper_benchmarks/paper_benchmarks.jsonl"
rows = [json.loads(l) for l in open(manifest) if l.strip()]
rows = [r for r in rows if r.get("split") == "train"]

def prompt(row):
    return canonical_json({"role": SOLVER, "task": cb._task_prompt(row), "bound_skills": [], "messages": [],
                           "previous_output": None,
                           "instruction": "Perform your role using the public task and feedback. Return your result."})

def one(args):
    row, k = args
    seed = int(hashlib.sha256(f"{row['task_id']}/{k}".encode()).hexdigest()[:12], 16)
    t = time.time()
    text, n_in, n_out = model.frozen_text(prompt(row), max_new_tokens=CAP, input_limit=16384, temperature=0.3, seed=seed)
    return {"task_id": row["task_id"], "kind": row["kind"], "k": k, "reward": cb._score(row["kind"], text, row),
            "out_tokens": n_out, "truncated": n_out >= CAP, "seconds": time.time() - t, "tail": text[-160:]}

jobs = [(r, k) for r in rows for k in range(K)]
t0 = time.time(); results = []
with ThreadPoolExecutor(96) as pool:
    for i, res in enumerate(pool.map(one, jobs)):
        results.append(res)
        if i % 200 == 0:
            print(f"{i}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
json.dump({"samples": K, "cap": CAP, "solver": SOLVER, "results": results}, open(OUT, "w"))
by = {}
for r in results:
    by.setdefault(r["kind"], []).append(r)
for kind, rs in sorted(by.items()):
    tasks = {}
    for r in rs: tasks.setdefault(r["task_id"], []).append(r["reward"])
    p = [sum(v)/len(v) for v in tasks.values()]
    mixed = sum(1 for x in p if 0 < x < 1)
    print(f"{kind:10s} n={len(tasks)} mean={sum(p)/len(p):.3f} mixed={mixed} all0={sum(1 for x in p if x==0)} all1={sum(1 for x in p if x==1)} "
          f"trunc={sum(r['truncated'] for r in rs)/len(rs):.3f} out_tok_mean={sum(r['out_tokens'] for r in rs)/len(rs):.0f} "
          f"out_tok_max={max(r['out_tokens'] for r in rs)}")
print(f"total {time.time()-t0:.0f}s")
