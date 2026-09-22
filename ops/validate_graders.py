import json, sys, time
sys.path.insert(0, "/workspace/evosteer/testcopy/src"); sys.path.insert(0, "/workspace/evosteer/prepvenv/pylib")
import pyarrow.parquet as pq
from skillev.experiments import curve_benchmarks as cb
rows = [json.loads(l) for l in open("/workspace/evosteer/SKILLEV-new-main/data/paper_benchmarks/paper_benchmarks.jsonl")]
code = {str(r["task_id"]): r["code"] for r in pq.read_table("/workspace/evosteer/datasets/paper_benchmarks/raw/mbppplus.parquet").to_pylist()}
bad, slow = [], []
for r in rows:
    k = r["kind"]
    if k == "hotpotqa":
        checks = [(r["answer"], 1), (f"Based on the passages, the answer is {r['answer']}.", 1), ("Unrelated Person", None)]
    elif k == "nq_open":
        checks = [(r["answers"][0], 1), (f"**Answer:** {r['answers'][-1]}", 1)]
    elif k == "medqa":
        wrong = next(l for l in r["options"] if l != r["answer"])
        checks = [(f"Reasoning ... Answer: {r['answer']}", 1), (r["answer"], 1), (f"The correct option is ({r['answer']}).", 1),
                  (f"Answer: {wrong}", 0), (f"To answer, a patient with this ... Answer: {wrong}", 0), ("I am not sure.", 0)]
    elif k == "aime_2026":
        checks = [(f"... so the result is \\boxed{{{r['answer']}}}", 1), (f"\\boxed{{{int(r['answer']) + 1}}}", 0)]
    else:
        t = time.time(); s = cb._score(k, "```python\n" + code[r["source_id"]] + "\n```", r); dt = time.time() - t
        if dt > 10: slow.append((r["task_id"], round(dt, 1)))
        checks = []; 
        if s != 1: bad.append((r["task_id"], "gold code", s))
        checks = [("def nothing(): pass", 0)]
    for out, want in checks:
        got = cb._score(k, out, r)
        if want is not None and got != want: bad.append((r["task_id"], out[:60], got, want))
        if want is None and got >= 1: bad.append((r["task_id"], "unrelated scored", got))
print("tasks:", len(rows), "| failures:", len(bad)); [print("  ", b) for b in bad[:25]]
print("slow MBPP+ gold runs (>10s):", slow)
