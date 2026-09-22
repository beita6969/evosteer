"""Seed-matched two-arm fair baseline.

arm A = the old pilot's canonical_json envelope (context_window 32768, input_limit 16384)
arm B = the REAL in-run path: FrozenTextExecutor + the run's own solver RoleSpec, policy
        context_window taken from the run config, truncation from finish_reason.
Both arms hit the same server with the SAME seed per (task_id, k), so the difference is
the prompt envelope and the budget plumbing, nothing else.

usage: pilot_9b_v4_execpath.py <out.json> <run_dir> <manifest> [samples] [url]
"""
import asyncio, hashlib, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from skillev.contracts.canonical import canonical_json
from skillev.orchestration.graph import NodeExecutionRequest, RoleSpec
from skillev.orchestration.model_executor import FrozenTextExecutor
from skillev.policy.sglang_text import SGLangFrozenText
from skillev.experiments import curve_benchmarks as cb

OUT, RUN_DIR, MANIFEST = sys.argv[1], sys.argv[2], sys.argv[3]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 8
URL = sys.argv[5] if len(sys.argv) > 5 else "http://127.0.0.1:31001"

run = json.load(open(RUN_DIR + "/run.json"))
cfg = run["config"]
role_value = next(r for r in cfg["application"]["roles"] if r["role_id"] == "solver")
ROLE = RoleSpec.from_value(role_value)
CTX = cfg["model"]["context_window"]
TEMP = cfg["executor_temperature"]
CAP = ROLE.model_maximum.output_tokens
tok = AutoTokenizer.from_pretrained(cfg["model"]["model_path"])

# arm B: exactly what the run's executor does
policy_b = SGLangFrozenText(URL, tok, reference_id="pilot-execpath", context_window=CTX)
executor = FrozenTextExecutor(policy_b, temperature=TEMP)
# arm A: exactly what pilot_9b_v2.py did
policy_a = SGLangFrozenText(URL, tok, reference_id="pilot-legacy", context_window=32768)
LEGACY_INSTR = "Perform your role using the public task and feedback. Return your result."

rows = [json.loads(l) for l in open(MANIFEST) if l.strip()]
rows = [r for r in rows if r.get("split") == "train"]
print(f"tasks={len(rows)} K={K} role={ROLE.role_id} ctx={CTX} cap={CAP} temp={TEMP}", flush=True)

def seed_of(task_id, k):
    return int(hashlib.sha256(f"{task_id}/{k}".encode()).hexdigest()[:12], 16)

def arm_a(row, seed):
    prompt = canonical_json({
        "role": ROLE.instruction, "task": cb._task_prompt(row), "bound_skills": [],
        "messages": [], "previous_output": None, "instruction": LEGACY_INSTR,
    })
    text, n_in, n_out = policy_a.frozen_text(
        prompt, max_new_tokens=CAP, input_limit=16384, temperature=TEMP, seed=seed)
    return text, n_in, n_out, n_out >= CAP

def arm_b(row, seed):
    req = NodeExecutionRequest(
        runtime_id="pilot", node_id="n0", role=ROLE, task_prompt=cb._task_prompt(row),
        skills=(), messages=(), previous_output=None, execution_index=0, seed=seed)
    res = asyncio.run(executor.execute(req))
    outcome = (res.metadata or {}).get("execution_outcome") or {}
    return res.output, res.usage.input_tokens, res.usage.output_tokens, outcome.get("failure_kind") == "truncated"

def one(job):
    row, k = job
    seed = seed_of(row["task_id"], k)
    rec = {"task_id": row["task_id"], "kind": row["kind"], "k": k, "seed": seed}
    for name, fn in (("a", arm_a), ("b", arm_b)):
        t = time.time()
        text, n_in, n_out, trunc = fn(row, seed)
        rec[name] = {"reward": cb._score(row["kind"], text, row), "in_tokens": n_in,
                     "out_tokens": n_out, "truncated": bool(trunc),
                     "seconds": time.time() - t, "tail": text[-160:]}
    return rec

jobs = [(r, k) for r in rows for k in range(K)]
t0 = time.time(); results = []
with ThreadPoolExecutor(64) as pool:
    for i, rec in enumerate(pool.map(one, jobs)):
        results.append(rec)
        if i % 200 == 0: print(f"{i}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
json.dump({"samples": K, "cap": CAP, "context_window": CTX, "temperature": TEMP,
           "role": role_value, "run_dir": RUN_DIR, "results": results}, open(OUT, "w"))

by = {}
for r in results: by.setdefault(r["kind"], []).append(r)
print(f"\n{'family':12s} {'armA':>7s} {'armB':>7s} {'B-A':>8s} {'truncA':>7s} {'truncB':>7s}")
import statistics as st
dall = []
for kind, rs in sorted(by.items()):
    ta = {}; tb = {}
    for r in rs:
        ta.setdefault(r["task_id"], []).append(r["a"]["reward"])
        tb.setdefault(r["task_id"], []).append(r["b"]["reward"])
    a = st.mean([st.mean(v) for v in ta.values()]); b = st.mean([st.mean(v) for v in tb.values()])
    d = [st.mean(tb[t]) - st.mean(ta[t]) for t in ta]; dall += d
    print(f"{kind:12s} {a:7.4f} {b:7.4f} {st.mean(d):+8.4f} "
          f"{st.mean([r['a']['truncated'] for r in rs]):7.3f} {st.mean([r['b']['truncated'] for r in rs]):7.3f}")
se = st.stdev(dall) / len(dall) ** 0.5
print(f"{'POOLED':12s} {'':7s} {'':7s} {st.mean(dall):+8.4f}  se={se:.4f}  t={st.mean(dall)/se:+.2f}  (n={len(dall)} tasks)")
print(f"total {time.time()-t0:.0f}s")
