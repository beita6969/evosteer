"""Time one orchestrator decision's pieces on the 9B policy (standalone, GPU0)."""
import json, time, cProfile, pstats, io, torch
from skillev.policy.evosteer import CausalLMOrchestrator
cfg = json.load(open("/workspace/evosteer_9b_paper_v7_config.json"))
t = time.time()
policy = CausalLMOrchestrator.from_pretrained(**{**cfg["model"], "device": "cuda:0"})
print("load", round(time.time() - t, 1), flush=True)
# A realistic legal-action set: 2 roles x (no skill) ADD, SET_OUTPUT/RERUN/DROP for 2 nodes, edges, STOP.
actions = [{"kind": "ADD_AGENT", "node_id": f"n{i}", "role_id": r} for i in (2,) for r in ("solver", "verifier")]
for n in ("n0", "n1"):
    actions += [{"kind": k, "node_id": n} for k in ("RERUN_AGENT", "DROP_AGENT", "SET_OUTPUT")]
actions += [{"kind": "ADD_EDGE", "source_id": a, "target_id": b, "protocol": p} for a, b in (("n0", "n1"), ("n1", "n0")) for p in ("feedback", "revise")]
actions += [{"kind": "STOP"}]
body = {"task_family": "aime_2026", "task": "x " * 400, "execution": {"graph": {"nodes": [{"node_id": "n0", "output": "reasoning " * 800}]}, "history": []}, "reference_value": 0.5, "reference_value_change": 0.0, "legal_actions": actions}
text = json.dumps(body)
def timed(label, fn, n=3):
    fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): out = fn()
    torch.cuda.synchronize(); print(label, round((time.time() - t) / n, 3), "s", flush=True)
    return out
menu = timed("menu", lambda: policy.menu(tuple(actions)))
ids = timed("encode_prompt", lambda: policy.encode_prompt(text))
print("prompt tokens", len(ids), "paths", len(menu.paths), "max path", max(map(len, menu.paths)))
timed("sample(actor)", lambda: policy.sample(ids, menu, reference=False, seed=1))
timed("sample(reference)", lambda: policy.sample(ids, menu, reference=True, seed=1))
pr = cProfile.Profile(); pr.enable()
for s in range(3): policy.sample(ids, menu, reference=False, seed=s)
torch.cuda.synchronize(); pr.disable()
st = io.StringIO(); pstats.Stats(pr, stream=st).sort_stats("tottime").print_stats(18); print(st.getvalue()[:5000])
