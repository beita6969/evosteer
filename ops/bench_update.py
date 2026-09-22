"""Profile the trainer's per-decision work on real v7 decisions (standalone 9B copy on GPU0)."""
import json, time, cProfile, pstats, io, sys, torch
from skillev.contracts.evosteer import EvoTrajectory
from skillev.policy.evosteer import CausalLMOrchestrator
run = sys.argv[1]
cfg = json.load(open("/workspace/evosteer_9b_paper_v7_config.json"))
policy = CausalLMOrchestrator.from_pretrained(**{**cfg["model"], "device": "cuda:0"})
rows = [json.loads(l) for l in open(f"{run}/trajectories/batch-000001.jsonl")]
trajs = [EvoTrajectory.from_value(r) for r in rows[:8]]
decisions = [d for t in trajs for d in t.decisions]
print("decisions", len(decisions), "prompt lens", [len(d.prompt_ids) for d in decisions][:12], flush=True)
def one(d):
    s = policy.score(d)
    with torch.no_grad():
        r, e = policy.reference_score_and_encoding(d)
    if s.requires_grad:
        s.backward()
    return r
for d in decisions[:3]: one(d)
torch.cuda.synchronize(); t = time.time()
for d in decisions: one(d)
torch.cuda.synchronize(); print("per decision", round((time.time() - t) / len(decisions), 3), "s", flush=True)
pr = cProfile.Profile(); pr.enable()
for d in decisions: one(d)
torch.cuda.synchronize(); pr.disable()
st = io.StringIO(); pstats.Stats(pr, stream=st).sort_stats("tottime").print_stats(25); print(st.getvalue()[:7000])
st = io.StringIO(); pstats.Stats(pr, stream=st).sort_stats("cumulative").print_stats(30); print(st.getvalue()[:7000])
