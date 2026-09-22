"""GPU kernel breakdown of the trainer's per-decision work (torch.profiler)."""
import json, sys, time, torch
from torch.profiler import profile, ProfilerActivity
from skillev.contracts.evosteer import EvoTrajectory
from skillev.policy.evosteer import CausalLMOrchestrator
run = sys.argv[1]
cfg = json.load(open("/workspace/evosteer_9b_paper_v7_config.json"))
policy = CausalLMOrchestrator.from_pretrained(**{**cfg["model"], "device": "cuda:0"})
rows = [json.loads(l) for l in open(f"{run}/trajectories/batch-000001.jsonl")]
decisions = [d for r in rows[:3] for d in EvoTrajectory.from_value(r).decisions]
def fwd_bwd(d):
    s = policy.score(d)
    if s.requires_grad: s.backward()
def ref(d):
    with torch.no_grad(): policy.reference_score_and_encoding(d)
for d in decisions[:3]: fwd_bwd(d); ref(d)
for name, fn in (("actor fwd+bwd", fwd_bwd), ("reference", ref)):
    torch.cuda.synchronize(); t = time.time()
    for d in decisions[:6]: fn(d)
    torch.cuda.synchronize(); print(name, round((time.time() - t) / 6, 3), "s/decision", flush=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for d in decisions[:3]: fwd_bwd(d)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=22, max_name_column_width=70))
