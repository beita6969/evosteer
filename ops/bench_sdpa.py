import json, time, torch
from torch.profiler import profile, ProfilerActivity
from skillev.policy.evosteer import CausalLMOrchestrator
cfg = json.load(open("/workspace/evosteer_9b_paper_v7_config.json"))
policy = CausalLMOrchestrator.from_pretrained(**{**cfg["model"], "device": "cuda:0"})
model = policy.model
ids = torch.randint(1000, 50000, (1, 6000), device="cuda:0")
def run(use_cache, grad):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = model(input_ids=ids, use_cache=use_cache, logits_to_keep=1)
        if grad:
            out.logits.float().sum().backward()
for use_cache, grad in ((True, False), (False, False)):
    run(use_cache, grad); torch.cuda.synchronize(); t = time.time()
    run(use_cache, grad); torch.cuda.synchronize()
    print(f"use_cache={use_cache} grad={grad}: {time.time()-t:.3f} s", flush=True)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run(use_cache, grad); torch.cuda.synchronize()
    names = [e.key for e in prof.key_averages() if any(k in e.key.lower() for k in ("flash", "attention", "fmha", "efficient", "bmm", "softmax", "cudnn"))]
    print("   attention kernels:", [n[:70] for n in names][:6], flush=True)
