import json, time, torch
from skillev.policy.evosteer import CausalLMOrchestrator
cfg = json.load(open("/workspace/evosteer_9b_paper_v7_config.json"))
policy = CausalLMOrchestrator.from_pretrained(**{**cfg["model"], "device": "cuda:0"})
model = policy.model
def sweep(tag, lengths):
    torch.cuda.synchronize(); t = time.time()
    for n in lengths:
        ids = torch.randint(1000, 50000, (1, n), device="cuda:0")
        with torch.no_grad():
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
    torch.cuda.synchronize(); print(tag, f"{(time.time()-t)/len(lengths):.3f} s/forward over {len(lengths)} distinct lengths", flush=True)
# fresh lengths each sweep so no plan is cached
sweep("cudnn on ", list(range(2001, 2001 + 12 * 37, 37)))
torch.backends.cuda.enable_cudnn_sdp(False)
sweep("cudnn off", list(range(2003, 2003 + 12 * 37, 37)))
torch.backends.cuda.enable_cudnn_sdp(True)
sweep("cudnn on ", list(range(2005, 2005 + 12 * 37, 37)))
torch.backends.cuda.enable_cudnn_sdp(False)
sweep("cudnn off", list(range(2007, 2007 + 12 * 37, 37)))
# grad path with checkpointing through the real score() on the same lengths is exercised by the training run
