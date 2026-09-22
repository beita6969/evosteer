import os, sys, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
sys.path.insert(0, "/workspace/evosteer/SKILLEV-new-main/src")
import torch
from skillev.policy.evosteer import CausalLMOrchestrator

t0 = time.time()
orch = CausalLMOrchestrator.from_pretrained(
    "/workspace/models/Qwen3.8-27B-FP8", reference_id="smoke", device="cuda:0", dtype="bfloat16",
    lora_rank=8, lora_alpha=16, target_modules=("o_proj",), context_window=8192, max_action_tokens=256)
print(f"load {time.time()-t0:.1f}s  mem {torch.cuda.memory_allocated(0)/2**30:.1f} GiB", flush=True)
types = {}
for name, m in orch.model.named_modules():
    if name.endswith(("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")) and "lora" not in name:
        types.setdefault(type(m).__name__, 0); types[type(m).__name__] += 1
print("mlp proj module types:", types, flush=True)
lora = sum(1 for n, _ in orch.model.named_modules() if n.endswith("lora_A"))
print("lora-wrapped modules:", lora, flush=True)
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
print("fast path:", mq.is_fast_path_available, "chunk_gdr:", mq.chunk_gated_delta_rule is not None, flush=True)
ids = orch.encode_prompt("hello there", reserve_tokens=16)
with torch.no_grad(), orch._active_model(True) as m:
    _, cache, clen = orch._next_logits(m, ids, cache=None, cached_length=0)
print("cache returned:", cache is not None, type(cache).__name__, "cached_length", clen, "of", len(ids), flush=True)

prompts = [
    ("gsm8k", "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Solve it and end with '#### <number>'.", 256),
    ("hotpot", "Which country is the birthplace of the physicist who formulated the theory of general relativity? Answer with just the country name.", 64),
    ("mbpp", "Write a Python function `is_prime(n)` that returns True if n is prime. Return only the code.", 256),
]
for tag, text, n in prompts:
    torch.cuda.synchronize(); t = time.time()
    out, pin, pout = orch.frozen_text(text, max_new_tokens=n, temperature=0.3, seed=1)
    torch.cuda.synchronize(); dt = time.time() - t
    print(f"--- {tag}: prompt {pin} tok, gen {pout} tok in {dt:.1f}s = {pout/dt:.1f} tok/s\n{out[:600]}", flush=True)
