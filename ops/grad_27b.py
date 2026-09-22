import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
sys.path.insert(0, "/workspace/evosteer/SKILLEV-new-main/src")
import torch
from skillev.policy.evosteer import CausalLMOrchestrator
orch = CausalLMOrchestrator.from_pretrained("/workspace/models/Qwen3.8-27B-FP8", reference_id="grad", device="cuda:0",
    dtype="bfloat16", lora_rank=16, lora_alpha=32, target_modules=("o_proj", "out_proj"), context_window=16384, max_action_tokens=256)
m = orch.model
lora = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
print("trainable tensors:", len(lora), "dtypes:", sorted({str(p.dtype) for _, p in lora}), "params:", sum(p.numel() for _, p in lora), flush=True)
wrapped = [n for n, mod in m.named_modules() if n.endswith(("o_proj", "out_proj")) and hasattr(mod, "lora_A")]
print("wrapped modules:", len(wrapped), wrapped[:2], "base type:", type(m.get_submodule(wrapped[0]).base_layer).__name__, flush=True)
from transformers.integrations.finegrained_fp8 import FP8Linear
layer = m.get_submodule("base_model.model.model.language_model.layers.3.mlp.gate_proj")
assert isinstance(layer, FP8Linear)
x = torch.randn(7, layer.in_features, device="cuda:0", dtype=torch.bfloat16, requires_grad=True)
g = torch.randn(7, layer.out_features, device="cuda:0", dtype=torch.bfloat16)
layer(x).backward(g)
bn, bk = layer.block_size
sc = layer.weight_scale_inv.float().repeat_interleave(bn, 0)[:layer.out_features].repeat_interleave(bk, 1)[:, :layer.in_features]
dense = layer.weight.float() * sc
xr = x.detach().float().requires_grad_(True); (xr @ dense.T).backward(g.float())
fwd_err = ((layer(x.detach()).float() - x.detach().float() @ dense.T).norm() / (x.detach().float() @ dense.T).norm()).item()
bwd_err = ((x.grad.float() - xr.grad).norm() / xr.grad.norm()).item()
print(f"FP8 layer check: forward rel err {fwd_err:.4f}, input-grad rel err {bwd_err:.4f}", flush=True)
filler = "The quick brown fox jumps over the lazy dog. " * 400
print('checkpoint layers:', len(orch._checkpoint_layers), flush=True)
for L in (1024, 2048, 4096, 8192, 4096):
    ids = tuple(orch.tokenizer.encode(filler, add_special_tokens=False)[:L])
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(0); t = time.time()
    with torch.no_grad():
        orch._forward(m, ids, logits_to_keep=1)
    torch.cuda.synchronize(); t1 = time.time() - t
    print(f"L={L:5d} fwd(no_grad) {t1:6.2f}s", flush=True)
    m.zero_grad(set_to_none=True); t = time.time()
    with orch._checkpointed():
        out = orch._forward(m, ids, logits_to_keep=1).logits[0, -1]
    lp = out[[11, 13]].float().log_softmax(-1)[0]
    lp.backward(); torch.cuda.synchronize(); t2 = time.time() - t
    gB = [p.grad for n, p in lora if "lora_B" in n]
    nz = sum(1 for g in gB if g is not None and bool(torch.isfinite(g).all()) and g.abs().sum() > 0)
    print(f"   L={L:5d} fwd+bwd {t2:6.2f}s  peak {torch.cuda.max_memory_allocated(0)/2**30:5.1f} GiB  lora_B grads nonzero {nz}/{len(gB)}", flush=True)
