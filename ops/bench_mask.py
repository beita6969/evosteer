import json, time, torch
from skillev.policy.evosteer import CausalLMOrchestrator
cfg = json.load(open("/workspace/evosteer_9b_paper_v7_config.json"))
policy = CausalLMOrchestrator.from_pretrained(**{**cfg["model"], "device": "cuda:0"})
model = policy.model
print("attn impl", getattr(model.config, "_attn_implementation", None), getattr(getattr(model.config, "text_config", None), "_attn_implementation", None))
for n in (3000, 8000):
    ids = torch.randint(1000, 50000, (1, n), device="cuda:0")
    for label, kw in (("ones mask", {"attention_mask": torch.ones((1, n), dtype=torch.long, device="cuda:0")}), ("no mask", {})):
        with torch.no_grad():
            model(input_ids=ids, use_cache=True, logits_to_keep=1, **kw)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(3):
                out = model(input_ids=ids, use_cache=True, logits_to_keep=1, **kw)
            torch.cuda.synchronize(); dt = (time.time() - t) / 3
            # one cached decode step
            cache = out.past_key_values
            step = {"attention_mask": torch.ones((1, n + 1), dtype=torch.long, device="cuda:0")} if kw else {}
            torch.cuda.synchronize(); t = time.time()
            model(input_ids=ids[:, :1], past_key_values=cache, use_cache=True, logits_to_keep=1, **step)
            torch.cuda.synchronize(); dd = time.time() - t
        print(n, label, "prefill", round(dt, 3), "s  decode step", round(dd, 3), "s", flush=True)
