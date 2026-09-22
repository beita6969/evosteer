import glob, torch
R = open("/workspace/evosteer_runs/active_9b_paper_v8.out").read().strip()
for d in sorted(glob.glob(R + "/checkpoints/batch-*")):
    state = torch.load(d + "/trainable.pt", map_location="cpu", weights_only=True)
    out = {}
    def walk(prefix, obj):
        if isinstance(obj, torch.Tensor):
            out[prefix] = obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
    walk("", state)
    groups = {}
    for k, v in out.items():
        if not v.is_floating_point() or "optimizer" in k or "state." in k: continue
        g = "lora_A" if "lora_A" in k else "lora_B" if "lora_B" in k else k.rsplit(".", 1)[0]
        groups.setdefault(g, []).append(v.float().norm().item() ** 2)
    print(d.rsplit("/", 1)[1], {g: round(sum(v) ** 0.5, 3) for g, v in sorted(groups.items()) if "lora" in g or "head" in g})
