"""cProfile the real rollout path (collect_batch) on a few tasks, standalone 9B on GPU0, no author."""
import argparse, asyncio, cProfile, io, json, os, pstats, sys, time
cfg = json.load(open("/workspace/evosteer_9b_paper_v7_config.json"))
# The author is served (no model load); collect_batch never calls it.
cfg["model"]["device"] = "cuda:0"
path = "/workspace/evosteer/bench_rollout_config.json"
json.dump(cfg, open(path, "w"))
os.environ.pop("EVOSTEER_SAMPLER_DEVICE", None)
from skillev import evosteer_cli
from pathlib import Path
args = argparse.Namespace(config=Path(path), tasks=None, task_factory="skillev.experiments.curve_benchmarks:task_factory")
app, bindings, config, _ = evosteer_cli._build_training(args)
by_family = {}
for b in bindings:
    by_family.setdefault(b.task.family, b)
batch = tuple(by_family.values())[:5]
print("tasks", [b.task.task_id for b in batch], flush=True)
pr = cProfile.Profile()
t = time.time(); pr.enable()
collected = asyncio.run(app.collect_batch(batch, batch_number=1))
pr.disable(); wall = time.time() - t
n = len(collected.trajectories)
print("trajectories", n, "wall", round(wall, 1), "s", flush=True)
st = io.StringIO(); pstats.Stats(pr, stream=st).sort_stats("tottime").print_stats(30); print(st.getvalue()[:8000])
st = io.StringIO(); pstats.Stats(pr, stream=st).sort_stats("cumulative").print_stats(45); print(st.getvalue()[:10000])
