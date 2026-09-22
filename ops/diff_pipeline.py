import asyncio, sys
sys.path.insert(0, "/workspace/evosteer/testcopy/tests/evosteer"); sys.path.insert(0, "/workspace/evosteer/testcopy/src")
import test_application as ta
seq = ta.application(); e = asyncio.run(seq.train_batch((ta.binding("task-1"),)))
app = ta.application(); app.set_rollout_policy(ta._replica(app.policy))
async def run():
    c = await app.collect_batch((ta.binding("task-1"),), batch_number=1)
    return await asyncio.to_thread(app.finish_batch, app.prepare_batch(c))
f = asyncio.run(run())
def walk(a, b, path=""):
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)): walk(a.get(k), b.get(k), f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}[{i}]")
    elif a != b:
        print(path, "|", str(a)[:120], "|", str(b)[:120])
for x, y in zip(e.trajectories, f.trajectories): walk(x.to_value(), y.to_value(), x.sample_id)
print("loss", e.metrics["anchor_tb_loss"], f.metrics["anchor_tb_loss"])
