"""cProfile EvoTrainer.update on real trajectories (all sources of 6 tasks) with the live config."""
import cProfile, io, json, pstats, sys, time, torch
from skillev.contracts.evosteer import EvoTrajectory
from skillev.policy.evosteer import CausalLMOrchestrator
from skillev.policy.state_value import StateValueHeads
from skillev.training.evosteer import EvoTrainer, EvoOptimizerConfig
from skillev.training.reference_statistics import ReferenceObservation, ReferenceStatistics
run = sys.argv[1]
cfg = json.load(open("/workspace/evosteer_9b_paper_v7_config.json"))
policy = CausalLMOrchestrator.from_pretrained(**{**cfg["model"], "device": "cuda:0"})
trajs = [EvoTrajectory.from_value(json.loads(l)) for l in open(f"{run}/trajectories/batch-000002.jsonl")]
tasks = []
for t in trajs:
    if t.task.task_id not in tasks: tasks.append(t.task.task_id)
keep = set(tasks[:6])
chosen = tuple(t for t in trajs if t.task.task_id in keep)
print("trajectories", len(chosen), "decisions", sum(len(t.decisions) for t in chosen), "sources", sorted({t.source for t in chosen}), flush=True)
snapshots = {
    ctx: ReferenceStatistics(ctx).snapshot(chosen[0].batch_id, tuple(
        ReferenceObservation(t.sample_id, t.task.task_id, t.task.family, t.reward, ctx)
        for t in trajs if t.source == "natural_reference" and t.statistics_context == ctx))
    for ctx in {t.statistics_context for t in chosen}
}
families = tuple(cfg["application"]["task_families"])
heads = StateValueHeads(encoding_dim=policy.encoding_dim, num_task_types=len(families)).to(policy.device)
opt = cfg["application"]["optimizer"]
trainer = EvoTrainer(policy, heads, families, EvoOptimizerConfig(**opt))
pr = cProfile.Profile()
torch.cuda.synchronize(); t0 = time.time(); pr.enable()
metrics = trainer.update(chosen, snapshots)
torch.cuda.synchronize(); pr.disable(); wall = time.time() - t0
print("wall", round(wall, 1), "s  per decision", round(wall / sum(len(t.decisions) for t in chosen), 3), flush=True)
st = io.StringIO(); pstats.Stats(pr, stream=st).sort_stats("tottime").print_stats(22); print(st.getvalue()[:6000])
st = io.StringIO(); pstats.Stats(pr, stream=st).sort_stats("cumulative").print_stats(40); print(st.getvalue()[:9000])
