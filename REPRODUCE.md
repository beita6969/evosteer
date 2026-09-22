# Reproducing the EvoSteer paper runs (v13 / v14) and ALFWorld

This file covers only the EvoSteer experiments of the paper. The rest of the
repository (legacy Bayesian SKILLEV, SkillFlow baseline, protocol_v10–v14
evaluation code) is kept as-is for provenance.

## What is where

| Path | Content |
|---|---|
| `src/skillev/evosteer_cli.py` | Entry point: `python -m skillev.evosteer_cli train ...` |
| `src/skillev/evosteer_application.py` | Composition root: rollouts, AnchorTB update, skill author, admission |
| `src/skillev/scoring/anchor_tb.py` | AnchorTB all-subtrajectory balance loss |
| `src/skillev/training/evosteer.py` | Trainer (actor LoRA, residual/value/outcome heads) |
| `src/skillev/experiments/curve_benchmarks.py` | Task factory for HotpotQA, NQ-Open, MedQA, AIME 2026, MBPP+ (and graders) |
| `src/skillev/experiments/curve_alfworld.py`, `src/skillev/alfworld_env.py` | ALFWorld text environment and task factory |
| `configs/paper_runs/evosteer_9b_paper_v14_config.json` | **Main run (v14)**: LoRA r64 on q/k/v/o + in_proj_qkv, β = 2.0, 32 tasks/step |
| `configs/paper_runs/evosteer_9b_paper_v13_config.json` | Replication run (v13): LoRA on o_proj/out_proj, β = 1.0, 44 tasks/step |
| `configs/paper_runs/evosteer_9b_alfworld_smoke_config.json` | ALFWorld smoke configuration |
| `ops/launch_executor_9b.sh` | Frozen Qwen3.5-9B executor (SGLang 0.5.15, port 31000) |
| `ops/launch_paper_v14.sh`, `ops/launch_paper_v14_ext.sh` | Start v14; resume v14 from its newest complete checkpoint |
| `ops/launch_paper_v13.sh`, `ops/launch_paper_v13_ext.sh` | Same for v13 |
| `ops/stopat.sh` | Stop a run cleanly after a given batch (used to stop on alpha-spending looks) |
| `ops/guardian.sh`, `ops/vram_floor.py`, `ops/lookclust.py`, `ops/curvedata.py` | Cluster supervision, task-clustered look statistics, curve export |

The `ops/` scripts are the exact scripts used on the training cluster and
assume its layout (`/workspace/evosteer/SKILLEV-new-main`, `/workspace/models`,
`/workspace/evosteer_runs`). Adjust those paths for your machine.

## Released artifacts (Hugging Face, private)

* **Dataset repo** `evosteer-data`: the task pools (`tasks/`), the per-step
  training logs of both runs (`training_logs/`, `metrics.jsonl` + per-batch
  JSON; per-episode trajectories, 17 GB, are not uploaded), the data behind every
  paper figure (`figure_data/`), final figures, metric reports and the cost
  comparison against TB / TTB / GRPO / PPO / FlowSteer / SkillFlow.
* **Model repo** `evosteer-qwen3.5-9b`: `v14/batch-000240` (final v14 checkpoint)
  and `v13/batch-000210` (final v13 checkpoint). Each directory is a complete
  resumable checkpoint: `trainable.pt` (actor LoRA + heads + optimizer),
  `state.json` (reference statistics, skill library, author ledger),
  `state.sha256`, `cli-context.json`.

## Environment

```bash
python3.13 -m venv venv313
venv313/bin/pip install -e . torch==2.13.0 transformers==5.14.1 peft==0.19.1 openai==2.6.1
# optional GPU kernels for Qwen3.5 linear attention:
venv313/bin/pip install causal-conv1d==1.6.2.post1 flash-linear-attention==0.5.2
```

The executor runs in a separate SGLang 0.5.15 environment. Model weights:
Qwen3.5-9B (set `model_path` in the config).

## Data

Download `tasks/` from the dataset repo into `data/` (this directory is
git-ignored on purpose):

```bash
hf download <org>/evosteer-data --repo-type dataset --include "tasks/*" --local-dir /tmp/evosteer-data
cp -r /tmp/evosteer-data/tasks/paper_benchmarks data/
export EVOSTEER_CURVE_DATA=$PWD/data/paper_benchmarks/paper_benchmarks_9b_mixed_large.jsonl  # the 145-task pool
```

## Running v14

```bash
ops/launch_executor_9b.sh 1 31000          # frozen executor on GPU 1
# skill author: any OpenAI-compatible endpoint
export EVOSTEER_AUTHOR_API_BASE=https://<your-endpoint>/v1
export EVOSTEER_AUTHOR_API_KEY_FILE=$HOME/.config/evosteer/author_api_key
ops/launch_paper_v14.sh                    # actor on GPU 0, rollout replica on GPU 1
ops/stopat.sh v14 240                      # optional: stop after the step-240 look
```

To continue from the released checkpoint, point `--resume` at the downloaded
`v14/batch-000240` directory (see `ops/launch_paper_v14_ext.sh`).

The paper runs were stopped on alpha-spending look boundaries after the paired
gap had converged: **v14 at step 240, v13 at step 210** (configs say 250).

## ALFWorld

ALFWorld is integrated as an additional task family (`alfworld`), using the
official text environment (TextWorld) — no RAGEN dependency.

* One task = one official game file (`alfworld/<split>/<index>`, index into the
  sorted split). The model sees only the observation text and the admissible
  commands; `won` stays in the evaluator (reward 1 if won, else 0).
* One agent node = up to `ALFWORLD_STEPS_PER_NODE` (default 20) environment
  steps with the frozen executor; the episode is shared by the team, so a later
  node (e.g. a RERUN) continues from the current state. An episode ends when the
  game is won or after `ALFWORLD_MAX_STEPS` (default 50) steps in total.
* Sessions of the same task start from identical states, so ALFWorld tasks are
  replay-safe and can be used for paired skill validation.

```bash
scripts/provision_alfworld.sh venv313_alfworld/bin/python /data/alfworld   # installs alfworld==0.4.2, downloads data
export ALFWORLD_DATA=/data/alfworld
pytest tests/experiments/test_curve_alfworld_official.py tests/experiments/test_curve_alfworld_real.py
MEM_FRACTION=0.45 ops/launch_executor_9b.sh 0 31000
ops/launch_alfworld_smoke.sh /tmp/alfworld-smoke
```

Role budgets. The orchestrator reserves `role.model_maximum` for every node,
and a multi-step ALFWorld node needs more than the one-call budget of the text
tasks. The ALFWorld config therefore declares 21 model calls, 80k input and 2k
output tokens and 300 s per node, and an episode cap of 786k tokens. The
executor checks each step against what is left of the reservation (exact prompt
length from the executor) and ends the node early rather than overspend.

Smoke result (2026-09-23, 1×H200, 2 steps × 2 games × (2 π + 2 ρ) = 16 episodes,
logs in the dataset repo under `alfworld_smoke/`): every episode ran in the
real environment (13–50 environment steps, 18–63 model calls), one episode won
(13 steps), step-2 AnchorTB loss 0.49, both checkpoints written. This checks the
integration end to end; it is not a benchmark result.

ALFWorld-only training: `--task-factory skillev.experiments.curve_alfworld:task_factory`
with `task_families: ["alfworld"]`. Mixed with the text benchmarks: keep
`curve_benchmarks:task_factory`, add `"alfworld"` to `task_families` and set
`EVOSTEER_MIX_ALFWORLD_COUNT=<n>`.

## What was removed before release

No credentials were ever stored in the tree. Before release the hosted
author/judge gateway host was replaced by `llm-gateway.example.org`, real GPU
UUIDs and cluster pod names were replaced by placeholders, and macOS AppleDouble
(`._*`) files were deleted. Code behaviour is otherwise unchanged from the
version that produced the v13/v14 runs, except for the ALFWorld integration
described above (separate commit).
