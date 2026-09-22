# SKILLEV-new

> **EvoSteer paper runs (v13/v14), released checkpoints and the ALFWorld integration:** see [REPRODUCE.md](REPRODUCE.md).

This repository now includes **EvoSteer**, adapted from the supplied
`EvoSteer-ICLR2027.pdf`, alongside the original Bayesian SKILLEV and SkillFlow
baselines. EvoSteer follows the existing `src/skillev` package architecture.
Its entry is `skillev.evosteer_cli` and its composition root is
`EvoSteerApplication`.

See the [EvoSteer implementation and run guide](docs/evosteer-implementation.md)
and the [paper-to-code method map](docs/evosteer-method-conformance.md)
for the complete method, explicit underspecified choices, configuration and
checkpoint recovery. Method validation is recorded in
[the current validation artifact](docs/evosteer-method-validation.json). A local CPU smoke run builds a
tiny Transformer with LoRA and trains on a synthetic environment without model
downloads:

```bash
PYTHONPATH=src python -m skillev.evosteer_cli smoke --output /tmp/evosteer-smoke --steps 2
```

Python >=3.11 and the `policy` dependency group are required. Choose a new output
directory for each run. This smoke run is an integration check, not a benchmark
result. Local model training is configured separately; no paper scores are
claimed.

The sections below describe the **legacy Bayesian method** and its dataset
infrastructure. Its scientific specification remains `idea.tex`, with
[`docs/method-v3-final-spec.md`](docs/method-v3-final-spec.md) as its engineering
specification. Those definitions do not govern the new EvoSteer objective.

## Current datasets

As of the owner's September 15 update, IID is **HotpotQA, TriviaQA, AIME2026,
HealthBench, ALFWorld, MBPP+**; OOD is **MuSiQue-Ans, NQ-Open, Math-Hard,
GPQA Diamond (Biology + Organic Chemistry), ScienceWorld, APPS Introductory**.
Native thinking defaults to **off**. See [downloads and setup](docs/current-datasets.md).
HumanEval, Omni-MATH, LiveMedBench and LiveCodeBench are historical only.

## Historical seven-domain training entry

The previously declared run was [BayesianImprove250/B28](docs/bayesian_formal_250.md):
seven domains, single Qwen3.5-9B owner, native reasoning thinking, one inference and
two gradient GPUs, cadence checkpoints every10steps. The former queued four-step
rerun and rotating eighth question are superseded. Configuration/launch status is
not a claim of completed250-step training or verified natural evolution.
Its population and schedule are not a launch configuration for the new six-IID catalog.

## Current status

The Protocol-v3 method graph is implemented behind one exact attempt boundary.
Every successful publication binds the builder kind, answer-free public method
identity, exact-input content hash, complete source-log set, and tagged outcome.
Failed or incomplete child bundles remain quarantined and cannot enter a
scientific audit API.

A formal run uses a preregistered phase-search interval followed by a fixed
closure-training tail. Phase transitions may be opened only in search slots;
therefore every committed library mutation and deterministic Z reset is followed
by at least one committed training step under the new library before the attempt
can succeed. Runtime snapshots and execution state use wire version `@6` and
carry the exact run cursor and immutable method/protocol identity. No primary,
ablation, benchmark, or paper result is claimed by this README.

The active application graph is:

```text
contracts → policy → scoring → rollout → training
                                  ↓
                       diagnostics + calibration
                                  ↓
                              evolution
                                  ↓
                         application builder

audit ← four scientific source events (offline only)
experiments/arms ← six explicit alternate implementations
```

The eight method layers are:

1. **Contracts** — canonical trajectories, rewards, edge scores, TTB batches,
   posteriors, phase transitions, sealed full-method evolution actions, and
   Protocol-v3 source-event payloads.
2. **Policy** — an explicit pinned Qwen causal-LM backend with a frozen base,
   independent forward/hindsight LoRA adapters, and a trainable query-only
   `log Z` head.
3. **Scoring** — canonical forward/hindsight rendering, exact recorded action
   token spans, per-token teacher-forced log probabilities, and
   `Δ = log Z + ΣP_F − β log R̃ − ΣP_B` with `(Δ/T)²`.
4. **Rollout** — a fresh atomic two-pass reasoning/action rollout using full
   retrieved skill content. Invalid agent actions remain data; infrastructure
   failures terminate the attempt.
5. **Training** — a fixed B-task population, one on-policy batch per AdamW
   update, zero weight decay, no gradient clipping, deterministic Z reset, and
   one `TRAINING_STEP_COMMITTED` scientific source per completed step.
6. **Diagnostics** — raw log-importance/state-flow/skill-flow projections and
   raw-`Δ²` stagnation windows, with immutable online preview/commit.
7. **Calibration** — uncapped flow-weighted Beta–Bernoulli updates and
   confidence bounds derived in the same online transaction as diagnostics.
8. **Evolution** — residual-and-entropy phase detection, complete nonempty
   five-action Φ, sealed frozen-base authoring, one immutable library mutation,
   deterministic Z reset, and continuation under the new library.

`SKILLEVApplication` is the sole scientific composition root for the full
BayesianImprove method packages. The historical GPU `run_training.py` path
does not call it and is therefore not an admissible BayesianImprove formal
entrypoint; unifying that runtime is pending behind the Protocol 10 execution
gate. Active full-method packages do not replay event logs. `skillev.audit`
accepts only a published successful
attempt bundle, reduces the four scientific sources as a closed state machine,
and independently verifies diagnostics, posterior updates, detected phases,
library mutations, Z-reset identity, and continuation under the new library:

- `LIBRARY_INITIALIZED`
- `TRAINING_STEP_COMMITTED`
- `EVOLUTION_PHASE_OPENED`
- `EVOLUTION_CYCLE_COMMITTED`

`RUN_ATTEMPT_FAILED` is optional operational failure telemetry and is never a
recovery input. Runtime checkpoints persist one tagged
`RuntimeExecutionState@6`, including the fixed run cursor and snapshot identity,
not source history plus derived state. Construction
from a checkpoint always names one completed exact snapshot path; no
latest-file search, compatibility reader, event-history proof, open-phase
resume, repair, automatic retry, cleanup, or live pruning exists. A failed
attempt may retain private forensic evidence, but that evidence is not an
executable state. Offline maintenance and event readers are separate packages.

One attempt runs in one child process. Its supervisor invokes that worker
exactly once and receives only a tagged success/failure IPC value; a failed
child cannot return half-mutated model, optimizer, library, projection, or
budget objects to the parent.

## Full method and ablations

The literal full method uses raw-softmax policy sampling, raw importance,
uncapped flow weights, confidence-bound decisions, the residual-and-entropy
AND trigger, no gradient clipping, and AdamW weight decay `0.0`.

Six single-axis alternatives live only under `skillev.experiments.arms`:

1. Bayesian calibration disabled (flow-only Retain/Generate);
2. no flow weighting;
3. capped flow weighting;
4. clipped importance;
5. posterior mean instead of confidence bounds;
6. residual-only phase detection.

There is no runtime arm switch inside the full core. In particular, the
no-Bayesian arm owns separate projection, training-source, decision,
execution, cycle, event-log, and application types; it cannot masquerade as a
full `TrainingStepCommit` or full evolution cycle. All seven builders have
distinct published identities and typed audit paths. The no-Bayesian arm owns a
separately hashed four-source log; the five full-shaped single-axis arms share
the source schema but are audited with their preregistered alternate kernel,
never with literal-full recomputation.

The target result-blind benchmark schema is Protocol 10, frozen in
[`evaluation.tex`](evaluation.tex) and
[`configs/evaluation/protocol_v10.yaml`](configs/evaluation/protocol_v10.yaml).
It is not yet executable: `src/skillev/experiments/protocol_v10.py` now owns the
typed population-level `@10` reader, while the 18-benchmark reader is retained
only for historical `@9` artifacts. Formal admission remains blocked until the
private datasets, trusted evaluators, and canonical runtime consume Protocol
10. Runtime/checkpoint identity uses `@6`, published
attempt identity uses `@7`, and formal execution freeze uses `@5`; these are
distinct wire contracts rather than alternate benchmark protocols. Historical
artifacts retain their original identities and are never relabelled as Protocol
10 evidence.

## Scientific and evaluation boundaries

- Rollout keeps the generated action's original content token IDs. It never
  infers a span from mixed completion text or substitutes re-encoded IDs.
- Policy rollout and forward scoring use the same raw-softmax distribution.
  Frozen-base skill authoring has a separate request type and sampler.
- `log Z` receives the canonical query only, not retrieved skills or assembled
  `H_0`.
- Both log-probability directions come from the training policy stack;
  serving/provider log probabilities are not scientific inputs.
- Terminal reward is the only correctness channel into training. Answer keys,
  hidden tests, verifier truth, and native evaluator payload never enter the
  evaluated model's context.
- Legal agent failures, including reward zero, remain in the fixed batch.
  Environment, evaluator, snapshot, budget, source-append, authoring, or
  checkpoint infrastructure failure ends the attempt without replacement or
  retry.
- A verified phase executes every proposal as one mutation in the same
  `run()` call, or the attempt fails. There is no executable open-phase state,
  pending-success return, no-action success, truncation, authoring skip, or
  subset commit.
- Production retrieval returns every applicable active skill with full
  content. H0 budget overflow is an error, not a top-N truncation.

The benchmark layer is governed by
[`docs/benchmark-evaluation-guide.md`](docs/benchmark-evaluation-guide.md).
Licensed task content, choices, answers, per-item outputs, GPU credentials,
private keys, and private GPU connection instructions must never enter Git. The
use of a configured direct GPU server may be documented; all private endpoint
and connection details remain local. Public
model-facing code ships in `skillev`; trusted evaluators and private dataset
builders, including the formal fixed benchmark worker, ship separately in
`skillev-private-evaluation`. Public completion cases remain correctness and
real-backbone vertical fixtures, not formal benchmark environments.

## Development

- Repository: `https://github.com/YJLi-new/SKILLEV-new`
- Python: `uv`
- Related tests during development; `make check` once before push.
- GPU work uses the privately configured direct GPU server; do not use NCSA Delta,
  Duo, Slurm, or scheduler commands.
- The server has eight NVIDIA H800 GPUs. Before every run, inspect all eight and
  select only GPUs with no other user's compute process. Never terminate or
  modify another user's process. Every CUDA launch must set
  `CUDA_VISIBLE_DEVICES` explicitly to the selected physical GPU or subset.
- A full SkillFlow-BayesianImprove run assigns three distinct GPUs at launch
  time: the verified SGLang inference GPU plus two healthy process-idle GPUs
  for trainer coordination/state and the gradient worker. Human role numbers
  are not hard-coded physical CUDA indices. CUDA OOM fails the uncommitted
  attempt; the supervisor does not reserve or add a standby GPU.
- Do not modify `idea.tex` or `AGENTS.md`.

See [`docs/reuse-boundary.md`](docs/reuse-boundary.md) for the exact boundary
between reused method-neutral infrastructure and retired historical method
code.
