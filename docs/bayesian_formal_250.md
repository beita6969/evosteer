# Current BayesianImprove formal training condition

The owner's 2026-09-08 instruction supersedes the queued four-step rerun and
B32/rotating-eighth-question schedule. The queued four-step launcher was canceled
before any new trainer started. Historical B32 runs remain unchanged evidence.

## One configuration, one application

`configs/training/bayesianimprove_250.yaml` declares:

- Qwen3.5-9B, frozen BF16 base, one task owner. Forward/backward LoRA rank4,
  alpha8, dropout0; train logZ. AdamW LR1e-4 for adapters and Z, decay0;
  no clipping/extra KL. TTB beta1/epsilon0.1.
- 250 steps, seed0, **B28**. Each step contains exactly one occurrence from
  HotpotQA, TriviaQA, AIME2026, HealthBench, ALFWorld, MBPP+, HumanEval, with four
  independent rollouts each. **1,750 occurrences, 7,000 trajectories.** No eighth
  question and no WebShop. Repeated source questions retain their labels but get
  distinct occurrence/rollout coordinates; these are not 7,000 independent questions.
- Latest owner revision: **six static domains allow at most 2 turns; ALFWorld
  allows at most 50**. All retain reasoning1024/action2048/input65536.
  Native thinking is enabled for the reasoning `c_t` request. The
  structured action request conditions on that actual `c_t`; its sampled tokens
  are scored unchanged. No consultants, votes, answer merging, shortened ALFWorld
  budget, IID timeout overlay or cross-step off-policy prefilling.
- On input overflow, **truncate input and continue**, never turn the overflow
  into a terminal failure or a negative label. The explicit
  `h0-head-tail-recent-tokens@1` condition keeps H0 plus recent input token IDs.
  At least half the window goes to recent history/current suffix; H0 uses up to
  the other half, with unused capacity returned to recent history. If H0 exceeds
  that share, retain its head (system/task) and tail (interfaces). Inputs within
  the limit are unchanged. Full trajectories, original H0 counts, generated
  action IDs and action denominators remain recorded unchanged. Logical,
  admitted and removed input counts are separate phase-sidecar fields.
- Window metadata is persisted in initial context and consumed by rollout,
  sealed scoring and provisional F/B preparation, including worker transport and
  restored artifacts. It is not inferred from a worker's local defaults. This
  conditioning change and the static two-turn budget are **owner-approved new
  conditions**, not equivalent continuation of the old full-history run.
  Completion guidance also retains the JSON envelope and code escaping rules
  previously lost when domain instructions overrode the base action contract.
- Beta(1,1), LCB k1, diagnostics W50/rho0.05. 249 search slots and one closure
  slot, at most two naturally triggered library mutations. Existing method v5
  and evolution v7 controls remain explicit in the resolved application. The
  per-cycle author envelope is explicitly 64 calls, not unlimited/fallback authoring.
- Retain cadence checkpoints at10,20,...,250. Separately, rolling last-three
  transaction snapshots support exact recovery from failures between cadence
  points; phase/final products remain independently retained.

Run the public composition entry with private bindings (paths, scorer deployment,
three physical GPU UUIDs, serving endpoint and namespace) outside Git:

```bash
CUDA_VISIBLE_DEVICES="$GRADIENT_GPU_A,$GRADIENT_GPU_B" \
  python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m skillev_private.experiments.bayesian_improve_training \
  --config configs/training/bayesianimprove_250.yaml \
  --bindings "$PRIVATE_BINDINGS" --run-root "$PRIVATE_RUN_ROOT"
```

The entry calls `SKILLEVApplication.build_formal()` or `resume_formal()` and the
existing `EvolutionLoop.run()`; it does not create a parallel training algorithm
or depend on retired admission/attestation scripts. `--resume` takes a complete
method snapshot of this same run. Changed batch, thinking, budgets or scorer
conditions must not be represented as an equivalent continuation of old B32.

`configs/serving/qwen35_9b_bayesian_250.yaml` reserves a67584 total context
(65536 input+2048 output), preserving the tested FP32 Mamba checkpoints,
page64-aligned hybrid cache, deterministic raw sampling and decode graph profile.
One GPU serves rollout/base-judge/author requests; two distinct GPUs actually
compute gradients, including the coordinator. Global canonical reduction and
complete-batch posterior normalization/commit remain unchanged.

## Measurement and honest completion criteria

4.2 ordinary steps/hour is a target.72 hours is a planning estimate, **not a hard
stop or a performance guarantee**. Per-process warmup, every actual step wall time,
preparation, service state, rollout phase detail and transaction state are recorded.
Resume forecasts do not divide all historical steps by only the new process time.
A below-target measured rate is reported as needing human performance decisions;
no batch truncation, threshold relaxation or reward change is an automatic remedy.

The previous complete334-request native-thinking FIFO/fair comparison retained
exact outputs/stops (804.63s vs792.96s;1.45% lower wall time). The previous fixed
B32 gradient/Adam/projection comparisons are execution evidence, **not new B28
training or proof of natural evolution**. The current domain-specific budgets
and context window are an explicit new condition; older four-step timing is not
its throughput result.

A running formal experiment is not a claim that calibration quality, real author
success or natural W50 evolution has already been verified. Actual posterior
coverage, phase reasons, mutation count, new-library continuation and recovery
must be read from the committed evidence. No-trigger/no-improvement are legitimate
outcomes; an author/infrastructure failure stops rather than becoming a fake no-op.

## Previous implementation validation (before the failed first launch)

22049 Linux CPU `CUDA_VISIBLE_DEVICES="" make check`: **3656passed,1CUDA opt-in
skip**,142warnings,pytest289.52s; Ruff/format1067files,mypy559sources,bothwheels
and model-wheel boundary passed. New eight-condition tests cover full schedule,
real budget hydration through the shared session builder (mocked ALFWorld process),
formal build/full-resume delegation, device-role/budget checks, incompatible
continuation rejection and torn monitoring logs/process-cold steps. The first
focused pass found a missing fake-evaluator constructor argument; it was fixed
before the final check. Unchanged numerical/GPU kernels were not re-profiled.
No review sub-agent, manual hash check, AGENTS or idea.tex edit was used.

## Failed first launch and authorized repair

The all-domain-50-turn attempt ended at **2026-09-09 01:23:24 UTC**, with
**zero committed optimizer/posterior steps**. Three model requests exceeded
65536 input tokens. The whole batch was rejected and CUDA owners drained; the
initial checkpoint and failed logs are preserved, not rewritten as training
success. Numerous malformed actions amplified the long histories. No timing
from that incomplete batch establishes accepted step throughput.

The owner then authorized repair/restart, input truncation instead of termination,
and a two-turn cap for static domains. The revised configuration has its own
identity and starts a new Step-0 run. Revised validation/launch status is recorded
in HANDOFF_PUBLIC; the older 3656-test result above does not qualify new edits.

### Revised condition validation

130 focused tests passed. An initial full check found the new input-window import
violated rollout's existing policy-interface boundary. The code moved into that
interface; the boundary test was not weakened.28 affected tests then passed.
Final22049 CPU `make check`: **3662passed,1CUDA opt-in skip**,145warnings,
pytest284.78s; format/lint1068files,mypy559sources,both wheels/model boundary pass.
The real-tokenizer CPU overflow probe also passed, using private prior history
plus clearly synthetic length extension; it is not on-policy or GPU training.
No unchanged kernel benchmarks were repeated. No subagents/manual hash checks.
