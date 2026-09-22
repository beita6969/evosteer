# Seven-domain training and gradient main-path follow-up — 2026-09-08

## Scope and execution batches

The owner requested seven training domains, excluding WebShop. The current
entry uses `seven-domain-rotating-eighth-8x4-seed0@1`: one question occurrence
from each allowed domain plus a rotating extra domain, four independent rollouts
per occurrence, B=32. The seven domains are HotpotQA, TriviaQA, AIME2026,
HealthBench, ALFWorld, MBPP+, HumanEval. Historical eight-domain files may supply
unchanged source lanes, but WebShop is neither hydrated nor sampled. A seven-only
source file no longer depends on the legacy eight-domain population validator.
This population change is explicit in the provider and runtime identity; old
checkpoints and their labels are not rewritten or called equivalent continuations.

Implementation was grouped by behavior rather than file-by-file gates:

- **Scoring execution (medium risk):** collect detached edge scalars together;
  preserve sequential backward and Python `fsum`; scope adapter/training-mode
  changes over each direction including checkpoint recomputation; remove a
  redundant gradient clone. Validate all three gradients and one Adam update.
- **Distributed ownership (high risk):** ready trajectories are assigned using
  actual prepared-edge token lengths. CPU edge-plan preparation overlaps CUDA.
  Workers send individual, globally B-scaled contributions in owned dtype buckets;
  a single coordinator CUDA owner adds only canonical positions. No rank-local
  shard SUM substitutes another floating-point order. Byte reservations and
  canonical priority bound buffering without head-of-line deadlock. CPU tests
  cover real two-process B32 math, cancellations and unexpected owner failures.
- **Preparation and visibility (medium risk):** bounded independent environment
  initialization preserves record order and drains cleanup on cancellation.
  Startup stages, live direction/edge/prefix progress, actual worker count,
  contribution/buffer capacity and distinct wait reasons are observable. A
  four-step report explicitly has too few post-warmup observations for the
  existing steady-throughput gate; that gate is not relaxed.
- **Memory and integration (high risk):** known resident frozen parameter storage
  is not repeatedly offloaded. Other no-grad tensors are not presumed immutable.
  Host-copy bytes/times are separately reported; ordinary training does not add
  per-edge CUDA synchronization for profiling. Existing bounded pinned storage
  remains opt-in; no unqualified asynchronous DMA implementation was promoted.

All changes were implemented and checked by the main coding agent; no independent
review agents, manual hash checks or repeated small-patch full checks were used.
The integrated CPU check was repeated after concrete boundary/test failures and
then the final seven-only source-entry correction. Historical successful kernel
probes were not rerun merely because documentation changed.

## Mathematical boundaries retained

Each edge keeps its original action token IDs and its own K denominator;
trajectory losses retain `(delta/T)^2` with global B=32 scaling. Scores and
residuals remain tied to the fixed policy and library. Ready-first *computation*
does not change canonical gradient addition, complete-batch flow normalization,
posterior preview/commit, phase detection, evolution or durability boundaries.
A failed artifact or worker invalidates the whole batch, not only its shard.

Microbatch size remains **1**, checkpoint threshold remains **1**, activation
offload threshold remains **32768**, and existing Torch convolution / deterministic
FLA selection remain unchanged. Earlier rejected microbatch/conv candidates were
not re-enabled. Full-vocabulary normalization was not replaced by target logits;
detached inference caches were not used as trainable prefix caches. Differentiable
prefix sharing, direction-split workers, fused vocabulary loss and selective
operator checkpointing remain unqualified research/engineering candidates, not
claimed fixes or default speedups.

## Fixed-input CUDA evidence (not new training)

The old fixed complete batch contains 32 trajectories and 205 execution edges
(410 directional edge scores). Its longest full edge is 25,565 tokens, so it does
**not** exercise >32K offload. Its historical population is retained only for the
execution comparison, not executed again as an eight-domain training run.

| Execution | Gradient wall time | Numerical comparison |
|---|---:|---|
| Baseline, one H800 | 866.666 s | Reference |
| New single-owner scoring, one H800 | 835.339 s | All compared quantities bitwise identical |
| Two gradient workers, global canonical reduction | 435.064 s | All compared quantities bitwise identical |

The single-owner observation is about **1.0375x** (3.75% higher throughput).
Two real gradient workers achieved **1.9920x** versus that baseline (about half
the gradient wall time), or **1.9200x** versus the new single-owner execution. All forward/backward/Z gradients, 410 directional
means, 32 residuals, parameters after one Adam update and Adam moments/steps were
bitwise identical. This is one cold fixed-input pair on two healthy selected H800s,
not steady-state on-policy throughput; no uncertainty or thermal-normalized
speed guarantee is inferred. The earlier 1705-second observation used another
runtime/thermal condition and is not the baseline for this comparison.

A separate controlled test extended an existing token prefix to 32,768 tokens
with its original 54 action tokens, using both adapters and opposite gradient
signs. It is **not** a sampled trajectory. Scores and both adapter gradients were
bitwise identical. Saved/restored traffic per edge fell from 11,821,392,736 to
9,787,155,296 bytes by retaining 2,034,237,440 bytes of already-resident frozen
parameter references. Peak allocation was unchanged. Individual cold timings were
15.625→14.152 s and 12.442→11.378 s; these two edges do not establish a workload-wide
offload speedup. The unavailable second batch's original long-tail artifact has
not been substituted or declared verified by this controlled case.

## Acceptance status

- CPU integrated check including the mandatory-three-card entry: **3546
  passed**, one explicit CUDA opt-in skip; Ruff/format, mypy (543 source files),
  both wheels and model-package boundary passed on 22049 CPU. The loader
  correction passed its seven-only/legacy-source equivalence test. The subsequent
  owner requirement for exactly two real gradient owners plus separate inference
  is checked before the training entry initializes CUDA; its ten-test affected
  group and final integrated check passed.
- Real fixed-source EvalPlus subprocess probes covered correct code, Base failure,
  Plus-only failure, native timeout, failed reference and missing profile. Missing
  official runtime dependencies were installed in a private evaluator overlay;
  infrastructure errors produced no success/failure label.
- GPU comparisons do not publish an adapter, update posterior cells, collect
  fresh samples or count as optimizer commits in a training run.
- New seven-domain **four-step 2+2 fresh-process integration is not yet completed**.
  The owner now requires three physical cards: one inference GPU and two actual
  gradient workers (coordinator participates). No two-card fallback is permitted.
  Resource checks initially found only two eligible cards; at 18:54 UTC a third
  briefly met the permitted shared-occupancy condition. The planned 22048 GPU7
  inference / GPU0+GPU4 gradient mapping was rejected by its launch check at
  18:58 UTC: another task had grown GPU4 to 45–79 GiB. The service wrapper exited
  before inference/CUDA startup; no new training or service began. A continuously
  available third card is required, not a two-card fallback. No other user's
  process was stopped.
  At 19:06 UTC the owner explicitly exchanged resources with another session: the
  current mapping is **22048 GPU5 inference, GPU4+GPU7 gradients**, releasing GPU0.
  GPU5 was confirmed free at 19:08 UTC and its inference service became healthy.
  The owner explicitly authorized stopping the old project training on GPU4;
  its launcher and training process exited after TERM, with capacity rechecked.
  New seven-domain B32 training launched at **19:13 UTC**. At 19:14 UTC both
  gradient ranks had loaded on GPU4/GPU7 and official session hydration was
  complete. Four committed steps and fresh-process recovery are still pending.
- At **19:40 UTC**, the first complete transaction was committed and the second
  batch was collecting. Step wall time was **1473.45 s** (24.56 min), rollout
  span **1022.73 s**, overlap **990.68 s**, gradient tail **448.45 s**, and
  durability **2.27 s**. Thirty of 32 gradient contributions completed before the
  last rollout; both ranks computed (host compute totals **485.09/401.57 s**).
  Two 50-step ALFWorld trajectories formed the tail, including real >32K
  activation offload. The checkpoint has **four posterior updates / three
  cells**, and its adapter/checkpoint transaction is committed. About 2.44
  steps/hour for this first warmup step is **below** the 4.2 target; the remaining
  three-step extrapolation was roughly 74 minutes, not a guaranteed ETA. Different
  domain composition prevents attributing historical whole-step timing changes
  solely to execution optimizations.
- Four steps cannot validate W=50 natural phase transitions, steady-state
  throughput (two warmup plus at least three measured steps), or long-run Bayesian
  calibration/evolution efficacy. Those claims remain outstanding.
