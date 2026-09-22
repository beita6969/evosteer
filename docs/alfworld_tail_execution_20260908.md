# ALFWorld long-tail execution repair — 2026-09-08

## Scope and invariants

Seven domains, seed 0, 32 independent trajectories, one Qwen3.5-9B task agent,
production W=50 and the existing Bayesian application remain unchanged. On
22048, GPU5 serves inference and GPU4/GPU7 compute gradients, including rank 0.
The preceding run retains its one committed step; its incomplete second batch
was discarded, not shortened or relabelled. The new execution condition is a
separate run, not a transparent continuation of that checkpoint.

## Implemented

- Environment IPC uses one absolute deadline for request writing and the entire
  newline response, with bounded buffered reads. Partial replies, trickling,
  oversized responses, EOF and protocol errors fail as infrastructure errors.
- Answer-free sidecars expose active trajectory/turn, queue age, phase tokens,
  environment timing/repetition and gradient progress. Unavailable backend
  timing stays null; non-streaming response latency is not called TTFT.
- `long-horizon-first-stable@1` starts the four long interactive trajectories
  early without changing canonical positions, seed coordinates or reduction.
- Completed ALFWorld steps can prepare F/B/Z gradient contributions while
  interaction continues. Each trajectory owns its partial state. Terminal
  reward, final horizon and global B scaling are applied only after validating
  the final artifact; posterior normalization and optimizer commit remain whole
  batch operations. The real slash-delimited ALFWorld task-family route is tested.

## Failures fixed before promotion

The first native cached configuration produced non-identical sampled tokens.
Two distinct causes were isolated on fixed real requests: intermediate GDN
checkpoints were rounded to BF16 before entering an FP32 cache, and unaligned
prefill chunk boundaries changed with concurrency. The opt-in service profile
now retains FP32 checkpoints while preserving the old BF16 output-dot operand,
and aligns page/prefill scheduling to 64 tokens. The shared SGLang installation
is not modified. Neither sampling controls nor token budgets were changed.

The corrected cached and uncached services matched exact output IDs and stop
boundaries for 16 fixed short/long reasoning/action inputs under base, Step-0
LoRA and an updated LoRA reloaded under the same alias, at concurrency 1 and 4.
Each condition had 64 comparisons, including its 16 reference rows. A separate
mixed base-judge/current-actor replay also matched all 16 actor outputs. These
are fixed-request diagnostics, not new benchmark results or training samples.
Cache hits were nonzero. Summed request latency at concurrency 4 fell roughly
20–23%; this is not a claim about end-to-end training throughput or measured
prefill time. The ordinary serving default remains cache-off.

## Numerical qualification

The complete committed B32 input contains 167 edges, including the actual
50-turn, >32K tail. Sealed single-owner replay took 872.43 seconds; provisional
single-owner replay took 871.45 seconds. F/B/Z gradients, edge scores,
residuals, Adam parameters and optimizer slots were bitwise identical. This
does **not** demonstrate faster edge computation: the intended benefit is
earlier work and less post-rollout waiting. Dual-owner replay finished in
485.23 seconds (1.80x the sealed single-owner baseline); all of the same
quantities, including Adam slots, remained bitwise identical. This is a
complete fixed-input gradient comparison, not whole-step throughput.

Targeted tests cover deadline/protocol failures, live waiting, launch order,
provisional failure cleanup and ordered gradient/Adam equivalence. The full
CPU check exposed a new test's environment-variable leak into spawned-server
tests; it was fixed, and their combined 19-test group passed. The final
integrated `make check` passed on 22049 CPU with CUDA explicitly disabled:
3601 tests passed, one CUDA opt-in skip, format/Ruff, mypy, both wheels and
the model-package boundary. The successful fixed GPU probes were not repeated
after test-only/documentation corrections. No independent review agents or
manual hash checks were used.

These qualifications preceded the real runs below. Four steps cannot establish
steady throughput or natural W50 phase detection. No
unqualified microbatch, reduced trajectory budget, special Generate trigger,
posterior change or inactive-environment expert optimization was introduced.

## First real attempt exposed an additional scheduling defect

The new attempt started at 21:04 UTC. Sidecars confirmed early ALFWorld
provisional gradients, but all four sparse interactive streams acquired rank 0:
first-available-worker ownership could recreate a single-card tail even though
both ranks initially computed other trajectories. The attempt was stopped at
21:07, before any optimizer/posterior commit; only the initial checkpoint exists.
Its outputs are retained separately. This was not an OOM or a shortened batch.

`balanced-live-trajectories@1` now counts unfinished actors even between edges
when assigning provisional affinity. A busy rank can receive its share of future
work; another idle rank cannot steal that trajectory's partial state. Finished
trajectories release scheduling load. Global canonical reduction is unchanged.
The affected 24-test group passed, including single-owner/state-only-coordinator,
sticky affinity, failure cleanup and complete two-process gradient/Adam checks.
The updated CPU `make check` passed 3605 tests (one CUDA opt-in skip),
format/Ruff, mypy (548 source files), both wheels and package boundary. The
updated complete fixed-input CUDA comparison also passed: 502.92 seconds,
1.73x the original sealed single-owner baseline, with bitwise-equal F/B/Z,
edge scores, residuals, Adam parameters and slots. It was not faster than the
previous already-balanced replay (485.23 s); this fix prevents the observed
sparse-arrival skew, not a universal per-edge speedup. V2 used a separate frozen
source and run identity after these checks, not a hot patch to the stopped run.

## Completed real four-step / fresh-process test

Training source `ed04e85`; seven domains, B32, seed 0, production W50. V2 ran
from about 21:23 to 22:30 UTC. Both two-step processes exited successfully.
The first process committed steps 1–2; a new process and new model/optimizer
objects restored that cutoff, matched the recorded method state, and committed
steps 3–4. This is real on-policy integration debugging, not formal benchmark
acceptance or a paired uninterrupted-GPU control experiment.

| Step | Complete step (s) | Rollout span (s) | Post-rollout gradient tail (s) |
| --- | ---: | ---: | ---: |
| 1 | 866.57 | 823.66 | 40.67 |
| 2 | 1013.48 | 877.54 | 133.26 |
| 3 | 1027.36 | 919.46 | 105.55 |
| 4 | 1032.27 | 861.75 | 168.13 |

Four complete step intervals total **3939.69 seconds / 65.66 minutes**, or
**3.66 steps/hour**. Hydration, process replacement and cleanup are separate.
The first step was 1.70x faster than the preceding run's 1473.45-second warmup
observation; that is not a controlled whole-run comparison. The 4.2 steps/hour
performance target is **not accepted** and remains a human performance decision.
Only two post-warmup steps exist, fewer than the declared throughput gate needs.

All **128 trajectories** appear in four committed 32-trajectory evidence batches.
Posterior update counts are **[3, 0, 5, 0]**, eight updates across six cells;
reconstruction passed. Zero-invocation batches advanced their evidence cursor
without fabricated credit. All four step transactions are committed, the task
cursor is 128, model/optimizer step is 4, and the final forward adapter was
published. The periodic progress sidecar's last pre-completion sample is not the
completion authority; the final summary and committed transactions are.

There were **zero phase detections and zero library mutations**. Library identity
and active IDs remain unchanged. Eight invocation events are sparse, correlated
training evidence, not eight independent questions or calibrated causal effects.
This run does not validate natural W50 evolution, a real new authored skill,
post-mutation continuation, or IID gains. No threshold, reward, sample count or
skill-credit definition was changed to manufacture those outcomes.

## Validation and resource closure

- Targeted IPC, live-progress, launch-order and provisional-gradient tests passed.
  The final affinity group was `test_provisional_affinity.py`,
  `test_streaming_step.py` and `test_local_gradient_stream.py`: **24 passed**.
  Bootstrap isolation used `test_launch_sglang.py` plus
  `test_sglang_sampling.py`: **19 passed**. CPU checks ran with CUDA disabled.
- Integrated milestones passed **3601**, then **3605 tests** after the observed
  affinity fix; each had one CUDA opt-in skip. Final format/Ruff, mypy, both
  wheels and package-boundary checks passed. GPU qualifications are reported
  separately above. Documentation-only updates reuse those successful checks.
- GPU jobs were the bounded cache-off/on and adapter-isolation services/replays,
  sealed/local/dual/affinity complete-B32 gradient comparisons, the stopped v1
  attempt, and completed v2 training plus its serving and monitoring processes.
  All used only **22048 GPU4/5/7**; the unrelated GPU7 process was not touched.
  Main-thread checks at **22:32 UTC** found no remaining process from this run:
  GPU4/GPU5 empty, GPU7 retaining only the pre-existing unrelated allocation.
  V2 occupied three cards for roughly **67 minutes** (about 3.3 card-hours);
  earlier fixed-input qualification is additional, not included in throughput.
- No independent review agents, manual hash checks, new IID round, hardware
  changes, `idea.tex` edits or `AGENTS.md` edits. Environment IPC was a small
  measured component, so no expert wrapper/domain-randomization changes were made.
