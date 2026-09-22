# Protocol 13 formal-training throughput gate

## Requirement

The exact-eight schedule remains scientifically unchanged: 250 optimizer steps, one question
from each of eight IID domains per step, four trajectories per question, and therefore 32
trajectories per optimizer update. The 72-hour wall-clock bound requires:

\[
250 / 72 = 3.4722\ \text{steps/hour}
\]

The deployment target is **3.8 steps/hour** (15.79 minutes/step), leaving margin for phase
transitions, cadence checkpoints, long interactive tails, and transient service variation. A
speed decision uses two warmup steps followed by at least three measured steps. Below 3.4722
steps/hour is `needs-human-decision`; it is not silently accepted as a formal launch.

## Measured pre-repair baseline

The superseded two-GPU debug attempt completed six contiguous optimizer steps before it was
stopped for performance work. It was not continued or combined with another run.

| Segment | Mean seconds/step | Effective rate |
|---|---:|---:|
| Complete step, steps 1--6 | 2,066.3 | 1.742 steps/hour |
| Rollout and terminal evaluation | 1,049.0 | — |
| TTB gradient | 1,014.8 | — |
| Post-gradient durability | 2.2 | — |
| Complete step, excluding two warmup steps | 2,017.7 | 1.784 steps/hour |

At that rate, 250 steps would require about 140 hours after warmup. The observation also rules
out checkpoint serialization as the main stable-state bottleneck: ordinary durability occupied
about two seconds per step, while rollout and gradient each occupied roughly half of the step.

## Implemented execution repairs

These changes affect deployment scheduling, not the BayesianImprove objective, samples, rewards,
or update count.

1. The debug/formal-shape runner exposes all workflow limits as command-line controls. Its tested
   profile uses 16 resident trajectories, 8 model requests, 8 environment calls, 8 terminal
   evaluations, 4 process graders, and 16 SGLang transport threads. The SGLang deployment admits
   eight running requests.
2. Session construction/reset now runs outside the asyncio event-loop thread. Official WebShop
   and ALFWorld blocking `step`, action-surface read, and process cleanup operations also run in
   worker threads. Session setup and cleanup have independent timing lanes and do not consume the
   SGLang transport executor.
3. The distributed TTB coordinator can explicitly participate in gradient computation. The
   sealed 32-trajectory batch is token-cost partitioned across rank 0 and every worker; each rank
   uses the unchanged global `1/32` scaling, collective reduction sums the disjoint shards, and
   detached trajectory math is reordered back to the original batch order before the one AdamW
   update. Legacy callers retain the previous worker-only mode unless they opt in.
4. Each completed step writes an answer-free `performance.jsonl` record containing rollout
   queue/service timings, setup/cleanup timings, model token counts, trajectory percentiles,
   gradient time, residual commit time, post-warmup throughput, and the hard-gate decision.
5. The progress monitor now reports the 3.4722 hard floor and 3.8 target. The obsolete one-step-
   per-hour threshold is no longer used.

The first optimization milestone deliberately does **not** merge reasoning and action generation,
change token or horizon limits, pre-generate the next policy's samples, or alter TTB floating-point
reduction semantics. Length-bucketed teacher forcing and within-step rollout/gradient streaming
remain escalation options only if this lower-risk profile fails the measured gate.

## Optimized real-run result

The first optimized deployment attempt (`perf-v1`) was a fresh run from step zero. It completed
two contiguous optimizer steps before the owner requested that the experiment stop. A third
batch had begun collection, but it was not committed and is not counted.

| Step | Rollout + evaluation | Exact TTB gradient | Residual commit | Total | Cumulative rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 570.7 s | 557.0 s | 7.9 s | 1,135.6 s | 3.184 steps/hour |
| 2 | 654.3 s | 485.1 s | 0.7 s | 1,140.0 s | 3.164 steps/hour |

This demonstrates an approximately **1.82x** end-to-end improvement over the six-step baseline,
and confirms that both training ranks concurrently execute disjoint gradient shards. It does not
pass or fail the formal 72-hour gate: both completed steps belong to the declared warmup, while a
decision requires at least three subsequent contiguous steps.

The answer-free telemetry also exposed the next serving bottleneck. The SGLang instance admitted
eight running requests but allowed only one model identity per batch. HealthBench's base-model
rubric requests therefore could not share a running batch with forward-LoRA rollout requests. A
replacement profile with 12 running requests and two model identities per batch was attempted,
but its new FlashInfer warmup path required a local JIT tool that was absent from that runtime.
The owner stopped the work before that deployment was repaired or a second training run began.
There is no surviving replacement service or training process.

Consequently, the completed implementation is a measured first-stage performance repair, not an
authorization to launch the 250-step formal run. Any continuation must start a new run from step
zero, retain the same frozen dataset and method, repair and validate the serving runtime first,
and then apply the two-warmup-plus-three-measured-step gate.
