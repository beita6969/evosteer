> Superseded plan: the owner canceled the queued new four-step before launch.
> Continue with [formal250/B28](bayesian_formal_250.md), not that queued launcher.
> Native-thinking complete-chain FIFO/fair qualification finished exact334/334
> outputs/stops:804.627s vs792.958s (1.45% wall reduction, not on-policy throughput).

# ALFWorld per-turn execution changes

This is a separate execution condition, not a reduction in training work. Seven domains,
B32, seed0, reasoning/action token budgets, raw categorical sampling, full terminal
scoring, TTB and weighted Beta updates remain unchanged. One Qwen3.5-9B owner per
trajectory; inference and two gradient workers use three distinct devices.

## Implemented

- Provisional-step admission queues immutable history without waiting for prefix
  encoding. Per-trajectory CPU publication remains ordered. Queued histories and
  retained edge plans have independent byte limits; preparation failure fails the
  batch. Terminal records must match before prepared edges are reused.
- Validated artifacts enter gradient preparation before environment cleanup finishes;
  cleanup must still succeed before the complete batch can seal or update.
- Optional bounded device-resident provisional gradients use owner-local LRU spill.
  They remain unscaled ∇Delta until the complete trajectory supplies reward/horizon.
  No cross-trajectory `.grad` sharing and no change to canonical accumulation.
- Optional `horizon-action-aging@1` request ordering gives declared long tasks and
  ready actions a bounded priority bonus. Waiting age prevents starvation; idle
  capacity is never reserved. FIFO remains the compatibility default.
- Wait reports distinguish missing edges, CPU preparation, other-owner affinity,
  buffer capacity and remote contributions. Artifact, gradient and canonical-merge
  counts are shown together with seal/transaction stage and stage age.
- Each committed batch retains its phase sidecar. Missing server timings stay null;
  an unpopulated backend queue-time zero is not reported as a measurement. Service
  Prometheus counters remain aggregate context, not invented per-request CUDA time.
- Warmup is process-local, with process/service identity retained. All step wall
  times are reported, including cold steps after fresh-process recovery.

The opt-in profile is `configs/training/protocol13_turn_latency.yaml`: CPU queue
128 MiB, retained plans 1 GiB, device provisional cache 512 MiB per gradient owner,
canonical host buffer 2 GiB. Existing qualified FP32 GDN state, page64, prefill4096,
extra-buffer and overlap settings are preserved. No speculative decoder, quantization,
new kernel, smaller reasoning allowance, forced skill call or truncated batch is used.

## Validation at this milestone

- Focused CPU tests cover early next-turn admission, preparation failures and byte
  bounds, final-plan reuse, interleaved gradient ownership, fair/cancelled requests,
  cleanup failure and process-local warmup. Passed unrelated kernel checks were not
  repeated. Local WSL test/type-check timeouts are not counted as passes.
- One completed final CPU `make check` on the approved Linux host: **3624 passed,
  1 explicit CUDA opt-in skip**; Ruff/format 1058 files, mypy 554 source files, both
  wheels and model-boundary check passed. Earlier lint/type failures were fixed
  before this successful check.
- Fixed complete **B32 / 167-edge** two-gradient replay: **501.234 s**, versus the
  existing sealed single-gradient reference **872.431 s**. Forward/backward/Z
  gradients, edge values, residuals, Adam parameters and optimizer slots are all
  **bitwise equal**. This is fixed-input execution validation, not new on-policy
  training or a fresh 1.74x gain over the already optimized dual-gradient version.
- Complete fixed multi-turn FIFO/fair comparison: **334/334 requests in each**,
  exact output IDs and stops. FIFO **733.112 s**, fair **730.437 s** (about 0.37%
  less time in this pair, not a demonstrated material speedup). These requests
  used the earlier native-thinking-off condition. A new native-thinking-on pair
  and the new four-step 2+2 run are separate, pending qualifications.

Four steps test correctness/recovery, not steady throughput. W50 needs at least
100 batches even to form both residual windows; natural evolution and calibration
quality cannot be certified by a four-step result. The previous completed run's
[loss/reward curves](machine-results/bayesian_four_step_curves_20260908.svg) are
historical observations, not results of this new execution condition.

## Synchronization with the evidence/evolution and IID sessions

The new training snapshot includes the first repair batch from `c6cb683`, rather
than continuing with the older execution-only frozen tree. Resolved state records
method semantics v5, evolution configuration v7, actual declaration/execution
links, and pre-batch calibration diagnostics. Raw author responses are durable
before validation; this does not mean a real author mutation has already occurred.
See [the remaining Bayesian work](BAYESIAN_EVOLUTION_REPAIR_STATUS.md).

A read-only extension of the saved real-Qwen **B32 / 167-edge** comparison now
passes through the current complete projection: diagnostics, weighted posterior,
public author evidence and detector state are exactly equal for sealed and
provisional two-owner scoring. Both rebuild the same four updates / three cells
and reject the same repeated complete batch. W50 returns insufficient residual
history; Phi proposals are **not** covered by this observation. No new model call,
optimizer update or training evidence was generated by this CPU comparison.

The requested **native thinking-on** run is an explicitly new sampling condition:
all seven domains enable Qwen native thinking for the existing reasoning `c_t`
pass. The following structured action consumes that exact reasoning but does not
add a second hidden thinking prefix; its sampled token range remains exactly the
one scored by forward/backward TTB. Budgets and raw categorical parameters stay
unchanged. Rollout config v4 / the paired prompt-encoder identity prevent silently
resuming a historical native-off condition as native-on. Legacy v3 remains off.

Hotpot's explicit evidence-deliberation guidance is shared with the updated IID
owner through one public constant and is enabled for this training run only as a
declared condition. No gold answers, extra solver or forced reasoning length are
added. The IID ALFWorld 20-minute overlay is **not** copied into training.

The seven-domain thinking/configuration/TTB-prefix bridge has **15 passing CPU
cases**. The combined Linux CPU `make check` passed: **3648 passed, 1 CUDA opt-in
skip**, Ruff/format 1063 files, mypy 556 sources, both wheels and model boundary.
The new four-step run has not yet started at this update; native-on request
qualification is still in progress. It cannot certify natural W50 evolution, real-author
continuation or calibrated causal benefit.
