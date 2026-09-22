# Protocol 13 execution acceleration — 2026-09-04

Historical run/configuration record. For the subsequent local stream,
ready-first scheduling, server action stop, microbatch/offload and grader-broker
changes, see [the 2026-09-08 execution update](execution-performance-20260908.md).
The availability and running-job statements below describe their dated checks,
not current GPU status or a standing resource restriction.

Status (2026-09-06 UTC): the corrected full-32 GPU gradient comparison and serving canaries
pass. The owner's latest instruction replaces the fresh eight-step prerequisite with a direct
full 250-step run on either approved endpoint once three exclusive cards are available.
No shared-memory trial or new training job is currently running. This authorization does not
turn unmeasured throughput or existing formal-admission results into a pass or a 72-hour claim.

## Preserved scientific configuration

`idea.tex` is unchanged. Keep seed 0, 250 formal updates, eight questions per update,
four trajectories per question (32 total), the original decoding budgets, frozen Qwen3.5-9B,
bf16, both rank-4 LoRA adapters, the trainable Z head, TTB normalization, reward contracts,
projection/Operator rules, and cadence 10. The authoritative IID list is HotpotQA, TriviaQA,
AIME2026, HealthBench, WebShop, ALFWorld, MBPP+ hard, and HumanEval. Private questions,
answers, rubrics, artifacts, and per-question outputs stay outside Git.

## Implemented execution changes

- One execution-only `TrainingPerformanceConfig`, persisted as resolved values and usable through
  the formal runtime binding as well as the integration driver. Legacy bindings remain readable.
- Serving schema v3 adds explicit running/model capacity. Candidate: resident 16, actor 8,
  environment 8, terminal 8, grader 4, transport 16, server running 12, base/LoRA slots 2.
- Actual-interpreter preflight checks official Ninja, CUDA build tools, package versions and
  supported SGLang flags. No package/framework upgrade or attention-backend substitution.
- Bounded instance-local LRU for detached, adapter-free frozen query features. The trainable
  Z head is recomputed every time; changing Z does not invalidate immutable base features.
- Canonical prepared edge inputs retain full repeated prefixes and original action token IDs.
  Sealed scheduling can use the true encoded cost; live streaming freezes partitions from public
  query/budget estimates before generation and records actual rank cost afterwards.
- Genuine single-artifact gradient contributions retain the global 1/32 denominator. Same-step
  streaming uses CPU-only bounded JSON IPC, one model owner per rank, fixed per-rank ordering,
  one barrier/SUM and one optimizer/projection/durability transaction. No next-step sampling.
- Failed batches drain their owners and discard partial gradients without compensating unknown
  usage or publishing a partial update. The sealed path remains available.
- Deterministic bounded gradient/parameter buckets reduce small collectives. Finite checks and
  gradient norms retain their semantics with fewer scalar transfers.
- HealthBench's trusted local broker acquires a per-case judge permit before the shared model
  permit, for each HTTP request rather than the whole grading subprocess. Rubrics, seeds,
  parsing attempts and answer isolation are unchanged.
- Physical generated tokens, admitted tokens and discarded suffixes are counted separately;
  no scientific token accounting or server-side stopping rule has been changed.
- Immutable durable-step timing records distinguish rollout span, gradient span, overlap, tail,
  commit time and rank metrics. Deadline estimates use complete elapsed plus future work;
  an expired deadline rejects the next step at the existing durable boundary.

## Findings that changed the candidate

1. Actual runtime: Torch 2.11.0+cu129, SGLang 0.5.15.post1, Transformers 5.12.1,
   FlashInfer 0.6.12 and FLA 0.5.2. Official Ninja 1.13.2 was installed in a private overlay,
   not into another workload's environment.
2. With the existing prefix cache, a synthetic 60k-token forward-LoRA request differed between
   serial and mixed execution. Disabling the radix cache made the unchanged short/8k/32k/60k
   base and LoRA requests, plus a JSON-schema judge, pass exact seeded token comparison.
   The checked-in candidate therefore explicitly disables radix caching.
3. The first fixed-32 comparison had exactly equal losses, residuals, edge scores and Z gradients,
   but forward/backward gradient relative L2 errors of 0.005847 / 0.002228, failing the preset
   0.001 threshold. It was not accepted. The cache recorded 24 hits and 8 misses, occupying
   131,072 bytes; total serial time was 1,098 s versus 1,090 s, not a demonstrated speedup.
4. Repeating one unchanged 20-turn training artifact under the old backend itself produced
   approximately 1.2% gradient differences. Restoring per-edge scalar synchronization did not
   remove this noise. PyTorch deterministic algorithms with a supported cuBLAS workspace made
   the two old-backend gradients exactly equal (about 70 s / 67 s). This execution condition
   is now explicit for gradient processes. Per-edge scalar batching remains deferred.
5. On the second endpoint, the complete new stream finished all 32 artifacts in 875.55 s,
   but failed equivalence: forward/backward gradient relative L2 errors were 0.3474/0.2476.
   Rank zero's 16 losses matched the reference exactly; worker losses did not. This time is
   **not accepted speedup evidence**. Direct tensor comparisons excluded different model
   weights, inputs and ordinary thread switching. Triton/FLA had selected different tile
   configurations during cold parallel autotuning. Reusing the reference's tuning records
   alone restored the diagnostic artifact's exact reference loss and Z, without changing
   model, seed, input, dtype or tolerance.
6. The deterministic profile now ships shape/dtype-only FLA kernel choices, independent of
   private cache paths or timing records. Known keys reuse the reference choices; a previously
   unseen key retains upstream eligibility pruning but selects a single configuration in a
   stable order, not by timing. Disk autotuning results cannot silently override this profile.
   This addresses an observed numerical failure, not an integrity or attestation mechanism.
7. The unchanged serving canary on the second endpoint initially differed for one 60k-token
   base request, even without chunked prefill. Restricting prefill to one request passed, but
   was not accepted as a general fix: decode can still mix judge and rollout requests.
8. A fixed-probability GPU probe with seed zero exposed a batch-dependent sampler path:
   SGLang's filtered PyTorch path indexes seeded noise by sorted probability rank, whereas
   raw-softmax uses vocabulary order. Co-batching a greedy judge can switch a raw request's
   path. The project serving entrypoint now preserves each unfiltered row's standalone
   sampler and leaves filtered/unseeded rows and upstream errors unchanged. Installation
   also runs in spawned scheduler processes; installed SGLang files are not overwritten.
   The original mixed/chunked capacities were restored, and the unchanged short/8k/32k/60k
   base/LoRA plus JSON-schema canary passed on both the dedicated diagnostic card and the
   actual shared inference deployment. All eight serial raw outputs also exactly match
   the retained original serial outputs; seeds and fixture contents were not changed.
9. The first fresh-mini launch exited before any training update because the old deployment
   reader required AppWorld, which is not in the authoritative IID suite. Protocol 13 now
   reads only its WebShop, ALFWorld and HealthBench deployment sections, retaining their
   existing asset checks; static QA/math and code evaluation retain their original routes.
   Legacy Protocol 10 validation is unchanged. No AppWorld/SpreadsheetBench evaluation or
   unrelated environment copy was added. The failed attempt is retained as zero-step startup
   work; a distinct fresh eight-step attempt started at 06:57 UTC on September 5.
10. That second attempt reached rollout but stopped before its first update: SGLang tried
    to construct an `int64` tensor directly from an unsigned 64-bit request seed. This was
    a representation overflow, not a shared-memory OOM (about 37.5 GiB was free at adapter
    loading). The serving compatibility layer now supplies the lossless two's-complement
    representation only during synchronous tensor construction, then restores request
    metadata even on failure. SGLang's sampler converts it back to `uint64`; no seed bits,
    scientific coordinates or wire values are changed. Actual shared-service base/LoRA
    serial/mixed requests pass at zero, the signed boundary and the uint64 maximum, with
    the original greedy JSON judge. Both failed mini attempts remain zero-step evidence.
11. The third attempt failed because the operator used an uppercase timestamp in the run
    directory, which also became the stable trajectory-ID prefix. This was a launch mistake,
    not a scientific or GPU failure. It surfaced only after sibling rollouts drained; that
    wasted attempt is retained as zero completed updates. The debug entrypoint now rejects
    invalid run IDs before distributed/model initialization. A new lowercase-ID attempt
    started at 07:39 UTC, retaining the same seed, full workload and shared GPU mapping.

These are bounded tests of actual failures, not a relaxation of numeric tolerances.
The retained deterministic old fixed-32 reference completed in 1,200.8 s. The corrected NCCL
stream completed in 635.50 s: all losses, residuals and edges exactly equal; forward/backward/Z
gradient relative L2 errors 1.06e-7 / 1.24e-7 / 8.63e-8, well inside the unchanged 1e-3
threshold. Optimizer step counts match; maximum optimizer-state difference is 5.96e-8.
Both ranks processed 16 trajectories with compute spans 616.73 / 606.28 s. These are fixed-input
single-stage observations with different initialization/cache states, not ordinary-step or
end-to-end throughput. No final benchmark panel is used for optimization.

## Measurement status and deployment target

- The final fixed-32 deterministic gradient/AdamW comparison now passes. The initial
  two-rank GPU attempt was stopped when a post-launch check found an
  external process on a selected card; the project's parent and both children were terminated.
  A shell `pipefail` / early `grep -q` interaction had misclassified the captured process list.
  Subsequent launch checks must consume the complete list before deciding availability.
- The original acceptance request was eight fresh updates, two warmup and six measured.
  That run never completed. The latest owner instruction explicitly skips that prerequisite
  and requests a fresh full 250-step run instead. Do not resume or combine historical attempts.
- Report actual throughput, overlap/tail, per-rank imbalance, initialization and end-to-end time.
  Ordinary-step target is 4.2/hour; 250/72 = 3.4722/hour is only the arithmetic floor.
- Report both growth-factor 1.0 and 1.1 forecasts. The eight-hour future reserve is an explicit
  planning assumption, not measured overhead or a completion guarantee.
- Eight steps cannot measure cadence 10 or prove the complete 250-step evolution cost.
  Formal admission and the later cadence/long-trajectory coverage remain separate.
- Larger concurrency, teacher-forcing batching, checkpoint/offload changes and server-side
  root stopping are conditional options, not unmeasured speedup claims.

## Verification and workflow

No subagents or manual hash checks were used. Local ruff was used for changed files; CPU-heavy
model/process tests were moved off slow WSL storage to the approved Linux local disk with
`CUDA_VISIBLE_DEVICES=""`. Targeted coverage includes cache gradients and invalidation,
configuration/deadlines/buckets, true two-process Gloo streaming (32 items, reverse arrivals,
duplicate rejection, missing final artifact and worker OOM), serving compatibility, broker
requests, objective math and module boundaries.

The first full `make check` found three genuine integration-boundary failures; its other
2,445 tests passed. Cleanup was expressed with `finally` and joined futures, not additional
broad exception handlers, and the scoring/model import boundary was retained. The affected
72-test group passed; subsequent deadline/profile/stream changes passed an 11-test group.
The final CPU `make check` passed: 2,449 tests in 127.80 s, format/ruff, mypy over
457 source files, both wheel builds and the model-wheel boundary check. Already passing unchanged commands were not rerun
per subtask; fixes used targeted tests before this final check.

The FLA correction adds four targeted tests for reference choices, cold-rank order independence,
eligibility-preserving fallback, idempotence and packaged-profile installation. Changed-file
ruff and two-source mypy passed. Its single milestone `CUDA_VISIBLE_DEVICES="" make check`
on endpoint 22049 passed 2,453 tests in 124.71 s, formatting/ruff, 458-source mypy, both wheels
and the model-wheel check. The actual wheel includes the kernel profile. The four-test and
two-source typecheck set unintentionally completed on both hosts while the slow local run
was being moved; this was not a required double-check and was not repeated again.

The sampling correction passed an 11-test targeted group and two-source mypy. Its first
full CPU check found a genuine import-boundary error (2,459 other tests passed): the Torch
compatibility implementation belonged under `policy`, not `runtime`. It was moved without
weakening the boundary rule; the affected 29-test group passed. The final CPU `make check`
passed 2,460 tests in 125.22 s, formatting/ruff, mypy, both wheels and the model-wheel check.
The passing full-32 gradient comparison is not repeated for this serving-only change.

The deployment-selection correction passed 13 relevant tests across its targeted runs,
two-source mypy and reading the actual required deployment assets. Three new rejection tests
initially assumed the wrong inherited exception type; only those assertions were rerun after
correction. Its single push-time CPU `make check` passed 2,465 tests in 123.35 s, formatting/ruff,
mypy, both wheels and the model-wheel check. Unchanged gradient and serving numerical checks
were not repeated for this CPU deployment-selection correction.

The uint64 correction passed nine focused sampling tests and two-source mypy, including
exact 64-bit round trips and request restoration on errors. Its single push-time CPU
`make check` passed 2,467 tests in 122.91 s, formatting/ruff, mypy, both wheels and the
model-wheel check. Additional real serving coverage addresses the observed large-seed
failure; unchanged fixed-32 gradients were not rerun.

The early run-ID check passed its focused regression. The push-time CPU `make check` passed
2,468 tests in 124.51 s, formatting/ruff, mypy, both wheels and the model-wheel check. No GPU
equivalence or serving fixture was repeated for this identifier-only change.

## Resource status — updated 2026-09-05

The owner authorized holding two idle cards on endpoint 22049 with SGLang, copying the
project environment/data, and subsequently trying GPU coexistence. Two shared placements
were tried without changing the validated code or per-request budgets: physical 0 and then
physical 2, with project-exclusive training on physical 3 and 5. Both used a 131,072-token KV
pool, static-memory fraction 0.60, context 65,536, 12 running slots and two base/LoRA slots.
The pool setting is not a hard bound on transient allocations or other users' future usage.

Each placement committed one warmup update before withdrawal:

| Shared inference card | Step wall | Warmup equivalent steps/hour | Rollout span | Gradient tail | Gradients finished before last rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| Physical 0 | 41.39 min | 1.45 | 36.62 min | 4.72 min | 20/32 |
| Physical 2 | 23.86 min | 2.52 | 15.49 min | 8.31 min | 10/32 |

These are separate fresh attempts, not two warmups of one accepted run. Neither reached
ordinary-step measurement. The wall-time difference is not an isolated architecture comparison.
The shorter rollout exposed a substantial remaining gradient tail; rank compute in the second
attempt was 628.20 / 660.22 seconds. Persistence was only 3.40 seconds. Gradient-span overlap
includes waiting and must not be presented as pure compute hidden behind rollout.

Both shared services were proactively withdrawn after external allocations grew: the first
placement had about 2.3 GiB free, and the second reached 80,986 MiB total usage with almost no
free memory. The second service's own usage remained stable while external usage increased.
Only identity-confirmed project processes were terminated, all with SIGTERM; other users'
processes were untouched. Previously empty-looking memory is not a reservation for future work.

At the main thread's 09:31 UTC check, physical 3 and 5 again held healthy SGLang services;
there was no training in progress and no third fully idle card on either endpoint. These
were temporary holds, subsequently replaced by the exclusive attempt below.

The original shared serving canary reproduced the 60k base-token mismatch; the corrected
sampler now passes on dedicated and shared cards. The migrated physical-2 service passed
the same long-context mixed base/LoRA and uint64-boundary canaries before training. All eight
synthetic serial token sequences directly matched the retained physical-0 outputs. Its 60k
mixed requests took about 10.05 seconds versus 28.7 seconds at the earlier shared placement;
this is bounded contention evidence, not training throughput. Existing outputs were compared
directly, without hashes. Unchanged full CPU and fixed-32 gradient checks were not repeated
for these placement-only changes. There is still no accepted eight-step result, formal launch
or additional benchmark evaluation.

### Exclusive placement and observed thermal slowdown

At 13:42 UTC, physical 0 and 2 became empty on endpoint 22049. After another complete
eight-card check, the project retained inference on physical 3, stopped only its physical-5
hold, and started a fresh eight-step attempt at 13:47 UTC with coordination on physical 0
and gradients on physical 2. The retained inference service passed the same long-context
mixed base/LoRA and uint64-boundary canaries before launch. Training budgets and source
were unchanged; no fourth GPU context was started.

This attempt committed one warmup in 2,087.43 seconds (34.79 minutes; 1.72 steps/hour
equivalent), not an ordinary-step measurement. Rollout took 840.27 seconds, gradient tail
1,243.95 seconds, and rank compute 1,376.51 / 645.45 seconds for 2.75 / 2.92 million edge
tokens. Only 4/32 gradients finished before the last rollout; durability took 3.22 seconds.
At 14:24 UTC the main thread observed physical 0 at 86 °C, 720 MHz and active thermal
slowdown, while physical 2 was at 27 °C and 1,980 MHz without that flag. This identifies a
real hardware confounder, not proof that every second of the step was throttled or that
thermal behavior explains the entire difference. High GPU utilization alone is not proof
of useful gradient compute: a rank may be waiting in a collective.

The project withdrew this attempt at 14:32 UTC using identity-confirmed SIGTERM only,
preserving the single warmup and partial next step separately. Read-only stack inspection
of the project's own ranks was denied by the host; no sudo or permission bypass was used.
No hardware controls or other users' processes were changed. Physical 0 is excluded from
the next placement pending thermal investigation.

A requested move to coordinator 5 / gradient 2 timed out during HTTP preparation **before
launch**. At 20:03 UTC the main thread confirmed no new launch record existed, the retained
inference service was healthy with an empty queue, and physical 2 and 5 were now occupied
by other users. The last complete dual-host check at 20:34 UTC still found no empty cards.
Current project GPU use is only one warm SGLang service on physical 3; there is no training
or numerical-test job running. The main thread continues checking both endpoints roughly
every three minutes. Low memory with a live external compute PID is not classified as idle.

All three one-step attempts remain separate: they cannot be combined into the required
two-warmup/six-measured run. Current training progress is zero while waiting and completion
ETA is unknown until capacity is available. Formal 250-step admission remains
**needs-human-decision** for a stable resource window; no 4.2/hour or 72-hour claim is
supported. Unchanged CPU full checks and fixed-32 numerical comparison were not repeated
for placement changes or thermal diagnosis. No hashes or review agents were used.

### Latest owner instruction: direct full 250 steps on either endpoint

The owner explicitly approved the private transfer of the validated source, initial checkpoint,
fixed-32 training fixtures and gradient reference outputs to a new private directory on
endpoint 22048. The CPU-only transfer completed; the destination directory is restricted to
the project account. A proposed allocator-capped sharing experiment was subsequently
cancelled before any GPU process started. No memory-cap trial or cross-host training was run.

The latest instruction is to keep the main thread watching both endpoints and start one
fresh full 250-step run when either endpoint has three project-exclusive cards. On 22049,
the existing physical-3 SGLang service can be retained with two newly empty training cards;
on 22048, all three cards must first be empty. No sharing test or separate eight-step mini
is required by this updated instruction. All selected cards still receive a fresh process
and thermal check; no other user's process may be modified or terminated.

CPU-only configuration checks passed on both hosts: 2,000 questions, exactly 250 from each
authoritative IID domain, one question per domain per update, four trajectories per question,
250 optimizer updates, 8,000 trajectories, seed 0, and 25 cadence checkpoints at interval 10.
The existing required deployment assets, local preparation/model paths and EvalPlus venv
entrypoint were readable. Both bindings use the validated same-step streaming source and
unchanged inference budgets. No unchanged full test suite or GPU equivalence test was rerun.

At the main thread's 2026-09-06 00:15 UTC check, only physical 3 was empty on endpoint 22048;
endpoint 22049 still lacked two empty training cards, and its retained inference service was
healthy. No full training run has launched. Progress remains zero while waiting and ETA is
unknown until capacity is available. The owner has authorized proceeding without an accepted
eight-step measurement, not changing the existing experiment result label or claiming that
formal scientific/performance admission passed. The current driver retains its 72-hour
durable-boundary deadline; an incomplete run must be reported as incomplete, not as 250 steps.
