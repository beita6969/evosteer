# Execution-only training acceleration — 2026-09-08

The method remains `idea.tex`: complete fixed-policy/library batches, original
action token IDs, per-edge action-length normalization, trajectory TTB loss,
global invocation-edge flow normalization, binary posterior labels, and one
optimizer/projection/evolution/durability transaction. None of the changes below
modify rewards, sampling temperature, task budgets, skill credit or phase rules.

## Implemented batches

| Batch | Implementation | Correctness boundary |
| --- | --- | --- |
| PR1 | Local `GradientStepStream`, one background CUDA owner; distributed coordinator may be state-only | External rollout model owner required for local overlap; compute ready artifacts first, reduce in canonical position order; bounded CPU gradient storage; seal all artifacts before any update |
| PR2 | Request-local SGLang post-sampling JSON-root stop plus incremental UTF-8 token decoding | No grammar/logit mask; never split a token, stop at a bare `}`, or truncate nested/string/code content; untagged requests retain upstream stopping |
| PR3 | Independent padded edge microbatches within one trajectory and adapter; configurable checkpoint/offload tiers; scoring telemetry and offline FLA profile support | Each action keeps its own denominator and original IDs; no detached inference KV cache; OOM fails the batch, not the sample population |
| PR4 | Existing HealthBench request broker connected to training; cancelled HTTP requests drain before releasing model capacity | Shared actor/grader request capacity is leased per actual rubric request, separately from process capacity; fixed base-model judge route and unchanged payloads |

The source of posterior state remains the committed projection. Ready-first
gradient computation does **not** update or normalize posterior evidence as
trajectories arrive. CPU buffering preserves gradient dtype; byte limits apply
to detached out-of-order contributions, separately from the device accumulator
and the fixed-size artifact inbox. A full buffer always admits the next canonical
position, avoiding a backpressure deadlock.

## Execution configuration

`TrainingPerformanceConfig` records resolved values, including these defaults:

```yaml
pipeline_mode: within-step
gradient_buffer_bytes: 2147483648
teacher_forcing:
  microbatch_size: 1
  microbatch_max_tokens: 8192
  checkpoint_min_tokens: 1
  offload_min_tokens: 32768
  pinned_memory_bytes: 0
  profile_cuda: false
fla_profile: null
```

Microbatch sizes 2/4 are explicit opt-ins, not newly enabled defaults. Groups are
length-bucketed, constrained by padded token count, and never cross trajectories
or adapters. Long edges run individually; they are not shortened. Checkpointing
still requires the backbone's checkpoint flag. Pinned storage is bounded by
**live saved tensors**; PyTorch's allocator may cache released pinned blocks,
so this is not a hard bound on all OS-locked memory. Default pinning stays off.

**Current CUDA acceptance: microbatching is not accepted for training.** The
fixed-input BF16 comparison at configured sizes 2/4 produced approximately
0.1003 forward and 0.02503 backward gradient relative L2 error, exceeding the
unchanged 1e-3 bound; maximum edge-mean error was 0.004120. The 3-trajectory
profiling subset kept global B=32 and contained 46 scored edges. At the 8192
padded-token cap, both candidate sizes formed the same actual groups, so this
did not independently exercise four-row CUDA scoring. Times were 97.13 s
(single edge), 97.66 s (size 2), 97.53 s (size 4): **no measured speedup**.
Longer-than-32K offload/pinned paths were not exercised by this subset. The
microbatch implementation remains a numerical/performance candidate; keep
`microbatch_size: 1` for real training. No tolerance or default was relaxed.

A separate fresh-process check reused the B=1 cumsum kernel configuration for
the new B=2 shape, keeping all other frozen choices and the same error bound.
It also failed acceptance. This rules out that cumsum fallback **alone** as the
fix; it does not identify the remaining batched-kernel/padding precision cause.
The private candidate table was not promoted, and an unchanged size-4 comparison
was not repeated because it formed the same groups as size 2 on these inputs.

FLA defaults retain the existing frozen table and deterministic legal fallback.
Counters distinguish calls, new-shape fallbacks and cached shapes.
`offline_fla_profiling()` temporarily measures eligible configurations on fixed
inputs and restores the training table even on failure. Its exported table must
pass a **fresh-process** numerical comparison before use; live training never
autotunes from wall-clock measurements. This change does not ship a newly tuned
table or claim that every fallback is fast.

## Measurement and interpretation

`scripts/benchmark_teacher_forcing.py` consumes a private frozen collected batch
and initial checkpoint. It does not generate trajectories, regrade answers,
create an optimizer update or update posterior state. A position subset retains
the original global B; it is explicitly a profiling sample, not a smaller
on-policy batch. CLI inputs and detailed outputs belong in private storage.

It reports per-direction edge/group counts, prefix/action/padded tokens,
length percentiles, checkpoint/offload hits and saved bytes, wall time, device
peak allocation and FLA counts. `profile_cuda` enables synchronized group CUDA
timing; absent measurements are null, not zero. Normal training retains the
existing step overlap/tail/durability timing and adds queue wait/buffer metrics.

Numerical bounds are declared before comparison: FP32 gradient relative L2
1e-5 and edge-mean absolute error 1e-6; BF16 respectively 1e-3 and 1e-3.
Forward, backward and Z gradients are checked separately, retaining the existing
fixed-32 CUDA gradient bound rather than relaxing it based on dtype precision.
These are execution-comparison bounds, **not** a posterior confidence interval
or proof of bitwise-identical Adam updates. Failed comparisons must not loosen
their own bounds or silently enable microbatching.

The actual deployed SGLang scheduler was exercised on synthetic JSON-format
requests with nested objects, Unicode and a brace inside a string. Three paired
request comparisons retained exactly the same admitted token prefixes. Native
generation fell from 229/193/203 tokens to 27/27/25 tokens; this is an intentionally
tail-producing format probe, **not** a representative training speedup. Client
boundary scan took approximately 0.2–0.5 ms on these outputs. No benchmark answers
or evaluation rounds were used. Training continues to use only seed 0.

## Validation status and operational limits

Focused CPU tests cover ready-first/canonical equivalence, late abort without
parameter changes, non-participating distributed coordinator, padded hybrid
Qwen attention/recurrent scoring, positive/negative residuals and varied K/T,
OOM cleanup, parser/token boundaries, FLA table restoration and broker draining.
The real multimodal Qwen configuration exposed a missing outer `pad_token_id`;
the executor now reads text configuration and has a regression test.

The existing two-plus-two training run stayed on its original frozen source
and scoring condition. It committed two steps and restored them in a new process,
then stopped during the third batch on invocation/action admission mismatch;
the incomplete batch did not update model or posterior. Its GPU processes were
automatically cleaned up. A corresponding parser gap is now fixed: persisted
invocation admission and live execution share the exact optional Action-header /
whole-JSON-fence projection. Wrapped executed skills no longer lose their identity;
the original action text/IDs and strict rejection of prose/fabricated credit remain.
No hot replacement or relabeling of the old checkpoint was performed.
GPU performance probes use a separate frozen execution condition.
Neither four training steps nor a synthetic author/gradient test establishes
natural evolution at production W=50. Thermal slowdown on the old run requires
an administrator/resource decision; no fan, power, clock or other user's process
was modified. No new IID round is part of this performance change.

## Follow-up: observed buffering and checkpoint tradeoffs

The separate local-stream run committed one step in 2284.35 seconds before
its first-half two-hour deadline expired in batch two. Collection took 727.97
seconds; 698.37 seconds overlapped gradient work, and eight contributions had
finished before the last rollout. Compute took 1705.44 seconds and queue wait
575.29 seconds. The 512 MiB host buffer peaked at 520,095,776 bytes: eight
65,011,972-byte contributions. Later ready artifacts could not proceed while
that buffer was full and the next canonical artifact was missing.

The H800 YAML now explicitly allocates a **2 GiB host-buffer cap**, sufficient
for all 31 out-of-order contributions of this measured batch32/rank4 model.
It is not preallocated, does not change the device accumulator or dtype, and
does not change legacy omitted-field defaults. Wider buffers still reduce in
the original order and discard the entire failed batch. New counters separate
`artifact_wait_seconds` and `buffer_backpressure_seconds`; remote dispatch
waiting no longer inflates rank-0 CUDA-owner waiting. CPU tests cover narrow
backpressure and completion of every other artifact before position zero,
with exactly unchanged final gradients and no optimizer update before seal.
No whole-training speedup is claimed before a new run measures it.

The fixed-input checkpoint comparison used the same three trajectories,
46 edges and global B=32 as the earlier numerical probe:

| Minimum checkpoint length | Seconds | Peak allocated GB | Numerical result |
| --- | ---: | ---: | --- |
| 1 (reference) | 97.99 | 32.04 | Reference |
| 4096 | 94.78 | 59.08 | Compared edge means and all three gradient components exactly equal |
| 8192 (separate pair) | Not completed | Out of memory | Rejected during warmup |

The single-pair 3.27% time reduction at 4096 costs 84.4% more peak allocation.
Therefore the memory-safe always-checkpointed **default stays unchanged**.
`scripts/benchmark_teacher_forcing.py --sizes 1 --checkpoint-min-tokens 1 4096`
supports explicit headroom-qualified comparisons without generating new data.
An OOM is a failed candidate, never a retry with shorter inputs in training.
Reports now remain explicitly incomplete until every requested variant finishes;
a saved baseline alone cannot mark an interrupted comparison as passed. They
also record effective runtime package versions, not merely the source lock.

The runtime had FLA 0.5.2, but lacked the optional
[causal-conv1d CUDA extension](https://github.com/Dao-AILab/causal-conv1d).
The import warning alone was not evidence that gated-delta scoring used its
Torch fallback. Convolution candidates are measured in isolated dependencies;
installing a package is not evidence that its numerical path is acceptable.

The sequential GPU3 convolution probes have completed. With the project-declared
causal-conv1d 1.6.2.post1 in isolated dependencies, the same fixed input gave:

| Convolution path | Seconds | Maximum component gradient relative L2 | Edge-mean absolute error |
| --- | ---: | ---: | ---: |
| Original Torch reference | 97.17 | 0 | 0 |
| Native convolution, original SiLU rounding, contiguous output | 90.47 | 0.01500 | 0 |
| Native forward, original Torch convolution backward and SiLU | 94.61 | 0 | 0 |

The unchanged gradient bound is 0.001 **per forward, backward and Z component**;
an exact forward score alone is insufficient. The native-backward candidate is
rejected. The private custom-backward probe isolates that discrepancy but offers
only a single-pair 2.64% time reduction, insufficient evidence to add a new default
autograd implementation. Both paths peaked at 32.04 GB allocation. A separate
official 1.7.0 probe also failed the bound (even after preserving forward rounding);
it was not installed into the training environment or substituted for the lock.

These are fixed-artifact numerical/performance probes, not extra on-policy steps,
natural evolution evidence or a steady-state throughput acceptance. All four GPU
probe processes and both CPU-only package builds ended. No server, optimizer,
posterior, sampling stream, thermal control or other user's process was modified.

The final integrated CPU `make check` on 22049 passed 3512 tests (one explicit
CUDA opt-in skip), format/Ruff, mypy, both wheels and the model-wheel boundary.
The initial run's inherited `PYTEST_ADDOPTS` interfered with nested pytest; the
runner now uses its own temporary root without exporting a fixed basetemp. One
pre-existing evaluation assertion was updated to the already implemented opening
system-budget metadata, without changing actor behavior. The successful GPU
comparisons were not repeated for these test/runner corrections. No new IID or
on-policy run was performed, so end-to-end buffer speedup remains unmeasured.
