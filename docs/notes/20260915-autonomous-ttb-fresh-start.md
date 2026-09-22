# Fresh autonomous TTB run — 2026-09-15

The owner requested a new start, not continuation of the previous run. The new
run is `bayesian250-autonomous-ttb-20260915-v1`. It entered training at
**04:30:28 UTC**, with optimizer step, task cursor, optimizer state and posterior
all zero/empty. The saved original pre-specialized-SFT initialization is used;
old training, supervised warmup and failed startup records are preserved.

## Frozen condition

- `bayesianimprove_autonomous_ttb.yaml`, runtime source `7211be0`.
- One Qwen3.5-9B owner per trajectory; 250 updates, B28, seed 0, checkpoint every
  10 updates and cooperative drain/save on stopping.
- Seven balanced training lanes; no WebShop or eighth question. Historical AIME
  training sources are not renamed AIME2026 IID data; closed-book Trivia training
  is not claimed equivalent to the historical reference-context IID evaluation.
- Static horizon 8, ALFWorld horizon 25; thinking on only for AIME and HealthBench.
  Expanded per-domain budgets are saved with the run.
- Autonomous catalog-then-read, seven frozen public method cards, native action
  wire, full F/B/Z TTB and ordinary Bayesian/Phi boundaries. No teaching prompts,
  specialized SFT, read quota, read reward, auxiliary loss or zero-coverage bypass.
- The source order is frozen before sampling, with all outcomes retained. Public
  task applicability is not evidence that a skill improves success.

This is an **owner-authorized, unqualified experiment**. Failed cold-start results
and outstanding architecture-matched IID/A0 acceptance are not relabeled passed.

## Actual execution and startup repairs

On `185.212.56.211:22049`, the current mapping is GPU4 for SGLang and GPUs6/0 for
two participating gradient ranks, rank 0 also coordinating. GPU5 was occupied by
a foreign high-memory job, so it was not reclaimed. The latest all-card resource
authorization was used; no foreign process was terminated or modified.

The maintenance recovery required isolated runtime repair: matching CUDA compiler
components, missing training imports, and restoration of Transformers 5.12.1 to
match the original tokenizer configuration. The normal tokenizer constructor now
accepts that unchanged configuration. Its check was not weakened or bypassed.
Two failed training startup attempts had **zero requests and zero commits**;
their logs and empty startup states remain separate. Serving/compiler startup
probes are not training-performance or numerical-equivalence results.

Active processes at the startup observation: actor parent 196926 / scheduler
197475; controller 291615; torchrun 291648; gradient ranks 291666/291667; CPU
metrics source 291686; local W&B uploader 1284292. Owned standby SGLangs were
drained only after the new gradient owners established their CUDA contexts.
The controller retains drain/save handling and restores actual standby services
on owned cards after exit when remaining resources permit.

At **04:31:34 UTC**, native generation had begun: 2 completed model requests and
12 dispatched requests, no completed trajectory or optimizer commit yet. GPU4
was 87°C with software thermal slowdown active. This is an explicit hardware
performance concern requiring administrator attention, not a reason to alter
protection, discard trajectories or claim a speedup. No complete-step throughput
or completion ETA can yet be inferred. The planning targets remain 4.2 steps/hour
and 72 hours, not measured guarantees.

At **04:35:48 UTC**, collection reached **22/28** complete artifacts and **20/28**
complete gradient contributions (2 canonically merged, preserving order).
Requests: 134 completed, 6 dispatched; no batch abort or startup exception.
The metric source was live with zero committed points, as expected before the
first full update. All three selected GPUs now reported thermal slowdown:
GPU0 86°C/345 MHz, GPU4 86°C/1305 MHz, GPU6 86°C/975 MHz. This has been escalated
as requiring human hardware/resource attention. Training remains running under
the unchanged safety protections; progress fraction is not a remaining-time
estimate, and no below-target throughput observation is hidden as warmup.

## Metrics and verification

[New W&B run](https://wandb.ai/lanlangcll-university-of-illinois-urbana-champaign/skillev123/runs/bayesian250-autonomous-ttb-20260915-v1)
uses committed core telemetry for TTB loss, native reward/success, action
validity, domain metrics, posterior/skill/evolution counts, timing, tokens and
GPU usage. No invented loss/reward zero is emitted before the first commit;
an uploader heartbeat is not training progress. Raw licensed task content is not
uploaded or committed.

Remote Linux CPU `CUDA_VISIBLE_DEVICES="" make check` completed: **5065 passed,
16 skipped**, with Ruff, mypy (697 sources), package builds and wheel checks.
The stale IID restart test fixture now uses canonical expanded controls rather
than `asdict`; runtime checks remain unchanged. Relevant targeted tests also
passed. Full tests were not repeated for this documentation-only update or the
private deployment version correction; actual GPU model/tokenizer startup is
reported separately from CPU tests. No subagent or new hash audit was used.
