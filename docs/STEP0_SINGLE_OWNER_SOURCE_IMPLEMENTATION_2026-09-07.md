# Single-owner source and resume integration

This is an implemented subset of the [repair worklist](STEP0_SINGLE_OWNER_REPAIR_WORKLIST_2026-09-07.md),
not completion of every proposed trained-policy/recovery feature or every score goal.

## Implemented

- Both active and legacy selector imports unconditionally reject public-check selection,
  label-swapped voting and baseline fallback. Historical types remain readable; the old
  implementation lives only in Git history. The historical composite table now has a
  withdrawal warning immediately beside its values, including its direct-policy column.
- The diagnostic inventory follows bounded, explicit local imports and aliased calls without
  importing code or recursively searching private directories. Tests cover the real clean CLI,
  explicit runtime factory, isolated actor and dispatcher dependencies. It is not a new gate.
- Actual broker outputs and transport-completion events are committed together before actor
  delivery. Private records retain raw output tokens, final-channel tokens/text, usage,
  termination cause, participant, purpose, revision, policy, attempt and call identity.
- New finals identify their actual attempt/owner call and terminal status. Submission and
  same-run recovery/first scoring check persisted source, scope, policy, parser and cost.
  The runtime verifies final-channel decoding against served tokens. A valid owner answer
  cannot be converted to an empty candidate or followed by replacement generation.
- Native actions link owner calls to their acknowledged environment executions. The submitted
  trajectory must match that action sequence; no earlier successful trajectory is substituted.
- New runs are explicitly fresh. Same-run resume is explicit, preserves frozen controls and
  definitive scores, and validates the source again. Cross-run SQL copying/run-ID rewriting
  is removed. Historical inspection is read-only and keeps origin IDs, scores and unverified
  labels. Old journals do not receive fabricated model-output records.
- Actor and broker share token-based native-channel parsing. Unclosed/malformed thinking has
  no final; equivalent scalar projection is retained. A frozen eight-benchmark thinking map
  reaches the profile, actor, template, broker and recorded conditions. New runtime configs
  must supply it explicitly; AIME is on and the other seven remain off in the new map.
  The legacy/global arm flag is not the per-benchmark resolved setting.

## Verification

- Selector/import-dependency/result-contract set: **71 passed, 3.51 seconds**.
- Source/channel/resume/broker integration set: **215 passed**, two failures due to an old
  test fixture omitting its new thinking-policy field. That fixture was corrected.
- Affected source/broker/ownership/pipeline follow-up: **49 passed, 14.84 seconds**;
  targeted mypy passed **11 source files**. Local affected-file Ruff passed.
- One integrated `CUDA_VISIBLE_DEVICES="" make check` on the approved22049 CPU host
  completed successfully (PID128398): **3,031 tests passed, 85 existing warnings,
  141.32 seconds for pytest**, full Ruff/mypy, both wheel builds and the model-wheel
  private-boundary check. No unchanged suite is repeated for these notes.
- A CPU-only check of the actual deployed tokenizer confirmed that each thinking marker is
  one token, the thinking-on template opens reasoning, and the off template closes it before
  generation. This made **zero model calls**; it is not a real thinking-on benchmark result.
- Main-thread implementation and one combined inspection of source, persistence and resume
  changes; no coding/review subagent, hash check or retired approval/receipt workflow.

## Remaining work and latest owner priority

The runtime in this milestone is still adapter-free Step-0. Explicit trained checkpoint/LoRA
binding and trained-policy schema remain pending; it must not be relabelled as trained
evaluation. Finished calls are durable/readable, but automatic reconciliation of a crash
between output delivery and candidate submission, or HTTP-loss recovery by service request ID,
is not implemented. Such incomplete episodes still stop for reconciliation rather than receive
a fresh budget or answer. The final-candidate commit follows actor return; immediate durable
output recording is not claimed to be automatic final-candidate recovery.

Broader historical launch/catalog tooling and complete new authority-report fields remain in
the worklist. None is a claim of observed selection in the current clean actor.

The previous frozen thinking-off `@13` AIME30 round independently completed **23/30 (76.67%)**
with no empty submission in28.27 minutes. The owner subsequently accepted this AIME result
as sufficient for now and redirected effort to **ALFWorld and WebShop**. Therefore the
prepared thinking-on AIME worker/canary **was not launched**. Those new AIME conditions remain
unmeasured. ALFWorld/WebShop will each retain128 tasks, one seed and native scoring; changes
must address observed interfaces or explicitly declared conditions, not scripted solutions.
Only the existing22048 GPU4/5/7 inference services are retained. No new GPU job or inference
daemon was created by this integration; the completed CPU full check used no GPU.

Subsequent owner update: thinking-on was requested again. The separate frozen-source30-task
round has now completed **25/30 (83.33%)** in53.76 minutes, including3 budget-exhausted
output failures; it is not a clean communication pass. See the
[thinking-on result](STEP0_AIME_THINKING_ON_2026-09-07.md). The earlier nonlaunch statement
describes the previous priority change, not the current execution state.
