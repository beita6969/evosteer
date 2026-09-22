## 2026-09-16 04:12 UTC — Current six-domain TTB run: Step37 committed

**Read this section first. Older Step27, seven-domain B28, full-inline, device
mapping and W&B/startup instructions below are historical, not current commands.**
This side conversation only updates the two handoffs. It directly inspected
the current remote controller, processes, frozen config, committed metric export,
checkpoint markers and local backup/upload state. It did not operate the main
thread's agents, start/stop training or services, edit code/config, or commit Git.

### Active method and condition
- Run `bayesian250-autonomous-ttb-20260915-v1`, frozen implementation `4b937b3`.
  This remains owner-authorized **unqualified diagnostic TTB**, not passed
  cold-start/A0 or verified completion of250 updates. The main thread's objective
  is still250 complete steps with monitoring, repair and local cadence backups.
- One Qwen3.5-9B task owner per trajectory; frozen BF16 base; F/B LoRA rank4,
  alpha8 and trainable Z; AdamW learning rates1e-4, WD0, no clipping/extra KL;
  TTB beta1/epsilon0.1; Beta(1,1), W50/rho0.05/k1;249 search+1 closure and at most
  two natural evolutions. No specialized SFT, distillation, auxiliary loss,
  read quota/bonus or zero-coverage bypass is introduced.
- Steps1–4 retain their original seven-domain B28 records. After full Step4,
  HumanEval was removed: Steps5–250 use six domains × four rollouts = **B24**.
  Keep the original cursor, consumed source identities, optimizer, posterior,
  diagnostics and library state. The mixed-condition run totals6,016 trajectories
  and1,504 question occurrences; do not relabel its historical denominators.
- Current IID list: **HotpotQA, TriviaQA, AIME2026, HealthBench, ALFWorld, MBPP+**.
  Current declared OOD list: MuSiQue-Ans, NQ-Open, Math-Hard,
  GPQA(Biology and Organic Chemistry), Scienceworld, APPS(Introductory subset).
  No new IID/OOD evaluation was run for this update. Historical AIME training
  sources and closed-book Trivia training are not their IID evaluation conditions.
- Actual training reasoning thinking ON: AIME/HealthBench; OFF: the other four.
  Evaluation defaults OFF except AIME2026/HealthBench/Math-Hard/GPQA ON.
  Static max8 turns, ALFWorld25; input65,536 with the declared history window.
  Reasoning caps: Hotpot1,024; Trivia4,096; AIME16,384; HealthBench/ALFWorld8,192;
  MBPP1,024. Action cap2,048 except HealthBench4,096.
- Native action wire@3, public-task-semantics@8, phase context/tool catalog/token
  notices enabled; initial public-method-cards@5. Current catalog-then-read@3
  applies fromStep3, not full-inline. Calls remain autonomous and counted;
  reading is not evidence of successful application or causal skill benefit.

### Directly checked progress and outstanding issue
At04:12:24UTC on22049, Step37 was the latest committed update. Complete rolling
checkpoints35/36/37 were present. Step38 was collecting16/24 artifacts and16/24
gradient contributions, canonical merge2/24, without an abort, pause or training
exit record. Both real gradient workers and the owner remained live.

**One Step38 HealthBench trajectory is marked `failed` in its rollout sidecar.**
The inspected entry contains no detailed error class/message, and the bounded
log tail did not expose the cause. The main thread must establish the underlying
failure and transaction state; this is not permission to drop it, label an
infrastructure failure as a task failure, or silently regenerate a replacement.
Intermittent SSH banner timeouts also occurred; this update connected successfully
with strict host verification. An observation timeout alone is not process death.

| Step | TTB loss | Mean reward | Success /24 | Reads / posterior events | Wall / rollout / gradient tail (s) |
| --- | ---: | ---: | ---: | ---: | --- |
| 35 | 0.610997796 | 0.627777778 | 14 | 4 / 4 | 1106.340 / 841.952 / 258.642 |
| 36 | 0.817702264 | 0.593137255 | 13 | 3 / 3 | 1272.321 / 1185.004 / 81.110 |
| 37 | 0.977058885 | 0.640447154 | 10 | 2 / 2 | 791.976 / 752.999 / 34.125 |

Both mutation counters are0 for these steps; natural-evolution qualification is
not claimed. Last-five active commit timing gives about2.855steps/hour and
roughly74.6 active hours for213 remaining steps at that rate. It includes recovery
work, is not new-hardware steady-state evidence, and excludes further downtime
and variable evolution costs. Below4.2 remains **a human performance-decision
risk**; the main thread has the owner's continuation instruction.

### Resources, recovery, backups and publication
- Current physical roles on22049: **GPU2 inference, GPU5 coordinator/gradient,
  GPU7 second gradient worker**. This is a full-Step35 continuation, started at
  03:33:01UTC, not a fresh run. At04:12 the selected cards used approximately
  63,058/67,249/54,505MiB total device memory, including any shared allocations;
  temperatures37/40/44C and clocks1965/1980/1935MHz, thermal slowdown inactive.
  Other cards' thermal events must not be attributed to this mapping.
- The main thread drained/saved Step35 before moving away from the earlier
  throttled/card-contention condition. This handoff update launched no GPU or
  resident process and did not touch another user's work. Private operational
  coordinates and PID ownership remain in the local private handoff only.
- Full local Step30, Step32 and paused-Step35 copies were directly found with
  COMPLETE and matching step metadata under **`G:\checkpoints-bf`**. Step35 is
  214,588,382bytes. These are local presence/metadata/size checks, not a new model
  restore test or hash audit. Retain older downloaded copies; future selection
  is Step30 onward, every10 steps plus complete pause/final snapshots.
- The CPU backup worker remains live. Its last status was `retry-needed` after
  an SSH banner timeout; already published copies remain intact. **Step40 is the
  next cadence backup, not already backed up.** Confirm the actual local complete
  checkpoint at every cadence, rather than trusting a heartbeat.
- The previously lost Step33 ALF reasoning response had one explicit owner
  retry authorization, now consumed and completed. It preserved other responses
  and finished artifacts. This is not blanket permission for future unknown
  requests, including the current HealthBench issue.
- Continue or pause via full-batch commit/checkpoint/drain. Do not terminate a
  rank, replay old launch scripts, roll back toStep32, or hot-swap latest main
  merely because of this time-stamped snapshot.

W&B: <https://wandb.ai/lanlangcll-university-of-illinois-urbana-champaign/skillev123/runs/bayesian250-autonomous-ttb-20260915-v1>.
The local uploader is live and its log records Step37; the remote committed
export contains the matching35–37 aggregates above. This side conversation did
not independently query W&B's API, so it does not claim a full remote-history
validation. Preserve one writer and all condition/hardware segments; do not fill
unknown metrics with zero or count exposure/no-op as invocation/mutation.

The main thread owns the requested GPT-5.6-Luna/xhigh Git/W&B-only role. This side
conversation did not spawn/contact agents or change Git state. Next main-thread
work is the Step38 failure investigation, continued complete updates through250,
and verified local40/50/.../250 backups. Only these two handoffs changed here;
no repeated tests, builds, evaluation, hashes or training modifications. Earlier
implementation validation is recorded in
`docs/notes/20260915-six-domain-step4-continuation.md`. Older sections below are
retained as history, not overwritten or promoted to current instructions.

---

## 2026-09-11 00:44 UTC — Current formal250 continuation: Step27 committed

**This section supersedes historical run-state/startup instructions below.**
It was updated through read-only checks of the live training process, committed
metrics, checkpoints, performance records and W&B API. This side conversation
only updates the two handoffs; it does not take over the main thread or operate
its agents, training processes, services or metric writers.

### Active condition
- Qwen3.5-9B, one main Agent per trajectory, skills enabled/full-inline;
  dual forward/backward LoRA and trainable Z with the existing complete-batch
  posterior/evolution transaction.250 steps, seed0, B28: seven domains × one
  question occurrence × four rollouts; no eighth question and no WebShop.
- Domains: HotpotQA, TriviaQA, AIME2026, HealthBench, ALFWorld, MBPP+, HumanEval.
  Static maximum8 turns; ALFWorld maximum25; input65536/action2048 tokens.
- Actual reasoning thinking ON: AIME, HealthBench, TriviaQA; OFF: HotpotQA,
  HumanEval, MBPP+, ALFWorld. Reasoning limits respectively AIME16384, ALF8192,
  Health4096, Trivia4096, otherwise1024. The newer preference for thinking-on
  scoring is not evidence that running actor/judge settings were changed.
- `native-single-tool-call@3`, `public-task-semantics@3`; not the separate ALFv4
  candidate. `token-budget-notice@1` was enabled at a complete Step23 boundary
  and applies fromStep24. Actual phase limits are shared by rollout/F/B context;
  the limits themselves, truncation and next-turn behavior are unchanged.

### Directly verified progress
At00:43–00:44 UTC, the latest complete checkpoint wasStep27. Step28 was collecting
24/28 artifacts and24/28 gradient contributions, with no abort or pause. GPU6
provides inference; GPU0 coordinates and computes gradients; GPU7 is the second
gradient worker on endpoint22048. All three role processes were live. Observed
total device memory was72,588/63,727/42,243 MiB onGPU0/6/7, including shared users,
not exclusive project allocations. No new GPU or resident process was launched
for this handoff update; no other user's process was touched.

| Step | TTB loss | Mean reward | Binary successes /28 | Parse / schema errors | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24 | 0.606008944 | 0.688449848 | 18 | 3 / 1 | 1131.732 |
| 25 | 0.421740009 | 0.965225564 | 28 | 0 / 0 | 475.752 |
| 26 | 0.174739672 | 0.872180451 | 23 | 4 / 0 | 955.856 |
| 27 | 0.508405582 | 0.928571429 | 26 | 7 / 0 | 1955.863 |

Step27 committed at00:37:25 UTC; rollout1885.722s and gradient tail67.425s.
Steps24/25 were process warmup. The two non-warmup observations26/27 average
2.47steps/hour: insufficient steady-state evidence and below4.2, **requiring
human performance decision**. The owner previously directed continuation despite
threshold misses.223 remaining updates imply roughly90.2h pure training at that
rate;45 quality checks at the latest24.4min/check add about18.3h. Rough remaining
total108h is conditional, excludes unforeseen failure/evolution costs, and is
not a72-hour guarantee. Do not use the unusually fastStep25 alone for an ETA.

The separate held-out Step25 check was17/28 success, reward0.711248439 and
first-action structural validity26/28, with warning/continue. This must not
overwrite trainingStep25's28/28 success. Full250 completion and production
natural-evolution validation are still unproven. Retained checkpoint cadence
remains every10 steps; rolling recovery checkpoints and durable event history
have different retention. Continue/stop through the existing complete-batch
checkpoint workflow, not by terminating a gradient rank mid-batch.

### W&B and pending metric work
Current run:
<https://wandb.ai/lanlangcll-university-of-illinois-urbana-champaign/skillev123/runs/bayesian250-token-notice-step23-20260910>

This update directly read back unique training points24–27 with matching loss,
reward and binary success. The CPU mirror and single uploader were live. Earlier
21–23 v3 and3–20 catalog condition segments remain separate; do not relabel their
history. W&B's running state describes its writer, not proof of training progress.

The main thread's pending request is continuous coverage of: TTB loss; reward;
binary success; action structural validity; action admission/execution validity;
per-domain reward/success; posterior update counts; actual skill invocations;
phase-change detection; actual mutations; step wall/rollout/gradient tail; tokens
and GPU hours. **This handoff does not claim the expanded list has all been
implemented or uploaded.** Preserve unknown values, evidence units and accounting
scope; distinguish invocation from exposure, mutation from no-op, and training
from held-out metrics. The main thread owns the requested Luna Git/W&B-only
agent; no agent was started or contacted by this side conversation.

The token-notice implementation `6d49d12` has previously recorded135 focused test
passes plus mypy/Ruff results. No tests/builds/hash checks were repeated for this
documentation-only update. See `docs/bayesian_training_repair_2026-09-09.md` for
the historical implementation and run evidence; private deployment/recovery
coordinates remain only in the local, untracked handoff.

---

## 2026-09-09 07:06 UTC — First new-condition step committed

- New8/20 mixed-thinking run: step1 committed all28 trajectories; TTB loss
  **2.20508338**, mean reward **0.33710623**, successes7/28. Durable commit
  events and the committed step journal agree; separate private CSV/JSON curves
  exported. Previous-condition checkpoints/results are unchanged.
- Wall715.68s (11.93min), observed5.03steps/hour; rollout681.04s,
  gradient tail32.27s,26/28 contributions finished before the last rollout.
  This is one cold step, not steady-state acceptance. At this rate remaining249
  steps would take49.5h; conditional arithmetic, not a completion guarantee.
- Two actual invocation updates have globally normalized weights summing to2;
  detector correctly reports residual window1/100. No natural evolution claimed.
  Step2 is collecting with the newly published step1 adapter. Actual per-domain
  decoding identities match the owner's thinking modes and turn limits.
- On22048 GPU0/6/7, processes live and thermal slowdown inactive. No new GPU
  jobs or code changes during monitoring; no repeated tests. Read-only inspection
  found invalid model submissions, but no confirmed transport/completion fault.

## 2026-09-09 — New owner-declared 8/20, mixed-thinking formal condition

- Supersedes the unstarted5/25 continuation: new run from Step-0, **static max8,
  ALFWorld max20**; native thinking **ON only HealthBench and AIME2026**, OFF for
  HotpotQA, TriviaQA, HumanEval, MBPP+ and ALFWorld. Seven domains/B28/250/seed0,
  one Qwen3.5-9B owner, TTB, optimizer and checkpoint cadence remain unchanged.
- Per-domain modes are now explicit immutable rollout configuration, saved in
  method snapshots and applied to each task's actual decoding identity. This is
  a declared sampling-condition change, not an equivalent performance patch.
  Reasoning/action phases remain; action generation remains non-thinking.
- The old run's step2 full checkpoint and two committed loss/reward records are
  preserved. At the owner's immediate-stop request, the unfinished numerical
  comparison was cancelled; it is neither a passed comparison nor a detected
  numerical failure. Unqualified chunked/pinned candidates remain disabled.
- Targeted formal/hydration/continuation tests:13passed; scoped mypy3passed.
  Final integrated CPU `make check`: **3,700 passed,2 CUDA opt-in skipped**
  (296.85s), Ruff/mypy, both wheel builds and model-wheel boundary passed.
  New training started from Step-0 at06:52UTC: B28,250steps, two participating
  gradient ranks and within-step overlap confirmed. No new committed result yet.
- On22048 GPU0, the refreshed inference service passed health checking; GPU6/7
  were handed from requested SGLang standbys to gradient rank0/rank1.
  Owner-requested Luna xhigh now handles only Git publishing, without edits or
  quality checks. No review agents/hash checks; unrelated GPU replay not repeated.
  Per-step loss/reward remain durable commit-event data, not partial-batch counts.

## 2026-09-09 — Saved step2; horizon continuation and long-tail qualification

- Target: **static max5 / ALFWorld max25**, seven domains, B28,250steps,
  seed0, one Qwen3.5-9B task owner, native reasoning thinkingON. This is a
  declared rollout-condition change, not an equivalent speedup. New-condition
  training has **not** started; old2/50 training stopped only after step2 was
  fully committed and checkpointed, with model/optimizer and method state saved.
- Implemented cooperative drain/checkpoint pause; completed-artifact and original
  budget-charge persistence for same-condition recovery; lazy current-batch
  environment preparation; persistent wall-time accounting; and explicit
  horizon-only continuation preserving original configuration and evidence.
- Added actual forward-input token reuse, per-rank wait and FLA telemetry,
  measured cache/prefill fields, read-only action failure classification and
  queued-work-aware initial owner assignment. No implicit state migration.
- Real installed SGLang CPU reproduction confirmed scheduler timestamps disappear
  on the second IPC pickle hop. Added scalar-only timing transport preserving
  unknown values and leaving receiving-process metric collectors disabled.
  Three-hop installed-version regression passed; request-ID timing logging is
  explicit in the formal serving profile. No sampling/kernel changes in this fix.
- Bounded pinned activation pool/DMA lifetime handling and chunked full-vocabulary
  target log-probs are opt-in execution candidates, not yet promoted. The CUDA
  retained-graph/noncontiguous activation test passed. Complete fixed B28 replay
  (238 edges, including real long ALFWorld prefixes) is running on22048 GPU6/7;
  GPU0 keeps the inference service. Replay does not generate training evidence or
  update posterior. LM-head fusion and live trajectory migration are not enabled.
- Final CPU `make check` on22049: **3,699 passed,2 CUDA opt-in skipped**
  (291.51s), with Ruff, mypy, both wheel builds and model-wheel boundary passing.
  An initial subprocess import-path mismatch was corrected. Later validation
  found a real preparation race: queue capacity was released before cancellation
  became visible. Failure is now latched before capacity release, with a
  deterministic delayed-Future-callback regression; original errors propagate.
  Scoped preparation/boundary tests passed20; related streaming tests passed30.
  No tests were relaxed. Unrelated GPU kernel tests were not repeated. No agents,
  manual hash checks or idea.tex/AGENTS.md edits.
- Actual committed rewards:step1 mean0.3316327,8/28 successes;step2 mean0.3072998,
  8/28 successes. Step1 loss1.7868633 and wall47.65min are cold-step observations,
  not stable throughput. Step2's timing sidecar was interrupted after durable
  commit; it is not silently filled with an estimate.4.2steps/hour and72hours
  remain unverified performance risks requiring realistic planning.
- No IID evaluation or new5/25 on-policy training is claimed. Remaining work:
  finish fixed full-batch numerical/speed qualification, retain safe execution
  for any rejected candidate, then explicitly restore checkpoint2 and publish
  the new condition boundary before step3. Private HANDOFF contains PIDs/paths.

## 2026-09-09 03:42 UTC — static cap remains2; explicit recovery process started

- The owner briefly requested static max10, then reverted to **max2**. The local
  edits were discarded before deployment or commit; source935762c and the frozen
  running condition are unchanged. ALFWorld retains max50; native thinking staysON.
- The stop signal issued for the intervening request had already taken effect.
  The interrupted batch had28 artifacts but no committed optimizer/posterior step;
  temporary gradients cannot be resumed. Logs and sidecars were preserved.
- A new two-rank process was explicitly launched with the **same run/config and
  complete initial checkpoint**. It is preparing sessions; successful application
  restoration and next-step commit still require observation. This is not a
  silent configuration change or an automatic retry of a failed training step.
- GPU0 inference stayed online; requested SGLang standby heldGPU6/7 during the
  interruption and was drained before the new gradient ranks launched. Private
  supervision and the GPU monitor remain active; other projects were untouched.
- Real overflow handling was observed in the interrupted batch:9 requests were
  truncated, with maximum original length69486 reduced to65536, and all28
  trajectories completed. No completed-step throughput can be inferred from an
  interrupted batch. The72-hour target remains at risk; no budget was reduced.
- No retained source changes, repeated full tests, subagents or manual hash checks.

## 2026-09-09 02:44 UTC — repaired formal run started on GPU0/6/7

- Owner reassigned the run to **22048 GPU0 inference / GPU6 coordinator and
  gradient / GPU7 second gradient**. All eight devices were checked before and
  after launch; existing unrelated processes were left untouched. The previous
  resource-blocked note below is historical.
- Qualified source **935762c** is unchanged. The inference service is healthy and
  its effective context/cache/determinism/LoRA configuration was read back. Both
  gradient rank processes have started; the entrypoint is preparing seven-domain
  sessions and loading the model. **No optimizer/posterior step is committed yet**;
  process startup is not a completed training transaction.
- Condition remains250 steps/B28/seed0, six static domains max2 turns, ALFWorld
  max50, native thinking ON, declared65536-token input window, every10 retained
  checkpoint. No new four-step or evaluation run was launched.
- At the owner's request, private process supervision retains the assigned cards
  with SGLang standby after training exits. Temporary startup standbys were
  stopped before the gradient workers launched. Training is **not automatically
  retried**; failures retain their logs and require attention. Standby status is
  never counted as training progress. A read-only GPU monitor is also running.
- No measured ordinary-step rate or empirical ETA yet;4.2 steps/hour implies
  roughly59.5 hours plus overhead, not a guarantee. Existing3662-pass CPU
  qualification was not repeated for deployment-only changes; no subagents or
  manual hash checks were used.

## 2026-09-09 — failed formal run; authorized truncation/static-horizon repair

- The first 250-step/B28 attempt (`924f882`) ended **01:23:24 UTC**, exit1,
  with **0 committed steps**. Input lengths65709/67201/66627 exceeded65536;
  the batch failed without partial optimizer/posterior updates. Failed evidence
  and the initial checkpoint are preserved. All its owned processes finished.
- The owner explicitly changed the next run: six static domains **max2 turns**,
  ALFWorld **max50**; input overflow is deterministically **truncated, not a
  terminal failure**. B28/250/seed0/native thinking/R1024/A2048 remain unchanged,
  with every10 retained checkpoint and production W50/evolution controls.
- Repair in progress: persist one input-window rule across rollout, full-batch
  F/B scoring, provisional gradients and artifact recovery; keep complete raw
  histories and action tokens; restore JSON/escaping completion instructions.
  This is a declared new condition, not an equivalent old-run continuation.
- Validation completed: 130 focused tests; the first full check found a forbidden
  rollout-to-policy-internal import. Moved the window into the existing token-only
  interface (no boundary-test relaxation); 28 affected tests then passed. Final
  22049 CPU `make check`: **3662 passed, one CUDA opt-in skip**, format/lint1068,
  mypy559, both wheels and model-wheel boundary passed (pytest284.78s).
- A CPU-only real-Qwen-tokenizer check covered private history with an explicitly
  synthetic overflow extension: 69347–69995 input tokens became65536, preserving
  the1183-token H0 prefix and exact suffix IDs in both chat modes. No model
  generation, learning, GPU context or posterior updates occurred in that probe.
- Source **935762c** is pushed and staged. **Restart is blocked, not completed**:
  the personal02:02–02:03 UTC resource check found22048 GPU4/GPU5 occupied by
  another project (about55GB/27GB), above the allowed shared-use ratio. Neither
  endpoint presently offers the three-card capacity required by this deployment.
  No other-project process was stopped; no new GPU service/trainer/monitor was
  launched. The CPU checks/probe have finished. Owner resource coordination is
  needed before launch; there is no automatic takeover queued.
- No accepted throughput or empirical ETA yet;4.2/h and72h remain planning targets,
  not guarantees. Unchanged kernels were not re-profiled; no subagents/manual hash
  checks/AGENTS or idea.tex edits. No four-step rerun is planned.

See [the current formal condition](docs/bayesian_formal_250.md). Older notes below
are historical, not live process status or proof of natural evolution.

## 2026-09-08 — training synchronization in progress

The requested other local session was read again through its latest `c6cb683`
completion: first-batch evidence/evolution repairs are included in the new frozen
training source, with method semantics v5 and evolution config v7. Its unfinished
formal-entry, same-family Split, complete mutation/H0 and real-author/natural-W50
work remains explicitly unfinished. Seven-domain native reasoning and shared
Hotpot guidance are declared new sampling conditions, not retroactive equivalence
claims. The 15-case bridge and combined Linux CPU validation pass: **3648 passed,
1 CUDA opt-in skip**, Ruff/format, mypy and both wheels pass. New native-on request
qualification is in progress. No new real four-step/IID run yet.

Saved complete-Qwen scoring now gives exactly equal full projection and restored
posterior/detector state on the current code (B32/167 edges, four updates/three
cells); repeated whole-batch evidence is rejected. No Phi decision is forced.
Earlier native-off FIFO/fair 334-request pair is exact, 733.112 vs730.437 seconds:
only about0.37% observed reduction, not a large ALFWorld speed claim. GPU5 serves
qualification; the GPU4/7 gradient replay has ended. Details in the
[turn-latency note](docs/alfworld_turn_latency_20260908.md).

## 2026-09-08 — new ALFWorld per-turn optimization task

Implemented asynchronous bounded provisional preparation, terminal-plan reuse,
optional GPU temporary-gradient residency and aging request fairness, cleanup
handoff, accurate wait/transaction/phase sidecars, and process-local warmup.
CPU total check: 3624 passed / 1 CUDA opt-in skip; complete B32/167-edge GPU replay
501.234 s with bitwise-equal gradients, residuals and Adam state. Full request-chain
qualification is ongoing; the new four-step rerun is **not yet launched** at this
entry. The previous four-step run remains complete and unchanged. See
[execution note](docs/alfworld_turn_latency_20260908.md). Only the approved endpoint's
GPU5 inference and GPU4/7 gradient roles are used; unrelated processes are untouched.

# SKILLEV public handoff

## IID serving moved to shared GPU6 — 2026-09-08

- The owner's latest instruction assigns this IID session to **22049 physical
  GPU6**, explicitly sharing with its existing process, not GPU1. At 19:27 UTC,
  the new Qwen3.5-9B BF16 SGLang passed health and synthetic generation checks.
  It exposes only GPU6; the existing process was left running. The service uses
  tested source `fdbe3ea`, context 98,304, KV capacity 196,608 and concurrency four;
  per-task output budgets are unchanged. Observed service memory was about
  26.9 GiB; shared-card capacity and throughput remain subject to fluctuation.
- This session's previous 22048 GPU5 service/evaluator were stopped at 19:07 UTC
  for the resource exchange. This supersedes the older HealthBench note below
  that its serving process stayed online; the other training session's new
  22048 services are separate and were not stopped by this IID session.
- Fresh HumanEval finished **29/32 (90.625%)**. AIME was operator-interrupted at
  **28/30 submissions, 26 correct**, not a completed 30-case result. No fresh IID
  round or answer-merging/resume was started as part of the GPU6 service launch.
  [Repair results and serving details](docs/step0_seven_iid_followup_2026-09-08.md).

## ALFWorld-tail repair and real four-step completion — 2026-09-08

- Implemented whole-request environment IPC deadlines, live turn/queue sidecars,
  long-horizon-first launch order and completed-step F/B/Z gradient precomputation.
  Fixed observed cache precision/concurrency failures and sparse-arrival affinity
  skew before continuing. The stopped v1 had **zero commits** and remains separate.
- Final fixed B32/167-edge dual replay: **502.92 s vs 872.43 s** single-owner
  baseline (1.73x), bitwise-equal gradients, edge scores, residuals and Adam states.
  Cache/base/LoRA/reload/mixed-judge fixed-request output-ID checks passed.
  These are diagnostic results, not new benchmark or whole-training speed claims.
- **V2 four steps and 2+2 fresh-process restore passed**, source `ed04e85`.
  Seven domains, B32, seed0, production W50, one Qwen3.5-9B task agent.
  All **128 trajectories / four transactions** committed; final optimizer step 4,
  task cursor 128, final adapter published, no pending transactions. Posterior
  updates **[3, 0, 5, 0]**, six cells, reconstruction passed; empty batches remained
  empty evidence, not invented invocations.
- Step times: **14.44 / 16.89 / 17.12 / 17.20 minutes**. Complete-step average
  **3.66 steps/hour**, below 4.2: **human performance decision still needed**.
  First-step gradient tail fell from the earlier 448.45 s observation to 40.67 s;
  this is not steady-state acceptance. Four steps supply only two measured steps
  after warmup. **Zero phase detections/mutations**: natural W50 evolution,
  authored-skill continuation, calibration quality and IID gains remain unverified.
- Final 22049 CPU `make check`: **3605 passed**, one CUDA opt-in skip,
  format/Ruff, mypy, both wheels and package boundary. Affinity/streaming affected
  group: 24 passed; bootstrap isolation: 19 passed. No unchanged GPU/full checks
  repeated after documentation-only updates; no agents/manual hashes.
- Personally checked **22:32 UTC**: all this run's diagnostic, serving and training
  processes ended; GPU4/GPU5 empty, GPU7 only its pre-existing unrelated process.
  Mapping was **22048 GPU5 inference, GPU4/GPU7 gradients**. Main v2 used three
  cards for roughly 67 minutes; earlier fixed-input qualifications are additional.
  No new IID round, hardware adjustment, `idea.tex` or `AGENTS.md` change.
  [Implementation, failures and evidence](docs/alfworld_tail_execution_20260908.md).

## Seven-domain / three-card training main path — 2026-09-08

- New training excludes WebShop: seven allowed domains plus a rotating eighth
  question occurrence, four independent rollouts each (B32, seed0). Current
  source loading accepts seven-only files without the old eight-domain gate.
- Training now requires **one inference card plus two actual gradient workers**;
  rank zero participates. No two-card/one-gradient fallback. Implemented
  ready-first work assignment with global canonical contribution addition,
  scoped adapter/backward execution, batched scalar reads, owned copy buckets,
  frozen-parameter offload avoidance, live long-tail/capacity metrics and bounded
  independent environment preparation. Full-batch Bayesian commits are unchanged.
- Fixed complete B32 gradient times: **866.7 → 835.3 → 435.1 seconds** for baseline,
  new single-owner and two-gradient execution. Dual is **1.99x** baseline; all
  compared gradients, scores, residuals and Adam states are bitwise identical.
  Controlled >32K storage checks also matched exactly. These are fixed-input
  diagnostics, not newly collected training or steady whole-step throughput.
- Final CPU integrated check passed **3546 tests**, one CUDA opt-in skip,
  format/Ruff, mypy, both wheels and package boundary. The latest seven-domain /
  mandatory-three-card entry changes also passed their ten-test affected group.
- All diagnostic CUDA processes have ended (about 0.8 GPU-hours). The first
  three-card launch was rejected before service/training startup when its third
  card became occupied. The owner subsequently exchanged resources with another
  session: this training session now uses **22048 GPU4/5/7 only**, with GPU5 for
  inference and GPU4+GPU7 for gradients. GPU0 is released. After explicit owner
  authorization, the old project training on GPU4 was stopped and capacity was
  rechecked. New seven-domain B32 training launched at **19:13 UTC**; both gradient
  ranks and the healthy inference service were personally observed on these cards.
  Initialization is not a completed training step; no new IID round is claimed.
- Four-step 2+2 recovery and W50 natural evolution remain unverified here.
  At 19:40 UTC, **step 1 was committed and batch 2 was collecting**: 1473.45 s
  (24.56 min), 30 gradient contributions completed before the last rollout,
  990.68 s overlap and 448.45 s gradient tail. The checkpoint contains four
  posterior updates across three cells and a committed adapter transaction.
  This single warmup observation is below the 4.2 steps/hour target; seven-domain
  timing is not a controlled speed comparison with historical eight-domain runs.
  No agents, manual hashes, idea.tex/AGENTS edits or unqualified kernel changes.
  [Implementation, numerical evidence and limits](docs/training-mainpath-seven-domains-20260908.md).

## HealthBench conversation/history repair — 2026-09-08

- Source `0937752`: original public dialogue is separately readable through
  history; execution logs explicitly are not external records. HealthBench asks
  for the next conversational reply, not a short-answer/code final. Optional
  clarification remains the owner's choice; no rubric hints or second agent.
- Official-source comparison matched all 32 conversations/rubrics and the pinned
  grader template. The previous completed repair was retrieved: **36.20596**, not
  an unknown score. Its 47 owner calls included 14 history reads and one repair.
- The fresh same-32 A2 round scored **45.03364/100**, 32/32 completed, 33 owner
  calls, one conversation read, no peer/adapter/thinking/skills, no failures.
  All 33 owner and 400 judge requests retained the full dialogue; all 32 native
  calculations and scoped owner submissions matched. Grader settings and owner
  budgets unchanged; no selection, baseline fallback or repeated favorable rerun.
- About **90.05 seconds** to the summary, versus the preceding 112.41 seconds;
  this is a small warm-service observation, not training throughput. Only existing
  22048 GPU 5 was used. The evaluation and CPU checks ended; SGLang stays online.
  No new GPU service, training, other IID round or resident monitor was launched.
- Main-thread work; 154 affected tests passed. The one 22049 CPU final `make check`
  passed **3518 tests**, one CUDA opt-in skip, format/Ruff, mypy, both wheels and
  model-package boundary. Local Python checks timed out without results; they are
  not pass claims. No agents/manual hashes; docs-only updates reuse the final check.
- **45 is met only narrowly on this development-exposed 32-case panel**; no robust
  margin, unseen-128 result, matched-backbone gain or official GPT-judge parity is
  established. Local judge reliability and answer-quality limitations remain.
  [Detailed aggregate report](docs/step0_healthbench_context_diagnosis_2026-09-08.md).

## Training speed follow-up — 2026-09-08

- Connectivity was personally restored at 11:42 UTC. The frozen four-step
  integration had hit its two-hour first-half timeout at 11:16: **one committed
  step**, not four. Batch two had 32 complete artifacts and 31 provisional
  gradient contributions, but no second update. Its service/trainer exited;
  another user's subsequent GPU4 process was identified and left untouched.
- The committed step took 2284.35 seconds (1.58 steps/hour), with 727.97 seconds
  collection, 698.37 seconds overlap and eight gradients finished before the
  last rollout. Local gradient compute was 1705.44 seconds; queue wait was
  575.29 seconds. The 512 MiB buffer held only eight out-of-order contributions.
  This establishes actual overlap, not steady-state throughput acceptance.
- The explicit H800 profile now budgets 2 GiB for canonical-order host buffering;
  legacy omitted-field defaults stay unchanged. Wait metrics separate missing
  artifacts from buffer backpressure and exclude remote dispatch from rank-0
  compute timing. The affected CPU group passed 50 tests and targeted mypy.
- On the same fixed GPU input, checkpoint threshold 4096 preserved all compared
  gradients/scores exactly and changed 97.99 seconds to 94.78 seconds, but raised
  peak allocation from 32.04 GB to 59.08 GB. Threshold 8192 exhausted memory.
  The small speed benefit does not justify changing the memory-safe default.
  These are component probes, not new on-policy training or IID results.
- Actual FLA 0.5.2 use was confirmed; the missing component was causal-conv1d.
  Isolated official-package probes on 22049 GPU3 are complete and cleaned up.
  Native fused convolution was faster but failed the unchanged numerical bound.
  The locked-version native-forward/reference-backward probe matched all compared
  scores/gradients exactly, but improved only 97.17 to 94.61 seconds (2.64%).
  No custom convolution or checkpoint tier was promoted to the training default.
- Comparison reports now stay incomplete until every requested variant finishes;
  an OOM after the baseline cannot leave an apparent comparison pass. Runtime
  package versions are recorded. Both report-lifecycle regression tests passed.
- No new training, inference service or IID round was launched during this tuning
  follow-up. Four sequential fixed-input GPU3 probes and two isolated CPU package
  builds ended; GPU3 retained only its pre-existing unrelated process afterward.
  The buffer improvement still needs a separately identified end-to-end run;
  no measured new steps/hour or completion ETA is claimed for the stopped training.
- Final CPU-only `make check` on 22049 passed **3512 tests**, one CUDA opt-in
  skip, format/Ruff, mypy 539 files, both wheels and model-wheel boundary.
  The initial final check exposed an inherited nested-pytest temporary-directory
  option and one stale HealthBench message assertion; the runner/test were fixed,
  not the evaluation behavior. Focused checks totaled 52 passing tests. All check
  processes ended; no agents, manual hashes or new IID/E2E run was used. Successful
  GPU comparisons were not repeated after documentation/test-only corrections.

## Execution acceleration implementation — 2026-09-08

- Implemented local within-step gradient ownership, ready-first computation with
  canonical byte-bounded reduction, request-local server JSON stopping, incremental
  decoding, execution-only scoring tiers/telemetry, and HealthBench request-broker
  wiring. Cancellation drains the CUDA/HTTP owner before clearing gradients or
  returning its shared capacity. Whole-batch Bayesian/optimizer commits are unchanged.
- Real SGLang synthetic format probes preserved all admitted token prefixes while
  avoiding the previously discarded suffix. This is not an end-to-end speed claim.
- **BF16 edge microbatching failed numerical acceptance** (forward/backward errors
  about 10%/2.5%, bound 1e-3); no measured speedup on the fixed subset. Default stays
  at one edge. A fresh-process cumsum-shape configuration check also failed;
  no new FLA table was promoted.
- Final CPU integration check passed **3502 tests**, one CUDA opt-in skip,
  format/Ruff, mypy 539 files, both wheels and model-wheel boundary. The subsequent
  historical-parser compatibility adjustment passed all 135 affected tests;
  old parse rejections are not relabeled as skill invocations. Earlier
  70/14/178/26 focused groups passed. Initial full-check boundary/stub failures were
  corrected without relaxing the dependency or no-fallback rules.
- The old frozen GPU0/6 run committed two steps, restored them in a fresh process,
  then failed the third batch on invocation admission; no third update. Its own
  service/trainer have exited. Live codec and persisted invocation admission now
  share optional Action-header / JSON-fence parsing, preserving original tokens
  and genuine invocation credit. The old checkpoint/history were not changed.
- Separate GPU3 stopping probes and GPU5 fixed-input gradient diagnostics have
  ended and cleaned up. The cumsum-only isolation did not resolve the BF16 error.
- New, separately frozen **9b12716** four-step integration started on 22049:
  physical GPU3 serves inference; GPU4 is the local gradient/coordinator owner.
  The GPU5 launch guard rejected a newly occupied card before training started;
  no other user's process was changed. This is a two-card debug deployment, not
  a formal three-role run. It retains batch32, seed0, W50 and explicit native
  EvalPlus 4s/4x; two steps plus a fresh-process two-step continuation are planned.
  SSH loss initially prevented monitoring. The 11:42 UTC reconnection confirmed
  the timeout and single committed update described above. There is no remaining
  ETA for that stopped attempt; its original 1–3 hour estimate is superseded.
- No new IID round or natural W=50 evolution claim. No agents, manual hashes or
  idea.tex/AGENTS edits. Remaining acceptance work includes steady-state overlap performance,
  four-step/new-process completion and a numerically acceptable microbatch path.

Details: [execution implementation and acceptance limits](docs/execution-performance-20260908.md).

## 贝叶斯评分与证据闭环修复（2026-09-08）

- 训练/IID 共用固定 EvalPlus 26d6d00、4s/4x profile 和 v2 请求/响应；移除训练隐式短限时分支。
  新 snapshot identity v7 同时绑定实际 scorer 与公开 task-feature mapping；旧历史不重标、不伪装等价续训。
- 完整批次 provenance 增加只读来源问题/H0 曝光坐标；报告分开显示曝光、真实调用、轨迹、来源问题、
  更新 cells、权重集中度与分组 pre-batch Brier。没有更改流归一化、Beta/LCB、相变阈值或信用单位。
- 新训练/IID 使用 benchmark-public-task@1；真实缺失工具/上下文不匹配有原因记录，不扩大适用范围。
  IID 的 checkpoint 来源、实际 mutation/后验数量与 C0–C3 轴显式区分；这些协调端信息不进 actor。
- 定向 50 项真实 scorer/合同检查、后续修订 21 项通过。最终 CPU 总检在22049本地盘分阶段完成：
  format 1018、Ruff、mypy 532、pytest 3467 passed/1 CUDA skip、两个 wheel 与 model boundary通过。
  首次全检在 collection 发现旧测试仍导入0.2常量；改固定版本常量后只补该范围并完成 test/wheels，
  没有重复成功的全量lint/mypy。WSL定向mypy超时改在同一CPU环境完成；未启动审查子代理或哈希检查。
- GPU仍只有原有22049 GPU0/6旧条件run，源码7ef421e未热更新。本轮亲查07:33 UTC为1/4步提交，
  第2批32/32已收齐，累计5 invocation边；GPU6热降频。首步1.18step/h，剩余粗估2–3h低置信，
  散热/持续吞吐需要人工处理。没有新增GPU任务或常驻监控进程。
- 尚不能宣布完整真实闭环验收：新评分条件GPU训练、真实作者后新库训练、真实新进程服务恢复仍需
  分别积累证据；4步W50不可能证明生产自然演化。不启动新IID轮，不选择性回退checkpoint或答案。


This is the **sanitized, Git-safe** handoff for the repository. The ignored private
`HANDOFF.md` carries operational details needed by the owner. Whenever project state changes,
update both files in the same work session; this file must contain only the public subset.

## GPU0/6 retry running; EvalPlus fix fully checked — 2026-09-08

- The first two-GPU attempt exited during first-batch grading: 20 completed trajectories
  from 21 started, **zero optimizer updates and zero posterior commits**. The batch never
  sealed. Infrastructure failure correctly stopped training instead of becoming a false
  negative label or a partially normalized posterior batch. Its own inference service was
  cleaned up; at 05:25 UTC physical GPUs 0/6 were both empty, and original run PIDs had exited.
- The worker imported `PASS`, but the pinned official EvalPlus 0.2.0 exports `SUCCESS`.
  The unit substitute repeated that incorrect API. Both are corrected without changing the
  package, scoring rules or Bayesian math. Three fresh-process tests use the real dependency
  with synthetic programs: full pass, base failure and plus-only failure. All **35 focused
  tests passed on 22049 CPU** (2.53 seconds). The originally failed generated submission was
  also graded CPU-only through the actual private worker and installed 0.2.0 interpreter;
  raw input/output stay private. The local WSL test timed out, not a pass.
- The final CPU-only `make check` on 22049's local memory filesystem passed: **3,396 tests,
  one CUDA opt-in skip**, 106 existing PEFT warnings (198.85 pytest seconds, about 17.1 tests/s),
  format (1,000 files), Ruff, mypy (523 source files), both wheels and model-wheel boundary.
  Exit zero was personally confirmed. Its code is the committed `a469f51` base plus this
  three-file fix; subsequent handoff-only changes do not repeat successful checks.
- Fresh committed source **`7ef421e`** is running in a separate directory. GPU0 inference
  started at 05:29 UTC, then health and physical mapping were checked before GPU6 training
  started at 05:30. The first **32/32 trajectories sealed at 05:52 UTC**, after 19.6 minutes
  (1.63 trajectories/minute), with 205 execution edges and two actual skill invocations.
  Real EvalPlus grading completed without the prior infrastructure failure. At 06:21,
  **zero optimizer/posterior commits** had completed; GPU6 remained active in scoring/gradients.
  No per-batch gradient progress counter is available, so completion cannot be inferred from
  utilization. The failed run is preserved, not resumed with changed grading. Production
  W=50/thresholds, single-agent topology, seed 0, full-batch weights and two-plus-two
  fresh-process recovery are unchanged; four-step/resume/controlled-author results remain pending.
- Both authorized cards show thermal slowdown under load: GPU0 reached 88 C during inference;
  GPU6 reached 89 C during gradients and was still 88 C/345 MHz at 06:21. GPU0/GPU6 used about
  62.0/41.5 GiB at that check. **Human cooling/resource decision requested**: continue the bounded
  test or pause for cooling. No clocks/fans/power limits or other cards were changed. Until
  instructed otherwise the existing bounded process remains active; rough total ETA is 4–8 hours,
  low confidence with zero completed steps. Each two-step half has a four-hour deadline and
  scoped cleanup. Four steps still cannot establish natural W=50 evolution.
  No AGENTS/idea.tex edits, independent agent, package upgrade or manual hash check occurred.

## Current evaluation scope: seven-IID existing-results summary — 2026-09-08

- The owner withdrew further WebShop work. The [seven-IID summary](docs/STEP0_SEVEN_IID_CURRENT_RESULTS_2026-09-08.md)
  and aggregate export list latest available records, not a best-of or one-version result.
  Latest static scores are HotpotQA F1 72.13, TriviaQA F1 82.62, local-Qwen HealthBench
  43.59, MBPP+ 72.66 and HumanEval 85.94; the last three used optional peers in the older
  condition. No complete current strict-single-controller seven-domain run is established.
  The separate fixed hard MBPP+ subset is unverified; HumanEval's historical accepted
  118/128 is retained separately rather than copied into the latest result.
- AIME thinking-on remains 25/30 (83.33%), including three output failures, with no actual
  peer calls. ALFWorld is 103/126 (81.75%) in the latest read-only snapshot, two unfinished;
  it is not a final 128-case mean. All reported conditions have zero optimizer updates.
- Repair batch `6bb921e` was committed and pushed. Its new mixed @6 run was stopped on
  the scope change with 44 ALFWorld candidates and zero scores; no WebShop candidate and
  no answer reuse. At 05:00 UTC the main thread confirmed it had exited with no run-specific
  residual, while the old @5 ALFWorld tail and existing 22048 GPU5/7 services remained.
  Recent tail progress was zero completions in about five minutes; rough token-budget ETA
  15–25 minutes, low confidence. No other session's training resources were operated.
- Summary-only validation checks source values, scope, JSON, links and diff. Prior successful
  source tests/builds are not repeated; no new benchmark, service, agent or hash check is
  added by this summary. Private operational details stay in the ignored HANDOFF.

## Historical first GPU0/6 attempt (later failed grading) — 2026-09-08

- The owner explicitly authorized **22049 physical GPUs 0 and 6**, superseding the preceding
  4/5-only restriction for this run. Fresh checks found both cards empty. GPU0 hosts inference;
  GPU6 hosts the coordinator and canonical local TTB gradients. UUID-scoped CUDA visibility
  prevents this host's previously observed numeric-device remapping. Other cards/services were
  not modified. This two-card sealed-batch integration run is not formal three-role deployment.
- Committed source `087a596` includes the continuous Bayesian fixes in `8c14dfe`; later uncommitted
  evaluation changes are excluded. The existing seed-0 checkpoint and native training bindings
  were checked CPU-only: four steps, batch 32, 128 trajectories. Production W=50, thresholds,
  terminal rewards and full-batch posterior normalization are unchanged; one Qwen3.5-9B task agent.
- A new bounded inference service started at 04:48 UTC; health and the physical GPU0 process
  mapping were personally confirmed before starting the trainer at 04:50 UTC. The test runs
  two steps, exits, then restores into a new Python process/model for two more. A separate
  controlled real-author/retrieval check follows only after core training succeeds.
- **Started, not passed:** there is no four-step, real-resume or natural-evolution result yet.
  The old rate was about 1.04 steps/hour; the new placement has no measured speedup yet, with
  a rough two-to-four-hour estimate to revise after its first commit. Natural eligibility needs
  at least 100 batches at W=50 and is outside this four-step claim. Low throughput remains a
  human resource decision, not permission to lower scientific thresholds.
- Private launchers have per-process deadlines and scoped cleanup; PIDs/logs are recorded in
  the ignored HANDOFF. No packages/shared environments, production source, AGENTS or `idea.tex`
  were changed. Only private syntax/import/binding checks were run; the prior full check was
  not repeated. No benchmark round, subagent or manual hash check was started.

## GPU restriction update — 2026-09-08 03:48 UTC

- The owner now permits **physical GPUs 4 and 5 only** for the pending run on 22048.
  The earlier GPU0 alternative is withdrawn; no other card will be used.
- A fresh all-card status check found GPU4 at 36,351 MiB / 96%, occupied by another
  project's training, and GPU5 at 69,342 MiB / 100%, running the existing project inference
  service. Neither meets the sharing condition. No existing process was stopped or changed;
  permission to pause the GPU5 service has been requested explicitly.
- The new four-step training/resume test remains **not started**. No new GPU task or persistent
  process was launched. This resource-only update changes no source, method or AGENTS rule;
  prior successful checks were not repeated, and no subagent or manual hash check was used.

## Continuous Bayesian evolution repaired; new GPU run blocked — 2026-09-08

- This entry supersedes the historical permanent no-op closure and “four-step run active”
  statements below. The existing Bayesian application remains the only training chain;
  no replacement posterior, reward adjustment, shard-local normalization or answer selector
  was introduced. `idea.tex` and `AGENTS.md` are unchanged; one Qwen3.5-9B task agent remains.
- **A — lifecycle:** method semantics v4 records a no-op's phase/batch/step cutoff. Rechecking
  requires a complete fresh residual/entropy comparison after that cutoff, with the original
  AND condition and thresholds. Duplicate evidence remains rejected across restore. No-op
  changes neither library, posterior nor Z. Longer Protocol 13 debug runs require an explicit
  run plan; all declared cycles receive checkpoints, per-cycle authoring limits are explicit,
  and resolved configuration/total budgets are written before training. A 250-step/two-cycle
  plan is supplied, not automatically launched; cycle-cap errors remain meaningful.
- **B — correctable skills:** immutable interface/security constraints are distinct from
  evolvable strategies. Refine/Split may revise or remove strategies only with recorded
  replacement IDs and evidence-based reasons; immutable constraints cannot be weakened.
  Retain preserves requirements. Prompts, validators and the actual SGLang JSON schema agree.
  New advisory seeds use new IDs; legacy untyped requirements retain their immutable wire
  identity. All modified/generated skill IDs start from prior, not inherited parent evidence.
- **C — evidence:** committed-source reports expose coverage, entropy, residual windows,
  uncovered high-importance edges and non-trigger reasons. Prequential invocation diagnostics
  use the cell's pre-batch posterior for every event, with Brier error, probability bins,
  distinct trajectories and weighted mass kept separate. Features remain post-hoc; these are
  neither routing inputs nor proof of causal calibration. The estimator's full-history scope
  and a hand-calculated skill-flow/Beta example are documented in `docs/bayesian-chain.md`.
- Controlled tests cover two mutations, both phase checkpoints, next-library training,
  Z-only Adam resets and a new CPU process/model's identical next update after public restore.
  Directed validation passed 121 tests, then 23 focused statistics tests and the fresh-process
  test; 72 affected final regressions passed after fixture updates. Final CPU-only `make check`
  on 22049 passed **3,366 tests, one CUDA opt-in skip** (179.75 pytest seconds, about 18.7 tests/s),
  formatting (997 files), Ruff, mypy (522 files), both wheels and model-wheel checks; final exit
  zero was personally confirmed at 03:44 UTC. This tested the `7e24b6e` base plus this task's
  changes, excluding another session's later uncommitted evaluation edits. Pure handoff changes
  do not repeat the check. Local WSL timeouts and the earlier failing fixtures are not passes.
- **D — real validation remains incomplete:** the previous GPU4 run was stopped after its
  first committed batch (32 trajectories), not after four successful steps. Its measured rate
  was about 1.04 steps/hour. At 03:44 UTC, a fresh check of all eight cards found physical GPU4
  occupied by another project's training (34,153 MiB, 40% utilization), above the permitted
  sharing threshold. That process was not terminated or modified. GPU0 had about 6.3 GB/0%,
  but switching from the explicitly requested GPU4 is awaiting the owner's decision.
- New source and bounded scripts are staged, **not launched**: four real steps with production
  W=50, batch 32 and seed 0, split into two steps then a new-process public resume for two more.
  A separate real-author/retrieval check is prepared and explicitly controlled, not natural
  evolution. Single-card cohosting is nonformal. No new GPU job or persistent service was
  started in this task; the old temporary trainer/service were stopped and the former owner
  service remains paused. At the old rate, reaching 100 eligible batches alone is roughly
  96 hours and requires a separate resource decision; eligibility does not guarantee a phase.
- Zero/one-skill entropy reachability remains an explicitly reported method limitation, not
  “fixed” by inventing invocations or replacing AND with OR. Production natural evolution,
  real-author/new-library continuation and real GPU cross-process recovery are **not yet
  established**. Old semantics/configs are not silently migrated. No independent review agent,
  manual hash check, new benchmark round or licensed dataset content was introduced.

## AIME equivalence and frozen learned-library evaluation repaired — 2026-09-08

- Issues 1, 2 and 4 are implemented in `3aebfbc`, `e09f895`, `0e20995` and `288ab7c`.
  Direct integer answers, explicit final declarations and native submissions share scalar
  parsing: whitespace, trailing periods and equivalent boxed wrappers do not trigger another
  model call. Conflicting values and alternatives are not selected by the framework; earlier
  intermediate numbers are not promoted to final answers. Raw responses and token usage remain.
- Clean evaluation now receives actual active skill content from an explicitly selected,
  completed checkpoint's canonical library snapshot. Policy identity, library identity and
  retrieval rule are separate configuration axes. No-skill, fixed initial and evolved libraries
  are distinct; no optimizer, trainer, training evidence or online evolution is restored.
  See `docs/clean-trained-policy-evaluation.md` for configuration format v5 and scope.
- The existing production retriever and skill-invocation observation are reused. The main
  model can list, retrieve, read and invoke skills, including discovering skills that did not
  automatically match. Basic tools remain available without skills. Current production
  retrieval uses active documents/applicability, not a learned ranker or calibration state.
  Diagnostics separately record permission, matches, actual post-packing body exposure,
  invocation, no-match, body-budget skips and context omissions. Text-skill invocation is an
  explicit adoption acknowledgement, not a hidden solver or evidence of following the advice.
- Behavioral tests observe two different evolved snapshots reaching actor prompts, forward
  LoRA transport with actual skill text, discovery/read/invocation and no-skill isolation.
  Directed sets passed **225 tests**, then **146 affected regression tests** after fixes.
  Final `288ab7c` source passed CPU-only `make check` on 22049: **3,336 passed, one CUDA opt-in
  skip**, 104 existing PEFT warnings, 175.84 pytest seconds; formatting (991 files), Ruff,
  mypy (519 source files), both wheels and model-wheel boundary checks passed. Final exit was
  zero at 02:49 UTC. Earlier type-export, intermediate-number and fixture failures were fixed;
  the local WSL type-check timeout is not counted as a pass. Documentation does not rerun checks.
- Own CPU runners exited; a fresh process check preceded removing only this run's copied
  source/environment/cache. Logs were retained privately. No GPU job, persistent service,
  benchmark round, subagent or manual hash check was started. `idea.tex`, `AGENTS.md`, scoring
  and AIME thinking rules were not changed. Other sessions' uncommitted changes and GPU jobs
  were not modified and are outside this verification claim.
- This verifies interfaces, frozen-library restoration and model-visible content, not real
  Qwen3.5-9B score gains. The current one-main-agent rule is preserved; no voting, candidate
  selection or fallback answers are introduced. IID remains the owner's eight named tasks;
  a separately defined **MBPP+ hard** subset is still needed before claiming that specific result.

## Mandatory single-main-agent topology

- The owner now requires **one Qwen3.5-9B main task agent** throughout the architecture,
  training/rollout and **all evaluations**, including Step-0, baselines and ablations.
  The two consultant/peer agents must not be configured, called, renamed into equivalent
  advisers, or used for voting or answer selection. Independent trajectory concurrency and
  the method's forward/backward, Z, Bayesian statistics and skill evolution remain distinct
  from task-agent topology.
- This mandatory rule is written into the locally ignored `AGENTS.md`; its ignore status
  is preserved. This update changes instructions only: implementation/configuration migration
  and already-running processes have **not** been changed or verified by this documentation task.
  Older multi-agent descriptions below are historical, not authorization for new runs.
- No tests, GPU commands, new processes, subagents or hash checks were run for this rules-only edit.

## Bayesian A–D complete; new real four-step run active — 2026-09-08

- Implementation is in `c8cdf13`: exact scored action spans and sealed source order; one
  committed posterior authority; explicit phase/no-op lifecycle; real weak-context evidence
  for Refine; new-ID prior isolation and next-library collection; initial-step rollback,
  durable received author drafts, and named optimizer-state restoration. Existing Beta
  weights/labels, confidence formulas, delta-squared windows and trigger thresholds remain.
- Draft acceptance explicitly preserves structured requirement IDs **and text** for
  retain/refine/split. Semantics v3 and schema 4.1 record the contracts; old unannotated
  optimizer snapshots and semantics-v2 configurations are not silently migrated. Unknown
  author responses stop for resolution; they are neither retried with another draft nor no-ops.
- Directed posterior/evolution/recovery checks passed, including 120-test and 22-test sets.
  After actual failures were corrected, the final CPU-only `make check` on 22049 passed
  **3,283 tests, one CUDA opt-in skip**, full formatting/Ruff, mypy (513 source files), both
  wheels and model-wheel checks. The later concurrent stable-prompt change passed 73 affected
  tests with the original stronger submission assertions. Earlier WSL timeouts are not passes;
  no fifth full check is repeated for that isolated change or these documentation updates.
  Temporary CPU processes exited and only this run's copied environment/cache/source were removed.
- At the owner's request, the old `a8ad9ee` run and its temporary service were stopped before
  any committed step; GPU 4 returned to 4 MiB, and no old-run processes remained on recheck.
  A fresh run using the **A–D training code from `c8cdf13`** started at 01:40 UTC on
  **22048 physical GPU 4 only**. Real Qwen3.5-9B, native training graders, four planned steps,
  batch 32, seed 0, eight IID domains and production window 50; no benchmark or synthetic rollout.
  Inference/local training cohost this card, so this is not formal three-role deployment.
- The new service uses bounded decode CUDA graphs (batch up to 12), with prefill graphs off.
  At 01:47 UTC, sampling had completed 24/32 trajectories in the first batch; **0/4 steps
  were committed**. The first 24 completions took 239 seconds versus the old run's 304 seconds
  (about 27% higher early sampling throughput), excluding long-tail trajectories and backward.
  GPU 4 was about 49.6 GiB / 100%; whole-run ETA remains roughly 2–4 hours, to be revised after
  step one. The formal throughput target still needs human decision. Four-step success and
  natural phase evolution are not yet established.
- Training/service are time-bounded, and the training exit script stops only its own service.
  The former owner service remains paused; other GPU processes are untouched. No subagent,
  manual hash gate, new benchmark, or modification to `idea.tex`/`AGENTS.md` was used.
  Private paths, PIDs, logs and recovery instructions are in the updated ignored HANDOFF.

## Owner response and residual selector repair — 2026-09-08

- Ordinary peer/review message bodies now own their complete text, including native-call
  examples; those examples neither execute nor prevent delivery. Python string/f-string
  contents are not final/tool/fence boundaries. Direct and enveloped code share literal-preserving
  projection, retaining the owner's raw response and unique final without choosing a program.
  Ambiguous nested native envelopes still need owner repair; plain message bodies can carry examples.
- Native calls accept the same explicit Chinese search/click operation aliases as the plain
  transport, without inferring or translating arguments. History pagination defaults are optional
  in both schema and decoder. Call budgets, native metrics, TTB and AIME thinking rules are unchanged.
- The exported legacy reward adapter now aborts evaluator errors instead of recording zero, and
  cannot configure positive credit for invalid submissions. The two old offline WebShop catalog
  builder/refiner scripts still contained voting and conservative fallback; their executable
  implementations are removed, with rejecting compatibility entry points and regression tests.
  They were not called by the clean controller; this is not a claim that an active run used them.
- Initial interface edits entered the concurrent `a8ad9ee` integration commit. Explicit pre-fix
  source reproduced 26 failures; the repaired directed set passed 263 tests. Final source
  (`a8ad9ee` plus this batch) passed one CPU-only `make check` on 22049: **3,237 passed, one CUDA
  opt-in skip**, full formatting/Ruff, mypy over 510 source files, both wheels and model-wheel checks.
  The local WSL timeout is not counted as a pass; three separate Python 3.13 literal-boundary probes
  passed. Unchanged full checks are not repeated.
- All this batch's temporary CPU processes exited; its copied environment/source/cache were
  removed and logs retained. No GPU job, service, training or benchmark was launched, and no other
  session's GPU state was changed or claimed current. No subagent or manual hash gate was used.
  `idea.tex` and `AGENTS.md` remain unchanged. These are interface/integrity repairs, not measured
  evidence that multi-agent or skills improve real Qwen3.5-9B scores; old answers are not reused.

## Bayesian consistency hardening validated — 2026-09-08

- This completes the code/engineering-validation work described as in progress below, using
  the existing application and immutable method in `idea.tex`, not a parallel Bayesian loop.
  Full-batch flow weights retain their existing normalization; unrepresentable underflow now
  rejects the batch with its source. Phase decisions consume one frozen projection cutoff.
- Common application resume now reconciles the step journal. Resolved configuration, posterior
  evidence counts and pre-update/current identities are recorded without changing decisions.
  No-op closure is explicitly `NO_OP_CLOSED`; absent invocation coverage is explained without
  relaxing the joint trigger. Distributed workers receive the exact forward/backward/Z versions,
  including Z-reset lineage, for both sealed and streaming gradient preparation.
- Focused cross-module verification: 114 passed, one CUDA opt-in skip; mypy passed. Final
  CPU-only `make check` on 22049: 3,171 passed, one CUDA opt-in skip, with formatting, lint,
  mypy and both wheel builds passing. Public-resume tests cover 11 interrupted commit stages.
  Earlier fixture failures were fixed; local WSL test/type-check timeouts are not counted as passes.
- Two bounded tests used only physical GPU 4 on 22048. The final Qwen3.5-9B BF16 run passed
  three 32-trajectory synthetic batches, actual TTB/Adam updates, evolution, Z-only optimizer
  reset, exact checkpoint restore and next-library continuation: about 105 seconds, 0.91
  trajectories/s and 18.5 GiB peak allocated memory. The repeat followed a real checkpoint-code
  change; it was not an additional seed or benchmark. Calls, labels, author text, short windows
  and adapter publication use test fixtures; this does not validate natural on-policy evolution,
  live SGLang adapter transport or formal three-role throughput. The existing GPU runtime was
  reused read-only, not asserted to match the CPU lock environment.
- At 00:34:55 UTC the main thread personally checked all eight 22048 cards: both test runners
  were gone and GPU 4 was back to 4 MiB. Its owner-authorized former service stays paused;
  services on 5/7 were not changed. No new resident service or benchmark was started. CPU
  test processes exited; this run's temporary CPU source/environment/cache were removed and
  logs retained. No review subagent or extra hash gate was used; unchanged full checks were
  not repeated per patch. Private operational details are in the updated ignored HANDOFF.

## Bayesian consistency hardening in progress — 2026-09-08

- This supersedes the earlier GPU-ownership blocker for single-card engineering validation:
  the owner handed physical GPU 4 on 22048 to the training thread. The former GPU 4 inference
  service is stopped; inference replicas on 5/7 remain outside this handoff's control.
- Uncommitted parent-thread work adds explicit flow-weight underflow reporting, one frozen
  projection read for phase decisions, transaction reconciliation at the common resume entry,
  resolved-method telemetry, and explicit no-op-closed/no-invocation diagnostics. These edits
  are work in progress, not a completed or fully validated delivery, and are not in this commit.
- The handoff thread personally checked 22048 at 00:24:02 UTC: GPU 4 used 19,913/81,559 MiB
  at 42% utilization. The parent-owned bounded Qwen3.5-9B fixture was still running, with two
  committed 32-trajectory batches, one committed evolution, and exact post-evolution restore
  reported; the third, new-library batch was underway. No test exit status was yet recorded.
  Approximate committed-evidence throughput was 64 trajectories/99s (0.65/s), with roughly
  1–2 minutes remaining if that pace held. This is an observation, not a speedup claim.
- The fixture uses synthetic calls, labels and author output through the existing application;
  it is neither an on-policy benchmark nor formal three-role validation. Its final result must
  be checked by the parent thread. The earlier focused run reported 34 passes and seven failures;
  follow-up edits exist, but this documentation update does not establish their final test status.
- Only these handoff additions are committed. Existing source/configuration edits and the
  unrelated public-handoff hunk are preserved. The private HANDOFF remains ignored. This side
  conversation started no GPU job, service, test, review agent or extra hash check, and did not
  alter or stop the running parent task. No unchanged tests/builds were repeated for documentation.

## Bayesian live-training debug blocked on GPU ownership — 2026-09-08

- The latest owner request is to finish and debug the Bayesian training chain against the current
  code and `idea.tex`; it supersedes the earlier note that no new training task had been selected.
  This is integration/debugging, not authorization for a new benchmark evaluation round.
- At 00:08:13 UTC the main thread checked all eight cards on 22048. Physical GPUs 4/5/7 used
  67,436 / 67,436 / 67,791 MiB of 81,559 MiB each, above the permitted 25% sharing ceiling.
  Another session had started three SGLang services after the earlier availability observation.
  These were not started by this thread and were not stopped, altered or reused.
- The real three-role run is blocked pending the owning session releasing the cards or explicit
  owner authorization to take over those services. No other GPU is assumed authorized. The main
  thread requested human coordination and paused rather than adding unrelated code while waiting.
- No new GPU job, resident process, training step or benchmark was launched. No live-chain pass or
  speedup is claimed; throughput/ETA cannot be measured before a run starts. Current work only
  inspected the latest composition/scoring/projection/distributed boundaries; no new method change.
  This handoff-only update skips unchanged tests/builds and extra hash checks; no subagent was used.

## Selected GPUs resumed by owner — 2026-09-07

- Only physical GPUs 4, 5 and 7 on the approved 22048 endpoint are resumed. All other GPU use
  remains paused. This partially supersedes the earlier pause; CPU/code work may continue.
- The main thread checked all eight cards: 4/5 were idle; 7 had about 2.44% total memory use and
  zero GPU utilization, eligible for permitted sharing. Recheck before each launch; leave other
  owners' processes untouched and explicitly set CUDA_VISIBLE_DEVICES to the required subset.
- Prefer useful three-card parallelism, batching and within-step overlap. Keep full-batch joins,
  global posterior normalization, scientific settings and one seed unchanged. Report measured
  throughput/ETA rather than treating reserved memory or artificial load as speedup.
- Resource permission does not select a new training/evaluation task or checkpoint. No GPU job
  or resident service was started for this authorization update; historical services are not
  automatically restarted. Actual workload and role mapping must be explicit before launch.

## GPU use paused by owner — 2026-09-07

- GPU use is paused until the owner explicitly resumes it. Code edits and CPU-only validation
  may continue; do not launch or automatically restart GPU inference, training or benchmarks.
  This instruction supersedes historical launch/continuation plans below.
- The main thread checked both approved endpoints. The three registered project inference
  services were already stopped, their endpoints refused connections, and no process remained
  under the registered project runtime root. No termination signal or new GPU job was needed;
  unrelated or unconfirmed-owner GPU processes were left untouched.
- Operational details remain in the private handoff. This documentation-only update does not
  repeat unchanged code tests, builds or additional hash checks.

## Bayesian chain closure — 2026-09-07

- The existing application chain was completed, not replaced: native metrics, continuous TTB
  reward and binary Beta outcome remain distinct. Scoring-time edge context and the complete
  batch join bind policy, library, adapter pair and identical action-token denominators.
  Shards never perform their own posterior normalization; log-domain flow math is unchanged.
- The committed projection snapshot is the only live posterior query authority. It retains
  ordered invocation updates with before/after counts, binary labels, weights and source cursors;
  repeated batches/trajectories and stale preview commits are rejected. Restore reconstructs
  actual weighted cell arithmetic. Calibration and evolution share confidence-bound arithmetic,
  while declared query/evolution/split confidence multipliers remain explicit.
- No-op keeps the previous once-per-library-segment behavior, now with a persisted closure reason.
  Changed/new skill IDs start at the prior; inactive documents/evidence remain historical. Z reset
  clears only Z gradients/Adam state, preserving both LoRA optimizer groups. The next collected
  batch uses the newly committed library, not the triggering batch.
- Failures stop the application instance and require exact restore. The existing transaction
  reconciles model/optimizer, posterior, diagnostic/detector windows, library, sampling cursor and
  adapter publication. Orphan snapshots from the filesystem-publication/journal-write gap are
  preserved separately before retry. Synthetic tests interrupt all eleven relevant stages.
- See [Bayesian chain semantics](docs/bayesian-chain.md). Projection snapshot v7 requires complete
  evidence; old ID-only snapshots need explicit migration from their original evidence or a fresh
  run. No historical run was migrated. `idea.tex`, reward definitions and thresholds are unchanged.
- Validation: affected-file local Ruff passed. Early local focused tests had 162 passes and two
  fixture failures, corrected subsequently; slow local mypy/expanded tests timed out and are not
  counted as passes. Remote recovery/vertical tests passed (13), then boundary/process/recovery
  tests passed (45). The first full check exposed one exception-boundary incompatibility and eight
  subprocesses using stale entrypoints in the copied environment; both were repaired without
  weakening method checks. Final isolated CPU-only `make check` passed: 3,117 tests (147.30s),
  full format/lint, mypy on 505 source files, both wheels and the model-wheel boundary check.
- Full validation was repeated only after those concrete fixes, not per small edit. Main-thread
  combined inspection only; no independent review agents or additional hash/attestation gates.
  No GPU task, resident service, training run or benchmark round was launched. CPU runner cleanup
  is recorded in the private handoff. These are code-behavior results, not 9B training or quality
  validation.

## Training/no-skill interface alignment — 2026-09-07

- Removed controller instructions that made solution strategies depend on retrieved skills or
  forced concise reasoning. Skill context is advisory; empty skill sections are omitted. Both
  no-skill profiles reject active/retrieved skills at the public context-assembly entry point.
- The action decoder accepts one exterior JSON fence / Action header and the explicit WebShop
  operation aliases 点击 → click and 搜索 → search. Arguments, original action text and sampled
  token IDs are retained; it does not select an action from prose or multiple objects.
- Protocol 13 training now shares the clean owner's answer-blind final projection for QA, AIME
  and Python. Explicit final/boxed integers and indented function completions are no longer
  rejected by the older training-only parser. Conflicting answers/programs are not selected or
  retried. Public interface descriptions were aligned without changing legacy Protocol 10.
- Prompt/codec/evaluator versions were advanced for new trajectories. No TTB/calibration/evolution
  formula, reward definition, task population, budget or AIME thinking setting was changed.
  Historical outputs are not relabelled; no voting, baseline fallback or targeted rerun was added.
- Focused local validation: 158 tests passed initially; two overbroad test assertions confused the
  skillev source-message marker with skill content. After fixing those assertions, all eight tests
  in that file passed. Affected-file Ruff passed; local mypy timed out at 210 seconds without a
  diagnostic. Final isolated CPU-only make check on 22049 passed: 3,087 tests (146.55s), full
  format/lint, mypy on 502 files, both wheels and the model-wheel boundary check.
- The only concurrent committed change during validation was documentation; runtime/test source
  is the tested version. Other uncommitted work was excluded. No repeated full suite, independent
  review agent or extra hash gate was used. The CPU runner exited and its copied environment was
  removed; no GPU task, deployment, training or benchmark round was launched by this repair.
- This is interface validation, not evidence that multi-agent/skills improve quality. Trained
  checkpoint/LoRA binding into the clean evaluator and matched-budget comparisons remain pending.

## Interface and training-metric repair — 2026-09-07

- Native-function arms now rely on the model's tool definitions rather than competing
  Action/JSON message-format instructions. Plain-text compatibility and exact native arguments
  remain available; no strategy or favorable action is selected by the framework.
- ID-like text in a peer message no longer cancels delivery as an invalid archive request.
  Only existing episode-public events are attached, and the original message body is preserved.
- Fresh training's three generic seed skills are optional advice, not mandatory plans,
  repeated checks, fixed action sequences or extra submission formats. Stored historical
  libraries and the restrictive legacy Step-0 reproduction library are not relabelled.
- Training evaluators now export QA EM and F1, retain raw HealthBench scores separately from
  clipped TTB reward, and preserve separate MBPP Base/Plus verdicts. MBPP reward remains their
  conjunction; malformed grader verdicts are infrastructure failures, not reward zero.
- Historical answer-selector results remain withdrawn. No benchmark round, GPU task or training
  was started for this repair; there is no new score or claim of multi-agent improvement.
  `idea.tex`, TTB, calibration/evolution math, the IID catalog and AIME thinking settings are unchanged.
- Local focused validation: 102 tests passed and affected-file Ruff passed. Local mypy reached
  its 240-second limit without a diagnostic; it is not recorded as a pass. The isolated CPU-only
  final check on 22049 passed: 3,017 tests (151.68s), full format/lint/mypy, both wheels and
  the model-wheel boundary check. Concurrent worktree edits were excluded from that source snapshot.
- Main-thread combined inspection only; no independent review agent, manual hash check,
  retired approval/receipt gate, or repeated per-commit full suite.

## Non-negotiable boundaries

- `idea.tex` is the scientific authority and must not be modified.
- The authoritative IID catalog is exactly: HotpotQA, TriviaQA, AIME2026, HealthBench,
  WebShop, ALFWorld, MBPP+ hard, and HumanEval.
- Prompts, answer keys, options, executable hidden tests, rubrics, per-record outputs, private
  manifests, credentials, user names, key locations, internal storage paths, process IDs, ports,
  and hardware UUIDs must not enter Git.
- Evaluation answers must never be exposed to the evaluated model.
- `docs/delta_gpu_使用instruction.md` and the private `HANDOFF.md` remain untracked.

## Current GPU index authorization

All physical GPUs `0`–`7` on both owner-approved SSH endpoints are eligible. All historical
fixed-index restrictions are revoked; selecting an index does not require separate approval.
Inspect all cards before launching and prefer idle devices. Sharing with existing external
processes is authorized when GPU utilization is low and total device memory usage is strictly
below 25% (`memory.used / memory.total < 0.25`), with enough remaining memory for this workload.
This supersedes historical idle-only rules or more permissive sharing thresholds. Explicitly set
`CUDA_VISIBLE_DEVICES`; never terminate, modify, or debug another user's processes. Historical
role mappings are not current allocations.
Default exclusion lists are empty; preflight no longer bans a fixed device or fixes inference to
one index, and the physical-GPU CLI accepts the full range. Role isolation and availability checks
remain in force. This update makes no claim about live resources and launches no GPU work.

## Current training dataset and schedule

- The final Protocol 13 training artifact contains **2,000 records**: 250 questions from each of
  the eight IID domains.
- Domains with fewer than 250 eligible training IDs use deterministic repeated occurrences with
  unique episode identities. The materialized records retain a common input/output envelope.
- Every optimizer step consumes one question per domain and four trajectories per question:
  eight unique questions and 32 trajectories per step.
- The formal schedule is 250 optimizer steps and 8,000 trajectories. Checkpoint cadence is every
  10 steps, yielding 25 named cadence checkpoints.
- Model-visible inputs are separated from private evaluator payloads. Released context is included
  where required by the benchmark; answers, aliases, rubrics, routes, and hidden tests remain on
  the trusted evaluator side.

The schema, selection rules, isolation checks, and public counts are documented in
`docs/PROTOCOL13_FORMAL_TRAINING_DATASET.md`.

## Performance implementation

Implementation commit `619afc8` added the first measured execution-layer optimization without
changing the BayesianImprove objective, rewards, samples, rollout count, or optimizer semantics:

1. Workflow concurrency limits are configurable rather than fixed at four.
2. Blocking official WebShop/ALFWorld session setup, environment interaction, action-surface
   retrieval, and cleanup no longer block the asyncio event loop.
3. Session setup and cleanup have independent resource lanes and telemetry.
4. The distributed TTB coordinator may compute a disjoint local gradient shard while the worker
   rank computes the other shard. Both retain exact global batch scaling before collective sum.
5. Each committed step can emit answer-free rollout, gradient, commit, queue, service, token, and
   straggler timing in `performance.jsonl`.
6. The progress monitor uses the formal throughput thresholds rather than the obsolete one-step-
   per-hour debug threshold.

Legacy callers retain worker-only gradient behavior unless coordinator participation is explicitly
enabled. See `docs/PROTOCOL13_FORMAL_TRAINING_PERFORMANCE.md` for the contract and measurements.

## Measured run state

The superseded baseline produced six contiguous committed steps at approximately 1.742
steps/hour overall, or 1.784 steps/hour after excluding its first two steps. It was stopped and
must not be resumed or combined with another run.

A fresh optimized attempt completed two contiguous warmup steps before the owner instructed the
run to stop:

| Step | Rollout + evaluation | Exact TTB gradient | Residual commit | Total |
|---:|---:|---:|---:|---:|
| 1 | 570.7 s | 557.0 s | 7.9 s | 1,135.6 s |
| 2 | 654.3 s | 485.1 s | 0.7 s | 1,140.0 s |

This is about a **1.82x** end-to-end improvement over the baseline. It is not a formal speed-gate
result because the gate requires two warmup steps followed by at least three measured steps. The
72-hour hard floor remains 3.4722 steps/hour and the operating target is now 4.2 steps/hour.

The stopped attempt has only two valid optimizer commits. A partially collected third batch is
invalid and must not be reused. No formal 250-step run is active.

## Known continuation issue

The next serving profile was intended to allow base-model judge requests and forward-LoRA rollout
requests in one running batch. Its first warmup exposed a missing local JIT build tool (`ninja`).
The owner stopped the task before that dependency was added, the serving profile was validated, or
a replacement training run began.

If work resumes:

1. Recheck the approved compute resources; never rely on this handoff as a current allocation.
2. Provide the official JIT build dependency in the server environment and validate base plus one
   LoRA identity under concurrent load.
3. Start a completely fresh run from optimizer step zero using the frozen 2,000-record artifact.
4. Verify exact 8-domain × 4-trajectory balance, finite/nonzero forward, backward, and log-Z
   gradients, one update per sealed batch, and checkpoint/publication behavior.
5. Use two warmup steps plus at least three subsequent steps for the throughput decision. Below
   3.4722 steps/hour remains `needs-human-decision`.
6. Only if that low-risk serving repair still fails should length-bucketed teacher forcing or
   within-step rollout/gradient streaming be considered.

Do not resume or concatenate any incomplete debug attempt.

## Validation and stopped state

- Targeted remote tests for the concurrency and exact distributed-gradient changes: **40 passed**.
- Final CPU-only repository gate on the same tracked source state: Ruff format/check passed, mypy
  passed for 449 source files, pytest reported **2,413 passed / 80 warnings**, both wheels built,
  and the model-wheel check passed.
- No independent review agent or hash check was used.
- At the last direct check on 2026-09-05, all project training, evaluation, and model-serving
  processes from this task were stopped. Existing unrelated workloads were not touched.

## Synchronization rule

For every future handoff update:

1. Write complete operational detail only to ignored `HANDOFF.md`.
2. Mirror all non-sensitive scientific, implementation, validation, and status changes here.
3. Run a focused secret/path check on this file before staging it.
4. Commit and push this public file; never force-add the private handoff.

## Acceleration implementation — 2026-09-04

See `docs/protocol13-training-acceleration-20260904.md` for the execution changes and measured
canary findings. Frozen-feature caching, same-step gradient streaming, mixed serving controls,
request-level judge scheduling, durable timings and deadline checks are implemented.
The candidate disables radix caching and enables deterministic gradient algorithms to address
observed failures; numeric thresholds were not relaxed. The deterministic fixed-32 reference is complete; new GPU equivalence and fresh eight-step
measurement are blocked by insufficient free GPUs. All GPU tasks from this work have exited. No formal 250-step run has been started.

## Explicit trained-forward evaluation — 2026-09-07

The clean runner now binds a caller-selected checkpoint to its observed, preloaded forward
LoRA route, carries the trained inference state into the isolated actor, and records the
actual policy/adapter for owner and peer outputs. Concurrent base/trained arms use separate
route pools; failed routes never fall back to base answers. Matched comparisons change only
one of topology, skill access or forward weights, with the same native scoring and budgets.
See [configuration and scope](docs/clean-trained-policy-evaluation.md).

Affected-file Ruff passed. The initial directed set had 90 passes and 4 failures; the
recovery-order regression and old fixture identity assumptions were fixed, then the affected
follow-up passed 49 tests. The initial isolated 22049 CPU `CUDA_VISIBLE_DEVICES="" make check` passed:
**3,125 tests, 84 warnings, 148.76 pytest seconds**, format (962 files), Ruff, mypy (503 sources),
both wheels and the model-wheel boundary check. CPU runner PID 103052 exited; its temporary
source/venv/cache were removed and logs retained. No GPU/benchmark jobs, review subagents,
new/manual hash checks or repeated unchanged full suites were used. Parallel uncommitted
training/hardware edits were not included or altered; `idea.tex` and `AGENTS.md` are unchanged.

Concurrent training-core commit `032c947` then changed the source. The combined source was
checked once more before push: **3,155 tests, 96 warnings, 160.23 pytest seconds**, full
format/lint, mypy, both wheels and the model-wheel boundary check passed. This was a changed
source integration check, not a repeat on unchanged code. CPU runner PID 157128 also exited
and its temporary files were removed; logs are retained. Uncommitted hardware edits remain separate.

GPU permission was subsequently restored only for **22048 physical GPUs 4, 5 and 7**,
as recorded above. This code batch started no GPU job and did not terminate other sessions'
services; its CPU cleanup is not a GPU-service shutdown claim.

This is forward-weight evaluation, **not** restoration of the evolved skill library or full
training application state. Real model comparison still needs an explicitly chosen checkpoint
and evaluation round under the current GPU authorization; no architecture/skill performance
improvement is claimed by these tests.

Delivery closeout (2026-09-08): this batch was pushed as `a289f1c` and `413bc59`.
The 3,155-test result does not cover later sessions' commits or uncommitted changes.
This documentation-only closeout starts no GPU job or persistent process and repeats no tests.
For the subsequent GPU4 handover to training and evaluation's remaining GPU5/7 allocation,
see the [owning session's dated record](docs/STEP0_NATIVE_INTERACTION_REPAIR_2026-09-07.md);
it is not a fresh live-resource check by this closeout. The pending checkpoint/round choice
above applies to forward-weight evaluation, not the separately requested training debug task.

## 2026-09-08 — Bayesian evidence / authoring repair (partial)

The current repair request is **not complete**. See
[the implemented/pending matrix](docs/BAYESIAN_EVOLUTION_REPAIR_STATUS.md).
The immutable method, seven IID and single-task-agent constraint remain in force.
This batch added declaration-to-execution links for posterior events, public
execution snippets for authors, bounded author material, transport-response
persistence before validation, static COMPLETE Generate candidates, related-skill
summaries, and additional dependence-aware calibration reporting. Method semantics
is version 5 and evolution configuration version 7; do not silently resume an old
condition as this one. Existing training and optimizer/posterior boundaries remain.

First directed Linux CPU verification: 40 passed. The first full check passed
format/lint but found an unrelated concurrent execution-stage Protocol omission;
the shared worktree already supplied that fix. The refreshed full snapshot passed
format/lint and mypy (555 source files); full pytest/build was still running when
this note was written. Final verification outcome is appended below when known.
No GPU task, service, review agent or manual hash check was started by this repair
thread. Other sessions' live GPU state has not been rechecked by this thread.

Do not report formal-entry readiness, arbitrary-size mandatory Generate prompt
capacity, same-family context Split, full mutation H0 executability, real-Qwen
parallel equivalence through Phi, actual author/mutation/restart/new-evidence
closure, natural W50 or calibration/effect acceptance as completed. The historical
four-step source `ed04e85` had zero phase/mutation and does not validate these edits.

Repair-batch validation closeout: final private Linux CPU snapshot passed
`CUDA_VISIBLE_DEVICES="" make check`: **3,633 passed, 1 real-CUDA opt-in skip**,
142 warnings, pytest 280.66 seconds; full Ruff/format, mypy (555 source files),
both wheels and model-wheel boundary check passed. Before that, the full test run
had 12 failures from old COMPLETE eligibility/no-op assumptions and 3,621 passes;
updated assertions and a below-floor synthetic no-op witness passed 26 affected
tests, then the complete suite. No production phase threshold changed. The
synthetic witness is not natural W50 evidence or a revived launch-admission gate.
All this thread's CPU command sessions finished; no GPU job or service was started.
The complete repair goal remains open with the pending work listed above.
Current GPU permissions are those in AGENTS.md (both endpoints, physical 0–7,
sharing only under the current utilization/memory rule), not older numbered-card
restrictions retained in historical sections. This is a permission statement,
not a new observation of GPU availability.

## 2026-09-09 — infrastructure batch in progress; training safely paused
The current B28 / 8-turn (ALFWorld 20) / thinking-on HealthBench+AIME condition
completed two optimizer/posterior transactions, then honored the saved pause:
loss 2.205083 → 1.086538; mean reward 0.337106 → 0.431319; success 7/28 → 11/28.
Step walls were 715.68 and 728.74 seconds (~4.98 steps/hour across two warmup
observations, not steady-state acceptance). The step-2 full checkpoint and
committed transaction were personally read; training exited zero. Inference
and two requested standby services remain on endpoint 22048, physical 0/6/7.

Implementation now includes explicit role placement, all-replica publication
and sticky version-bound sessions, private durable actor/author/judge responses,
unknown-dispatch refusal, per-endpoint request/token admission and role counters.
Actor token reservations use admitted input plus output budget; trusted chat
requests without exact input IDs conservatively reserve the full configured
token allowance. This is not a measured physical-KV predictor. No new grammar
mask, fused scoring kernel or GPU topology has been enabled in training.
Targeted Linux CPU checks: 41 tests passed; affected-source mypy passed.
A final integrated check, real multi-replica numerical qualification, active
environment crash recovery and fused-backend qualification remain outstanding.
Do not call the six-PR proposal complete or restart with unqualified candidates.

Infrastructure milestone check completed on the private 22049 Linux CPU snapshot:
`CUDA_VISIBLE_DEVICES="" make check` passed: **3,726 tests passed, 2 explicit GPU
qualification skips**, 154 warnings, pytest 282.76 seconds; Ruff/format, mypy,
both wheels and model-wheel boundary passed. The initial full run exposed two
legacy AST rules conflating resource capacity return/rethrowing cleanup with
budget compensation; updated rules still prohibit budget release and swallowed
unknown outcomes. The affected 29 tests passed before the final full run.
SQLite/WAL privacy was separately covered by 9 journal tests. Unknown Judge
outcomes now block all subsequent repair prompts for that case, including a new
broker process/object; known malformed-response repair remains unchanged.
No unrelated GPU kernel comparison was repeated. Real actor-pool qualification
is the next step, not covered by these CPU results. The full six-package task
remains open; see `docs/infra_execution_status_2026-09-09.md`.

### 2026-09-09: serving registration and real replica follow-up

Actual model/tokenizer/context/cache/backend settings are now registered privately
and checked again at actor publication/restore. Deployed flattened server-info
responses are supported; missing values are not invented, and logging toggles
are not numerical identity. Startup failure closes the gradient coordinator
before any rollout/model update. Full 22049 CPU `make check` passed again for
this integrated runtime change: **3,736 passed, 2 explicit CUDA skips**, pytest
288.30s; Ruff/format, mypy, both wheels and model-wheel check passed. Affected
formal-entry tests passed after updating their runtime fixtures.

Real existing services on 22048 physical0/6 passed eight paired short requests
across two adapters, then forty paired reasoning/action prefixes from a saved
20-turn interaction. Exact IDs/stops and original-response restoration passed;
all temporary adapters were removed. The longer replay reached19,556 input
tokens but capped diagnostic output at32; it is not full-budget training or
full-B28 numerical qualification. Two harness startup failures (input wrapper
and omitted existing dependency path) were corrected, not converted to rewards.

Training remains fully saved atstep2. At08:24UTC a non-project compute process
was using the intended gradient card7; other processes are untouched. The
remaining safe gradient memory must be rechecked before resume. No new formal
training was started by these probes; original inference/standbys are retained.
The six-package implementation remains incomplete as documented above.

At08:35UTC formal training successfully resumed from the saved new-condition
step2 using the tested published source, with1 inference GPU and2 participating
gradient GPUs on22048 physical0/6/7. The first capacity handover had safely
refused a transient100% post-shutdown utilization reading before torchrun; a
bounded stability wait fixed the launcher without weakening memory or utilization
limits. Standby was restored after that refusal. No other project process was
stopped. At08:39UTC the new B28 step was collecting and computing gradients;
14 completed artifacts were saved, with no new complete optimizer commit yet.
First two pre-pause steps averaged4.98steps/hour (both warmup); remaining248
steps would take roughly50hours only if that average persisted, not a steady-state
claim or guaranteed ETA. Every committed step retains loss/reward in the event log.
No full check was repeated for these operational notes/private-launcher changes.

09:38UTC monitoring: formal step7 committed; step8 collecting on22048 physical0/6/7.
First three post-resume warmup-excluded steps5–7 average4.756steps/hour, existing
three-sample gate passes. Step7 alone took962.47s (3.74steps/hour), flagged for
human performance review, not silently excluded. It had230 executed edges and
186457 completion tokens versus174/131246 atstep6; rollout946.81s dominates,
gradienttail13.09s. No thermal slowdown, FLA fallback or offload groups observed
for this step. Continued original B28/8–20/thinking2 condition, no restart or
newGPUjob. Conditional remaining~51.1h using measuredsteps; persisted-origin
forecast~53.8htotal, not guaranteed. Full metrics and limitations appended to
infra_execution_status_2026-09-09.md. No source changes or redundant tests/hash checks.

10:22UTC: formal step10 committed, step11 collecting; retained cadence-step10
checkpoint COMPLETE with optimizer/policy/run/projection state personally checked.
Posterior32cells/229invocationupdates across10 B28 batches; not independent
sample counts or natural-evolution proof. Six postwarmupstepsaverage4.438steps/h,
latestthree~4.16 belowtarget; human performance risk retained. Conditional
remaining54.1h/total57.5h includingpastdowntime. No training restart/newGPUtask.
Private monitor batch-transition display repaired after observed stale-progress /
new-artifact mismatch; per-source batch IDs and file ages now explicit. Public
source unchanged, no redundant fullcheck. Detail appended to infra status note.

11:16UTC: step13 committed and step14 collecting, same run and processes.
Personally verified checkpoint/optimizer/projection/detector13, taskcursor364,
and actual served adapter13. Aggregate curves saved through13. Nine postwarmup
steps average3.981steps/h; actual gate now needs-human-decision, latestthree3.30.
Conditional remaining59.5h, persisted-origin total63.9h, not guaranteed. Step12
last artifact AIME, step13 ALFWorld; low loss does not prove improved reward.
No restart, newGPUjob, source change or redundant fullcheck. See infra status note.

12:56UTC: cooperatively paused after complete step17 due sustained action-format
and reward degradation; step18 notstarted. Checkpoint/optimizer/projection/
detector17,taskcursor476 verified; gradient ranks drained,launcherexit0.
Fullbatch validaction fraction fell to5.5% atstep16; lowTTBloss notqualityproof.
Last postwarmup3.297steps/h; conservativeprepause forecast77.2hremaining/83.2htotal,
notliveETAwhilepaused. ExistingGPU0server now runs boundedfixedrequest diagnostics
(noenvironment/reward/trainingevidence); ownerstartedGPU6/7SGLangstandbyservices.
Results pending; noautomaticresume ormethodchange.250goal remains incomplete.

13:39UTC: diagnostic pause remains at complete step17. A fresh Qwen backbone
reloaded actual step16 weights AND named Adam state, then replayed all28 saved
trajectories/268edges in canonical sealed order, without new sampling, evaluation,
posterior updates or run checkpoint writes. F/B edge means and residuals match
original streaming results exactly; the resulting F/B/Z parameters and both Adam
moments match committed17 exactly. Gradient norms also match; gradient vectors
inferred from saved first moments differ by at most7.81e-7 relativeL2 (inversion
rounding, not a stored original-gradient comparison). Existing1e-3 bounds unchanged.
Independent CPU comparison of all268 actual generation inputs/action-token prefixes
against reconstructed scoring inputs/recorded actions passed. These results do not
prove learning quality: the observed action-format/reward decline remains real,
and no actionable cache/transport/gradient-update discrepancy was found in scope.
Earlier seven-domain fixed action requests reproduced recorded trained outputs and
cold/warm tokens exactly; comparing initial adapters on those trained-policy
conditioned prefixes was not a Step0 benchmark or a population quality estimate.
Finite GPU6 replay took1084.1seconds of gradient work, exit0; it is not on-policy
throughput. GPU4 admission was rejected when external load changed before launch;
no external processes touched. OwnGPU6 standby was drained and restored; inference0
and standby7 remained. Personal health readback confirms allthree services200;
replay and its finite owner exited. Formaltraining remains paused, awaiting owner
decision whether to continue unchanged from17; no implicit grammar/LR/reward change.
Historical postwarmup3.297steps/h is below4.2, requiring human performance decision;
recent-rate remaining233steps roughly77.2h, excluding current pause/future overhead.
Only private diagnostic scripts and this aggregate note changed; no repeated full
suite/build/lint, no independent review agent or added hash checks. Full250 training
and natural-evolution validation remain incomplete.
