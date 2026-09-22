# Execution infrastructure: implementation and qualification status

The scientific algorithm is still `idea.tex`: complete B28 on-policy TTB,
canonical gradient addition, shared committed posterior, phase/evolution and
atomic outer-step publication. No grammar mask, PPO/GRPO objective, sample
filtering or cross-step rollout is introduced by this batch.

| Work package | Current delivery | Remaining qualification/work |
| --- | --- | --- |
| Role topology | Explicit service catalog, shared-role references, distinct gradient devices; legacy three-card binding retained | Actual additional replica deployment and readiness qualification |
| Actor pool | All-member publication barrier, policy-bound sticky sessions, workload-estimated placement; no generation retry/base fallback | Full-output-budget / full-B28 numerical qualification beyond the controlled replica tests below |
| Durable requests | Private SQLite actor, author and trusted Judge responses; original route restoration; ambiguous dispatch blocks resampling | Active environment snapshot/recovery; server-side resolution of genuinely unknown outcomes |
| Admission | Per-physical-endpoint count/token capacity; shared actor/Judge capacity only when actually colocated; cancellation drain and role counters | Measured KV/Mamba resource model and per-role physical token telemetry |
| Fused scoring | Existing reference remains enabled; previous unqualified candidates remain disabled | Independent Liger/CCE integration and full B28 F/B/Z, residual and Adam qualification |
| Structured actions | Existing raw-softmax actor unchanged | Separate optional scientific condition with common schema and matched F/B mask normalization; not part of this rollout restart |

Token admission reserves full admitted input plus maximum output, without
subtracting assumed cache hits. A trusted chat request without exact token IDs
reserves the entire configured token allowance. This conservative bound is not
an estimate of actual prefill work or permanent cache occupancy. Large aged
requests wait for outstanding work to drain rather than being starved by an
endless stream of small requests. Per-service counters retain actual aggregate
concurrent high-water marks, not a sum of unrelated peaks.

The durable response store does not itself restore an ALFWorld environment,
settle an optimizer transaction or authorize another sample. A completed
response restores the original logical budget usage; it is not reported as new
physical generation. Unknown dispatches remain infrastructure failures. Native
Judge parse-repair prompts remain the pre-existing declared scorer behavior;
transport retries are not added.

The new-condition training was drained and fully saved at step 2, then resumed
from that checkpoint on the qualified reference path. Its first two warmup steps took 715.68 / 728.74 seconds. No steady-state throughput,
full natural evolution or six-package completion is claimed. Resume must use
this condition's checkpoint, not the earlier static2/ALF50 run. Current condition:
250 steps, static8/ALF20, thinking-on HealthBench+AIME2026 only, seven domains,
four independent rollouts each, seed0, checkpoint every10 and on saved pause.

## Follow-up: actual serving registration and controlled replica tests

The controller now reads the deployed `/get_server_info` representation (both
flat and nested variants), matches model/tokenizer locations and context budget,
and registers numerical execution settings privately. Actor members must agree;
publication and restore re-read the registered settings. A same-run server
restart cannot silently change the prefill/cache/backend profile. This is not
a content attestation, and unreported project hooks are not inferred from null
fields. Logging switches do not become numerical identity.

Real services on endpoint 22048, physical GPUs 0 and 6, passed eight paired
controlled requests across step0/step2 adapters, thinking off/on, with identical
output IDs and stopping reasons. Completed responses restored without a new
generation. A second controlled replay used 40 reasoning/action prefixes from
a saved 20-turn interaction: paired IDs/stops also matched, up to 19,556 input
tokens. Each diagnostic request allowed only 32 output tokens; this is **not**
full-budget on-policy training or a full-B28 scoring/Adam comparison. Temporary
probe adapters were unloaded, leaving the saved training adapter untouched.

The first short harness failed on a Transformers BatchEncoding/list mismatch;
the first longer harness omitted an existing private CPU dependency path. Both
failed before generation, were repaired and retained as failed attempts. Neither
was turned into a negative reward or a replacement training sample.

At 08:35 UTC the formal run resumed from optimizer step2 on physical0 serving
and physical6/7 participating in gradients. No fresh Step-0 run was substituted.
The initial handover safely refused a transient post-shutdown utilization sample
before torchrun; the private launcher now waits at most90 seconds for three
consecutive safe capacity samples, with the same memory/utilization limits.
Standby was retained/restored until the successful handover. Other projects'
processes were not stopped. The source remains the tested published snapshot;
no candidate scoring implementation or grammar mask is enabled.

## Monitoring update, 09:38 UTC

The formal run has committed step7 and is collecting step8; it was not stopped
at four steps. Checkpoint completeness, the committed transaction record,
optimizer/projection/detector cursors and the served forward-adapter version were
checked at the completed boundaries. Loss/reward exports preserve all points.

| Step | TTB loss | Mean reward | Complete step seconds |
| --- | ---: | ---: | ---: |
| 5 | 0.401329 | 0.392857 | 585.23 |
| 6 | 0.568484 | 0.428571 | 723.25 |
| 7 | 0.065852 | 0.345347 | 962.47 |

These are the first three post-warmup steps in the resumed process. Their
aggregate rate is 4.756 steps/hour, passing the existing three-observation
throughput gate. This is a short observation, not proof of sustained 250-step
performance or natural W50 evolution. Step7 alone is only 3.74 steps/hour and
is explicitly flagged for human performance review; the original condition
continues without dropping trajectories or changing budgets.

Step7 spent 946.81 seconds in rollout and 13.09 seconds in its gradient tail.
Compared with step6, it contained 230 rather than 174 executed edges and
186,457 rather than 131,246 reported completion tokens. All four ALFWorld
trajectories reached the declared 20-turn limit in each of steps5–7. The longer
workload is consistent with the observed slowdown; it is not a controlled causal
speed comparison. Neither rank reported FLA fallback or activation-offload
groups for step7; coordinator buffer backpressure was zero. Live GPU sampling
showed no software/hardware thermal slowdown. Invalid model actions remain
unfiltered training evidence, not infrastructure retries.

Using all three measured steps, the runtime forecast is approximately 51.1
additional hours, or 53.8 total hours including the persisted wall-clock origin
and past downtime. This is conditional on subsequent workload and recovery
costs, not a completion guarantee. No training/service restart, additional GPU
job, code change or repeated test/build was performed during this monitoring
update. Physical0 serving and physical6/7 gradients remain active on22048.

## First scheduled checkpoint, 10:22 UTC

Step10 committed and step11 is collecting under the same condition and live
processes. The separate ten-step cadence checkpoint is complete: optimizer
state, policy directory, run step10 and projection revision10 are present.
Rolling transaction checkpoints are distinct from this retained cadence copy.
No process restart or candidate scoring/grammar change occurred.

| Step | TTB loss | Mean reward | Complete step seconds |
| --- | ---: | ---: | ---: |
| 9 | 0.239802 | 0.379252 | 793.26 |
| 10 | 0.133744 | 0.497330 | 901.14 |

All ten committed batches contain28 trajectories. The saved posterior has32
cells and229 invocation-update events with finite positive Beta parameters;
the accumulated evidence mass is229 under the existing normalization. This is
not229 independent problem samples and does not establish calibration quality.
The detector remains untriggered at cursor10, before the production W50 window
can qualify for natural evolution.

The six post-warmup steps average4.438steps/hour; the latest three average about
4.16. The recent slowdown remains a human performance-decision risk even though
the existing cumulative gate passes. Conditional remaining time is54.1hours
using all six measured steps, approximately57.5total hours including the
persisted run origin and prior downtime. No guarantee is implied.

A monitoring-only issue was also corrected: a batch transition can expose an
older progress snapshot beside a newer persisted-artifact directory. The private
monitor now displays each source's actual batch identity, file age and whether
its counters refer to the same batch. It does not infer lost trajectories from
those independently refreshed snapshots. Syntax and live-output checks passed;
training state and public source code were untouched, so full checks were not
repeated. No new GPU job, resident process, review agent or hash check was added.

## Sustained monitoring and throughput warning, 11:16 UTC

Step13 is committed and step14 is collecting. The step13 transaction journal,
checkpoint completion marker, optimizer/projection/detector cursor13,
task cursor364 and the actual served forward adapter13 were personally checked.
All committed batches remain B28; aggregate loss/reward exports include all13
steps. No training condition or process was changed.

| Step | TTB loss | Mean reward | Complete step seconds |
| --- | ---: | ---: | ---: |
| 11 | 0.058544 | 0.302296 | 832.65 |
| 12 | 0.072825 | 0.232143 | 1285.70 |
| 13 | 0.076383 | 0.243584 | 1154.17 |

The nine post-warmup steps now average3.981steps/hour, below4.2; the actual
throughput gate reports **needs-human-decision**. The latest three average3.30.
Conditional remaining time is59.5hours, or63.9total hours from the persisted
origin including past downtime; neither estimate guarantees completion in72h.
The authorized original run continues while the performance warning is visible.
Low TTB loss has not established improved reward or task quality.

Step12's last artifact was AIME; step13's was ALFWorld. Rollout spans were
1236.85s and1139.04s, with gradient tails45.42s and12.30s respectively. A live
step13 phase snapshot showed about87% token-weighted input cache reuse. Its
summed server prefill spans were about50s versus7029s after prefill; these are
concurrent request aggregates, not batch wall time or pure CUDA decode time.
Frequent non-JSON actions remain an output-quality risk, not evidence justifying
silent retries, grammar masking, action repair or removal of failed samples.

Physical0 inference and6/7 gradients remain active on22048. The latest hardware
check in this monitoring interval found no SW/HW thermal slowdown; bounded recent
training-log inspection found no traceback, CUDA OOM or terminal worker failure.
No new GPU task, daemon, source change, review agent or hash check was introduced.
Already successful checks were not repeated for this monitoring/documentation
update. The250-step objective and unqualified infrastructure candidates remain
incomplete; this is not natural-evolution or end-to-end acceptance.

## Saved diagnostic pause after step17, 12:56 UTC

The run was cooperatively paused after sustained output-quality degradation,
under the standing instruction to diagnose problems before blindly continuing.
No in-flight batch was truncated: step17 completed all28 trajectories, committed
optimizer/posterior state, published its adapter, and wrote a complete paused
checkpoint. Optimizer/projection/detector cursors17 and task cursor476 were
personally checked. Both gradient ranks drained and the launcher exited0;
step18 was not started. The250-step objective is **not complete**.

| Step | TTB loss | Mean reward | Complete step seconds |
| --- | ---: | ---: | ---: |
| 14 | 0.069698 | 0.214286 | 1199.03 |
| 15 | 0.048381 | 0.131148 | 1673.22 |
| 16 | 0.042358 | 0.021429 | 1611.43 |
| 17 | 0.036129 | 0.071429 | 1572.32 |

Frozen-parser checks on complete batches12–16 found structurally valid action
fractions27.7%,20.5%,17.4%,12.3%,5.5%. Structural validity is not environment
success. This deterioration is not a healthy training result merely because
TTB loss is small. Most invalid actions stopped normally rather than at the
output limit. No grammar mask, answer repair, sample filtering or reward change
was introduced.

The last measured post-warmup rate was3.297steps/hour. Before the diagnostic
pause, the existing conservative forecast indicated about77.2remaining hours,
83.2total hours including past downtime, and capacity-shortfall against72h.
These are conditional workload estimates, not a running ETA while paused.
The forecast uses the slower of all measured steps and the latest ten.

Read-only checks found finite saved F/B/Z tensors, distinct F/B adapters, and
an exported forward tensor mapping identical to its checkpoint counterpart.
All28 first-action requests from step16 retained their action interface,
non-thinking suffix, raw sampling parameters, root-boundary request, exact
sampled token prefix and decoded action text. These checks rule out some
specific faults; they do not establish the cause of quality degradation.

A bounded fixed-request diagnostic is now running on the existing physical0
inference service: cold/warm cache and original/trained adapter comparisons.
It executes no environment or scorer and creates no training evidence. Results
are pending. Following the saved pause, the existing owner started its declared
SGLang standby services on physical6/7; those cards are no longer gradient ranks.
No other project's process was changed. No public production-source change,
new benchmark evaluation, repeated fullcheck, review agent or hash check occurred.
