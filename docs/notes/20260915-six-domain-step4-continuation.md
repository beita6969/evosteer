# Step4 complete checkpoint → six-domain B24 continuation

Owner request: save Step4, remove HumanEval from the current IID/training
condition, then continue the same complete checkpoint. No fresh initialization.

## Saved original condition

Step4 committed at **2026-09-15 06:28:07 UTC**; the owner exited successfully
at **06:28:22 UTC**, after the complete paused-step-4 checkpoint was written.
The snapshot contains policy, AdamW and complete method/runtime state.

| Step4 metric (original seven domains, B28) | Value |
| --- | ---: |
| TTB loss | 0.7895757648 |
| Mean native reward | 0.6758241758 |
| Binary success | 18/28 |
| Skill reads / posterior events | 2 / 2 |
| Structural validity | 121/131 |
| Step wall / rollout / gradient tail | 1381.76 / 913.98 / 462.11 s |

These original records stay unchanged, including the four HumanEval trajectories.
Reading a skill and subsequent success are not proof of a causal skill benefit.
The run still has no accepted A0 or qualified cold-start claim.

## Declared continuation

`configs/training/bayesianimprove_autonomous_ttb_six_domain.yaml` selects:
HotpotQA, TriviaQA, AIME2026, HealthBench, ALFWorld, MBPP+.
Only AIME and HealthBench use native thinking; all retained budgets, horizons,
scorers, optimizer/TTB settings and the proactive catalog protocol stay unchanged.
Historical AIME training sources and closed-book Trivia training inputs remain
explicitly distinct from their IID evaluation populations/inputs.

- Steps1–4: 28 trajectories each; Step5 onward: 24 each.
- Preserve the consumed 112-task prefix and retained future source identities.
  Seed remains zero; future sampling is bound to the declared new ordered schedule,
  not claimed identical to the old seven-domain draws. Use the actual saved task
  cursor (112), not 4 times the new batch size (96), for the next sampling position.
- Keep F/B/Z, AdamW, library, posterior provenance, diagnostics and run cursor.
- Continue to global Step250; checkpoint cadence remains 10 plus rolling complete
  transactions and cooperative save/pause.
- Mixed-condition total: **1,504 question occurrences / 6,016 trajectories**.
  A fresh all-six-domain 250-step run would instead contain 6,000 trajectories.
- IID freezing and aggregation use the declared/frozen panel, not an assumed
  seven-domain denominator. Old IID panels/results are not rewritten or relabelled.
- W&B retains the same run, with B28 and B24 segments explicitly distinguished.
  Aggregate reward/success across the condition boundary are not directly comparable.

## Validation and deployment

The real Step4 checkpoint passed CPU-only restoration-identity and ordered-source
checks: saved step4, cursor112, unchanged consumed prefix, next batch six × four.
The focused recovery/config/evidence suite passed 63 tests, including complete
optimizer and posterior/library restoration followed by another complete update.
Expanded regression suite: **76 passed**, including the true saved-cursor sampling
position test and current/historical IID panel aggregation. Final remote Linux
`CUDA_VISIBLE_DEVICES="" make check`: **5,089 passed, 16 skipped**; formatting,
Ruff, mypy (699 source files), both wheels and model-wheel separation passed.
GPU/tokenizer-dependent skips are explicit, not claimed GPU validation.
Earlier test-runner temporary-directory/child-pytest issues were corrected;
obsolete entrypoint mocks now exercise the domain-aware scheduler. No production
scoring or failure handling was relaxed to make those tests pass.
W&B was read back directly: all four original committed scalar rows are present.

No subagents, hash audits, model quantization, reward repairs, new skill quotas,
or alternative learning objectives were used. Unrelated concurrent worktree edits
were excluded from this deployment. GPU role mapping remains inference4,
coordinator/gradient6 and gradient0 on the approved endpoint22049; only this run's
standby SGLang processes are handed back to the real training ranks.

## Live continuation observation

At **2026-09-15 07:22:21 UTC** the frozen implementation started from optimizer
step4 with B24. The published domain boundary is effective from Step5. Direct
comparison of the source and boundary checkpoints found their complete
`execution_state` identical; both have COMPLETE markers. Original AdamW and
policy were loaded by the normal full-restore path.

At **07:26:27 UTC**, the Step5 sidecar contained exactly 24 trajectories, four
per active domain and no HumanEval. Twenty artifacts and 18 gradient contributions
had completed; canonical merge was four, waiting for the earlier interactive
trajectory, not silently reordering additions. Step5 was not yet committed.

Process ownership: controller601250, torchrun601307, gradient ranks601396/601397,
remote CPU metric exporter601785, local W&B uploader1322196. Inference remains
SGLang parent196926 / scheduler197475. Only owned standby parents312988/312989
were stopped after the actual training ranks acquired their CUDA contexts.
No foreign process was stopped. These are private deployment PIDs, not stable IDs.

W&B continuation: [same training run](https://wandb.ai/lanlangcll-university-of-illinois-urbana-champaign/skillev123/runs/bayesian250-autonomous-ttb-20260915-v1).
The source exporter retains all four original records and explicitly labels the
B28 steps1–4 and B24 steps5–250 segments. The uploader remains running.

At that observation, GPUs0/4/6 were busy at 87C with software thermal slowdown
active (450/1320/945 MHz respectively). **Administrator cooling action is needed**;
no protection, power, clocks or other users' workloads were changed. The original
four B28 steps average about2.41 steps/hour; at that old rate, the remaining246
steps would take about102 hours. This is a conditional comparison, not a measured
B24 forecast or a 72-hour guarantee. The owner-authorized run continues.

## First complete B24 update

By **07:30 UTC**, Step5 had committed and its full checkpoint had a COMPLETE
marker. At **07:31:22 UTC**, Step6 was collecting (7 artifacts and 6 gradient
contributions ready, no abort). The first Step5 artifact's actual sampling
position was112, confirming use of the restored cursor rather than96.

| Step5, six domains / B24 | Value |
| --- | ---: |
| TTB loss | 1.0746329424 |
| Mean native reward | 0.7023809524 |
| Success | 16/24 |
| Unique source questions | 6 |
| Skill reads / posterior updates | 2 / 2 |
| Structural validity | 63/71 (8 parse errors, 0 schema-invalid) |
| Wall / rollout / gradient tail | 407.97 / 355.62 / 47.84 s |

W&B API readback confirmed the actual Step5 row (trajectory_count24), current
batch24, running status and paused=false. A stale display-only paused flag was
corrected by restarting only the local CPU uploader; its replacement parent PID
is1324361. The training and gradient workers were not interrupted.

This single B24 step corresponds to8.82 steps/hour; repeating that speed for the
remaining245 steps would take about28 hours. It is a one-step conditional estimate,
not steady-state qualification or evidence that domain removal alone caused the
speed change. The task mix changed, and the observed thermal issue still requires
administrator attention. No new IID benchmark run was launched for this request.

## Steps6–7 monitoring continuation

Direct observations through **2026-09-15 08:05 UTC** confirmed full COMPLETE
checkpoints for Steps6 and7; Step8 started collecting without a restart or a
configuration change. The same controller, actor, two gradient workers and CPU
metric processes remain in use. No new GPU task or persistent process was launched.

| B24 step | TTB loss | Reward | Success | Skill reads / posterior events | Wall / rollout / gradient tail (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.9815107323 | 0.5833333333 | 14/24 | 3 / 3 | 1196.25 / 978.48 / 213.60 |
| 7 | 0.4769934603 | 0.7019622834 | 15/24 | 5 / 5 | 885.53 / 831.91 / 48.42 |

Step7's five reads occurred in five trajectories, all at their first action;
all five returned skill bodies, with no read failure. Two of those trajectories
eventually succeeded. This is observed use, not demonstrated causal application
benefit or a cold-start qualification. No phase trigger/library mutation occurred.
Step6 structural validity was95/115 (19 parse errors and one schema error);
Step7 was94/107 (13 parse errors). Existing raw records retain those failures.
Step6's long tail advanced through AIME reasoning turns before full gradient
drain; idle gradient workers were waiting for the remaining trajectory, not
evidence of a deadlock. No infrastructure abort was observed.

W&B API readback independently confirmed the Step7 loss, reward, success,
trajectory_count24, skill_count5, wall time, running state and paused=false.
The three completed B24 steps average **4.34 steps/hour**; at that rate the
remaining243 steps would take about56 hours. This remains a small-sample,
conditional estimate, excludes future interruptions and is not steady-state
qualification. The last individual step was about4.07 steps/hour, below4.2.

All eight GPUs were inspected; the selected roles remain22049 GPU4 inference,
GPU6 coordinator/gradient and GPU0 gradient. One inference sample reached93C;
an immediate follow-up read was86C with software thermal slowdown active.
Cooling remains **administrator/human-action needed**; the owner has authorized
continued training. No hardware protection or foreign process was changed.
This update is a monitoring note only: no source edits, repeated test suite,
subagents or hash audits.

## Local checkpoint backup requirement (owner update)

Every ten-step complete checkpoint must also be copied to the owner's local
machine, not left solely on the training server. Include complete pause/final
and named condition/phase snapshots. Keep local backups; never delete an older
copy just to make space. The next deployment must bind this same backup policy
to its actual checkpoint directory before training starts.

The owner-selected local destination is **`G:\checkpoints-bf\<run-id>`**
(`/mnt/g/checkpoints-bf/<run-id>` in WSL).
The local backup worker was switched to that mounted drive on September15;
write, fsync, atomic rename and file locking were checked there. The previous
destination was preserved and held no checkpoint directories at the switch.

On September15 a CPU-only local backup worker was installed independently of the
training and W&B processes. It polls every five minutes, discovers complete
cadence snapshots, copies the entire checkpoint directory (both adapters, Z,
optimizer and runtime state), checks file presence/length and run/step metadata,
fsyncs the local copy, and atomically publishes it. Interrupted transfers remain
private staging directories and are retried; no successful-backup claim is made
until local publication. This is transfer completion, not a model restore test.

Seven targeted CPU tests passed: cadence/pause selection, complete copy and
idempotence, disconnect/retry, truncated copy rejection, wrong-run rejection,
status recovery after publication, and unreachable-source status preservation.
No training code or methods changed; no full GPU/evaluation suite was rerun.
The private operational worker/configuration are outside Git; credentials and
checkpoint contents must never be committed.

At installation the old server was unreachable. W&B last reported Step32, but
its metrics do not establish that Step30/32 checkpoint files remain accessible.
No Step30 local full backup was found in the inspected current-run/transfer
directories. The worker therefore starts with backups pending, not successful;
on connectivity recovery it attempts all still-available cadence checkpoints,
including10/20/30. New server access alone cannot recover inaccessible old disks.

## September16 recovery: local backups complete, retry needs owner decision

Direct checks at02:10–02:12UTC confirmed22049 is reachable after a host reboot.
Complete Step30 and Step32 checkpoints are now locally published under the
owner's G-drive destination, including both adapters, Z, optimizer and runtime
state. Going forward the backup worker selects Step30 and later only, retaining
already copied older snapshots without deletion. Explicit Step32 backup is also
complete. Eight targeted backup tests passed; no hash checks were used.

The original frozen code and GPU dependency versions were restored to volatile
runtime storage. The full Step32 restore entered training at02:06:22UTC, but
exited before committing Step33. Its persistent request journal contains6,396
complete responses and one unresolved ALFWorld reasoning dispatch from the
interrupted Step33. Its response is absent, and no retry authorization exists.
The23 completed Step33 artifacts remain preserved. We did not clear the journal,
resample silently, or report a new optimizer update. An explicit decision to
regenerate the missing response is needed; this is not exact response recovery.

Current services: GPU2 actor(parent81216), GPU4 standby(parent98812), GPU6
standby(parent98813). The attempted training owner96762 and ranks96955/96956
have exited; no training is currently advancing. Local checkpoint backup
worker1506197 remains active. The W&B uploader recorded the failed attempt and
exited rather than continuing a false running heartbeat. No foreign process was
changed. All eight GPUs were inspected directly.

The latest committed Step32 retains loss0.164304, mean reward0.744898,
16/24 successes and4 skill reads. Historical six-domain steps5–32 averaged
2.83 steps/hour; the last five averaged2.29. If those rates recur,218 remaining
steps would require roughly77–95 active hours, excluding this interruption.
Both rates miss4.2; the blocked restart has no current completion ETA and needs
an owner decision. This is still unqualified diagnostic TTB, not passed A0.
No training-source edit, full-suite rerun, benchmark or subagent was needed for
this operational recovery; the only tests were the changed private backup logic.

### Owner-authorized single-request recovery

The owner explicitly authorized regeneration of only the missing ALFWorld
reasoning response, preserving all other records and resuming full Step32.
A private SQLite backup was taken first. The existing single-use retry API
registered exactly one permission, retaining6,396 completed responses and the
original unresolved dispatch. No response was fabricated and no batch was reset.
At02:21UTC the permission was consumed and its response durably COMPLETED.

Three focused existing retry tests passed on the frozen remote CPU runtime:
successful retry, failed retry cannot consume a second permit, and rejection of
completed/missing requests. The initial test setup hit an old shared temporary
directory ownership mismatch; a fresh private tmp directory fixed the test
infrastructure without changing permissions on existing directories.

Training actually started at02:19:53UTC from full Step32. New owner135733,
torchrun135759, ranks135772/135773 and CPU metrics135971 are active; GPU2 remains
the actor and GPU4/6 compute gradients. Both ranks established CUDA ownership
before the previous owned standbys were drained. Local W&B uploader1509432
resumes the same run; checkpoint backup1506197 retains the Step30+ policy.
The23 previously completed Step33 artifacts are retained. Step33 is not yet
claimed committed; its lost ephemeral gradients must be recomputed.

Direct02:21UTC GPU observations show GPU4/6 software thermal slowdown active,
86C and480/390MHz, versus the actor at32C and1980MHz. Cooling requires
administrator attention; continued execution is owner-authorized. No fan,
power, thermal protection or foreign process was modified. Previous77–95-hour
active-time estimates remain historical, not a prediction at these throttled
clocks. No source-method edits, new evaluation, full-suite rerun, subagents or
hash audit were performed; only private launch/telemetry wrappers changed.

### Step33 committed after recovery; Step34 collecting

Direct controller/checkpoint inspection confirmed full Step33 completion and
Step34 collection at02:45UTC. The33rd transaction reports TTB loss0.996643,
mean reward0.672276,14/24 successes,4 skill reads and4 posterior events; mutation
counts remain0. Its1465.40s active attempt includes174.02s rollout span and
1286.07s post-rollout gradient tail. Because23 artifacts and earlier request
responses were reused, this recovery attempt is not a fresh-batch throughput
benchmark and does not erase the earlier failed-attempt/downtime cost.
Root directly confirmed W&B state running and committed Step33 aggregates after
the exporter caught up; the existing uploader was reused, not duplicated.

GPU6 acquired a foreign26GiB allocation while our worker remained live, leaving
roughly5GiB free at the sample. GPU4 remains thermally throttled. These resource
risks require administrator coordination; foreign processes were not inspected
beyond GPU/PID occupancy or modified. The22048 connection probe timed out, not a
claim that the endpoint is permanently down. Current training was not restarted
because of that unrelated observation failure.

The owner requested a GPT-5.6-Luna/xhigh publishing-only agent; it is retained
solely for Git/W&B operations and may not edit files or operate training/GPU.
The main agent keeps responsibility for monitoring, repairs and local cadence
backups. No new tests or training-code changes accompany this result note.

## Step35 full save and thermal/resource migration

Step34 completed with loss1.163131, reward0.474576,11/24 successes,3 reads and
3 posterior events; wall1669.48s, rollout1084.65s, gradient tail579.65s.
Step35 completed with loss0.610998, reward0.627778,14/24 successes,4 reads and
4 posterior events; wall1106.34s, rollout841.95s, gradient tail258.64s.
The last five complete steps average2.438 steps/hour;215 remaining steps would
need about88 active hours at that historical rate. This misses4.2 and excludes
additional interruptions; it is not a prediction for the newly selected cards.

GPU4/6 repeatedly reached86–90C and345–750MHz. A foreign allocation on GPU6
pushed total usage close to80GiB. When GPU5/7 became available, the main agent
requested cooperative stop after the current complete batch, not cancellation.
Step35 was committed, full method state and evidence saved, and both old ranks
exited0. The local G-drive backup worker published the complete paused Step35
snapshot(214,588,382bytes). No prior artifact, response, optimizer update or
posterior/library state was discarded. Step30+ cadence backups remain active;
the next ordinary cadence copy is Step40.

A fresh all-eight-device check confirmed the new mapping before launch:
GPU2 actor, GPU5 coordinator/gradient, GPU7 second gradient worker. GPU5 had a
small existing allocation below the authorized25% sharing threshold; remaining
memory was sufficient at launch. Other processes were untouched. Both real new
ranks established ownership before the old project standby on GPU4 was drained.
The actor was retained. At03:34UTC, both selected gradient cards were1980MHz,
29–35C, without software thermal slowdown. This validates the immediate hardware
condition, not a measured steady-state speedup.

New full-Step35 continuation actually started at03:33:01UTC with Step36/B24.
Owner534171, torchrun534183, ranks534286/534287, CPU metrics534484 are the new
processes. The previous W&B uploader ended cleanly after Step35; a single new
uploader1520699 resumes the same run, preserving all historical metric points.
Local checkpoint backup1506197 remains active. Only private operational wrappers
and physical device bindings changed; frozen training code/method/configuration
were retained. Syntax and real clean-checkpoint resume were checked; no repeated
full test suite, evaluation, hash audit or source-method modification was needed.
The publishing-only Luna/xhigh agent remains responsible for Git/W&B duties.

## Step37 saved; Step38 external Judge network failure

Root directly observed Step36 and37 complete. Step36: TTB loss0.817702,
reward0.593137,13/24 successes,3 reads/posterior events, wall1272.32s,
rollout1185.00s, tail81.11s. Step37: loss0.977059, reward0.640447,
10/24 successes,2 reads/posterior events, wall791.98s, rollout753.00s,
tail34.12s. Together these two steps average3.488 steps/hour:213 remaining
steps would require about61 active hours, excluding failures/downtime. This is
below4.2 and is not a steady-state or total-time guarantee.

Step38 retained23 complete artifacts. Position9 HealthBench failed because both
of its two rubric requests returned APIConnectionError, with no settled grades.
Other started trajectories drained; the original owner exited1 without a partial
optimizer/posterior commit. Root confirmed its own SGLang standbys111306/111307
on GPU5/7 and the retained actor onGPU2. No foreign process was changed.

The owner authorized local-network forwarding to the same OpenAI API. A private,
authenticated, loopback-only CONNECT proxy permits only api.openai.com:443;
TLS remains end-to-end and verified. Four targeted CPU tests passed (destination
restriction, authentication, plain-HTTP rejection, exact tunnel bytes). A real
remote HEAD request through local egress returned401 with TLS verification0;
this establishes transport connectivity, not a completed Judge result. Original
candidate/rubrics/model settings are unchanged. One scoped recovery operation
was dispatched, but its SSH acknowledgment timed out; its outcome must be read
before any further attempt. No successful repair or Step38 commit is claimed.

At04:31UTC fresh root SSH probes instead returned connection reset/refused.
Transport reconnection only is supervised; this does not retry Judge generation.
New local CPU services: CONNECT proxy1534969 and SSH transport supervisor1535390.
The backup worker was replaced with1535208 to reuse the observation SSH master
and include an extra Step37 copy. Root verified the complete local Step37 copy
(215,820,768bytes), including optimizer,F/B adapters,Z and runtime state. Step30+
every10-step backups remain configured, next cadence40. No restore execution or
hash check is claimed. W&B uploader1520699 exited after preserving Step37; it
correctly reports failure, not a running/committed Step38. The publishing-only
Luna agent remains reserved for Git/W&B work. No method code, dataset, task
policy, full test suite or evaluation changed in this operational repair.

### Local-egress repair completed; Step38 resumed

At04:45UTC root again connected and checked all eight devices. The missing
HealthBench grade is now complete: exactly the original two rubric criteria were
judged via the authorized local transport, in7.62s; the individual native
result is retained only in the private grading ledger. The original failed requests remain preserved. An earlier CPU-only
repair attempt failed importing blobfile before any API dispatch (zero requests);
the successful attempt reused the existing dependency-complete training Python
with CUDA visibility empty. No candidate, rubric, model or grading rule changed.

Root started owner187728, torchrun187802, gradient ranks187844/187845 and CPU
metrics187967 at04:45UTC. Mapping remains actorGPU2, coordinator/gradientGPU5,
gradientGPU7. Both real ranks acquired CUDA ownership before only our standbys
111306/111307 were stopped. Root observed at04:48UTC: all24 Step38 artifacts
ready,4 gradient contributions complete, remaining gradient work active on both
ranks, no failure. The23 earlier artifacts and all settled responses were reused;
this recovery is not a fresh-rollout throughput benchmark.

One new local uploader1543999 resumes the same W&B run; Luna remains solely on
publishing duties. The old uploader had already exited, so no duplicate writers
were introduced. Local backup1535208, proxy1534969 and SSH-only reconnection
supervisor1535390 remain live. Step37 is fully backed up locally; next scheduled
cadence is40. Training is resumed, not declared complete or newly A0-qualified.

### Step38 committed; Step39 exposes intermittent tunnel failure

Root's05:01UTC checkpoint/stream check confirmed full Step38 and active Step39.
The existing uploader subsequently published Step38 loss1.175920,
reward0.708333 and4 skill reads. These metrics include recovered original
trajectories, not a fresh sampling speed benchmark.

At05:07UTC root found one Step39 HealthBench submission with10 rubric requests:
8 preserved responses and2 APIConnectionError outcomes (criteria2/3). One failed
after79s, so server-side completion is unknown. The other3 HealthBench submissions
completed. Original responses/candidates and remaining active work are retained;
no partial update or fabricated zero label is allowed. Separate permission was
requested to retry only the2 failed criteria and to replace the fragile opaque
SSH tunnel with a durable local request/response relay. Neither is assumed
approved. No new Judge requests were issued for this failure.

SSH itself is intermittently reachable: root connected at05:07, but the05:12
probe timed out during banner exchange. A successful interactive login does not
establish continuous transport health. The extra Step38 local backup is queued,
not yet verified; Step37 remains the latest verified full local copy. Backup
worker1551135 retains the10-step cadence and retries transport only. Local
proxy/tunnel/uploader processes remain the previously recorded ones. Training
was still draining at the last successful observation; no later exit is claimed.
This network reliability issue requires an explicit recovery decision, not more
unreviewed resampling or repeated whole-batch restarts.

### Durable local Judge delivery and Step39 recovery

After the owner's authorization to repair the grade and reconnect automatically,
root completed only the two missing Step39 rubric requests locally (3.21s and
3.09s). The original eight responses were retained. The existing native parser
and aggregation retained the original native result in the private ledger;
aggregation made zero API calls. The result is not repeatedly judged for success.
The complete Step38 checkpoint is now also backed up locally (216,507,939bytes).

The opt-in Judge spool persists a single operation and its local raw response
before SSH delivery. Connection recovery transfers the saved result, not a new
sample. Unknown local API outcomes fail visibly; credentials stay local. The
previous CONNECT proxy/tunnel were retired after draining. Local CPU worker
1565452 handles this transport, backup1551135 retains the every10-step cadence,
and uploader1574281 resumes the same W&B run. No raw tasks or Judge bodies are
published.

One restart attempt stopped before model release because process handoff waiting
raised OSError. Explicit owned-process liveness polling replaced that wait; no
training update occurred in the failed launch. At06:32UTC root observed the new
owner51550, torchrun51628, ranks52035/52036 and metrics53003 restoring full Step38.
Mapping is actorGPU2, coordinator/gradientGPU5, gradientGPU7. Only our standby
services were retired after the CUDA ranks became ready. All24 Step39 artifacts
then became ready (23 reused plus the repaired original HealthBench trajectory),
and both ranks recomputed gradients. No new actor generation was needed for this
recovery; its wall time must not be presented as fresh rollout throughput.

Targeted spool tests:6 passed; private local-delivery lifecycle tests:4 passed.
Targeted ruff/mypy passed. CPU-only remote full check passed formatting, lint and
mypy, but exposed a test-temp ownership issue and a missing math-verify dependency
in the restored CPU environment. After setting an owned temporary directory,
5100 tests passed,16 skipped,3 native-math tests still failed for that missing
package. After installing the declared math-verify0.9.0 dependency, all three
affected tests passed (2.41s); both wheel builds and model-wheel check passed.
The final aggregate is5103 passed,16 skipped. Already-passing tests were not
repeated after this environment-only dependency repair. No method, training data, reward, model or decoding changes;
no independent review agent or hash checks. The publishing-only Luna agent is
retained. Training remains an owner-authorized unqualified diagnostic, not a
new A0-qualified experiment.

### Step39/40 committed; delivery queue deadline correction

Root observed Step39 commit: loss0.759150, reward0.449405, success9/24,
2 skill reads/posterior events. Its510.74s is recovery/recomputation time, not
fresh sampling throughput. Step40 subsequently completed normally: loss0.131324,
reward0.913510, success20/24, no skill reads/posterior events;690.85s wall,
634.53s rollout and48.51s gradient tail. The zero skill usage is genuine and is
not repaired with fake credit. All52 Step40 rubric responses were durably saved
locally, with no failed API outcomes. W&B received both commits.

A remaining queue issue was identified: a disconnected durable response could
occupy the four-request semaphore long enough for later requests to exhaust the
old queue deadline. The opt-in spool route now waits for infrastructure capacity
separately; the actual API execution retains its120s timeout and concurrency4.
Direct API behavior is unchanged. A600s simulated queue wait verifies no API-time
budget is consumed;25 affected tests passed. The current batch was not hot-patched:
Step40 drained and checkpointed cleanly at06:52UTC before the execution-only
continuation. Every10-step local backup remains automatic; the Step40 copy is
still pending verification at this note's observation time.

At06:56UTC root verified continuation owner178183, torchrun178277,
ranks178321/178322 and metrics178600. Only our Step40 standby parents158112 and
158117 were handed over; actor81216 remains onGPU2. New uploader1582894 continues
the same W&B run after the previous uploader closed the paused segment. Local
Judge1565452 and backup1551135 are unchanged. No extra GPU role was introduced.

At06:59UTC root verified Step41 collecting on both training ranks and the local
Step40 cadence checkpoint (COMPLETE, optimizer_step40,217,520,221bytes). The
backup worker is copying the additional paused/full representations; the cadence
backup is already complete. The final CPU-only remote `make check` now exited0:
formatting, lint, mypy,5104 tests passed/16 skipped, both wheels and wheel-content
validation passed. An intermediate rerun stopped at formatting because generated
test fixtures were inside its source directory; the temporary root was moved
outside the checkout, without editing product code or skipping those tests.
The slow WSL typecheck was stopped and covered by the final Linux typecheck.

### Post-repair continuation through Step42

Root checked the live run at07:34UTC: Step42 fully committed and Step43 started;
no training/monitor failure, and no thermal slowdown on the three selected GPUs.
Step41: loss0.744967, reward0.566667, success13/24,6 reads/6 posterior events,
1108.02s wall/1033.03s rollout/68.14s gradient tail. Step42: loss1.583943,
reward0.503268, success11/24,1 read/1 posterior event,1042.97s wall/944.18s
rollout/91.74s tail. These real variations are not relabeled or smoothed away.
Steps40–42 average about3.80steps/hour (not a steady-state qualification);
208 remaining steps imply roughly55 active hours at that rate, excluding future
outages/recovery. This remains below4.2 and needs an owner performance decision;
existing authorization is to continue, not to stop solely for missing the target.
The local complete Step40 cadence, paused and full copies are now all present.
Next10-step backup is50. No new GPU/CPU services or method changes since the
previous entry; transport reconnection, Judge result durability and W&B continue.

### Step44 committed and transient transport recovery

At08:12UTC root observed Step44 committed and Step45 collecting under the same
live owner/rank PIDs. Step43: loss0.430681,reward0.680556,success13/24,3 skill
reads/posterior events,1555.24s wall/1443.93s rollout/105.02s tail. Step44:
loss0.247986,reward0.654215,success13/24,6 reads/events,680.88s wall/651.41s
rollout/23.99s tail. The five fresh steps40–44 average3.54steps/hour;206 remaining
imply roughly58 active hours, not a completion guarantee. The below-target
performance remains flagged for owner decision while authorized training proceeds.

One SSH observation/delivery timeout recurred near08:11UTC. The existing workers
retried transport and root reconnected at08:12; training was never declared dead
or restarted for this observation timeout. The same durable Judge worker resumed
delivery, with new operations active, not a replacement trajectory or score.
SelectedGPU2/5/7 showed no thermal slowdown. GPU7 reached79,229MiB usage/reservation,
close to capacity, but both completed steps had no CUDA OOM. No new service or
method/code change was introduced. Step40 remains fully backed up locally, next50.

Individual recovered Judge verdicts/scores have been removed from this public
note; those remain only in private ledgers. Batch-level aggregate training
metrics remain public. This correction does not alter training labels or erase
old Git history; earlier note revisions still contain those isolated numbers.

### Step50 milestone and local cadence backup

At10:22UTC root observed Step50 committed and Step51 collecting under the same
owner, two training ranks and actor process. Steps45--50 respectively reported
losses0.594049,0.470223,0.252809,0.296070,0.327369 and0.164784; rewards0.682350,
0.802083,0.617903,0.555952,0.631548 and0.848148; successes14,17,12,12,13 and18
out of24. Skill reads/posterior events were7,1,7,8,3 and8. These observations
are preserved as-is and are not interpreted as a monotone learning curve.

The six step wall times were766.28s,454.96s,1040.37s,2844.19s,1459.55s and
1096.45s. Step48 was a real long-tail batch:2510.05s rollout and327.86s
post-rollout gradient tail. The latest six-step rate is about2.82steps/hour;
200 remaining steps imply about71 active hours at that short-window rate,
excluding outages, future workload variation and recovery. This remains below
the4.2 target and requires an owner performance decision; the standing owner
instruction is to continue rather than stop for throughput alone.

Root verified both complete remote Step50 checkpoint representations. The local
cadence worker then copied `cadence-step-00000050` to the approved local backup
root: `COMPLETE` is present, `runtime_state.json` records optimizer_step50, and
the copy contains224,086,766bytes. W&B also received Step50 and remains running.
GPU2/5/7 continued as actor, coordinator/gradient and gradient respectively;
no thermal slowdown, CUDA OOM, training exit or monitor failure was observed at
the milestone. No method, task, reward or decoding condition changed.

### Step60 checkpoint milestone

At13:30UTC root observed Step60 committed and Step61 collecting under the same
three-GPU mapping and process set. Step60 reported loss0.155277,
reward0.912281, success20/24, and5 skill reads/posterior events;653.63s wall,
625.38s rollout and17.87s gradient tail. The latest six completed steps55--60
ran at about3.77steps/hour;190 remaining steps imply roughly50 active hours at
that short-window rate. This is not a steady-state guarantee and remains below
the4.2 target, so the performance flag stays open while the explicit continue
instruction remains in force.

Root verified both remote Step60 checkpoint representations as complete with
optimizer_step60. The local cadence worker copied
`cadence-step-00000060` to the approved backup root: `COMPLETE` is present,
`runtime_state.json` records optimizer_step60, and the copy contains
229,918,334bytes. W&B received Step60 and remained running. No thermal
slowdown, CUDA OOM, training exit, monitor failure, resampling or method change
was observed at this milestone.
