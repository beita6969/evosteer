# ALFWorld / WebShop native interaction repair

Focus: repair explicit model intent and public feedback, then evaluate each frozen128-task
development cohort. AIME's separate thinking-on result is not rerun or combined with this round.

## Evidence and changes

- Completed `5a1517a` traces show WebShop owners copying a menu entry `click[target]`
  into the `click` tool's target parameter. The transport wrapped it again and rejected it.
  The adapter now accepts one complete same-operation call in its corresponding argument,
  preserving the exact target/query. It does not infer a target, choose among commands,
  accept an unavailable target or bypass the current-state revision. Public parameter
  descriptions match this interface.
- An environment response with no separate `action_valid` field was described as execution
  `unknown`, despite its acknowledged observation. Public feedback now distinguishes
  **acknowledged** from confirmed-valid, rejected and unknown. It does not invent success,
  expose rewards or alter the observation. The underlying durable acknowledgement is unchanged.
- The same completed cohort had58/128 WebShop episodes stopped at the10-action cap.
  ALFWorld had55 unsuccessful terminal episodes at its20-action cap; another1 succeeded there.
  An exhausted action limit is not proof of a model/transport bug or proof that more steps solve it.
  These data motivate an explicit new development budget, not a same-budget improvement claim.

## Frozen new condition

`A2-owner-native-interaction@2` uses no skills, demonstrations, handcrafted decisions,
catalog automation, pruning, training, votes or fallback. Both interactive benchmarks retain
thinking-off and the existing native reward/SR scorers. Task identities, order and seed remain fixed.

The separate `step0_native_interaction_horizons.yaml` declares **WebShop50 / ALFWorld100**
actions for every case. Existing160-call,163,840-output-token,32,768-history-input-token and
98,304-context allowances are not reduced. These horizons are engineering choices, not claimed
official benchmark defaults. The effective ALFWorld limit reaches both the actor and simulator;
changing only the outer loop would leave the shorter native cap in effect. Original manifests
remain unchanged, while the actual per-task budgets are frozen in each new run's controls.

## Batches and validation

1. Literal argument and acknowledgement semantics: medium risk; transport and real actor/broker
   tests, including immutable owner output and feedback/peer delivery.
2. Explicit horizon plumbing: medium risk; budget preservation and actor/native-cap tests.
3. Integration: one full22049 CPU check before push, followed by a separate native service canary
   and A2-only128+128 evaluation. No coding or review subagents; main-thread combined inspection.

Affected-file Ruff passed. The10-file directed set passed **129 tests in9.24 seconds** on22049 CPU.
The complete check and real interactive rounds are pending at this note's creation. No new
ALFWorld/WebShop score is claimed yet. No hashes, repeated unchanged suites or retired gates.

Before launching the native round, direct inspection of the deployed official
`WebAgentTextEnv.step`, `get_available_actions` and `browser.search` exposed two additional
API mismatches. The engine accepts search from the current page regardless of HTML search-bar
presence, and lowercases click arguments before key lookup. The projection now always exposes
the real search API; click transport accepts only that native lowercase equivalence while
preserving the owner's literal wire text. It does not fuzzy-match, infer or select a control.
The public tool reference is corrected accordingly. The intermediate `@1` condition was
not evaluated; `@2` identifies this additional interface change before any native run starts.

The already-started intermediate `027bbce` full CPU check finished successfully:
**3,044 tests,85 warnings,142.67 pytest seconds**, full Ruff/mypy (502 source files), both
wheels and the model-wheel boundary check. Affected API tests and the final updated-source
check follow the newly discovered native-API contract change; this is not a repeat on unchanged code.

The native-API/public-surface/broker/factory follow-up passed **93 tests in10.85 seconds**
on22049 CPU. Affected-file Ruff also passed.

The final `42a33b4` archive passed `CUDA_VISIBLE_DEVICES="" make check` on22049 CPU:
**3,056 tests,85 warnings,142.92 pytest seconds**, format check (959 files), full Ruff,
mypy (502 source files), both wheels and the model-wheel boundary check. Its CPU-only
coordinator PID548336 has exited successfully. This verifies the frozen source, not unrelated
concurrent working-tree edits. No unchanged full suite is repeated for result documentation.

The native6+6 canary started at22:57 UTC on22048 (coordinator PID12402), with the same
three resident inference services on physical4/5/7. Main-thread postlaunch checks confirmed
all three healthy and serving work. There is no additional GPU allocation, inference daemon,
training job or AIME regeneration in this round. The canary is integration evidence only,
not a substitute for the complete fixed128+128 development evaluation.

## Corrected WebShop catalog deployment

The canary exposed a concrete data-wiring fault, not just a low model score: the
1,181,430-product store was paired with a 99,995-document search index. All six canary
target products existed in the store, but none existed in that configured search index.
Those target checks stayed in the trusted, private diagnostic process; no target IDs,
answers or retrieved candidates were supplied to the evaluated actor.

The existing complete-corpus index contains 1,181,370 documents and covers all 10,014
unique target products across the 12,087 official goal records. No index rebuild or
goal-conditioned index was needed. New private manifest copies change only the index
path; one structural comparison confirmed that task content, IDs, ordering and every
other deployment field stayed unchanged. Original manifests and run records remain intact.

The wrong-index canary was stopped, not published as a completed benchmark result;
unfinished records are not converted to zeros. Its coordinator and process group are gone,
while the three inference services remain healthy. A new independent fixed WebShop128 +
ALFWorld128 run started at23:14 UTC (PID45275), using the already-checked `42a33b4` source,
the same `A2-owner-native-interaction@2` policy and budgets, and corrected catalog controls.
The dataset correction is distinct from an architecture improvement. No prior answer is
reused, no AIME case is regenerated, and no new GPU or inference service is allocated.

The catalog repair changes private data configuration, not executable source. The successful
3,056-test full source check is reused rather than rerun for a path correction and documentation.
The complete128+128 result remains pending. Historical WebShop runs using the reduced index
must retain that limitation and cannot stand in for the corrected full-catalog condition.

## Owner-requested pause and scoring of the completed subset

At23:54 UTC the owner paused GPU use. The coordinator and all three registered inference
services were stopped; 237 committed trajectories and their original model outputs survived.
The remaining19 episodes were not silently regenerated or converted to zero scores.

The owner then requested scoring of those237 records. The frozen `42a33b4` native scorers
read the retained environment outcomes on CPU, with no new model call or environment action.
All237 scored successfully in2.99 seconds; scorer PID15073 exited. The original256-task plan,
candidate texts, model-output records and acknowledged executions are unchanged.

| Benchmark | Scored / planned | Native result | Native success |
|---|---:|---:|---:|
| WebShop | 123 / 128 | average reward37.52 / 100 | 21/123 =17.07% |
| ALFWorld | 114 / 128 | success92/114 =80.70% | 92/114 =80.70% |

ALFWorld's observed split results are seen68/85 =80.00% and unseen24/29 =82.76%.
These are **completion-conditioned development subsets**, not complete128-case results.
In particular ALFWorld's subset mean exceeding80% does not establish the full-panel goal.
WebShop has5 unfinished cases and ALFWorld14; none enters the above denominators as zero.

Both subsets retain `communication-not-complete`: WebShop has216 unresolved action decisions;
ALFWorld has99 unresolved action decisions and8 unresolved control attempts. The latter is not
evidence that8 delivered peer requests were lost: the inspected failures include native `look`
and `inventory` calls misaddressed as function names rather than arguments to `act`.
Acknowledgement, feedback and action-surface mismatch counts are zero for the scored subsets.
Actual peer calls are1 for WebShop and0 for ALFWorld; declared multi-agent capability is not a
claim of actual peer participation in every episode.

Aggregate-only data: `docs/machine-results/step0_a2_native_interaction_partial237_2026-09-07.json`.
Per-item trajectories, targets and scores remain private. No full suite is repeated to score
existing records with the already-validated frozen source.

The owner subsequently reauthorized only22048 physical4/5/7. Their three SGLang services
were restored with the same serving parameters. A missing virtualenv executable directory in
the startup PATH caused an initial `ninja` lookup failure; correcting PATH fixed startup without
installing dependencies or changing the model. The successful parent/scheduler pairs are
23123/29185,23125/30049 and23126/28954 respectively; main-thread health checks passed.
The original evaluation remains paused rather than implicitly replaying its19 interrupted episodes.

The owner then reassigned GPU4 to a separate training session. Main's handoff check found its
registered evaluation service already exited and GPU4 at4MiB, so no signal was necessary.
This evaluation session now owns only physical5/7; GPU4 is excluded from its service registry
and upcoming inference mapping. Other sessions' processes are not touched.

## Next independent native condition

The bounded interface repair exposes `look()`, `inventory()` and `help()` as literal zero-argument
ALFWorld calls, equivalent to the same command passed to `act`. The current native action menu
and state revision still govern execution; no object, location, action sequence or success is inferred.
This addresses actually observed explicit calls, not a generated expert policy. No additional
shopping instructions are added: the existing public reference already explains Buy Now semantics.

`step0_native_interaction_repairs.yaml` declares a separate `A2-owner-native-interaction@3`
development condition, using presence penalty1.5 for both native benchmarks and64 coordinator
workers across the two remaining replicas (32 per replica). The penalty follows the backbone's
[general non-thinking recommendation](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices).
It is a hypothesis for reducing repeated output, not a proven accuracy or speed improvement.
All call/token/context/action budgets, fixed task identities and the single seed stay unchanged.
Old answers are not imported into this condition, and AIME is not regenerated.

This batch has medium interface/configuration risk: targeted transport, capability and broker
tests plus one final CPU `make check` before push. Main performs the combined inspection;
there is no coding/review subagent, hash check or per-fix full-suite gate.

Affected-file Ruff passed locally. A local40-case directed invocation reached100% but then
timed out with exit124 during completion/teardown at180 seconds; it is **not** recorded as a
successful test command and was not retried on WSL. The final frozen `2d9b2cb` source instead
passed one `CUDA_VISIBLE_DEVICES="" make check` on22049 CPU: **3,164 tests,96 warnings,
153.78 pytest seconds**, format check968 files, full Ruff, mypy506 source files, both wheels
and the model-wheel boundary check. CPU coordinator PID273703 exited0. This includes the new
literal-call, real isolated actor/broker and decoding-budget preservation tests. No separate
unchanged remote directed suite or second full check was added.

The independent `@3` fixed WebShop128+ALFWorld128 run started at00:20 UTC onSeptember8
(coordinator PID3824). It uses only physical5/7,64 episode workers and8 CPU scorers, with the
already-validated frozen source. This is a new development condition, not resumption/splicing
of the237-record subset and not a new AIME run. Results and measured throughput are pending.
