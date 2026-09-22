# ScienceWorld persistent-focus condition: development regression

This continues the [OOD follow-up](ood-step0-followup-20260916.md), not an unseen
final evaluation. The other five accepted OOD64 results are not rerun. This session
does not modify Judge code/configuration or send Judge requests.

## Frozen condition

Code `5aacc30` adds only public focus persistence/replacement semantics through
`public-task-semantics@13` / `scienceworld-commands-persistent-focus@5`. It does not
choose objects, repair actions, expose private scores, or change the native scorer.
The same previously frozen panel10 IDs are used, with one adapter-free
Qwen3.5-9B owner, skills off, thinking off, 8,000 total output tokens, 200 owner
actions, the independent native 200-move horizon, batch32 and presence penalty0.
Seed0 is requested; existing inference remains nondeterministic, without a restart.
Training updates, posterior updates, skill evolution and training-evidence writes
are all zero. There are no consulting agents or training/W&B runs.

## 32-sample result and complete trajectory review

`science32-persistent-focus-panel10-v42` scores **65.09375/100 signed native mean**,
strictly above the **50.553** gate. All32 native results are present:23 full
successes, four negative terminals and five partial outcomes. The auxiliary
zero-clipped mean77.59375 is not the primary score. Relative to @41, nine cases
improve, five regress and18 are unchanged; this is not a deterministic causal
estimate of the API-help effect.

All970 owner replies,967 actions and their public-state changes were read. Literal
commands, execution acknowledgements and delivery of required public observations
show no mismatches. Three final length-stopped replies exhaust the declared total
budget without a complete action; they remain recorded output failures, not
network failures, and receive no extra generation. Native command rejections134
and all four negative terminal scores are retained.

Remaining behavior includes exploratory task selection, container/content
confusion, unavailable-object references, cumulative timing errors, unvalidated
circuits, and guesses without completing experiments. Some native100 outcomes
also contain such reasoning mistakes. The score increase therefore does not
establish that scientific reasoning improved by the same amount.

Costs are13,213,824 input and98,341 output tokens, with zero peer calls. Generation
completed at522 episodes/hour; ETA0, coordinator exit0. The32-result panel and all
raw records remain private and unchanged. After completing this review, the
pre-frozen64 request was launched under the identical condition, not by selecting
successful32 cases or reusing their generated answers.

## 64-sample execution and full manual review completed; gate failed

`science64-persistent-focus-panel10-v42` scores **22.50/100 signed native mean**:
32 full successes,22 negative terminals and10 partial outcomes. All64 native
scores are present, with no scorer infrastructure failure. It does **not** pass
50.553; the higher32 result is not substituted. All64 trajectories have now been
read before any subsequent generation.

The repeated32 IDs score22.3125 in this fresh run; the additional32 score22.6875.
Among the repeated IDs,12 regress, two improve and18 are unchanged. For31/32
initial requests, the actual messages and input-token sequences are identical
between runs, but all32 first replies differ. The remaining reset differs only
in public paint-cup listing order. This demonstrates response nondeterminism and
an upstream object-order difference, not an established causal explanation for
the entire score change. No decoding setting, service restart or answer reuse
was introduced for64.

The full wire check covers1792 owner responses and1787 actions, with zero command,
acknowledgement or required-observation mismatches. Five replies stop at length;
188 native rejections remain separate from delivery failures. The manual review
covers every response, action and public-state change. Two final length-stopped
responses contain unfinished tool carriers and were correctly not executed; the
other three length stops contain no executable action. None receives extra budget
or replacement generation. Five partial cases hit the native horizon and five
exhaust owner output. The22 negative terminals remain native failures.

Real reasoning failures include treating focus as inspection, confusing fruit with
a plant, incomplete growth experiments, cumulative stopwatch comparisons, missing
instrument pickup, and unvalidated circuits. Some100-score outcomes also use
invalid experiments or guesses; native success alone does not certify the reasoning.
The review does not establish a communication defect explaining the score collapse.

Two narrower API defects are supported by the observed traces:

- The footer reports remaining owner action invocations, but not the separate
  simulator-move limit. A wait can cross that limit while many calls remain.
- The contact help restricts names to terminals printed by object inspection,
  although ordinary material descriptions need not list their contacts. The native
  connect API accepts such an object name and chooses a free contact; the harness
  must neither prohibit that native shorthand nor choose a contact itself.

Costs:24,354,090 input /178,886 output tokens;1792 owner calls,1787 tools, zero
peers. Generation completed at741 episodes/hour, ETA0; coordinator exit0 and no
longer running. The same three existing services remain healthy and were not
restarted. This failed64 remains part of the development record.

## Independent API corrections prepared for controlled comparison

`public-state-clock@3` adds the returned native move counter and declared limit to
the unchanged public-state@1 observation. The original native termination rule is
still `moves > limit`, not an owner-call countdown. Missing move evidence is an
incomplete interface, not an invented zero. No score, private goal state, extra
observation action, automatic finish or changed ticking is introduced. The rule
was checked in the deployed wrapper and the [official Python API](https://github.com/allenai/ScienceWorld/blob/main/scienceworld/scienceworld.py).

Separately, `public-task-semantics@14` / `scienceworld-commands-material-contacts@6`
corrects only generic electrical-contact help. Polarized components and wires need
named terminals; other objects with native contacts also support the native
object-name shorthand. This was checked against the deployed JVM implementation
and [official connection action](https://github.com/allenai/ScienceWorld/blob/main/simulator/src/main/scala/scienceworld/actions/ActionConnectElectrical.scala).
It contains no circuit recipe, material label, target choice or answer-container
mapping. All old help versions and literal tool bindings remain unchanged.

The first32 clock condition enables **only the clock observation**, keeping semantics13
and commands5. Both32 and64 retain the same pre-frozen panel10 IDs, full budgets,
thinking-off and decoding. Contacts6 is not silently enabled in that contrast.
The material-contact change is available as a separate later condition, not a
claimed cause of an unmeasured improvement.

Correction validation:192 targeted tests passed. One worker-SSD CPU `make check`
completed with5444 passed,16 skipped, Ruff/Mypy and both wheels successful
(tests343.86 seconds). The exact saved clock/contact trajectories were also checked
with61 original actions: zero differences in native observations, public state,
score, termination or moves; zero model calls and no candidate replacements. The
new clock projection was present after reset and every action. This verifies the
interface does not alter those native transitions, not that a new policy will solve
the tasks. No independent review agent or hash check was used; unchanged tests are
not repeated for these documentation updates.

## Clock-only32 result and complete trajectory review

`science32-native-clock-panel10-v43` scores **38.53125/100 signed native mean**,
below the strict50.553 gate. It has16 full successes,7 negative terminals and9
partial outcomes. All32 native scores are present; no scorer infrastructure
failure occurred. The zero-clipped60.40625 is auxiliary only. The pre-frozen64
request was **not launched** after this failed32.

All32 trajectories,891 owner replies and882 native actions have been read, including
successful trajectories. The wire checks find no command substitution, missing
acknowledgement or required-public-observation mismatch. The clock matches the
native move count at all32 resets and882 transitions. Eight replies exhaust the
8000-token episode budget without an executable final action; these remain output
failures, not transport failures. There is one owner finish and no native-horizon
termination. The absence of horizon failures does not establish better task solving:
one owner finishes after visible growth while its persistent selection is a pot.

Remaining failures include exploratory focus, container/content confusion,
unequal-duration friction comparisons, missing instrument pickup, absent circuit
indicators, and repetitive prose/actions. Some100-score cases also guess correctly
without a valid experiment. The framework does not replace their conclusions,
rescore their reasoning or select alternative commands. Material descriptions that
omit contact labels still cause confusion, supporting the already-tested generic
contact-help correction as a **separate** interface condition.

Costs:13,213,154 input /109,529 output tokens;891 owner calls,883 tools including
one finish, zero peers. Native command rejections131 and all7 negative scores remain
in the record. Training, posterior, skill evolution and training-evidence writes
remain zero. All attempted panels are development regression evidence.
Generation completed at458 episodes/hour, ETA0; coordinator exit0 and no longer
running. The same three existing inference services are healthy and not restarted.

After this complete review, the next32 condition will enable semantics14/contacts6
with clock3 held fixed. It will use the same pre-frozen panel10, thinking-off,
8000 tokens,200 action calls/native200-move horizon, presence0, batch32 and scorer.
Both32 and64 are frozen before generation;64 runs only after32 passes. This is not
an assertion that the narrower API correction fixes the model's experiment design
or repetition. No Judge code, configuration or endpoint is involved.

## Verification and resources

The code revision already passed136 related tests and a worker-SSD CPU `make check`:
Ruff, Mypy, both wheels,5394 tests passed and16 skipped. Documentation/private
review-viewer updates do not repeat that unchanged full verification. No independent
review agent or hash check was used. Only the three existing worker08 inference
replicas on physical GPU0/1/4 are used; no service restart or new resident inference
service is involved. Per-item data and private execution paths are excluded from Git.


## Material-contact32 result; user-accepted stopping point

`science32-material-contacts-panel10-v44` completed32 fresh generations using the
pre-frozen panel10, with only semantics14/commands6 enabled relative to the clock-only
condition. Clock3, single Qwen owner, skills-off/thinking-off,8000 output tokens,
200 action calls, native200-move horizon, presence0, requested seed0 and batch32
were unchanged. Services were not restarted; deterministic inference was not enabled.

The signed native mean is **38.09375/100**, with16 full successes,7 native negative
terminals and9 partial outcomes. All32 scores are present; no scorer infrastructure
failure occurred. The zero-clipped59.96875 is auxiliary, not the primary score.
This does **not** exceed the previous strict50.553 gate and is not evidence that
contact help improved overall performance. Both failed interface comparisons remain
in the development record; no item-wise best-of or alternative commands are selected.

On2026-09-16 the user explicitly accepted a ScienceWorld result of38 and requested
that work stop. Accordingly **this32-sample result is accepted by user exception;
no64-sample confirmation is claimed or launched**. The optimization/generation loop
is closed. The other five already-accepted64 conditions are not restarted.

Automatic checks cover all891 owner responses and883 native actions: zero literal
command substitutions, acknowledgement gaps or required-observation mismatches. The
public clock matches all32 resets and883 transitions. At the stop request, manual
reading covered16/32 trajectories' owner outputs and native transitions; the other16
have not received complete manual reading. This is not presented as a completed
whole-panel manual review. Already-read cases still include exploratory focus,
container/content confusion, repeated rejected commands, cumulative stopwatch
comparisons and ignoring the visible simulator horizon. Some100-score cases also
use exploratory focus rather than justified task-object selection.

Costs:12,446,476 input /106,469 output tokens;891 owner model calls,885 tools
(883 native actions, one history read, one owner finish), zero peers. Six episodes
exhaust the8000 output allowance; two reach the native simulator horizon; one partial
outcome uses owner finish. Generation finished at515 episodes/hour, ETA0; evaluation
coordinator exit0. No new resident service or service restart occurred; GPU use was
limited to the three existing worker08 replicas on physical0/1/4. Training updates,
posterior updates, skill evolution and training-evidence writes remain zero.

### Diagnostic classification correction only

The two budget-truncated final tool carriers were not executed. Previously their
`control-parse-failure` events also counted as unresolved delivery failures, which
misleadingly suggested a valid command had been lost in transit. The new
`communication-report@2` keeps unresolved/repaired control-parse counts separate from
actual delivery failures; unexplained attempts and explicit delivery failures remain
visible. It does not complete malformed commands or change generation, native scores,
budgets or any Judge path. Reprojecting the saved journal produces two unresolved
control-parse failures, four unresolved action decisions and zero unresolved delivery
failures; status remains **unresolved-output-failure**, not a clean pass. Original
run reports and the new diagnostic report are retained separately.

Validation: local scoped Ruff formatting/lint and33 worker-SSD CPU targeted tests
passed. The already-started final CPU `make check` also passed:5447 tests,16 skipped,
345.21 seconds for pytest, Ruff/Mypy and both wheels successful; coordinator exit0
and stopped. No new model generation is authorized by this closure. There is no
independent review agent or hash check, and no repeated full check for documentation.
Judge files/configuration remain exclusively assigned to the other session.
