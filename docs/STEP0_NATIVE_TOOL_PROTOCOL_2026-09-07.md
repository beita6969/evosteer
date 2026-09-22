# A2 model-native tool protocol (explicit new condition)

`configs/evaluation/step0_integrity_native_tools.yaml` declares
`A2-no-thinking-native-tools@13` (initial native-template condition: `@9`), with the v3 arm control `tool_call_mode: qwen-xml`.
Historical v1/v2 arms keep the plain-text template and their original identities.
This is a new public communication affordance, not proof that earlier unused peers
were dropped in transport or that optional collaboration must improve a score.

The newer [single-owner repair worklist](STEP0_SINGLE_OWNER_REPAIR_WORKLIST_2026-09-07.md)
adds remaining policy/output provenance, resume and AIME native-thinking work. Its planned
thinking-on condition is separate from the frozen thinking-off `@9`–`@13` history below.

## Observed defects and implementation

The preceding complete `5fa9b0e` development round contained rejected `Action: search`
requests with an explicit capitalized `Query:` label, and complete JSON tool calls
following discussion. The syntax-only decoder now accepts these forms without
changing literal argument values. It counts calls before public-surface validation;
it does not choose among candidates, change target case, strip target quotes, infer
an action from discussion, or replace unavailable actions. Licensed trace contents
and individual results remain private.

The deployed Qwen3.5 template has a native function schema branch, but the old clean
runtime never supplied its `tools` argument. The new condition supplies definitions
for the *existing* public interfaces: history, native simulator actions where present,
and optional advice-only peers. Static tasks additionally have `submit_answer`, an
explicit envelope for the owner's own response or code, not another model call.
It adds no external solver, search access to reading-comprehension tasks, skill,
handwritten policy, vote, selector, fallback or automatic environment action.
Direct chat answers and existing native action/message forms remain accepted.

Tool definitions are included in actual tokenizer packing and context accounting,
actor/broker RPC, the rendered request, and the trusted model-transport trace. The
broker re-encodes the actual template and verifies its definitions against the
condition's public capabilities. Peers receive no executable function definitions;
their isolated replies remain advice. Optional peer calls and interface repairs use
the same shared call/token limits. No native thinking or extra serialization model
is enabled, and no sampling, total output budget or environment horizon is reduced.

A single Qwen XML envelope preserves literal parameter bodies, including LaTeX,
quotes, newlines and source indentation. Unknown or ambiguous calls require an
ordinary budgeted owner repair. The unique owner response is independently projected
again at the broker boundary. No grader answers or tests enter actor context.

The previous AIME empty submission actually spent its full 81,920 output tokens in
one call. That is not evidence that a correct final answer was lost, and this change
does not rescue or replace that old answer.

## Validation and execution

All code is main-thread work. The only subagent watches resources; no review/audit
or implementation subagent is used. The affected 246-test set passed on the approved
CPU host in 7.72 seconds, including actual isolated actor/broker peer, owner-final
and native-environment tests. Local Ruff passed. Local mypy was stopped after WSL
filesystem I/O stalled; the complete type check runs on the approved CPU host as part
of the final `CUDA_VISIBLE_DEVICES="" make check` rather than repeating that wait.
The first final attempt stopped at a typed `apply_chat_template` keyword-expansion
error before tests/builds; it was fixed by passing the named `tools` argument.

No checksum, retired approval/seal/attestation gate, or separate review agent is used.
A fresh disjoint/native-train canary will exercise the real model-native path before
any complete A2 round. Only the existing three registered inference replicas are
eligible; historical candidates and grades are never reused or silently updated.
Scores and real peer usage remain unverified for this new condition at this stage.

The tokenizer fix passed the 31 affected native-protocol/actual-broker tests in
5.65 seconds. The subsequent final CPU `make check` passed: **2,926 tests** in
128.60 seconds (85 existing warnings), full Ruff/mypy, both wheel builds and the
model-wheel private-boundary check. That source stayed fixed during the first native-template canary. Result-only
notes did not require another identical full validation; the subsequent argument
clarification is verified separately below.


## First real native-template canary and argument clarification

`canary-a2-native-tools-f79a70b-20260907T124441Z` completed **15/15** in **172.07 s**,
with 161 completed model calls (646,910 input / 22,939 output tokens), and all 145
native actions acknowledged. A real owner-requested researcher call occurred in
HealthBench; the peer received no tools and returned advice to the owner. The other
14 episodes voluntarily used no peer. This establishes observed execution, not a
causal multi-agent score benefit. The [aggregate-only export](machine-results/step0_a2_native_tools_canary_2026-09-07.json)
contains costs and communication, not individual or single-canary-task scores.

There were ten budgeted interface repairs and no unresolved communication failure.
Seven rejected calls used unavailable actions. Three others passed a complete
`click[...]` expression as the function's `target` value. The new schema had not
explained the distinction between a displayed action notation and its literal
argument clearly enough. Condition `@10` documents the native parameter meanings:
a click target is the label itself, a search query is the query text, and an ALFWorld
command is the complete command. It does **not** strip wrappers or alter a model's
target automatically. A fresh canary precedes the complete panel of that condition.

At 12:51:18 UTC, all three resident services received an unexplained SIGTERM and
shut down gracefully, with no OOM in their log tails. The finished canary was not
interrupted, and no full native-template round had started. The main thread initiated
one owner-authorized same-card/same-argument recovery, not a restart loop. The signal
source remains unknown. A separate observed PID reuse by a kernel worker was handled
by comparing the coordinator command, never by terminating the reused PID.

Recovery was **not launched**: after fixing a private launcher syntax error that had
no side effects, its prelaunch resource check found the three devices already above
the allowed 25% sharing threshold. At 12:58 UTC the main thread observed 39,025/47,861/40,995 MiB on GPUs 1/3/7, no registered service PID, and no listening service
health endpoint. These occupancies are not attributed to this project's recovery.
No process was terminated or modified and no additional GPU was used. Human resource
coordination and confirmation of the SIGTERM source were requested. Condition `@10`
has not yet run a canary or full panel; its scores remain unverified.


The `@10` argument-description change passed 24 affected native-protocol/capability
tests in 1.87 seconds, local Ruff, and a subsequent CPU `make check`: **2,926 tests**
in 130.02 seconds, with the same 85 existing warnings, full Ruff/mypy, both wheels
and the model-wheel private-boundary check. This validates the changed implementation,
not real-model compliance with the clarified descriptions or any new benchmark score.
At the main-thread 13:04:44 UTC resource check, all three original services remained
offline; the new occupancies were 47,685/47,861/41,283 MiB. No GPU recovery process
was started. The next real canary/full evaluation was then dependent on resource
availability and coordination; no completion or all-goal claim was made.

## Resumed `@10` canary and owner-approved GPU replacement

The main thread recovered the original GPU7 service after it became available.
Two startup attempts failed because the launch environment omitted the installed
CUDA compiler path, then the existing virtual environment's executable path. These
specific environment omissions were corrected before the next attempt; no package,
model, decoding or budget change was needed, and no other process was terminated.

`canary-a2-native-tools-b31c562-20260907T174551Z` then completed **15/15** in
**284.15 seconds**, using GPU7 only. The [separate `@10` aggregate](machine-results/step0_a2_native_tool_arguments_canary_2026-09-07.json)
retains 163 completed model calls, 663,519 input / 25,385 output tokens, 146
acknowledged native executions and one owner-requested advice-peer call. All twelve
budgeted interface repairs resolved. Ten attempts proposed unavailable actions; two
were prose without an executable native action. No wrapped `click[...]` target was
observed among these rejections. This is interface evidence, not a score improvement
claim or an eight-IID result. No old canary answer or grade was reused.

The owner explicitly approved replacing unavailable original GPUs1/3 with GPUs4/5.
The main thread checked all eight devices before launch: both replacement devices
were idle at 4 MiB used, and GPU7's registered service was retained. At 17:58:31 UTC,
the main thread confirmed healthy services on **22048 physical GPUs4/5/7** with the
expected commands and explicit device mappings. Two new SGLang parents were started
(31957 and 31958; schedulers 33012 and 33009); GPU7 retained parent14566/scheduler15978.
No fourth GPU or GPU job on 22049 was used. These are time-stamped observations,
not a guarantee against another external service termination.

The fresh full A2 run `full-a2-native-tools-b31c562-20260907T175910Z` started at
**17:59:13 UTC**, coordinator PID55506, using only those three replicas. It freezes
`b31c562`, condition `@10`, one seed, seven 128-task cohorts plus all 30 AIME tasks,
generation concurrency64 and scoring concurrency8. The same per-episode budgets,
native horizons and local-Qwen grading condition remain in force. All 926 answers
must be retained before scoring; prior answers and grades are not used. Results and
all-goal acceptance remain pending this run.

Only private operational launch/monitor helpers and these result notes changed after
the successful `b31c562` full CPU validation. Their syntax and actual launch, device
mapping, health and completed canary paths were checked; the unchanged 2,926-test
suite was not rerun for documentation or a GPU replacement. All work was main-thread
work except the read-only resource watcher; no hash checks were performed.

## Observed long-tail feedback defects (`@11`, not applied to the running `@10`)

At the owner's request, the main thread inspected the six pending `@10` episodes,
without grading, replaying a request or modifying the frozen run. Three ALFWorld
episodes and one HumanEval episode had multiple 8,192-token responses containing
repeated discussion or code comments, not completed native actions/submissions.
Two AIME episodes were also pending: one still in its original 81,920-token request;
the other had an ambiguous legacy final, then an owner-requested history read and
another long generation. There is no unfinished-response content available for the
first case, so repetition is not asserted for it. Transport completions continued
while completed-episode throughput fell below 40/hour; this was escalated, and the
original budget was retained. None of this establishes a correct discarded answer.

Two concrete public-feedback defects were identified. A model interpreted the
internal `not-on-current-public-surface` code as a statement about an object's
physical location. Feedback now explains that the proposed command is absent from
the environment's current available-action list and was not executed. The structured
internal diagnostic and complete menu remain unchanged; no action is selected or
rewritten. Separately, an incomplete native answer envelope received an instruction
to send a message to `solver`. Feedback now distinguishes a native-call decoding
failure from an ordinary control message, describes complete arguments/closing tags,
and retains the owner's actual direct submission option without redirecting to a peer.

When the backend reports `length`, repair feedback now explicitly identifies the
per-call output-token limit. That backend finish reason is retained in both the
trusted transport-complete event and actor diagnostic response. The old traces did
not preserve it per call; old causes are not backfilled from token counts. No repeat
detector, early cutoff, penalty/sampling change, budget change, forced collaboration,
answer selection, native action rewrite or benchmark-specific solving advice is added.
Condition `@11` records the new feedback before a separately frozen evaluation.

The affected nine-file test set passed **248 tests in 7.96 seconds** on the approved
CPU host; local Ruff passed. Tests cover literal command/menu preservation and an
actual isolated owner/broker repair from an incomplete answer, with unchanged shared
call accounting, no peer call and matching trusted/diagnostic termination causes.
The necessary final CUDA-disabled CPU `make check` completed successfully (PID654280):
**2,930 tests passed in 129.61 seconds**, with 85 existing warnings, full Ruff/mypy,
both wheel builds and the model-wheel private-boundary check. No unchanged suite was
rerun for the result note. Real `@11` execution remains pending. No implementation or
review subagent is used.

The main thread also checked the installed serving EOS logic rather than changing
stop tokens speculatively: it checks the tokenizer's chat-end token as well as model
EOS IDs. None of the 4,422 completed owner/peer responses inspected contained an
unconsumed chat-end/start or end-of-text token. A missing optional generation-config
file alone therefore does not establish a stop-token defect; no model file or serving
parameter was altered by this diagnosis.

The separate `@11` canary `canary-a2-native-tools-5a1517a-20260907T184053Z`
completed **15/15 in 366.98 seconds**, with 165 completed model calls, 699,924 input /
27,008 output tokens, 144 acknowledged native executions and one actual advice-peer
call. All 165 backend finish reasons were `stop`, not `length`. The
[aggregate-only record](machine-results/step0_a2_native_feedback_canary_2026-09-07.json)
explicitly retains **eight unresolved action attempts in one WebShop episode**: the
owner repeatedly returned shopping prose instead of an executable command. The other
eight rejected attempts resolved. Fifteen budgeted repair calls were made; there were
no lost executions, route/feedback mismatches or failed grader invocations. This is
not an all-clear communication result and does not prove that the wording fixes
eliminate model repetition or improve task performance. No new syntax failure or
rejection of a legitimate submitted action was found in these eight pending attempts.

The fresh full `@11` run `full-a2-native-tools-5a1517a-20260907T185427Z` started at
**18:54:30 UTC**, coordinator PID14898, with the same 926-task scope, seed, budgets,
sampling, native horizons, generation concurrency64 and scoring concurrency8. It
uses only the resident 22048 GPUs4/5/7 and overlaps the unchanged `@10` long tail;
each run has independent candidates, context, metadata, controls and scoring. The
canary coordinator has ended. No new inference daemon, fourth GPU or training job
was created. This full round measures the changed condition, not a claimed fix for
every model-origin failure. Both full results and the all-eight goal remain pending.

## Complete `@10` full result (not a result for `@11`)

`full-a2-native-tools-b31c562-20260907T175910Z` finished **926/926 candidates and
scores**, with no missing grade, in **59.67 minutes**. The main thread confirmed
the actual coordinator had ended before exporting the [aggregate-only record](machine-results/step0_a2_native_tool_arguments_full_2026-09-07.json).
The original `b31c562` actor, parser, budgets and controls were retained throughout,
including its known feedback defects. This table is neither a best-of combination
nor a replacement of any earlier result.

| Benchmark | Count | Native primary, percent | Goal met |
|---|---:|---:|---|
| HotpotQA | 128 | Answer F1 72.13 | No (>75) |
| TriviaQA released RC | 128 | Answer F1 82.62 | Yes (>75) |
| AIME2026 | 30 | Accuracy 73.33 (22/30) | No (>=80) |
| HealthBench, local Qwen rubric | 128 | Mean rubric score 44.21 | No (>53.17) |
| WebShop | 128 | Native reward 19.61 | No (>=80) |
| ALFWorld | 128 | Native SR 43.75 (56/128) | No (>=80) |
| MBPP+ hard, actual MBPP v0.2 population | 128 | Base AND Plus 72.66 (93/128) | Yes (>70) |
| HumanEval original | 128 | Pass@1 85.94 (110/128) | No (>=118/128) |

WebShop SR was 10/128; ALFWorld seen/unseen were 41/97 and 15/31. MBPP Base was
109/128. Two empty AIME submissions and one each in MBPP/HumanEval remain native
candidate failures, not missing grades. The historical accepted HumanEval score is
not substituted here. HealthBench is not the official external-judge track.

All **4,432 model calls** completed; input/output cost was **16,652,406 / 1,252,210**
tokens, including **749 budgeted repairs** and **50 actual advice-peer calls**
(HealthBench 37, MBPP 4, HumanEval 9). All 50 peer requests returned replies, and all
2,796 native executions were acknowledged; 73 additional tool reads accessed public
history. There were no recorded missing native acknowledgements, stale surfaces or
route/feedback mismatches. This does not make the unresolved model outputs a PASS:
WebShop retained 270 unresolved rejected-action attempts and ALFWorld 80, with other
unresolved control/final outputs detailed in the export. Attempt counts are not
episode counts. Actual peer use does not establish a causal performance benefit.

Local grading made 1,499 requests/responses, including two permitted semantic repairs,
with 2,270,010 input / 158,839 output tokens, zero failed invocations and no unknown
usage. No earlier missing HealthBench grade was retried. Only **two of eight** goals
were met. The separately frozen `@11` full round is still running; its results cannot
be inferred from this table. No additional validation run is needed for this
aggregate-only documentation of already validated, unchanged source.

## Equivalent AIME scalar representations (`@12`)

Read-only inspection of the two empty `@10` AIME submissions found one concrete
false rejection: a single explicit final field contained a plain integer followed
by the same integer in a boxed representation. This corrects the preliminary
diagnosis of conflicting boxed values; the complete response did not support that
claim. The other response contained multiple active final fields and intervening
discussion, which remains ambiguous under the existing field-ownership rules.
No reference answer or grade was used to choose a value, and neither old submission
nor its grade is replaced.

Within an already selected explicit final field, the scalar parser now accepts
whitespace-separated plain/boxed integer representations only when every value is
identical. Leading zeros are formatting, not a different answer. Conflicts, prose,
decimals, out-of-range values and ambiguous multiple final fields remain rejected;
there is no majority rule, answer search through derivations or extra model call.
The AIME parser/projection versions and condition identity are updated. Other
benchmarks, model requests, total budgets, sampling and native horizons are unchanged.

Main-thread regression tests use synthetic values and cover the actual isolated
owner/broker path: the equivalent scalar is accepted from one owner call, without
repair or a peer. Local Ruff and **63 targeted tests in 7.98 seconds** passed. The
necessary CUDA-disabled CPU `make check` (PID170620) passed **2,938 tests in 128.23
seconds**, with 85 existing warnings, full Ruff/mypy, both wheels and the model-wheel
private-boundary check. No hashes or independent implementation/review agents were
used. The source-frozen `@11` full run continues unchanged; these tests are not a new
AIME benchmark score or evidence that the all-eight goal has been met.

## Complete `@11` full result; `@12` remains separate

`full-a2-native-tools-5a1517a-20260907T185427Z` completed **926/926 candidates and
scores in 59.45 minutes**. The main thread confirmed normal coordinator completion
before the [aggregate-only export](machine-results/step0_a2_native_feedback_full_2026-09-07.json).
It used its original frozen `5a1517a` implementation throughout, not the later scalar
parser. Only **two of eight** goals were met; a communication repair is not evidence
that every remaining low score must have an implementation cause.

| Benchmark | Count | Native primary, percent | Goal met |
|---|---:|---:|---|
| HotpotQA | 128 | Answer F1 72.13 | No (>75) |
| TriviaQA released RC | 128 | Answer F1 82.62 | Yes (>75) |
| AIME2026 | 30 | Accuracy 73.33 (22/30) | No (>=80) |
| HealthBench, local Qwen rubric | 128 | Mean rubric score 43.59 | No (>53.17) |
| WebShop | 128 | Native reward 19.09 | No (>=80) |
| ALFWorld | 128 | Native SR 49.22 (63/128) | No (>=80) |
| MBPP+ hard, actual MBPP v0.2 population | 128 | Base AND Plus 72.66 (93/128) | Yes (>70) |
| HumanEval original | 128 | Pass@1 85.94 (110/128) | No (>=118/128) |

ALFWorld seen/unseen were 47/97 and 16/31; WebShop SR was 10/128 and MBPP Base
109/128. All **4,268 model calls** completed, using **15,689,185 input / 1,188,911
output tokens**, including **635 budgeted interface repairs** and **50 actual
advice-peer calls** (37 HealthBench, 4 MBPP, 9 HumanEval). All 50 peer requests
returned replies. All **2,742 native executions** were acknowledged; 80 additional
tool reads accessed public history. Backend finish reasons were **4,242 stop / 26
length**. Native scoring made 1,499 local-grader requests/responses, with 2,276,358
input / 158,246 output tokens, two permitted semantic repairs and no failed or
unknown-usage invocations. No old missing grade was retried.

No missing model response, native acknowledgement or route/feedback mismatch was
recorded. Unresolved outputs still include 263 WebShop action attempts, 67 ALFWorld
action attempts, eight ALFWorld and four HumanEval control-encoding attempts, and
four AIME terminal-parse attempts. These are attempt counts, not episode counts or
proof of transport loss. Two AIME submissions and one each in MBPP/HumanEval remained
empty candidate failures. During the ALFWorld tail, the current observations,
complete action menus and public command semantics were present, yet the model
repeated discussion or unproductive commands. Neither context loss nor missing API
semantics was established for those inspected requests. The known equivalent-AIME
scalar rejection is addressed only in the later condition.

The `4dd26d3` **complete fixed AIME30 focused** round started at **19:32:27 UTC**,
coordinator PID32655, concurrency30, on the same resident GPUs4/5/7. It generates
all 30 answers afresh with unchanged per-episode budgets; it is not an eight-IID
result, a selected hard-case retry, or a source of replacement answers for `@11`.
The private subset runner uses the runtime's `canary=True` panel flag while explicitly
requiring the entire AIME30 cohort. An earlier helper launch (PID21513) exited before
any model call because of a stale arm-name assertion; that zero-call failure remains
recorded, and only the helper assertion was corrected. No inference service was
restarted. The focused result is pending. No unchanged full test suite was repeated
for these result notes or operational helper changes.

### Completed `@12` AIME30 focused result

The [separate aggregate](machine-results/step0_a2_native_scalar_aime30_2026-09-07.json)
records **30/30 candidates and scores in 53.43 minutes**, with **23/30 correct
(76.67%)**, below the80% goal. One empty submission remains. All **39 model calls**
completed, costing **78,138 input / 338,337 output tokens**, including seven budgeted
interface repairs and two history reads. Actual peer calls were zero. Backend finish
reasons were 38 stop / one length. There were no model transport failures, missing
grades or external/local-model grader calls; AIME used its native integer scorer.
Two terminal-parse attempts remained unresolved, so this is not a communication PASS.

A separate read-only comparison, without reading grades or rewriting candidates,
confirmed that one previously empty episode freshly produced the **identical initial
owner response** and submitted it through the new scalar projection in **one call,
44,673 output tokens**, without repair. This verifies the observed syntax fix rather
than attributing a different generated answer to parsing. The remaining ambiguous
response was not resolved by a vote, last-value selection or reuse of an old answer.
The last long request continued to its original token limit; neither the budget nor
the denominator was reduced to finish sooner.

At the main thread's **20:26:25 UTC** post-run check, the focused coordinator had
ended, all three SGLang services on22048 GPUs4/5/7 were healthy and idle, and no new
training or evaluation job was launched. Parent PIDs31957/31958/14566 remain resident
as explicitly requested. The full `@11` score table remains unchanged; the focused
AIME score is not spliced into it. The latest complete eight-IID evidence meets only
two goals, and the focused AIME repair still does not meet its goal. Documentation-only
publication reuses the successful 2,938-test final check; no repeated suite, hashes,
independent review agent or new GPU service is added for this note.

## Repeated explicit-final compatibility (`@13`)

Further read-only diagnosis corrected an overly broad earlier conclusion: multiple
active final headers do not by themselves prove conflicting answers. The remaining
empty `@12` AIME response's initial owner text was accepted by the unchanged legacy
unique-value projection, but the newer owner dispatcher rejected it solely because
there were two final headers. No reference or score rows were read for this diagnosis.

The AIME-only compatibility branch now requires every final header to begin with a
complete scalar declaration at a line boundary. All such declarations and any
standalone integer lines in their fields must agree, and the unchanged legacy
projection must independently yield that same value. Thus an invalid later final,
out-of-range value or conflicting box cannot disappear into an earlier valid answer.
There is no last-header selection, majority rule, candidate vote or semantic rewrite.
Other benchmark projections and the single explicit-payload path remain unchanged.
The arm and AIME parser versions distinguish this new condition from frozen history.

The proposed parser accepted that historical raw string with its original text and
the same legacy projection in a read-only diagnostic. This made zero model calls,
changed no candidate or grade, and is not a new benchmark result. Synthetic tests
also exercise the real isolated owner/broker: repeated equivalent declarations need
one owner call, while conflicting declarations require the owner's subsequent new
submission within the shared budget. **77 targeted tests passed in 9.29 seconds**;
local Ruff passed after removing an extra import-block blank line. The main thread
performed one combined code inspection; no coding/review subagent or hash check was
used. The necessary CUDA-disabled CPU `make check` (PID556421) passed **2,952 tests
in 134.45 seconds**, with 85 existing warnings, full Ruff/mypy, both wheel builds and
the model-wheel private-boundary check. A subsequent comment-only clarification
needed only the affected-file Ruff check, not another unchanged full suite. The
fresh complete AIME30 evaluation is still required; no score is inferred from the
parser diagnostic or synthetic tests.

The fresh `@13` AIME30 run `focused-aime-a2-native-tools-7235e93-20260907T205956Z`
started at **20:59:59 UTC**, coordinator PID4994, concurrency30, on the same resident
GPUs4/5/7. All 30 tasks are newly generated under the original thinking-off condition,
with unchanged budgets. The runner is copied into this run's private directory.
An earlier launch (PID7539) failed at import before any model call: the source archive
producer was still running when transfer began, producing an incomplete destination.
The main thread waited for archive completion, confirmed the subsequent transfer and
extraction finished successfully, then used a new directory and run. The failed attempt
is retained separately; it produced no candidates and is not part of the valid run.
No service restart, fourth GPU, training job or hash check was introduced.

The newer owner request adds pending no-selection/provenance/resume and AIME thinking-on
work in the linked worklist. It does not hot-change this running thinking-off experiment
or authorize mixing its answers with a future thinking-on round. This documentation-only
update does not repeat the successful unchanged-source full validation above.

### Completed `@13` AIME30 result

The [separate aggregate](machine-results/step0_a2_native_repeated_aime30_2026-09-07.json)
records **30/30 candidates and scores in 28.27 minutes**, with **23/30 correct (76.67%)**,
still below the80% goal. There are **no empty submissions**, and all six budgeted interface
repairs resolved; the original condition reports `complete-with-repairs`. This removes
the observed submission failure without establishing a correctness improvement. It is
not evidence that the subsequent persisted-source/thinking changes have already run.

All37 model calls finished with `stop`, costing **75,446 input / 272,670 output tokens**;
there was one public-history read and no actual peer call. No grader-model invocation,
missing native grade, vote, answer reuse or selected-item rerun occurred. The complete
fixed30 denominator and the original budgets were retained. The same23/30 score is not
spliced into the older eight-IID table or attributed to the later thinking-on condition.

At **21:29:55 UTC**, the main thread confirmed PID4994 had ended normally, all30 scores
were present and the three registered GPU4/5/7 services remained healthy and resident.
End-to-end throughput was about **63.68 records/hour**, final ETA zero. The earlier
single-request tail had zero incremental completions and was explicitly escalated below
the40/hour threshold; no budget was shortened to finish sooner. No inference daemon was
restarted and no additional GPU was used. The newer integration is verified separately.
