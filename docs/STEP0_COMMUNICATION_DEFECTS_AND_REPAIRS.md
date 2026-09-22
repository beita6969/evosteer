# Step-0 communication repairs

| Defect | Clean implementation |
|---|---|
| Skill ID invokes a handwritten policy | Advisory registry contains text only; no-skill never queries it |
| Grammar suppresses back/previous/look/inventory | Native surface supplies the complete legal set |
| Last-mentioned action mistaken for intent | Parse one explicit decision field; discussion alone, negation and ambiguity do not execute |
| Chinese verb changes the target | Normalize click/search verbs only; preserve exact target arguments |
| An attempted action becomes a completed fact | Durable intent, transport acknowledgement and semantic outcome are distinct |
| Task failure inferred from an action name | Preserve native observations and terminal status; do not infer state changes |
| Old state or another episode receives a reply | Program-owned scope, revision, parent and recipient validation |
| Two passes described as multi-agent | Separate peer contexts, delegation/reply records and actual-call counts |
| Truncated history replaces the task | Full original task; whole-event history eviction with accessible archive references |
| Final answer gets replaced after grading | Owner-only insert-once final; read-only scorer handle, no selection callback |
| Server identity asserted by caller flags | Expand actual service settings before and after generation |

Runtime summaries compare native/advertised/grammar action sets, decisions against durable
executions, returned observations against public transitions, and peer request/reply routes.
Semantic `unknown` is not a lost transport acknowledgement and is not invented success.
Only aggregate counts are public; messages, source tasks and detailed traces remain private.

Deployment validation found two concrete interpreter namespace issues: an absolute Python link
through a CPython directory alias, and a private `/dev` mount hiding a `/dev/shm` virtualenv.
Mount only the necessary interpreter alias and create private special filesystems before dependency
mounts. Do not solve these failures by exposing the host root, evaluator data or host network.

The real native canary also exposed missing JVM configuration, an EvalPlus companion-runtime
dependency hidden by isolation, the old worker's nonexistent `SUCCESS` import (official `PASS`),
and a HumanEval request type that rejected the empty prefix needed for a complete module. These
are evaluator/deployment repairs, not prompt changes. Namespace startup failure is distinct from
candidate runtime failure. Resume retains submitted answers and definitive scores; an interrupted
unsubmitted episode cannot silently receive a fresh model budget.

Scorer transport attempts, known tokens, unknown usage and native scoring wall time are recorded
separately from actor inference cost. Older canary scores without that instrumentation remain
explicitly unmeasured rather than being regenerated or regraded to fill the accounting fields.

The nonfinal canary exposed messages being mistaken for executed actions and repeated review-only
turns. Clean generation now accepts native action lines without forcing a JSON-only decoder.
Message/review replies explicitly acknowledge communication without claiming tool execution.
Reports distinguish observed native execution from merely intact request/reply routing.

An actual parser defect remained after removing the decoder restriction: it compared analysis
plus a final `Action:` line against the native action set, and rejected explicit `Action` plus
JSON `Argument` fields. The codec now projects that unique final field without changing its
argument. Quoted/example/negated blocks, conflicting action fields and tools absent from the
current capability surface remain non-executable. These checks use syntax and public state only.

Validation is batched and risk-based. No review subagent, hash verification or retired
approval/receipt gate is used. The GPU-monitoring subagent is read-only operational monitoring.

The earlier interactive-only nonfinal canary completed four candidates and 59 native executions.
Both arms preserved the full surface, decision/execution target, acknowledgement and observation
routes with zero recorded mismatches. A2 made 19 actual independent peer calls. Rejected model
decisions remain in the communication counts; no action was substituted to make a task succeed.
See the [answer-free diagnostic](machine-results/step0_integrity_2026-09-06.json).

The final CPU-only `make check` ran on a Linux-local copy of precisely the proposed commit scope:
2,552 tests passed in 120.37 seconds, plus Ruff, mypy, both wheels and the model-wheel boundary
check (exit 0). The last codec change had 53 targeted tests pass. An earlier native-wire milestone
also passed a full check on the then-current worktree; its larger test count includes unrelated
worktree tests and is not substituted for the final commit-scope result. Repeated HealthBench/code
canary grading and unchanged test commands were skipped after the interactive-only codec repair.

## Full-run transport failure and repair

The first full attempt (`3f903e3`) stopped with 559 of 1,852 candidates and no scores. Some long
AIME requests exceeded the 2,400-second HTTP deadline; repeating the non-streaming call discarded
roughly 40 minutes of generation each time. After three attempts, cancellation waited silently for
other HTTP threads. Those aborted attempts have unknown token usage: the old successful-response
totals are not total compute cost. The immutable partial run is **incomplete**, not a quality result.

Clean calls now make one transport attempt, recording start/completion/failure and unknown usage
instead of silently generating again. The execution timeout is extended to 7,200 seconds; model
call/output-token budgets, native thinking and scoring are unchanged. Replica selection uses
current in-flight call counts with rotating ties, not fixed task parity. The observed failure had
four long requests on one replica while the other was idle. Routing, concurrency, transport limits
and CUDA-graph settings now appear in the paired runtime controls.

On generation failure the pipeline reports failure immediately, stops admitting new episodes, and
preserves final answers from already-running actors before closing. It cannot score a partial
panel. A fresh frozen attempt must not splice the old partial candidates into its results; no
quality scores were available to select this engineering repair.

A separate four-candidate synthetic arithmetic canary completed on the two real services: two
requests per replica, four recorded starts/completions, no failed transport and no extra attempt.
This checks actual routing and accounting, not full-budget long-generation completion or AIME
quality. The corrected full run must still provide that execution evidence.

The integrated check caught a broad exception handler at the wrong production boundary. The
pipeline now inspects bounded task results, retains active actors on failure, and still cancels
them on explicit external cancellation; the boundary rule was not weakened. The final targeted
set passed 36 tests. The corrected CPU-only `make check` passed 2,559 tests in 121.84 seconds,
Ruff, mypy, both wheels and the model-wheel boundary check (exit 0). The earlier failed check is
not counted as a pass. Slow local mypy was stopped and replaced by that Linux-local typecheck;
unchanged native environment and scorer canaries were not repeated.

The fresh frozen `7f80f26` full run completed all 60 AIME generations by 2026-09-06 12:01 UTC,
including a normally completed 81,920-token call. No transport failures or automatic replays were
recorded before the later explicit interruption described below. This supplies the long-request
evidence missing from the short synthetic canary, not quality evidence.

## Literal native-call compatibility

Read-only communication diagnostics on the full run found a concrete codec omission:
`search(query="...")`, `Argument: query: "..."`, inline-code calls and an action accompanied
by its identical JSON representation were rejected despite expressing one unambiguous decision.
This was established from public syntax and communication traces, without generating or reading
quality scores. Other rejections, including unavailable actions and conflicting decisions, remain
legitimate; a high rejection count alone is not proof of a transport bug or task failure.

The codec now parses literal function syntax without executing code, accepts a single literal
argument or named argument, and compares duplicate representations rather than selecting one.
Arguments, current action permissions, channel ownership and state revision remain unchanged.
Expressions, multiple conflicting actions and wrong tool namespaces remain non-executable.
No model call, serializer, answer fallback, target substitution or additional budget is introduced.

The immutable `7f80f26` run was not hot-patched. Its coordinator alone received SIGINT at
12:44 UTC and had exited by 12:48 UTC, retaining 866 candidates, zero scores and four interrupted
requests with unknown usage. The inference services remain online. Old partial candidates are
not silently spliced into a corrected run. Whole-domain interactive-first scheduling is supported
for the next frozen attempt; task membership and within-domain ordering do not change.

The targeted codec/controller/order set passed 89 tests. The final CPU-only `make check` passed
2,595 tests in 132.12 seconds, Ruff, mypy, both wheel builds and the model-wheel boundary check
(exit 0). Four earlier failures were test-wrapper access errors, corrected before the successful
targeted run; they are not counted as passes. Unchanged static/scorer canaries and duplicate full
checks are skipped. There is no independent review agent or hash check.

The repaired native canary completed four candidates and 60 executions in about 283 seconds
(50.8 candidates/hour for this small diagnostic only). All 120 model transports completed;
A2 made 19 actual independent peer calls. All recorded surface, execution, acknowledgement,
observation and peer-route mismatch counts were zero. Fourteen inline-code decisions executed;
literal function and named-argument compatibility is covered by the controller tests, not claimed
as observed in this canary. Its remaining 22 rejected decisions were unavailable actions or
ambiguous fields. See the [answer-free canary record](machine-results/step0_integrity_literal_calls_2026-09-06.json).
Native quality scores remain private and were not used to select the condition. The new frozen
`b140e33` full paired run started at 13:04 UTC with all old candidates excluded.

The next full-run communication sample confirmed 10 accepted function calls and three accepted
colon-named arguments, with intact execution and peer routes. It also exposed two additional
literal formats: a bare operation header followed by matching complete tool JSON, and
`Argument: query="..."`. These remained wrongly rejected. The `b140e33` coordinator was stopped
at 13:53 UTC, retaining 66 WebShop candidates and zero scores, without changing its source or
stopping inference services. Four interrupted requests have unknown usage.

The follow-up codec accepts the matching operation header plus JSON and quoted assignment
arguments. Explicit named parameters and a single quoted positional argument also support
space-delimited call notation; this is whole-field parsing, not action extraction from prose.
Conflicting operations, expressions, negation and multiple intents remain rejected. The
targeted controller/codec/order set passed 119 tests. A subsequent typecheck caught reuse of
a local variable with incompatible optionality; a variable-only rename fixes it without changing
runtime behavior. The first check failed and is not counted as a pass.

Expanded native validation uses six final-disjoint WebShop goals and the six existing ALFWorld
train development games, with matched original budgets in both arms. Its first preparation
attempt used raw WebShop JSONL offsets for task text, whereas the native simulator uses shuffled
goal ordering. The unchanged task-consistency check rejected the attempt before any model call.
The fixture builder now obtains public instructions from the official manifest worker at the
selected native indices, rather than weakening the check or changing selected tasks. This is
a development fixture error, not a scored policy failure; the frozen full manifests are unchanged.

After the type-only correction, the final CPU-only `make check` passed 2,625 tests in
163.42 seconds, Ruff, mypy, both wheels and the model-wheel boundary check (exit 0).
The corrected 24-candidate native canary started at 14:07 UTC. No unchanged static/scorer
canaries, successful full checks, independent review agents or hash checks were repeated.

The expanded canary completed all 24 candidates, 289 native executions and 631 completed model
transports, with 69 independent peer calls and zero recorded surface, execution, acknowledgement,
observation or peer-route mismatches. Generation took about 52 minutes (27.7 candidates/hour),
below the 40/hour operational alert threshold; this is a speed risk requiring human attention,
not a reason to cut frozen model budgets. See the
[answer-free expanded diagnostic](machine-results/step0_integrity_expanded_native_2026-09-06.json).
These are development-tree diagnostics, not a claim that the complete development source tree
equals the immutable full-run snapshot. Full evaluation uses its own complete committed snapshot.

Syntax-only inspection of the development traces found three quoted search arguments followed
by a sentence period outside the quotes. The literal codec now ignores that outside punctuation
while preserving a period inside the argument. Targeted controller tests cover this last
projection, duplicate representations and rejection of conflicting continuations. The unchanged
native execution routes do not require rerunning the entire 24-candidate GPU canary. No native
quality scores were read to select this repair.

The punctuation-specific runtime module passed 100 targeted tests. The final CPU-only
`make check` then passed 2,633 tests in 157.01 seconds, Ruff, mypy, both wheels and the
model-wheel boundary check (exit 0). The already-tested controller/order modules were not
rerun separately, and the 24-candidate native canary was not repeated for this string projection.

## Owner-requested A2-only continuation

On 2026-09-06 the owner explicitly cancelled A1 and requested only A2, without increasing
sample counts or repeating completed answers. The paired coordinator stopped at 19:28 UTC
with 126 completed answers per arm and zero scores. A2-only continuation started at 19:41 UTC:
seven IID panels contain 128 tasks each and AIME2026 contains its 30 tasks, for 926 A2 answers.
All 126 completed A2 answers, their costs, communication traces and native execution outcomes
are retained. A1 is neither generated nor scored in the continuation. Unfinished episodes are
explicitly interrupted attempts in the old private run, not completed answers or zero-cost work.

This is an owner-directed scope/scheduling change, not splicing results across a policy repair.
The model-facing actor, parser, ordered public inputs, sampling, seed, tool space and model
budgets remain those of `1d09159`. Only orchestration changes: one selected arm, three existing
matched inference replicas on physical GPUs 1, 3 and 7 at the approved 22048 endpoint, and
coordinator concurrency 12 instead of 4. The old paired source and records remain unchanged.
Completed answers are copied transactionally into a new private run with explicit source-run
lineage; no scores are consulted and no unfinished environment state is replayed. Native
scoring still starts only after all 926 unique final answers exist. The single-arm report has
native means and exact denominators, not paired deltas or an unsupported improvement claim.

The scope-reduction/pipeline/results set passed 34 targeted CPU tests in 1.40 seconds. The
push-stage CPU-only `make check` passed 2,636 tests in 130.08 seconds, Ruff, mypy, both wheels
and the model-wheel boundary check (exit 0). A local E-drive test startup stalled importing
Torch and was stopped before collection; verification moved to the approved Linux-local CPU
workspace. An initial full-check launch used the parent directory rather than its repository
subdirectory and failed before testing; neither attempt is counted as a pass. No unchanged
GPU canary, independent review agent or hash check was added for this scheduling-only change.

### Useful three-replica capacity and parallel native scoring

Owner-requested capacity expansion on 2026-09-06 keeps the same model-facing `1d09159`
policy, seed and task budgets. Coordinator concurrency increased from 12 to 24, then 96;
each of the three 22048 serving replicas now admits 48 requests instead of 12 and sizes
its KV pool automatically instead of the former 196,608-token cap. Static memory fraction
is 0.82; FA3, deterministic inference, disabled CUDA graphs and sampling stay unchanged.
Completed answers remain insert-once, never selected using scores. The capacity-only
reuse change passed five targeted tests and the CPU-only full check (2,638 tests).

The capacity continuation at 21:10 UTC retained all 776 completed A2 answers. A raw socket
port check incorrectly treated TIME_WAIT as a listener; checking actual listeners corrected
that maintenance error. Serving and generation resumed after roughly eight minutes, with
no completed answers lost. At 21:13 UTC, 103 additional answers had completed in about
142 seconds; that short, mostly static/HealthBench interval is not a whole-panel speed claim.
At 21:23 UTC, 920/926 answers existed and no scores had been read. Physical GPUs 1, 3 and 7
used approximately 68 GiB each with sampled utilization 87–88%. Allocated pool memory is
not the same as occupied KV tokens: the remaining long outputs use a smaller active batch.

Native scoring now has bounded case concurrency (default eight), after **all** answers
exist. A failed grader stops new admission while other active scores finish and persist;
resuming never regenerates finals or regrades completed records. HealthBench cases share
the existing least-inflight scheduler across observed identical judge replicas. Each case
and its rubric repairs stay on one replica, and a separate external judge endpoint is not
redirected. Actual private grader routes and independent grading costs remain recorded.
The rubric, answer extraction, native execution and aggregation rules do not change.

The scoring change passed 25 targeted tests in 1.81 seconds, followed by CPU-only
`make check`: 2,642 tests in 123.80 seconds, Ruff, mypy, both wheels and the model-wheel
boundary check (exit 0). An initial targeted launch omitted the existing environment
selection and failed before pytest started; it is not a test pass. No unchanged GPU canary,
duplicate full check, independent review agent or hash check was added. The prepared runtime
source is a complete prior coordinator snapshot plus the changed files, not an asserted
full Git archive. Generation remains on its existing process until all final answers exist.

The retained A2 run subsequently completed all 926 answers and native scores. It exposed
20 undecodable AIME peer-message attempts and a separate standalone-final-integer defect
in a real synthetic communication canary. Both boundaries were repaired after the run;
the frozen IID scores were not rewritten or rerun. The full results, exact limitations,
resource/cost evidence and final 2,659-test verification are recorded in
[the A2 retained-results report](STEP0_A2_IID_RESULTS_2026-09-06.md).

## Visible-memory and short-answer interface batch (2026-09-06)

Read-only inspection of the retained pre-repair run found QA outputs were commonly complete
explanatory sentences (median 25 words for HotpotQA, 20 for TriviaQA). Native EM/F1 scored those
sentences as submitted. The clean actor now explains the short-answer interface explicitly, with
an optional `Final answer:` line when the owner includes explanation. The parser and official
scoring formulas are unchanged; no reference-based span extraction or additional serializer is used.

A second reproducible defect discarded visible owner reviews and peer advice at the next native
environment step when Qwen thinking was off: only the empty native-thinking channel was retained
as working notes. Visible conversations now persist with author and public revision, labelled as
advice/attempts rather than environment facts. Token budgeting evicts whole chronological messages
and old observations, preserves the complete task/current observation, records omissions, and does
not overwrite the underlying archive. A rejected environment action remains rejected after advice.

A third defect could abandon an episode despite one owner call remaining, when there was no longer
room for both a peer reply and a final owner decision. Remaining per-turn and episode budgets are now
communicated; an undelivered peer request gets visible feedback and leaves the remaining owner call
available. No extra call budget is added. Peer contexts no longer inherit the owner's review-control
instructions. An aggregate counter records requests not delivered because of insufficient budget.

The synthetic regression file passed 21 tests before the final episode-budget variant was added.
The batched CPU-only `make check` then passed **2,664 tests / 85 warnings in 119.08 seconds**, plus
Ruff, mypy, both wheels and the model-wheel boundary check. Initial failures in newly written tests
were fixture mistakes (empty request, insufficient synthetic character budget, wrapper introspection),
not passed validations; they were fixed without additional review agents. Unchanged targeted suites
were not rerun after the final fixture-only correction; the integrated check covers the final state.
No hash verification was performed.

Two real fictional QA canaries used the existing three inference services: **2 model calls, 78
output tokens, 3.97 seconds**. Both returned short payloads (two words and one word); strict literal
synthetic answer checks passed **1/2**, because one response included a descriptive noun. That
failed check is retained; these examples are interface evidence, not IID quality or target attainment.

The owner's active development targets are now
[`step0_integrity_targets.yaml`](../configs/evaluation/step0_integrity_targets.yaml): HotpotQA and
TriviaQA F1 >75, MBPP Base+Plus >70, AIME/WebShop mean reward/ALFWorld success >=80, HealthBench
Qwen-local rubric mean >53.17, HumanEval >=92.1875 (118/128, subsequently accepted by the owner).
Secondary EM, WebShop success and ALFWorld splits
remain reported. Historical targets and historical scores are not retroactively replaced. The
already-inspected panel remains development-exposed, and changing acceptance targets never changes
the native scorer or actor inputs. The running `17b918b` evaluation is immutable and does not yet
contain this visible-memory/short-answer batch.

## Remove unsolicited review sections (2026-09-07)

The independent `28b850a` round has completed two full 128-task QA cohorts: HotpotQA F1/EM
**72.7180/54.6875**, TriviaQA **80.3239/75.7813**. TriviaQA exceeds the updated >75 F1 goal;
HotpotQA does not. Its complete WebShop cohort scores **15.5078 mean reward / 7.0313 success**,
not the >=80 goal. These native scores were saved by a CPU-only scorer against insert-once
candidates while unrelated long generations continued. The main scorer reuses those saved rows;
no additional model call, answer regeneration or alternative answer extraction was performed.
A short SQLite write transaction kept unfinished generations from entering the main scoring phase
while these rows were committed. No score was sent to an actor. The full round remains incomplete.

Format-only inspection, without selecting code by hidden tests, found **55 MBPP+ and 94 HumanEval**
nonempty submissions that were not standalone Python syntax. Many contain an appended `Review:`
section solicited by the generic controller prompt; four long HotpotQA outputs exhibit that same
suffix. The corrective batch removes the unsolicited review instruction rather than adding more
format restrictions. Explicit review controls remain compatible. It does not strip review prose
from saved answers to inflate their scores.

The code projection also accepts a single payload with an unmatched opening or closing boundary
Markdown fence, preserving the program and indentation and still rejecting different candidate
code blocks. HotpotQA renders every released passage verbatim as readable text and places the
question after the evidence; no passage is selected or shortened. Interactive actors now see the
remaining environment-action budget, which was previously a silent stopping limit. Native horizons,
model-call/output-token budgets, sampling, skills-off state and scorers are unchanged.

The batch passed **131 targeted regressions**. The first integrated check stopped at a missing
regex-result type annotation, before pytest. After that annotation-only correction, the final CPU
`make check` passed **2,673 tests / 85 warnings in 118.36 seconds**, Ruff, mypy, both wheels and the
model-wheel boundary check. The already-passed behavioral tests were not separately repeated for
the annotation. No review agent or hash check was used.

Three real synthetic interface canaries consumed **3 model calls / 112 output tokens / 3.54 s**.
Both code payloads parsed as Python and no final contained unsolicited trailing review text. The
same fictional QA case still returned a two-word noun phrase rather than the one-word literal
expected by that synthetic check; its literal failure remains recorded. These checks establish
interface behavior, not IID goal attainment or functional correctness of generated programs.

## Owner submission feedback and chronological state (2026-09-07)

The complete 128-task QA cohorts from `c25b6e4` score **75.1898 HotpotQA F1 / 57.8125 EM** and
**82.0413 TriviaQA F1 / 78.1250 EM**. Both F1 means exceed the owner's updated >75 targets on this
development-exposed panel. The same round's complete WebShop cohort remains below target at
**20.5134 native mean reward / 8.5938 success**, as does ALFWorld at **33.5938 success**. These
512 scores were saved once by the native
CPU scorer, using zero model calls; the main scorer reuses them. The full eight-task-family round
is not yet complete, and the small HotpotQA margin is not evidence of unseen generalization.

Answer-blind inspection found 15 completed code submissions whose owner emitted multiple different
code blocks. The interface returned an empty candidate without reporting its submission error,
although these cases had used only one of their available calls. The owner now receives that public
error and may submit its final program within the original call/token budget. The same rule covers
an ambiguous explicit short answer or integer. There is no separate transcriber, mandatory extra
pass, test feedback, selection among drafts, or fallback to another model's answer. Valid first
submissions return immediately; failures and actual repair calls are counted in the trace.

The interactive batch also:

- Interleaves visible model discussion with environment observations in revision order. Previously,
  all old proposals appeared after the latest real observation. The latest feedback now follows
  the attempted decision that produced it; no history is synthesized and no task text is dropped.
- Accepts the observed `Action` plus a named parameter line or a single literal `Argument` field.
  This only transports the supplied parameter, with no query rewriting or target substitution.
- Explains WebShop's public simulation and completion semantics: there is no real payment, and
  the displayed purchase control commits the selected item/options. A recommendation in chat is
  not an environment purchase. This supplies missing interface information, not a shopping policy.

The WebShop trace had 50 episodes ending on a rejected submission before the environment horizon;
ALFWorld had 64 ending on an out-of-surface decision. These observations motivate the interface
repairs but do not establish that all such failures will be fixed. Native horizons, sampler,
model-call/token budgets, skill state, final-answer ownership and scorers remain unchanged.

The terminal-feedback subset passed **136 targeted tests**. After batching the native-wire,
chronology and simulation-interface changes, **142 targeted tests passed in 0.49 seconds**.
The batch's CPU-only `make check` passed **2,684 tests / 85 warnings in 118.41 seconds**, Ruff,
mypy, both wheels and the model-wheel boundary check. Earlier successful commands were not
repeated on unchanged code; the subsequent targeted run covered additional interface changes.
No independent review agent or hash check was used. The earlier `17b918b` round has 925/926 saved scores: one HealthBench
rubric grading exhausted its JSON repair allowance. That infrastructure failure is retained as
incomplete, not converted into a zero or omitted from a claimed complete result.

## Message roles and retained complete development rounds (2026-09-07)

Both `28b850a` and `c25b6e4` subsequently completed all **926/926** native scores. The earlier
partial reports above remain time-specific observations, not missing tasks removed from a result.
These are independent full A2 rounds, with one final answer per task; no per-benchmark best-version
combination is reported.

| Native percentage | `28b850a` | `c25b6e4` |
|---|---:|---:|
| HotpotQA F1 / EM | 72.7180 / 54.6875 | 75.1898 / 57.8125 |
| TriviaQA F1 / EM | 80.3239 / 75.7813 | 82.0413 / 78.1250 |
| AIME2026 accuracy, 30 tasks | 66.6667 | 70.0000 |
| HealthBench Qwen-local rubric mean | 38.8198 | 42.4427 |
| WebShop native mean reward / success | 15.5078 / 7.0313 | 20.5134 / 8.5938 |
| ALFWorld success, 128 tasks | 20.3125 | 33.5938 |
| ALFWorld seen success, 97 tasks | 19.5876 | 37.1134 |
| ALFWorld unseen success, 31 tasks | 22.5806 | 22.5806 |
| MBPP+ Base+Plus pass@1 / base-only | 39.8438 / 48.4375 | 69.5313 / 80.4688 |
| Original HumanEval pass@1 | 17.9688 | 79.6875 |

The actor costs were respectively **7,801 / 7,017 model calls**, **1,182 / 1,653 peer calls**,
**1,609 / 2,081 native tool calls**, **37,262,972 / 30,554,012 input tokens**, and
**2,592,080 / 1,788,555 output tokens**. All seven non-AIME cohorts contain 128 tasks. Empty or
invalid submissions remain in these denominators. HealthBench uses the same local Qwen rubric
judge, not an official external-judge-comparable score. These development-exposed results do not
establish matched-control superiority or unseen generalization. In `c25b6e4`, only the two QA
F1 goals are met; MBPP+ is **89/128**, below the strict >70 requirement. AIME, HealthBench,
WebShop, ALFWorld and HumanEval also remain below their active targets.

The oldest `17b918b` result remains incomplete at 925 saved scores. One explicitly logged retry
of its sole missing HealthBench grade retained the immutable actor answer and unchanged grader
profile. It completed seven rubric responses, then exhausted transport retries on a read timeout:
10 requests, seven responses, 8,060 known input tokens, 640 known output tokens and three
unknown-usage calls. No replacement score was saved, and no further retry or timeout extension
was launched. This failed infrastructure attempt is not a zero or a complete HealthBench result.

The next communication batch addresses a reproduced role error. Framework capabilities and budget
notices were appended as new user turns, and peer/environment returns also impersonated users.
Inspection of the first five sorted public conversations, without selecting by score or reading
rubrics, found replies addressed to the framework or discussing what the researcher recommended
instead of simply continuing the original conversation. This is evidence of a message-boundary
problem, not proof that every wrong domain answer is an engineering defect.

Framework controls now belong to the leading system message; original user/assistant content and
order are unchanged. Public environment observations, peer advice, history reads and submission
feedback use tool messages. Peer/environment text is never elevated into system instructions.
The evaluated owner still chooses its final response and tools, and all existing budgets and
sampling parameters remain unchanged.

The initial role batch passed 144 targeted tests and a CPU-only full check (2,686 tests in
113.69 seconds), but its real native-template canary found that Qwen rejects multiple system
messages. That failed canary was retained and stopped before any recorded model request. The
helper now merges trusted controls into the single leading system message; it does not append a
new user request or alter the underlying conversation. The changed regression explicitly checks
this single-system invariant. One attempted targeted command referred to a nonexistent test path
and collected no tests; that invocation is not counted as successful validation.

After that template correction, **144 targeted tests passed in 0.49 seconds**, local Ruff passed,
and the necessary final CPU `make check` passed **2,686 tests / 85 warnings in 118.76 seconds**,
Ruff, mypy, both wheels and the model-wheel boundary check. The full check was repeated because
native-template compatibility changed after the previous success, not as an unchanged-code gate.
A real fictional dialogue plus scripted peer-transport canary then passed the native role and
thinking-off checks: **3 model calls, one peer call, 50 output tokens, 2.57 seconds**. Both owner
answers retained the public location, and the actual tokenizer emitted its native tool-response
wrapper and closed thinking prefix. This is interface evidence, not an IID performance claim.
No review agent or hash verification was used; documentation-only additions do not trigger another
full check. Only the already running three inference services were used, without new CUDA contexts.

## Complete split and failure diagnostics (2026-09-07)

The private frozen ALFWorld manifest already labels all 128 cases as 97 seen and 31 unseen,
but the pipeline summary exposed only their combined mean. The report now includes per-arm
seen/unseen counts, successes and native means, plus an explicit count for missing split metadata.
It never guesses a split from a task ID or path. Per-benchmark candidate/infrastructure/scored
counts and verifier versions are also retained. This is reporting-only: no actor message, candidate,
score formula, task membership, denominator or execution budget changes. Already running frozen
rounds are not hotpatched; their split aggregates can be derived separately from saved scores.

The reporting batch passed **37 targeted tests in 1.33 seconds**, covering single/paired execution,
resume behavior, failed-candidate denominators and missing split metadata. A formatting-only pass
followed; the same targeted tests were not rerun just for formatting.

The independent `3ab098e` round's complete cohorts now include HotpotQA **75.1898 F1 / 57.8125 EM**,
TriviaQA **83.9944 F1 / 79.6875 EM**, WebShop **37.1873 mean reward / 13.2813 success**, and
AIME2026 **24/30 = 80.0000 accuracy**. The two QA goals and AIME goal are met in that frozen round;
WebShop is not. At this observation, one ALFWorld actor is unfinished and code/HealthBench scoring
has not completed, so the four cohort values are not an eight-benchmark completion claim. The
new `7069132` role-corrected full round runs independently with no answer reuse.

The reporting batch's final CPU `make check` passed **2,687 tests / 85 warnings in 114.32 seconds**,
Ruff, mypy, both wheels and the model-wheel boundary check. No GPU canary was rerun for this
answer-free aggregation change. No independent review agent or hash verification was used.

## Native tool reference separate from returned state (2026-09-07)

The `7069132` round has four complete 128-task native-scored cohorts: HotpotQA **75.9587 F1 /
58.5938 EM**, TriviaQA **81.8179 F1 / 78.1250 EM**, WebShop **31.1217 mean reward / 12.5000 success**,
and ALFWorld **32.8125 success**. Its role repair improved generation throughput but did not solve
the interaction performance problem; WebShop regressed relative to `3ab098e`. Both rounds remain
retained, and the current round is not replaced with the earlier WebShop answers.

The native controller previously supplied command spellings without explaining ALFWorld command
effects. Its fixed API reference now describes navigation, observation, inventory, transfers,
opening/closing, object-state changes, toggling and slicing. These are the released
[ALFWorld command meanings](https://github.com/alfworld/alfworld/blob/master/alfworld/data/alfred.twl2)
and [action effects](https://github.com/alfworld/alfworld/blob/master/alfworld/data/alfred.pddl),
not an expert policy or a task-type recipe. Placement is distinct from cleaning/heating/cooling;
a confirmed command does not by itself certify task success. WebShop similarly explains the
search form, current-page controls and simulated purchase completion.

The framework-owned API reference now belongs to the single system message. Task text remains
user content, and current observations/actions remain tool content. Task-specific object names,
results and peer advice do not enter the system reference. The same interface is supplied to
single- and multi-agent conditions, with skills off. No target selection, automatic navigation,
extra actions, changed parser, added demonstration or modified environment horizon is introduced.
The owner has been asked separately whether to permit a declared higher-action-budget condition;
no budget increase has been made while that question is unanswered.

Local Ruff passed. The existing **144 targeted regressions passed**; the two new tests initially
failed because their fixture omitted the required input message. After correcting that fixture,
**both new tests passed in 0.38 seconds**; the unchanged 144 were not separately repeated.
The batch's full CPU `make check` passed **2,689 tests / 85 warnings in 119.20 seconds**, Ruff,
mypy, both wheels and the model-wheel boundary check. Three real Qwen synthetic interface canaries
all selected their intended fictional API actions, using **3 calls, zero peer calls, 203 output
tokens and 4.27 seconds**. They exercised instruction rendering and model-to-action transport,
not real environment execution or IID scoring. No review agent or hash check was used.

## Complete round and live reset task wiring (2026-09-07)

The `3ab098e` frozen round subsequently completed all **926 candidates and scores**. These
are eight results from that one round, not a combination of best results across revisions:

| Benchmark | Native score (%) | Current goal met? |
|---|---:|---|
| HotpotQA | F1 75.1898; EM 57.8125 | Yes |
| TriviaQA | F1 83.9944; EM 79.6875 | Yes |
| AIME2026 | 80.0000 (24/30) | Yes |
| MBPP+ hard | BaseANDPlus pass@1 74.2188 (95/128) | Yes |
| HumanEval | Original pass@1 90.6250 (116/128) | No |
| HealthBench | Local-Qwen rubric mean 42.5151 | No |
| WebShop | Mean reward 37.1873; success 13.2813 | No |
| ALFWorld | Success 40.6250 (52/128) | No |

HealthBench includes two candidate failures in its denominator. It is not an official
external-judge-comparable result. Meeting four goals does not complete the requested task.

A public-input-only check of the newer `3f37ec0` round found that all 128 ALFWorld actors
received the manifest annotation as their user task, rather than the task printed by the native
reset. Some annotations are harmless paraphrases; others add source-location constraints absent
from the live instruction. This is not evidence of 128 distinct semantic errors. The reproducible
wiring error is that the isolated actor bypassed the authoritative-reset helper already used by
the direct runner. The actor now uses that same helper before initializing its controller; owner,
peer, later turns and input traces share the live public instruction. No hidden goal fields,
answers, action policy, task membership, sampler, horizon or scoring rule are introduced.

Local Ruff passed; **16 targeted tests passed in 0.38 seconds**, covering the direct runner and
the isolated actor, single/multi-agent task delivery, later turns, unchanged WebShop tasks and
missing reset instructions. No separate GPU synthetic canary is needed for this deterministic
wiring change: the focused real ALFWorld rerun will exercise it with the native environment.
To avoid regenerating seven unaffected benchmarks after this ALFWorld-only fix, the next
development rerun uses the complete existing 128-case ALFWorld cohort. It remains separately
labelled as a focused cohort, not an eight-benchmark completion or unseen-test result.

The batch's pre-push CPU `make check` passed **2,694 tests / 85 warnings in 119.12 seconds**,
Ruff, mypy, both wheels and the model-wheel boundary check. Documentation-only additions do
not repeat that check. No review agent or hash verification was used.

The live focused rerun confirms that **128/128 initial owner requests now use the native public
reset task**. It uses the existing inference services on physical GPUs 1 and 3; GPU 7 remains
online for already-running requests but receives no new focused-cohort traffic because another
process increased that card's total memory occupancy. Per-episode budgets are unchanged;
coordinator concurrency is 64 rather than the full round's 96. Earlier ALFWorld aggregates are
`3ab098e`: seen **45/97 = 46.3918%**, unseen **7/31 = 22.5806%**; `7069132`: seen
**35/97 = 36.0825%**, unseen **7/31 = 22.5806%**. These are separate rounds, not pooled scores.

## Explicit parameter spelling and actionable transport feedback (2026-09-07)

Actual WebShop responses repeated the public operation signature and supplied its same-named
quoted argument. The transport now accepts that fully matched representation, preserving the
literal query/target. Conflicting parameter names, expressions, multiple calls and negated or
example calls remain unexecuted. It does not infer a search query or choose an item.

An ALFWorld paragraph ending in a bare command still does **not** become an action: mining its
last command would reinstate the prohibited historical fallback. However, this response used to
receive a misleading unavailable-surface error even when the mentioned command was on the menu.
It now receives a call-boundary error with the already-public submission interface. Wrong-resource
feedback names the public resource instead of repeatedly asking for a different menu action.
Both are ordinary interface feedback to the same owner inside its original call/token budget;
there is no additional serializer model, selected candidate, task recipe or environment action.

Local Ruff and **162 targeted tests in 0.50 seconds** passed. The final check also covers a small
subsequent narrowing of representation help to actual representation errors, so stale-state and
other unrelated errors are not misdescribed as formatting failures.

The pre-push CPU `make check` passed **2,705 tests / 85 warnings in 119.24 seconds**, Ruff,
mypy, both wheels and the model-wheel boundary check. The next real focused rerun covers
all 128 WebShop and all 128 ALFWorld cases; it does not regenerate the six unaffected domains
or reuse any answers. No hash verification or independent review agent was used.

## Declared non-thinking sampling condition (2026-09-07; not a bug claim)

The [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B) recommends presence penalty
1.5 with temperature 0.7, top-p 0.8 and top-k 20 for non-thinking general tasks, and notes that
presence penalties can reduce endless repetition but can also hurt performance. The earlier
frozen rounds use presence penalty 0 and retain that setting. Their long-tail responses are
not interrupted or replaced.

The next independent full A2 condition uses that **general** sampler uniformly for all eight
domains, changing only evaluated-model presence penalty to 1.5 and giving the decoding profile
a distinct identifier. It does not claim to use the card's separate math-specific recommendation.
Thinking remains off; call, output-token and environment-action budgets, seed 0, sources and
native scorers are unchanged. The HealthBench grader is not retuned. All 926 candidates are
generated afresh, with no voting, baseline fallback or answer reuse. This is declared sampler
tuning on a development-exposed panel, not proof that the previous sampler was an implementation
bug or evidence of unseen generalization.

The complete 30-case AIME cohort is scheduled first to overlap its long responses with other
domains; no individual task is prioritized using answers or difficulty labels. The coordinator
uses concurrency 64 and the existing GPU 1/3 endpoints after GPU 7's earlier external memory
pressure. That extra occupancy had cleared by the launch-time check; the declared two-endpoint
condition was retained, while GPU 7 continued serving older frozen work.
These scheduling/resource differences are reported rather than hidden as an exact throughput
ablation. No unchanged source tests or builds are repeated for this private runtime configuration.

The owner subsequently accepted **HumanEval 118/128 = 92.1875%**. Its development acceptance
threshold is now inclusive at that value, rather than strictly above the rounded backbone
reference. This changes no native score, candidate, denominator or sampler. No HumanEval-only
rerun will be launched merely to chase 119 successes; the already-frozen full sampling condition
retains all eight cohorts and its existing denominator.


## Completed follow-ups and grading reliability (2026-09-07)

The remaining native-API full round completed **926/926 candidates and scores**. Its
HumanEval result is again **118/128**, meeting the owner's updated acceptance threshold.
The separately declared presence-1.5 condition also completed **926/926**. Neither table
mixes answers or best scores from different runs:

| Native metric, percent | Native-API `3f37ec0`, presence 0 | `9a6c4c1`, presence 1.5 |
| --- | ---: | ---: |
| HotpotQA F1 | 75.9587 | 71.8627 |
| TriviaQA F1 | 81.8179 | 80.5992 |
| AIME2026 accuracy | 73.3333 (22/30) | 70.0000 (21/30) |
| HealthBench local-Qwen rubric mean | 48.6539 | 44.9323 |
| WebShop native reward | 24.0365 | 27.2368 |
| ALFWorld success | 36.7188 | 39.8438 |
| MBPP+ Base-and-Plus pass@1 | 72.6562 | 67.1875 |
| HumanEval original pass@1 | 92.1875 (118/128) | 91.4062 (117/128) |

The first condition meets four current development targets, the second only TriviaQA's.
The later sampler experiment does not withdraw the owner's acceptance of the earlier
HumanEval result, but its own 117/128 is not relabelled as 118/128. Presence 1.5 reduced
observed generation time to about 16 minutes, but did not provide the required quality.
Different scheduling and concurrency prevent treating this as an isolated throughput ablation.
There is no claim that all eight goals are met, or that low scores alone prove an implementation bug.

The completed transport-help focused cohorts score WebShop **25.9635** native reward
(SR **11.7188**) and ALFWorld **39.0625** SR. ALFWorld splits are seen **39/97** and
unseen **11/31**. The presence-1.5 full round's ALFWorld splits are **38/97** and **13/31**;
the native-API full round's are **38/97** and **9/31**. Complete aggregate-only results,
including the separately completed reset-only ALFWorld cohort, are retained in
[`step0_a2_separate_rounds_2026-09-07.json`](machine-results/step0_a2_separate_rounds_2026-09-07.json).
All question text, model answers, rubric details and per-task outcomes remain private.

### HealthBench failures remain failures, not selected replacement scores

The earlier `7069132` round retains **925/926** scores after an original rubric transport
failure. One explicitly recorded, grader-only retry used the same immutable answer and
unchanged judge profile on an already-declared equivalent service. All **30** retry requests
returned: **28** rubrics parsed successfully; the remaining rubric's first response was malformed
and its one schema repair ended at the unchanged **2,048-token** limit without valid JSON.
The 29 rubric strings were distinct. No second repair, answer regeneration, zero-score fill,
replacement of an existing score, or further automatic retry was performed. Retry cost is
**77,687 input / 6,199 output tokens**, with zero unknown-usage calls for this retry. The original
failed invocation's unavailable cost cannot be reconstructed retroactively.

Two additional, explicitly synthetic serving probes—plain JSON and the existing schema—both
returned in about **1.5–1.7 seconds**. They contain no benchmark material, are not scored,
and do not establish that every rubric request can finish inside its transport budget.

### Implemented transport and accounting repair

The repeated real failures justified one bounded reliability batch, not a change to grading:

- Each HTTP attempt now receives the smaller of the declared request timeout and the remaining
  transport deadline; the deadline is checked again after backoff. This prevents starting another
  full-length retry after the available budget is exhausted. SDK timeouts are I/O timeouts,
  not a guarantee against arbitrary process or operating-system stalls.
- Failed rubric invocations retain known token usage, request/response counts, unknown-usage
  counts and elapsed time in the private coordinator trace. They still produce no native score
  and no feedback to evaluated actors. A later completed report includes previous failed costs
  separately and in its totals, with one trace scan rather than a scan per candidate.
- A blocked second semantic repair is no longer counted as an actually admitted repair.
- Transport classes moved out of the oversized HealthBench module; existing import paths remain
  supported. Judge model, prompts, rubrics, token limits, temperature and one-repair policy remain
  unchanged. Completed frozen runs were not hot-patched.

Validation: local Ruff passed; **49 targeted tests passed in 1.99 seconds**. The initial CPU
`make check` passed **2,713 tests / 85 warnings in 119.05 seconds**, Ruff, mypy, both wheels
and the model-wheel boundary check. Subsequent examination of the actual exhausted repair
revealed the blocked-repair counter issue; **17 affected tests passed in 0.78 seconds** after
that fix. The single-pass accounting refinement passed **20 affected tests in 1.83 seconds**.
The final pre-push CPU `make check` passed **2,713 tests / 85 warnings in 114.19 seconds**,
Ruff, mypy, both wheels and the boundary check. These two complete checks cover different
code states; unchanged documentation/export additions do not repeat them. No independent
review agent, hash verification, additional evaluation arm, seed or training run was used.

The existing SGLang services on physical GPUs **1, 3 and 7** remain resident independently
of completed evaluation coordinators and are not released when evaluation ends. The owner
reaffirmed this requirement and explicitly selected **gpt-5.6-terra / xhigh** for the read-only
GPU monitoring subagent. No other user's process is modified. Further action-budget changes
remain a separately declared experiment, pending the outstanding owner decision.
