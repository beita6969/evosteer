# Step-0 no-thinking integrity and communication repair

Implementation of `SKILLEV_STEP0_NO_THINKING_INTEGRITY_COMMUNICATION_REPAIR_PLAN_2026-09-06.md`.
The plan inspected `9a2d3a2`; this batch starts from `216edc2` and preserves its newer
HealthBench deadline and failed-grader-cost repairs. Validation, the frozen evaluation
attempt and its remaining limitations are recorded below; no unexecuted result is claimed.

The subsequent owner-authorized single-owner/no-voting/AIME-thinking plan is merged into
the [pending repair worklist](STEP0_SINGLE_OWNER_REPAIR_WORKLIST_2026-09-07.md).
Its future AIME thinking-on condition does not change the historical thinking-off runs
recorded here; listing the work is not a claim that its provenance/resume repairs are complete.

## Scope and scientific boundaries

- `idea.tex` and `AGENTS.md` are unchanged. No optimizer, LoRA, TTB or training changes.
- The selected real evaluation remains **A2 only**, Qwen3.5-9B, step zero, skills off,
  native thinking off, optional advice peers, and one seed. A1/A3 are supported API
  conditions, not additional authorized experiments in this batch.
- Seven IID domains have 128 tasks each; AIME2026 has all 30: 926 per full arm.
- The owner-facing name remains **MBPP+ hard** under the current project instructions.
  Its actual population is EvalPlus MBPP v0.2, Base AND Plus; there is no invented hard split.
- No vote, selector, baseline-answer fallback, score-guided regeneration, private-answer
  context, handwritten task recipe or automatic environment action is introduced.
- Low scores do not establish cheating or a transport defect. Correct interfaces do not
  guarantee that optional peers or generic advice improve a backbone's native score.

## Implemented findings

| Finding | Implementation |
|---|---|
| F01 | Keyword-only bindings; actual panel position and skill budget travel in separate fields; zero budget remains zero. |
| F02 | One closed, typed YAML/JSON arm decoder; explicit v1 compatibility and complete v2 fields, including precomputed context and legacy state. |
| F03 | Required explicit arm; clean and named historical-reproduction entry points cannot silently switch architectures. |
| F04 | Control attempt/decode/delivery/reply/resolution records; unresolved requests and missing observed boundaries cannot become a communication PASS. The real run exposed a remaining terminal/native-output omission, corrected separately in `871bf92` below. |
| F05 | Catalog of actually implemented owner tools, paged history and optional advice peers, with permissions, effects and costs; no fictitious TriviaQA search. |
| F06 | Advice retrieval uses the source task or an explicit query, not framework prompts, task IDs, benchmark labels, panel positions or scores; common-word abstention and whole-body budgets. |
| F07 | Advice-only peer permissions are separate from shared native API semantics; original requests, public state, references, revision and reply parentage are retained. No forced peer invocation. |
| F08 | Immutable public views; complete chat-template packing including framework controls, current observation/actions, recent feedback and output reservation. Whole historical entries move to paged environment/discussion/tool-result archives. Required input never gets truncated. |
| F09 | Owner, peer, review and admitted repair calls share one ledger; failed/unknown transport is not refunded. Decisions carry their pre-generation revision and are rechecked by the serialized broker. |
| F10 | Owner-explicit final payloads are distinct from drafts; raw response and projection are retained. Multiple active finals remain ambiguous. Legacy whole-response projection remains a separately identified compatibility path. |
| F11 | Syntax repairs are actual budgeted model calls, not free transport. Deterministic projection, model repair, self-discussion and private scorer retries remain distinct. |
| F12 | Closed native metric schemas, fraction/binary units, required secondary metrics, cross-metric relations and frozen verifier binding. Health raw negative items are preserved and only the aggregate is clipped. |
| F13 | Required/optional/source/runtime public fields and structural validation; private-label substitutions cannot change actor export. Hotpot retains ten passages and Trivia retains released RC context. |
| F14 | Origin-labelled SQLite events; actor RPC cannot write native/scorer/transport authority. The broker checks final scope, policy, parser, owner message and the projection of the actual served owner output. Native action logs come from acknowledged executions. |
| F15 | [Separate run index](STEP0_RUN_INDEX.md), complete denominators, versioned conditions and explicit incomplete records; historical paired notes are not a claim of an active or completed comparison. |
| F16 | Existing panels remain development-exposed, including prior syntax/communication debugging exposure. Old scores are never overwritten with a new parser or combined into a best-of headline. |

The peer topology remains owner plus optional advice-only peers. Autonomous tool-using peers
would be a separately declared condition, not a silent A2 change. Source tool semantics are
available to peers without telling them they can execute those tools themselves.

## Persistence and execution

Existing trace rows without provenance stay `legacy-unverified`; migration does not upgrade
them into trusted evidence. Read-only old databases remain readable. Scope/stage indexing
avoids repeatedly scanning the entire trace table during a full-panel report. This is ordinary
SQLite provenance and indexing, not a signature, attestation or digest-check protocol.

An acknowledged native action survives a later model-budget failure. Public episode close
does not accept success/reward from the actor. Unacknowledged effects are not replayed.
The real environment supplies the outcome privately to the scorer after owner submission.

Whole-template packing uses measured token counts and logarithmically many tokenizer calls
when old entries must be removed. A too-large required source/current surface produces
`context-capacity-incomplete`, not a silently shortened task or action menu.
Long owner drafts remain available verbatim in the discussion archive; they do not force
the current interface feedback out of the prompt. HumanEval candidate failures retain their
native syntax/runtime/test/timeout/resource category; evaluator outages produce no score.
EvalPlus verdicts must be actual booleans. The native scoring formulas are unchanged.

## Verification record

Development validation is limited to affected modules. A local WSL targeted pytest command
timed out after 150 seconds without output; local targeted mypy likewise timed out after
90 seconds. These commands were moved to the approved Linux CPU validation environment
instead of repeatedly interrupting and diagnosing WSL startup.

- First integrated CPU set: 252 passed, 12 failures identified stale fixture/API assertions
  and missing shared peer API semantics. The latter was fixed in the implementation.
- After integration repairs: 194 passed, two fixture role-classification assertions corrected.
- Latest communication plus actual isolated-broker boundary set: **45 passed**.
- Targeted type check: **8 source files passed** before the final accounting refinements.
- Expanded affected set: **318 passed**, one stale scorer-trace test double then corrected;
  its affected scoring/conformance set: **15 passed**.
- Real isolated actor/broker tests, including Chinese and literal-function click syntax,
  native acknowledgement and peer receipt of the resulting public page: **5 passed**.
- Context and shared-budget tests after oversized-draft handling: **17 passed**.
- Final native-score/aggregation/failed-grader-cost set: **53 passed**, with a targeted
  **3-source-file mypy pass**.
- Whole-suite execution then found three failures (2,835 passed): one real source-type
  incompatibility, one stale verifier fixture and the project's no-broad-fallback boundary.
  The actual 128 TriviaQA sources were checked structurally: they contain lists of public
  passages, now all preserved in order. Failed/cancelled call accounting now uses `finally`
  without catching or replacing the original exception. The affected source, communication,
  cancellation, broker and boundary set subsequently passed **90 tests**.
- New tests cover public-field noninterference, arm schema, binding budgets, whole-template
  packing, paged history, explicit final ownership, native metrics, event origins and the actual
  isolated actor/broker boundary with synthetic serving and native-environment fixtures.

The first final CPU `make check` passed formatting/lint and identified four type-narrowing
errors in the new metric decoder; these were corrected. Its next execution passed the
498-source-file type check and found the three full-suite issues above. The final rerun on
the approved Linux CPU host completed successfully at **2026-09-07 07:10:01 UTC**:
**2,841 tests passed, 85 existing PEFT warnings, 126.77 seconds for pytest**; full Ruff
format/lint, mypy on 498 source files, both wheel builds and the model-wheel private-boundary
check passed. CUDA was disabled for this validation.

The main thread completed the combined implementation/diff review. Unrelated pre-existing
hardware/run-control edits were preserved and excluded from the repair commit. No unchanged
targeted set was rerun solely for review, and no full local WSL check was attempted. Full
checks were concentrated at this integration milestone; failed stages were followed by
affected tests and the final complete rerun, not per-function gates.

## Frozen real execution (`7926eb9`)

The complete source archive passed the actual 926-record public-source preflight with zero
model calls. A preliminary uncompressed transfer timed out and left an incomplete directory;
it was not used for evaluation. A complete compressed archive was imported into a new
directory without digest checks.

The first real canary retained 9/15 candidates and no scores. All six WebShop environments
failed before reset, action or model execution because the launch environment omitted
`JAVA_HOME`; the installed Java runtime itself was present. After an official-environment
reset smoke test succeeded, an attempted resume correctly refused the changed frozen runtime
environment before any further generation. Neither the frozen controls nor the nine retained
candidates were overwritten. A new run froze the corrected environment from the beginning.

`canary-a2-integrity-repair-7926eb9-20260907T073158Z` completed **15/15 candidates and
15/15 scores**, from 07:32:01 to 07:34:37 UTC on 2026-09-07 (about 156 seconds).
It used six disjoint WebShop tasks, six ALFWorld train tasks covering the six task types,
and one disjoint task each for MBPP+, HumanEval and HealthBench. This is an interface
canary, not an eight-IID performance estimate; single-task scores remain private.

- 172 owner model calls; 528,529 input and 15,574 output tokens.
- 159 native actions and 159 acknowledgements; no missing acknowledgement, stale surface,
  feedback mismatch or unobserved required boundary was reported.
- Ten budgeted communication repairs, all resolved. These calls are included in the 172,
  not labelled free serialization or removed from model cost.
- All 15 episodes voluntarily used no peer. The real canary therefore does **not** establish
  real peer-call coverage; that boundary passed the separate isolated-broker integration tests.
- Health scoring used 19 local-Qwen requests, 48,149 input and 1,941 output tokens,
  with zero failed invocations, semantic repairs or unknown-usage calls.
- All five represented native metric contracts passed; there were no missing scores.
  ALFWorld train canaries do not have the full panel's seen/unseen population split.

The full run `full-a2-integrity-repair-7926eb9-20260907T073806Z` started at
**07:38:09 UTC** with A2 only, 926 planned records, generation concurrency 96 and
scoring concurrency 8. It uses the three existing inference services, the same seed,
decoding, per-episode budgets, native horizons and local-Qwen grader. All candidates
must be retained before scoring. It reuses no earlier answers and is not a paired
backbone comparison. The attempt ended after **81.91 minutes**, retaining **926 candidates
and 925 definitive scores**. One HealthBench rubric remained malformed after its original
call and one permitted semantic repair. No further grader invocation was launched, the
missing score was not replaced with zero, and no 127-case HealthBench mean is published.
The [separate result and limitations](STEP0_INTEGRITY_REPAIR_RESULTS_2026-09-07.md) retain
all denominators and distinguish native performance from execution/communication status.

## Real-run reporting correction (`871bf92`)

The real run revealed that terminal-parse failures and rejected native decisions were
counted but not included in the unresolved-output state. An earlier accepted action, or
a successful repair elsewhere, could therefore hide a later unresolved output. The main
thread corrected this remaining F04 defect: chronological terminal/native resolution now
reports `unresolved-output-failure`, and unused peer execution is explicitly `not-observed`.
Accepted native execution still requires its separate acknowledgement; a wrong but valid
action is not relabelled a transport defect.

The original `7926eb9` candidates, scores, events and communication values are unchanged.
The aggregate export preserves those original communication values alongside separately
labelled **post-run `871bf92` diagnostics**. This reporting-only analysis made zero model
or scoring calls and is not a fresh evaluation of a new actor revision.

Targeted communication/pipeline tests: **26 passed**; targeted mypy: **two source files
passed**. Because reporting code changed after the earlier full verification, one necessary
Linux CPU `make check` followed: **2,845 tests passed, 85 existing PEFT warnings, 123.40
seconds for pytest**; Ruff format/lint, mypy on 498 source files, both wheels and the
model-wheel private-boundary check all passed. No unchanged full suite was rerun just
for documentation or aggregate export, and the main thread reviewed the combined diff.

The engineering implementation does not establish all-goal acceptance. The full run used
**zero actual peers**, retained one unresolved AIME final and unresolved native action
decisions, and lacks one definitive HealthBench grade. Further grading beyond the frozen
repair allowance requires a separately approved, labelled scoring condition; it cannot be
silently resumed until a score appears. No extra evaluation or training was started.

No audit/review agents or digest checks are used. Two implementation agents contributed early
in the batch; they were stopped when the owner requested main-thread-only coding. Thereafter
all code, integration and validation are handled by the main thread; subagents only watch GPUs.

## Follow-up: observed signature and code-fence defects

Read-only inspection of the frozen actor outputs found two additional, reproducible
transport defects, not evidence that every low score is an interface failure:

- Eight WebShop decisions repeated the published `search[query]` signature with a separate
  `Arguments` field. The whole signature was incorrectly treated as a tool name. Exact
  named placeholders now bind only the supplied literal argument; concrete targets,
  public permissions, revision checks and ambiguity rejection remain unchanged.
- One HumanEval response reopened a Python fence before closing its first draft. The old
  regex mistook that second opening for a closing delimiter and submitted only the draft.
  Whole-line fence decoding now leaves such an output unsubmitted; the owner can clarify
  its own final within the existing shared budget. No AST/test-based program selection,
  evaluator-triggered regeneration or source rewriting is used. Legacy projection is
  versioned at `@3`; explicit-final projection remains `@2`.

Condition `A2-no-thinking-integrity-repair@6` identifies these changes. Real frozen prompts
contained the advertised peers and sufficient call budgets; zero actual peer requests
are not evidence of a dropped request, nor are they evidence of multi-agent benefit.
A focused rerun of the changed interfaces will be recorded separately, never spliced into
an eight-IID result. The incomplete HealthBench grade is not retried by this work.

Follow-up verification: local Ruff formatting/lint passed. The initial CPU targeted set
passed 140 tests and exposed one new test's incorrect use of a facade-only fixture; that
assertion was replaced by the actual isolated actor/broker path. The resulting signature
and broker set passed **22 tests**; final-payload, ownership, message and broker tests then
passed **70 tests**, with targeted mypy passing all three changed source modules. Read-only
application of the new parser to the old public outputs found eight corrected WebShop
rejections in one episode and one newly unsubmitted malformed code response in each of
HumanEval and MBPP+. This changed **no stored candidate or score** and made no model calls.

Both complete checks ran on the approved CPU host with CUDA disabled: the signature-only
milestone passed **2,862 tests** (118.16 s), then the newly discovered code-fence repair's
final check passed **2,870 tests** (121.74 s). Each included complete Ruff, mypy and both
wheel builds; 85 existing PEFT warnings remained. The second full check covered newly
changed code, not an unchanged-state review rerun. Main-thread self-review only; the sole
active subagent watched existing inference services. No digest checks or retired gates.

The eight-task nonfinal canary `canary-a2-signature-fence-db680d9-20260907T094538Z`
completed in **141.79 s**, retaining eight candidates and eight scores. It made 64 model
calls (250,598 input / 10,437 output tokens), acknowledged all 53 native actions and resolved
all nine budgeted interface repairs. No peer invocation was observed. Canary single-task
scores stay private. The subsequent whole-cohort focused round is a separate run; the
other five benchmarks, including the missing HealthBench grade, are not regenerated.

The focused run `focused-a2-signature-fence-db680d9-20260907T094943Z` ended with
**384/384 candidates and scores**, in **1,454.48 s** (24.24 min; about 950 records/hour
end to end). WebShop native reward is **21.8229/100**, SR **11/128**; MBPP+ Base AND Plus
is **91/128**, Base **109/128**; HumanEval is **110/128**, with 15 native test failures
and three runtime errors, no syntax or infrastructure failures. These scores are separate
from `7926eb9`, not matched causal estimates or a substitute eight-IID headline.

All 1,330 model requests completed and all 869 native actions were acknowledged. Cost was
4,180,222 input and 333,545 output tokens, including 189 budgeted interface-repair calls.
Four terminal failures resolved through owner submission. WebShop retains 128 unresolved
rejected action attempts and the communication status correctly remains
`unresolved-output-failure`; zero route mismatches do not override this. No real peer call,
skill, selector, fallback or answer reuse was observed. Only MBPP+ meets its current goal
among these three cohorts; the all-eight goal remains unmet.

One WebShop episode dominated the late tail. The main thread reported the window rate
falling below 40 records/hour as **needs human decision**, retained the frozen budget and
waited for its completion rather than clipping work. During this round, GPU7's remaining
memory fell below 1 GiB; all three registered services stayed healthy. Only the existing
22048 physical GPUs 1/3/7 were used, with high utilization during the bulk phase and the
services kept resident afterward. Both evaluation coordinators and both CUDA-disabled
CPU check processes finished; no new inference daemon or training job was created.
Final result-only documentation does not trigger another unchanged-source full check.

## Follow-up: advertise the actual submission channel (`1119025`)

Read-only inspection found no unparsed peer-address headers in either preceding real
round, including after prose or Markdown. The hypothesis that peer requests were lost
by header placement was not supported, so no speculative parser rule was added.

A separate, concrete capability contradiction was present: interactive owners were
told they could submit a final chat answer, although the broker accepts only native
environment actions. The shared owner prompt and malformed-control feedback now name
the actual submission channel: native actions for interactive tasks, owner finals for
static tasks. The optional peer description consistently says peers cannot submit on
the owner's behalf. This does not prescribe an action sequence, force collaboration,
change a target, add a skill or reduce the existing per-episode budget.

Condition `A2-no-thinking-integrity-repair@7` records the change. Actual isolated
actor/broker tests cover both native environments and static final submission after
malformed control feedback. The affected test set passed **55 tests** in 4.81 seconds;
targeted mypy passed two source modules and local Ruff formatting/lint passed. The
necessary final Linux CPU `make check`, with CUDA disabled, passed **2,873 tests**
(133.76 seconds, 85 existing warnings), full Ruff/mypy, both wheels and the model-wheel
private-boundary check. No review agent, digest check or unchanged full-suite rerun.

The new real canary used the existing physical GPUs 1/3 for actor and grader traffic.
GPU7's registered service remains resident, but new traffic excludes it because only
724 MiB remained free at the main-thread launch check. Generation concurrency is 64
(planned full round; 16 for the completed disjoint canary), scoring concurrency eight;
model, seed, decoding,
per-episode budgets and native horizons are unchanged. All three existing services
remain online; no extra GPU, inference daemon or training process is started. This
separate condition neither retries the old missing HealthBench grade nor reuses answers.

`canary-a2-submission-channel-1119025-20260907T103540Z` completed **15/15** in **121.81 s**,
with 152 model calls (441,168 input / 14,525 output tokens) and all 135 native actions
acknowledged. There were 13 budgeted interface repairs, but eight rejected action attempts
remained unresolved. Health grading completed with 19 requests and no semantic repair;
actual peer calls were zero. The full round was **not launched**: inspection first found
that those eight failures rejected standalone native calls following discussion solely
because the owner had omitted an additional `Action:` label. Canary scores stay private.

## Follow-up: unique standalone native API call (`5fa9b0e`)

The main thread fixed this observed serialization defect rather than launching another
full panel with known false rejections. A unique standalone WebShop/public-search API
call after discussion now supplies its own submission boundary. Multiple calls, quoted
examples, negation, trailing discussion, unavailable targets, and channel/state mismatches
remain unsubmitted; no target is inferred or substituted. ALFWorld ordinary command words
are not newly mined from discussion. The WebShop capability reference makes the optional
label explicit. Condition `A2-no-thinking-integrity-repair@8` records these changes.

Local Ruff passed; the affected parser, prompt, submission and actual isolated-broker set
passed **200 tests** in 5.56 s, and mypy passed both changed source modules. A new final
CPU check covered this subsequent public-interface change: **2,890 tests passed**
(124.57 seconds, 85 existing warnings), with full Ruff/mypy, both wheels and the
model-wheel private-boundary check passing. This was not a duplicate review of the
already verified `1119025` source. No code-writing or review subagent was used.

Read-only application of the new decoder to the preceding canary accepted exactly the
eight previously rejected standalone calls; the six unavailable-action rejections stayed
rejected. This diagnostic made zero model calls and changed no stored candidate, native
execution or score. A fresh canary, rather than substituted historical answers, tests the
actual new actor/broker path.

`canary-a2-standalone-native-5fa9b0e-20260907T104849Z` completed **15/15** in **137.84 s**.
All 156 model requests completed (493,256 input / 16,451 output tokens), all 144 native
actions were acknowledged, and all nine budgeted communication repairs resolved.
The status is `complete-with-repairs`; actual peer use remains zero, not evidence of
multi-agent benefit. Health grading used 19 requests with no semantic repair or failed
invocation. The separate canary exports retain both the unsuccessful `@7` communication
condition and this `@8` condition, without publishing canary per-task scores.

The independent full run `full-a2-standalone-native-5fa9b0e-20260907T105437Z` started at
**10:54:40 UTC** on 2026-09-07 with A2 only, 926 planned records, concurrency 64 and
scoring concurrency eight. Its complete source archive and two-replica routing are frozen
before generation. It reuses no previous answer or grade. The old incomplete HealthBench
run remains incomplete; this run is not a retry or a replacement of that missing grade.

The full round subsequently completed **926/926 candidates and scores** in **61.59
minutes**. Its [separate native result](STEP0_STANDALONE_NATIVE_RESULTS_2026-09-07.md)
meets only the HotpotQA, TriviaQA and MBPP+ criteria. All 4,053 model calls and all
2,900 native acknowledgements are accounted for, but AIME retains one unresolved
terminal failure, WebShop 72 unresolved action attempts and ALFWorld 120. Actual peer
use is still zero. This is not all-goal acceptance or proof of multi-agent benefit;
the complete new HealthBench result does not fill any older run's missing grade.
No source code changed after the successful final check, so result documentation
does not trigger another complete suite, build or review cycle.

## Explicit model-native tool-template condition

The [native-tool protocol note](STEP0_NATIVE_TOOL_PROTOCOL_2026-09-07.md) describes
`A2-no-thinking-native-tools@9`. It fixes two newly observed syntactic false
rejections and exposes existing optional peer/history/native-action interfaces
through Qwen3.5's actual tool template, with an owner-only final envelope for static
tasks. The new v3 control is frozen separately; old conditions, budgets, answers and
grades are unchanged. It is not a claim of observed multi-agent benefit or all-goal
acceptance. Code is written only by the main thread, with resource-only watching.
