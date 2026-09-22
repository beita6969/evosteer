# OOD regression repair, fresh original panels (2026-09-12)

## Evidence and changes

The preceding A10 run is retained, including regressions. Its inspected seven
64-case panels are development evidence, not an untouched final evaluation set.
The next condition, A11, keeps all 448 original IDs and one seed; no answer is
selected from competing runs. The single frozen Qwen3.5-9B owner remains the
only answer/action producer. No skills, consultants, training updates, reference
feedback, hidden-test feedback or automatic environment actions are introduced.
`idea.tex` and `AGENTS.md` remain unchanged.

- **QA:** A10 had no empty QA submissions or terminal parse failures, but
  MuSiQue F1 fell from 70.24% to 54.60%, and NQ EM from 23.44% to 17.19%.
  The earlier allowance for brief reasoning had been replaced by instructions
  forbidding discussion. Restore brief owner analysis before one explicit final
  short-answer line, without changing final-answer projection or native metrics.
  This is a plausible contributor, not a proven sole cause of the regressions.
- **ScienceWorld:** A10 selected trajectories contain 5,713 native rejections.
  A 200-feedback repeated-command trajectory was re-encoded with the actual
  local model tokenizer: all observations survived, and the resulting 33,521
  input tokens match the logged count. Feedback loss and accidentally enabled
  native thinking do not explain that trajectory. Replace the direct-call-only
  instruction with one short observation-grounded sentence followed by the
  owner's complete action. Keep literal command parsing, numeric disambiguation,
  native observations and negative final scores unchanged. No task solution,
  object chooser, valid-action oracle, automatic recovery or peak-score rescue.
- **LiveCodeBench:** Nine A10 trajectories had no complete final submission.
  Inspected capped outputs spent the 12,000-token allowance on draft code,
  repeated dry runs and self-checks before their final program was complete.
  Ask for at most one short introductory paragraph and one complete program,
  rather than unlimited explanatory drafts. The framework still cannot assemble
  snippets, choose alternative programs or repair algorithms.
- **Judge metadata:** Remove stale Omni `high` labels from new runtime settings.
  A10's effective API judge was already Luna medium, as recorded in its ledgers;
  this metadata correction does not retroactively change the judge or the score.

Single-owner reasoning before an action is distinct from native thinking.
[ReflAct](https://arxiv.org/html/2505.15182v2) provides this agent-level comparison;
the [Qwen template](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/chat_template.jinja)
supports text before native tool calls. We do not copy SwiftSage's automatic
action repair, gold paths, demonstrations or modified failure scores.

## Frozen generation and scoring condition

- One adapter-free Qwen3.5-9B owner, native thinking off, skills off, seed 0.
- Code: one uninterrupted first response with a 12,000-token cap and the same
  total episode allowance. Other tasks: 8,000 output tokens total. Actual
  remaining and per-call limits stay visible to the owner. No chunking/reserve.
- A10 decoding retained: code/Omni temperature 1.0, top-p 0.95; other tasks
  0.7/0.8; top-k 20, presence penalty 1.5, repetition penalty 1.0.
- Static concurrency 32. ScienceWorld concurrency **16**, explicitly adopting
  the previous successful deadline-recovery scheduling condition instead of
  repeating batch-32 cancellations. No other generation phase overlaps it.
  ScienceWorld keeps 256 tokens/call, 200 native interactions, 600 seconds per
  episode and the unchanged no-simplification environment.
- The existing official MuSiQue/FiD metrics, isolated APPS/LCB execution checkers,
  Omni official judge prompt and LiveMedBench per-criterion rubric are retained.
  External judges remain **gpt-5.6-luna / medium / official API**, terminal-only.
  [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna).
- Only the newly requested physical GPU7 SGLang on endpoint 22049 is used.
  GPU6's temporary service was stopped; co-resident processes were left intact.
  Model and KV-cache memory settings differ from A10's retired GPU5 service and
  are recorded privately. This is not a matched throughput experiment.

Seven reported domains are MuSiQue-Ans, NQ-Open, Omni-MATH, LiveMedBench,
ScienceWorld, LiveCodeBench and APPS Introductory. GPQA is not reinstated.
All attempts, cancellations, missing scores and actual costs remain visible;
no automatic replay follows a request with unknown serving usage.

## Validation and execution

Local Ruff lint/format checks passed for the three changed Python files. The
local targeted pytest command hit its 180-second wall limit without producing
a test result; local targeted mypy likewise exceeded 90 seconds. Neither is
counted as a pass. The same targeted pytest suite on the approved Linux CPU host
passed **137 tests in 8.29 seconds**; mypy is covered by the remote full check.
Full `CUDA_VISIBLE_DEVICES="" make check` runs once on the approved CPU host,
without overlapping timed code scoring. No independent review subagent or hash
check is used. The dedicated Luna agent handles Git only; there is no formal
training to log to W&B. Runtime details and licensed evidence remain private.

Fresh generation started at **20:39:19 UTC** on the existing GPU7 service. The
private coordinator PID is 615533; the code-phase launcher is 615536. A separate
CPU-only process 615534 waits for timed code scoring to finish before the full
check. Actual owner outputs have been observed, not merely a launch command.
Measured A11 scores remain pending; this note does not claim that the repairs
improved scores or met the reference thresholds.

## Owner redirection: stop adding prompt restrictions

At 20:53:40 UTC, the owner questioned restrictive skills, prompts and scoring
interfaces. Only the phase coordinator was paused; already-started static
generation/scoring was left to drain with its original settings. ScienceWorld
had not started and will not auto-start under A11's one-sentence instruction.
The frozen A11 source remains private; subsequent worktree edits are not silently
deployed into this run.

Completed A11 code scores are APPS **53/64 (82.81%)** and LCB **30/64 (46.88%)**.
Thus the shortened LCB explanation did not establish an improvement. A diagnostic
comparison of first replies found 19 LCB responses rejected by the strict owner
decoder; 11 normally stopped replies yield syntactically valid Python under the
[upstream final-fence rule](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/utils/extraction_utils.py).
This is a transport mismatch, **not proof those 11 programs pass hidden tests**.
No answer was replaced or rescored using the diagnostic extraction.

Current code/static evidence has zero skill discovery, reads, injections,
invocations and peer calls. The restrictive historical seed builder is separate
from the current optional-advice defaults; this does not prove all historical
skill-on runs were compliant.

The worktree now removes paragraph/sentence limits, exact final-label demands
and the instruction forbidding a supported submission tool. This is preparatory:
the submission parser still needs correction before another condition launches.
Multi-agent reintroduction requires resolving the conflict with the owner's
preceding explicit single-owner requirement; it has not been silently enabled.

The old static phase finished normally before shutdown. At 21:09:30 UTC the
paused coordinator was terminated without starting ScienceWorld, and its
temporary API credential was removed. GPU7 SGLang remained healthy and resident.
A11 is explicitly **partial: 384/448 generated and scored**, not a completed
seven-domain evaluation. The six 64-case results retained without replacement are:

| Metric | A11 |
| --- | ---: |
| MuSiQue F1 | 66.02% |
| NQ closed-book EM | 21.88% |
| Omni Luna-medium accuracy | 62.50% |
| LiveMedBench rubric mean | 45.10/100 |
| LCB pass@1 | 46.88% |
| APPS Intro pass@1 | 82.81% |

There were 412 owner requests and 412 responses, 598,527 input tokens and
980,544 output tokens, with no unknown owner usage. All 64 Omni and 64 medical
cases completed terminal judging (64 and 360 API calls respectively).
These scores belong to frozen A11, not the later relaxed worktree prompts.

The frozen A11 CPU `make check` passed **4,451 tests, 12 skipped**, Ruff, mypy,
both wheel builds and the wheel-boundary check. The subsequent prompt-only
worktree changes passed Ruff lint/format, but have not received the next batch's
functional/full validation. No commit/push of unfinished changes is claimed.

## A12: open instructions, deterministic final-code transport

The observed final-code rejection is now corrected for all Python submission
paths, including owner and trusted-broker projection. The last fenced block or
last explicit code declaration is submitted unchanged. Earlier examples, drafts
and language labels no longer trigger mandatory regeneration. There is no syntax
or hidden-test-based selection, block concatenation, execution-based repair or
fallback to an earlier executable program. Literal backticks remain source data;
an unfinished/empty final carrier cannot silently select an earlier draft. This
follows LCB's final-block convention with those documented transport differences,
not a claim that the entire APPS evaluation is LCB-equivalent.

MuSiQue/NQ instructions describe short-answer semantics without requiring a
particular final label. Coding and ScienceWorld instructions allow reasoning and
available tools without paragraph/sentence limits. No skill or consultant is
introduced. ScienceWorld also removes the separate 256-token response ceiling:
each request can use the remaining **8,000-token episode** allowance, while the
200-action and 600-second limits remain. The earlier inspected ScienceWorld
partition had only one length-stopped response out of 4,998, so this ceiling is
**not established as its principal failure cause**; removing it keeps the open
reasoning interface consistent rather than promising a score increase.

A12 uses a new private output directory and fresh generation on the same original
seven 64-case development panels, one seed, single owner, skills off, native
thinking off. Code keeps the uninterrupted 12,000-token first-call/episode limit.
Static/code concurrency is 32; ScienceWorld is 16. Native scorers, isolated Luna
medium judges and metric definitions remain unchanged. Historical A11 candidates
are not replaced or rescored into this condition.

Local Ruff passed. The expanded projection, real actor/broker, submission,
native-tool, OOD and budget tests passed **215 tests in 13.03 seconds** on the
approved Linux CPU host. Full validation for this source batch runs once after
timed code grading, before any push. No independent review agent or hash checks
were added. New results and their limitations will be recorded after execution.

A12 started at **21:24:15 UTC**, reusing GPU7 SGLang. CPU orchestration PID
200090 launched code PID 200093, then static PID 245650; CPU-only check PID
200091 waits until native timed code grading finishes. These are not additional
GPU services or answer agents. No other physical card was used.

The code partition completed all 128 generations and native scores in about
797 seconds. LCB is **34/64 (53.13%)**, up from A11's 30/64 but below the
diagnostic 40/64 threshold. APPS is **51/64 (79.69%)**, down from 53/64. This is
not a claim that every recovered extraction is correct or that all regressions
are fixed. There were 129 owner calls, 609,754 output tokens and one format
clarification, versus A11's 150 calls, 488,523 output tokens and 22 clarifications.
Thus fewer forced calls did not imply lower token consumption or universally
higher scores.

All 128 candidates were checked against their actual persisted owner outputs.
Ten LCB and three APPS candidates exhausted the budget without a submitted
program. Two other normally stopped LCB replies contained explanation but no
program; five further submitted text bodies were capped prose, and one APPS
reply ended with a fenced example after its program. The fixed last-block rule
submits that example, not an earlier program selected using tests. These cases
remain failures and are not silently repaired or replaced. They identify a
remaining distinction between permissive transport and a genuinely complete,
unambiguously designated owner answer. Other five-domain results are pending.

The first A12 full check exposed three older tests that still required balanced
multiple-code-block responses to be rejected: **3 failed, 4,453 passed, 12
skipped**. The remaining nested/unclosed-carrier rejection and unframed-source
preservation checks still apply. Those tests are updated to the declared final-
block contract, including a training/scoring test proving that pass/fail feedback
cannot change which block is executed. Production generation/scoring source is
unchanged during A12. The failed validation log is retained; affected tests and
one final full check are rerun before committing this batch.

## Completed A12 results and remaining limitations

All **448/448 original cases** were freshly generated and scored; all three
partitions exited successfully. Total evaluation wall time was **30.81 minutes**.
No within-condition sample replacement, score-driven answer selection,
hidden-answer feedback, skill access/injection/invocation or peer call occurred
in the recorded run.

| Metric, 64 cases each | A12 |
| --- | ---: |
| MuSiQue answer F1, supplied candidate passages | 48.55% |
| NQ-Open closed-book EM | 20.31% |
| Omni-MATH Luna-medium accuracy | 60.94% |
| LiveMedBench Luna-medium rubric mean | 45.40/100 |
| LCB pass@1 | 53.13% (34/64) |
| APPS Intro pass@1 | 79.69% (51/64) |
| ScienceWorld **raw final native score** | **12.75/100** |
| ScienceWorld zero-clipped reward, separately reported | 51.81/100 |
| ScienceWorld success at 100 / at least 70 | 43.75% / 46.88% |

ScienceWorld completed 28 tasks at 100, versus A10's 9, and native command
rejections fell to 338. However, 25 final native scores were negative; **51.81
must not be presented as its raw native average**. There were 2,069 acknowledged
environment executions, no missing acknowledgements, and one unresolved malformed
control message at the end of an 8,000-token episode. The model's recorded budget
and format failures remain separate from HTTP/model-transport failures (zero).

Relaxation was not uniformly beneficial: MuSiQue regressed substantially from
A11's 66.02 F1. In A12, 59 of its 64 responses already used a recognized explicit
final field; only one inspected fallback used an unsupported bold `Answer:`
header. That formatting loss is real but cannot explain the whole regression.
Do not replace whole-text answers using the reference or claim that deleting
instructions alone fixes the architecture. Remaining work must distinguish the
necessary public submission semantics from arbitrary restrictions on reasoning.
The current LCB score also remains below the requested diagnostic threshold.

Actual owner usage was **2,529 requests and 2,529 responses**, 17,203,012 input
tokens and 1,257,387 output tokens, with no unknown usage. Isolated external
judging completed 424 calls (64 Omni, 360 medical), using 1,228,246 input and
68,647 output tokens. The temporary API credential was removed after all 128
judged cases completed; ScienceWorld did not require it. One monitor SSH
connection closed and recovered on reconnection without restarting evaluations.

Final validation: **71 additional affected tests passed**; final CPU-only
`make check` passed **4,457 tests, 12 skipped**, Ruff, mypy, both wheel builds and
the wheel-boundary check. The initial failed full-check log is preserved.
Already-passing targeted suites were not rerun after the test-only alignment;
the expanded final check covers the milestone. No independent review agent or
hash check was added. The dedicated Luna agent is Git-only; there is no formal
training/W&B run.

This closes the A12 seven-panel experiment, not all outstanding regression work.
Its output is development evidence, not a held-out claim or proof of threshold
attainment. Multi-agent wiring has not been reinstated without resolving the
owner's earlier single-owner prohibition.

## A13: clarify the public submission wire, not the reasoning strategy

Paired A11/A12 inspection found **no metric change for identical QA answers**.
MuSiQue had 22 identical answers, four improved scores and 22 regressions; NQ
had 15 identical answers, three improvements and four regressions. A12 MuSiQue
had 20 submitted answers longer than eight words versus A11's six. Some replies
put explanatory sentences in the declared short-answer field; others genuinely
changed their answers. This is not evidence that every loss is a parser bug.

The bounded correction accepts ordinary Markdown `Answer:` / `Short answer:`
fields, preserves final-field precedence and accepts identical repeated finals.
Conflicting same-priority fields still need the owner's clarification. A new
synthetic test also exposed a legacy fallback extracting an `Answer:` line from
a fenced discussion example; the owner boundary now leaves such unlabelled
discussion intact. Neither correction consults references, tests or judges.

The public QA instruction now explains that the submitted value is a short
phrase and that explanation can remain outside it; a bare answer remains legal.
The code instruction explains the existing last-fenced-block/final-code-section
submission convention. Neither imposes paragraph counts, sentence limits,
mandatory deliberation length, a task-solving recipe or hidden answer selection.

Only the four affected domains are regenerated: **64 each, 256 total**, on the
same original IDs and seed 0. A12 Omni-MATH, LiveMedBench and ScienceWorld remain
separate, retained results; they are not represented as new A13 generations.
One frozen Qwen3.5-9B owner, skills off, native thinking off, decoding and native
scorers are unchanged. Code retains one uninterrupted 12,000-token first response
and episode allowance; QA retains 8,000. No external judge is needed this round.
All inspected panels remain development evidence, not an independent final test.

Local Ruff passed. The first targeted run retained one failure and 218 passes;
after correcting the fenced-example fallback and adding whitespace-label cases,
the affected actor/broker, ownership, budget and projection suites passed
**223 tests in 12.55 seconds** on the Linux CPU host. Fresh generation started
at **22:25:53 UTC**, using the existing GPU7 service only. CPU coordinator PID
545348 schedules code32 then QA32. CPU validation PID 545350 waits until timed
code grading exits before the single final `make check`. No service restart,
other GPU use, training, W&B run, review agent or hash check was introduced.
Scores and full-check results are pending; threshold attainment is not claimed.

During A13, the owner explicitly reaffirmed that consultant agents remain
prohibited. There is no pending permission to restore them: all task reasoning,
tool decisions and final submissions remain with the single Qwen3.5-9B owner.

## Completed A13: gains and regressions both retained

All **256/256** planned cases completed fresh generation and scoring, with one
attempt per original ID, in **15.90 minutes**. Both evaluation partitions exited
successfully. There were zero transport failures, unknown-usage calls, grader
infrastructure failures, skill accesses or consultant calls.

| Metric, 64 cases each | A12 | A13 |
| --- | ---: | ---: |
| MuSiQue answer F1 | 48.55% | **63.94%** |
| NQ-Open closed-book EM | 20.31% (13/64) | **15.63% (10/64)** |
| LiveCodeBench pass@1 | 53.13% (34/64) | **60.94% (39/64)** |
| APPS Introductory pass@1 | 79.69% (51/64) | **78.13% (50/64)** |

LCB remains **one pass below** the required 40/64 diagnostic threshold. MuSiQue
recovers but remains below A11's 66.02%; NQ and APPS regress. These are joint
prompt/transport results, not a causal estimate of parser changes alone. No
failed case is replaced with an earlier answer or selectively replayed to cross
the threshold. Omni, medical and ScienceWorld results remain the separate A12
measurements, including ScienceWorld's **12.75/100 raw native final mean**, not
its 51.81/100 zero-clipped reward.

Remaining failures are concrete rather than all attributed to transport:

- LCB has nine empty, budget-exhausted submissions and 17 episodes reaching the
  12,000-token ceiling. Six other submitted payloads have invalid Python syntax;
  all six also reach that ceiling. Other failures include wrong algorithms,
  execution errors and time limits. The one owner format clarification stays
  within the original allowance; there is no budget extension or draft rescue.
- APPS has two empty, budget-exhausted submissions, and two submitted syntax
  failures. One normally stopped program contains an unmatched `try` block;
  it is not an HTTP truncation and was not repaired by the framework.
- QA has no empty or infrastructure-failed candidates. NQ has 21 unlabelled
  replies, which are retained as whole text rather than shortened by guessing
  the answer. This and genuine answer changes require further investigation;
  low scores alone do not establish a communication defect.

Actual owner usage: **257 calls**, **320,635 input tokens** and **683,909 output
tokens**. No external judge or environment tool was called in these four static
panels. All per-case outputs, budget stops, native verdicts and costs remain in
private artifacts, including the first failed targeted-test log.

The single final CPU-only `make check` passed **4,483 tests, 12 skipped**, Ruff,
mypy, both wheel builds and the wheel-boundary check. Pytest took 321.07 seconds;
the complete check took 438.34 seconds. Passing local/targeted checks were not
repeated after this source-freeze; documentation-only reporting does not trigger
another full run. There was no independent review agent or hash check.

The personal 22:48 UTC check found coordinator 545348, code launcher 545404,
QA launcher 584147 and validation launcher 545350 all exited. The existing GPU7
SGLang PIDs 472775/474607 remain healthy and resident, about 39,058 MiB reserved
with 0% utilization after evaluation. All eight cards were checked; no other
card or another user's process was used or modified. The overall target is
**not met**; this closes A13, not the remaining regression investigation.

## A14/A15: test the existing native submission channel without consultants

The owner again explicitly prohibits consultation. Each trajectory continues to
have one Qwen3.5-9B owner; the live native tool definitions contain only public
history access and `submit_answer(answer)`. There is no second solver, search
agent, answer-extraction model, candidate selection or skill injection. Direct
chat remains accepted; the owner decides whether to use native submission.

A14 changes only the existing tool channel from plain text to Qwen native XML.
The original 64 MuSiQue and 64 NQ IDs, seed, public task instructions, scalar
sampling parameters, 8,000-token first-response/episode limits and official
scorers remain unchanged from A13. There is no generation chunk or reserved
finalization allowance. This is a new exposed-development condition, not a
holdout or selective retry.

All **128/128** fresh A14 candidates were scored in **241.33 seconds** (31.82
cases/minute). MuSiQue F1 rises **63.94% → 67.36%**, while its EM changes from
38/64 to 37/64. NQ EM falls **15.63% → 6.25% (4/64)**. Both results are retained.
Identical submitted answers receive identical grades: 42 MuSiQue and nine NQ
answers are unchanged; none has a changed score.

The actual communication records explain the limits of this experiment:

- All 137 owner requests have recorded replies and usage; transport failures,
  scorer infrastructure failures and consultant calls are zero.
- Two MuSiQue and five NQ episodes use native final submission. Eight additional
  NQ calls read public history, not outside knowledge or another model.
- One MuSiQue ambiguous final is clarified by the same owner within its original
  allowance. This is the only repair; the nine extra calls are not nine repairs.
- NQ has 34 unlabelled whole-text submissions, and 40/64 submitted values exceed
  eight words. Actual replies reach the scorer intact. They include explanatory
  prose and genuine knowledge errors, not simply text dropped in transit. No
  reference-informed phrase extraction is applied.
- Four MuSiQue and two NQ episodes reach the token ceiling; all still have
  nonempty submitted payloads under the unchanged answer-blind projection.

A14 uses **269,563 input / 87,211 output tokens**, zero external judge calls and
zero skill accesses. Private artifacts retain raw replies, literal submissions,
repair events, per-case scores and the paired comparison.

Two synthetic actor/broker tests cover native QA submission, literal owner
payload preservation and the absence of another solver. Local Ruff passed;
the three directly affected suites passed **149 tests in 7.44 seconds**. The
single Linux CPU-only `make check` passed **4,485 tests, 12 skipped**, Ruff,
mypy, both wheel builds and the wheel-boundary check; pytest took 307.73 seconds.
Production code is unchanged from the A13 commit. No independent review agent,
hash check, training or W&B run was introduced.

A15 was configured before inspecting A14 scores. It separately regenerates all
64 APPS Intro and 64 LCB cases with the same native-channel change, batch 32,
and **one uninterrupted 12,000-token first response and total episode allowance**.
Its timed code scoring starts only after A14 generation and the full CPU check
finish. Identical production and test source reuse that successful full check;
it is not run again merely for a second evaluation configuration.

A15 started at **23:15:00 UTC**, coordinator PID 186936, using the existing GPU7
service PIDs 472775/474607 only. The personal prelaunch check verified all eight
cards, GPU7 mapping, healthy frozen base-model service and no other GPU7 compute
PID. About 39,058 MiB remains reserved by that service. No service was started or
restarted, and other cards/processes were untouched.

A15 completed **128/128** fresh generations and scores in **742.17 seconds**
(10.35 cases/minute). APPS is **51/64 (79.69%)**, versus A13's 50/64; LCB is
**36/64 (56.25%)**, versus 39/64. Both improvements and regressions are retained.
There were 139 owner requests and responses, 191,565 input / 591,069 output
tokens, seven history reads and four interface repairs. Consultant calls, skill
accesses, external-judge calls and HTTP/scorer infrastructure failures are zero.
No new source changes or repeated full check were needed for this frozen run.

APPS retains two empty budget-exhausted submissions, four episodes at the
12,000-token cap and two submitted syntax errors. LCB retains seven empty
budget-exhausted submissions, 15 capped episodes and six syntax errors. Two
unresolved control-delivery diagnostics (one per benchmark) are malformed
control messages in first responses that already consumed all 12,000 tokens;
their HTTP responses arrived. They are not network outages and had no remaining
clarification allowance. No code was completed or selected by the framework.

The NQ and LCB regressions mean the native channel is not a demonstrated general
fix. The subsequent [native-contract repair](ood-native-contract-repair-20260912.md)
is a different code milestone; it does not relabel A14/A15 as results from the
new scoring, semantics, budget or observation conditions.
