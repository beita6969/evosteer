# Step-0 OOD evaluation

Historical medical OOD slot: **LiveMedBench**, replacing GPQA by the owner's
2026-09-12 instruction. The new 64-case protocol and measured result are in
[LiveMedBench evaluation](ood-livemedbench.md). The September 15 current catalog
restores GPQA Diamond BioOrganic-91 and replaces Omni-MATH with Math-Hard;
LiveMedBench and LiveCodeBench are no longer active. See
[current datasets](current-datasets.md). The runs below remain historical;
historical scores remain unchanged. The subsequent seven-panel fresh-generation
condition is documented in [trajectory fixes](ood-trajectoryfix-20260912.md).

This engineering extension reuses the current IID single-owner runner, isolated
actor/broker, complete-output parsing, insert-once candidates, native score schema
and scoring-only resume. It does not replace the method in `idea.tex` or enable
training updates. One frozen Qwen3.5-9B owner handles each episode; no consultants,
voting, second solver or evaluator feedback enters an episode.

For the subsequent scoring-only upstream alignment and the distinction between
raw ScienceWorld score and clipped reward, see
[official scorer integration](ood-official-scorers.md). That change does not
regenerate or overwrite the historical results below.

## Fresh generation with upstream scorers (2026-09-12 UTC)

The owner requested actual new model generation after scorer integration, not
another scoring-only comparison. This round ran from committed snapshot
`da2895b`, independently of concurrent workspace changes. All six original
64-item panels were generated anew; no previous answer or best-scoring cell was
imported. These remain inspected **development panels**, not untouched holdouts.

The declared A8 condition combines the existing plain-text code interface with
native XML for the other four domains and the brief-reasoning QA instructions.
These choices were fixed before generation. Therefore, differences from earlier
conditions cannot be attributed solely to scorer changes. Every trajectory used
one frozen Qwen3.5-9B owner, thinking off, skills off and seed 0. The concurrency
ceiling remained 32. Code's first request received the whole **12,000 tokens**,
with no chunking or reserved continuation; other episodes retained **8,000 total
output tokens**. Actual requests already told the model its remaining token and
call allowances, explicitly describing them as upper limits rather than required
work. ScienceWorld retained 256 tokens per response, 60,000-token history, 200
environment steps, no simplification and the 600-second episode deadline.

The original attempt completed all five static panels and **51/64 ScienceWorld
episodes**. Thirteen ScienceWorld episodes hit the 600-second wall limit; seven
had an in-flight model request canceled, while six stopped at other runtime
boundaries. These were not scorer crashes or evidence of a server disconnect.
The seven canceled-request histories inspected retained approximately 98–99% of
their preceding prompt prefix, so the earlier history-truncation diagnosis did
not explain these timeouts. The incomplete attempts had received 78–171 model responses
and used 6,812–7,786 observed output tokens before the deadline.

After the original processes drained and the service was idle, **only those 13
unfinished episodes** were restarted once, as A8r1, without changing generation,
token, environment or time limits. The configured concurrency ceiling was still
32, with only 13 available trajectories. All completed within 332.48 seconds;
the recovery job took 5m 38s and had no new transport failures. Selection used
completion/timing metadata before scoring, never correctness. The original
failed attempts and exit status remain recorded. Thus the following ScienceWorld
result is **51 original + 13 operational-recovery records**, not a uniform fresh
64-episode condition; the other five rows are complete fresh A8 conditions.

| Metric, all 64 planned items retained | Result |
| --- | ---: |
| MuSiQue-Ans answer F1 | **70.24%** |
| NQ-Open closed-book EM | **23.44% (15/64)** |
| Omni-MATH Luna high equivalence | **50.00% (32/64)** |
| LiveCodeBench pass@1 | **46.88% (30/64)** |
| APPS introductory pass@1 | **60.94% (39/64)** |
| ScienceWorld raw final-score mean, including negative failures | **21.84/100** |
| ScienceWorld zero-clipped final-score mean / bounded reward ×100 | **53.09/100** |

MuSiQue's separate EM is **57.81% (37/64)**. ScienceWorld now retains all 64 raw
final scores: **20 are -100**, explaining the 31.25-point difference between raw
and zero-clipped means. Strict score-100 success is **23/64 (35.94%)**; the
separately named score-at-least-70 rate is **30/64 (46.88%)**. Neither replaces
the other, and the clipped mean must not be advertised as the raw native mean.

**The overall reference threshold remains unmet:** LCB's 46.88% is below
62.32% (= 95% of 65.6%). APPS no longer reproduces the erroneous approximately
5% result, but scorer alignment does not guarantee higher scores everywhere.
Input, date-window, thinking, budget, judge and sample-size limitations still
prevent claiming official-model-card replication. No additional successful
answer was selected or regenerated to reach a threshold.

Record-level findings:

- All 128 code candidates were submitted and graded, with no empty submissions
  or output-protocol errors. Four programs have syntax errors; ten hit the
  12,000-token cap. APPS failures comprise 22 wrong answers, two runtime/timeout
  outcomes and one compilation/loading failure. LCB failures comprise 21 wrong
  answers, five test timeouts and eight runtime/loading failures.
- MuSiQue retains two empty submissions, and Omni-MATH one. Output caps were
  reached by five MuSiQue, one NQ, 20 Omni-MATH and 14 selected ScienceWorld
  episodes. A parsed or token-capped output is not a completeness guarantee.
- Selected ScienceWorld records contain **1,890 unrecognized native commands**,
  zero bare completion narratives dispatched as commands, and 14 episodes with
  unresolved control/output-interface diagnostics. Such diagnostics must not be
  mislabeled as HTTP failures. All selected execution acknowledgements and
  observation projections match; the 13 interrupted original attempts separately
  retain two missing acknowledgements and two feedback-projection mismatches at
  cancellation boundaries. Their incomplete evidence is not reported as clean.
- There were **zero scorer crashes** and no non-owner answer-generation calls.
  All 64 new Omni candidates received resolved, scope-matched Luna high verdicts
  under the exported official equivalence template; the empty candidate remains
  a failure in the 64-item denominator.

Measured costs include unsuccessful original attempts: **397 episode attempts**
for 384 final records, **6,460 recorded owner requests / 6,453 responses**,
**86,337,406 observed input tokens** (including cached inputs), **1,032,003 observed
output tokens**, and **5,999 executed owner tool calls** (5,973 native environment
executions and 26 logged history reads), plus 128 terminal code checks. One more
history-control request was decoded but not executed and was resolved by an
owner action after clarification. Seven canceled requests have unknown usage, so token totals are
observed lower bounds, not complete billing totals. Luna's token usage is also
unavailable through the subagent interface, not zero. Launch through recovery
exit took **36m 11s**; final CPU scoring/aggregation completed afterward. The
original non-code partition's failure exit is preserved rather than overwritten.

The executable snapshot had already passed 76 affected tests and a remote Linux
CPU `CUDA_VISIBLE_DEVICES="" make check`: **4,357 passed, 12 skipped**, with Ruff,
mypy, both wheels and the model-wheel boundary check. This round changed no
production evaluation code: private orchestration scripts were syntax-checked,
and all final source identities, budgets, scorer versions and complete panel
joins were checked during aggregation. Repeating the unchanged full suite for a
new generation run or this documentation update was deliberately skipped. No
independent review agent or hash check was used. Only the existing GPU5 service
on endpoint 22049 was used; no standby, other GPU or training/W&B job was started.
All questions, outputs, per-item diagnostics, judgments and journals stay private.
GPQA, `idea.tex`, `AGENTS.md` and concurrent unrelated changes were untouched.

## Earlier development comparisons (2026-09-12 UTC)

These original 64-item panels have been inspected and used to adjust interfaces
and prompts. They are **development evidence, not untouched final OOD tests**.
Three distinct conditions are retained below; do not combine their best answers
or present the best cells as one frozen six-domain experiment. Every populated
cell contains all 64 original tasks, including failures, with seed 0.

- **A5, full six-domain rerun:** explicit ScienceWorld action carriers and native
  rejection feedback; direct-code instructions and one-line-only QA instructions.
- **A6, complete code-interface comparison:** all 64 APPS and 64 LCB tasks use
  the existing plain-text capability representation rather than native XML.
  The comparison was declared before reading A5 native code scores. Sampling,
  budgets, tasks, checker inputs and checker limits remain unchanged.
- **A7, complete QA regression check:** all 64 MuSiQue and 64 NQ tasks permit
  brief owner reasoning before one final short-answer line. This corrects the
  one-line-only instruction's observed quality regression; it is explicitly a
  development adjustment, not a new independent test set.

All conditions use one frozen Qwen3.5-9B owner, thinking off, skills off and batch
32. Thinking off does **not** prohibit the owner's own reasoning or public tools;
skills-on can also be a legitimate separately declared, read-only condition.
Initial code requests receive the whole 12,000-token allowance without chunking
or a reserved continuation. Any format clarification uses only the remaining
episode allowance; A5 LCB had one additional call, whereas every A6 episode used
exactly one call. Other episodes retain 8,000 total output tokens, including
repairs. ScienceWorld retains 256 output tokens per response, a 60,000-token input
history, 200 environment steps and no simplification. Request/episode deadlines
remain 480/600 seconds. No model, posterior, skill library or success threshold
was updated during any condition, and no judge feedback entered an owner context.

| Native metric | A5: full six | A6: code plain text | A7: QA brief reasoning |
| --- | ---: | ---: | ---: |
| MuSiQue-Ans answer F1 | 48.74% | — | 73.86% |
| NQ-Open closed-book EM | 17.19% (11/64) | — | 23.44% (15/64) |
| Omni-MATH Luna high equivalence | 50.00% (32/64) | — | — |
| LiveCodeBench pass@1 | 35.94% (23/64) | 51.56% (33/64) | — |
| APPS introductory pass@1 | 62.50% (40/64) | 60.94% (39/64) | — |
| ScienceWorld zero-clipped final-score mean | 51.02/100 | — | — |

A7 MuSiQue EM is separately **62.50% (40/64)**, not its F1. ScienceWorld strict
score-100 success is **26/64 (40.63%)**; the report-only score-at-least-70 rate is
**30/64 (46.88%)**. Neither changes the native reward contract. The all-domain
reference objective remains **unachieved**: even A6 LCB is below **62.32% =
0.95 × 65.6%**. The existing version, context, judge and sample-size limitations
still apply; these numbers are not an official-model-card replication.

Record inspection distinguished actual transport from output syntax. The
previous selected records had 43 bare completion narratives dispatched as
ScienceWorld commands. The adapter now requires an explicit act/Action carrier,
without filtering the argument's meaning or consulting an oracle action list.
It preserves the simulator's explicit unrecognized-command observation as a
rejection instead of an ambiguous execution acknowledgement. A5 contains zero
such dispatched completion narratives, but still has **1,216 native unrecognized
commands**: correct transport does not make the owner's actions correct.

A5 retains six empty code submissions (APPS one, LCB five), two submitted code
syntax errors, and ten ScienceWorld episodes with unresolved interface errors.
A6 has **zero empty submissions and zero protocol errors**, but nine outputs
reach 12,000 tokens and three programs have syntax errors; all failures remain
in pass@1. A7 retains four empty MuSiQue answers and one empty NQ answer; six QA
episodes reach 8,000 tokens. Neither truncated output nor a successfully parsed
submission is advertised as a correct or necessarily complete solution.

| Condition | Episodes | Owner calls | Input tokens | Output tokens | Launch-to-exit wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| A5 | 384 | 3,845 | 43,550,749 | 806,019 | 23m 15s |
| A6 | 128 | 128 | 96,516 | 311,589 | 10m 34s |
| A7 | 128 | 164 | 427,108 | 111,174 | 4m 14s |

Total: **640 generated episodes**, 4,137 observed owner requests/responses,
44,074,373 input tokens (including cached inputs), 1,228,782 output tokens,
3,457 owner tool calls (3,426 native environment executions and 31 history reads),
plus 256 native code candidates graded after submission. History reads were seven
in A5 and 24 in A7; A6 had no owner tool calls.
There were **zero transport-failure events, scorer crashes, missing execution
acknowledgements or crossed observations**. Malformed/truncated model output is
reported separately above, not relabeled as a network failure. All episodes
stayed within their output cap and 600-second deadline; maximum observed episode
time was 480.06 seconds. Condition wall times include scoring and overlap, so
they must not be summed as serial elapsed time. Luna high graded 64 fresh final
math candidates; its token usage is unavailable through the subagent interface,
not zero. Every record, scope, judgement and journal snapshot remains private.

Validation: 135 affected tests passed, followed by one remote Linux CPU
`CUDA_VISIBLE_DEVICES="" make check`; after the QA prompt correction, five
affected actor/broker cases and a final full check passed. Both full checks had
**4,332 passed, 12 skipped**, plus Ruff, mypy, both wheels and the model-wheel
boundary check. Initial scoped failures exposed an observation-field typo and
a shared pytest temporary-directory issue; both were resolved before launch.
Local WSL mypy timed out and was not retried there. No independent review agent
or hash check was used, and documentation-only updates did not trigger another
full check. Only the existing GPU5 service was used; no standby or other GPU was
started. GPQA, `idea.tex`, `AGENTS.md` and unrelated work were left untouched.

The declared comparisons are complete. Further final-generalization claims need
a separately frozen holdout not used for these adjustments; this round did not
start one or keep retrying to select a satisfactory score.

## Authorized 2026-09-11 round

- Six domains, 64 episodes each, one seed (0); GPQA-Diamond health is deferred.
- Thinking off for all six; single-owner, skills-off Step-0 arm, matching the
  current IID architecture before training. No benchmark-specific handwritten
  solution policies, demonstrations or task-context precomputation.
- Only physical GPU5 on endpoint 22049. The active loopback-only SGLang service
  has a 65,536-token context, 1,114,112 KV-cache tokens, 336 Mamba state slots,
  and 65 effective request slots (one scheduling headroom slot). The qualified
  one-shot limit is 64; the current multi-turn ScienceWorld recovery uses
  **32 independent single-owner trajectories** to preserve state-cache reuse. Initially two services
  each handled one request; batching now uses **one service only**. The standby
  remains shut down at the owner's request. No other GPU is used.
- Reuse the IID non-thinking sampling profile (temperature 0.7, top-p 0.8,
  top-k 20). The owner subsequently capped cumulative episode output at **8,000
  tokens**, including repairs, and requested item-level recovery rather than
  restarting all six domains. An incomplete answer is never replaced with a guess.
- This 64-item panel is an infrastructure/development evaluation, not the full
  official leaderboard population. Report source split, sample size and exposure.

## Six-domain correctness repair (2026-09-12 UTC)

The owner requested a fresh six-domain Step-0 run after investigating the low
scores. Reuse the exact original 64 records per domain and seed 0, not a new or
result-ranked sample. This is an inspected **development panel**. GPQA is not
part of this rerun. Freeze one condition with batch/concurrency 32, one frozen
Qwen3.5-9B answer owner, thinking off and an 8,000-token cumulative episode cap.
The existing single GPU5 service remains in place; no standby is started.

The APPS scorer had a reproduced entry-point bug: the replacement for the old
`pyext.RuntimeModule` imported stdin programs under a non-main module name.
APPS wraps the submitted program in `code()`, so a valid original
`if __name__ == "__main__"` body never ran. A synthetic addition program passed
without a guard and failed with one. The adapter now supplies script semantics
for stdin tasks only; callable tasks retain import semantics. Candidate source,
all released tests, comparison logic, sandbox and native time limits are unchanged.
The new verifier is `apps-official-introductory-script-entry@ood2`. Scoring-only
diagnosis of the identical historical answers changed **3/64 to 41/64**; it is
not a new generation result and does not overwrite historical journals.

Static task instructions now explicitly request short answers or complete code
without extensive explanatory comments; math instructions request a concise,
complete solution without repetition. These are answer-free output instructions,
not demonstrations or task-specific solvers. The ScienceWorld public API reference
now includes the released command syntax templates (not valid-action/object
combinations or gold paths), including `focus on`, `move ... to ...` and `use ...
on ...`. Its per-response output limit is 256, still within the same cumulative
8,000-token allowance, with up to 200 environment interactions. It remains the
unsimplified environment. Strict success stays **100/100**; a separately named
**score >=70** rate may be computed in reports for literature comparison, without
changing TerminalReward or the primary native mean.

Comparison limitations remain material: MuSiQue receives every released candidate
paragraph with no supporting flags, decomposition or answer fields; NQ is
closed-book. Omni uses the full-source panel and the owner's isolated Luna high
equivalence judge, not Omni-Math-Rule. LiveCodeBench uses `code_generation_lite`
release v6, not an independently aligned official-model-card time window. The
[Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices) recommends
larger output budgets than the owner's 8,000-token cap. Public reference scores
are comparison goals, not grounds to change a score, drop an item or choose a
better repeat answer. See the [APPS checker](https://github.com/hendrycks/apps/blob/main/eval/testing_util.py),
[ScienceWorld API](https://github.com/allenai/ScienceWorld/blob/main/scienceworld/scienceworld.py),
[LiveCodeBench evaluation](https://github.com/LiveCodeBench/LiveCodeBench), and
[ReflAct's distinct success definition](https://arxiv.org/html/2505.15182v2).

### Completed repair-round results

All six original panels have 64 scored records. The five static domains each use
one fresh condition. ScienceWorld retained 50 completed records from that condition
and recovered only 14 unfinished, wall-time-limited episodes. Thus the combined
six-domain result is **mixed-condition operational recovery**, not one uniform
fresh condition, a paired architecture comparison, or a full leaderboard result.

| Benchmark | Native result |
| --- | ---: |
| MuSiQue-Ans answer F1 | 68.42% (secondary EM 57.81%) |
| NQ-Open closed-book EM | 21.88% (14/64) |
| Omni-MATH Luna high equivalence | 53.13% (34/64) |
| LiveCodeBench pass@1 | 45.31% (29/64) |
| APPS introductory pass@1 | 60.94% (39/64) |
| ScienceWorld mean native score | 51.38/100 |

ScienceWorld strict score-100 success is **26/64 (40.63%)**. The separately named,
report-only score-at-least-70 rate is **28/64 (43.75%)**; it does not replace strict
success or change TerminalReward. The LiveCodeBench sampled problem dates span
**2023-05-13 through 2025-04-06**, drawn from the cumulative 1,055-record release-v6
lite population without an additional date filter.

The ScienceWorld timeout diagnosis found six long prompts repeatedly hitting the
32,768-input-token history cap. Removing their earliest history on each step
reduced the common prefix to roughly 1,800 characters in approximately
130,000-character prompts. The recovery raised **input history**, not output, to
60,000 tokens within the existing 65,536-token service context. It retained every
other limit and the full public archive. Observed late recovery prompts retained
97.7–99.6% common-prefix characters. All 14 recovery records finished without a
new transport failure. No completed incorrect answer was selected for regeneration.

Every selected record stayed within 8,000 output tokens and 600 seconds; the
maximum measured selected-episode duration was 598.85 seconds. Nineteen static
records still lacked a valid submission and remain in their denominators as
failures: APPS four, LiveCodeBench twelve, MuSiQue two, and NQ one. Completion is
therefore not guaranteed by the concise-output instructions.
Of LiveCodeBench's 52 submitted programs, 29 passed; the native checker reported
14 wrong answers, five test timeouts and four runtime errors.

**The requested all-domain 95%-of-reference objective was not achieved.** APPS
60.94% exceeds 95% of the approximate 42% reference, and ScienceWorld's 51.38 mean
exceeds 95% of 51.8, numerically but not as matched-protocol replications.
LiveCodeBench remains below **62.32% = 0.95 × 65.6%**. The other references also
require their stated input/judge conditions; they are not interchangeable targets.
No sample was dropped, no answer was selected by score, and no scoring threshold
was relaxed to meet a reference.

Validation for the code repair comprised 147 distinct affected tests and one
remote Linux CPU `make check`: **4,312 passed, 12 skipped**, with Ruff, mypy, both
wheels and the model-wheel boundary check passing. The later input-history
configuration and aggregate-only documentation did not change executable code,
so the full check was not repeated. Private source panels, journals, verdicts and
per-record results remain outside Git. GPQA, `idea.tex` and `AGENTS.md` were not
modified by this repair round.

### Record-level communication diagnosis and owner-requested code rerun

The subsequent inspection covered all 384 selected records, their 3,714 model
responses and 3,230 native environment executions. Request/response counts,
execution acknowledgements, returned observations and owner payload projections
matched; no missing acknowledgement or crossed environment feedback was found.
This does not establish that every model answer or action was correct.

The communication report incorrectly treated ScienceWorld's free-form
`act(command)` schema as a missing enumerated action list. All 3,392 recorded
surfaces were consequently marked mismatched despite both the native and
advertised surface being `act`. The report now recognizes that interface without
obtaining oracle action/object lists. Regression cases retain detection of an
actually changed or missing advertised surface. This reporting fix does not
change a native reward or rerun ScienceWorld. Separately, 1,253 executed commands
received the simulator's unrecognized-command response; these observations were
delivered, not lost in transport. The model often repeated an invalid command.

Sixteen code episodes spent the full 8,000 tokens inside an unfinished native
submission envelope, typically with lengthy explanatory comments. There were no
remaining tokens for an interface repair. One submitted LCB program also contained
non-Python comment syntax in the actual model output, not introduced by extraction.
One other LCB program encountered frozen tests contradicting the statement's
even-length input constraint. This is recorded as a data-quality caveat; neither
the test cases nor its official score were altered.

The owner subsequently authorized **one whole 12,000-token generation**, with no
chunking or reserved repair allowance, **only for code records that reached the
old 8,000-token cap**. The selection is four APPS and thirteen LCB records,
including the capped LCB record that already had a submission, regardless of its
score. All seventeen old/new scopes remain private and immutable. Use the same
tasks, seed 0, thinking-off profile, native checkers, batch ceiling 32 and existing
GPU5 service; retain the 480-second request and 600-second episode deadlines.
The other 367 selected records are retained. Any combined result is explicitly
**mixed-budget development recovery**, not a uniform 12,000-token pass@1 run or
best-of-two selection. No reserve/chunk implementation was introduced.

All seventeen reruns completed and were natively scored, using exactly seventeen
model calls and 162,964 output tokens. The maximum episode time was 313.70 seconds;
none exceeded 12,000 tokens or the existing deadlines. Nine submitted a program,
of which two passed, three had model-generated syntax errors and four failed
native tests. Eight still exhausted 12,000 tokens without a valid submission.
The mixed-budget totals are **APPS 40/64 (62.50%)** and **LiveCodeBench 30/64
(46.88%)**, versus 39/64 and 29/64 before this subset rerun. Other domain scores
are unchanged. LCB still does **not** meet the 62.32% reference threshold. Longer
generation alone did not resolve the remaining completion and correctness losses.

Validation of the reporting fix: 13 targeted tests passed on remote Linux CPU,
followed by one `CUDA_VISIBLE_DEVICES="" make check`: **4,315 passed, 12 skipped**;
Ruff, mypy, both wheels and the model-wheel boundary check passed. The initial
WSL targeted-test attempt stalled in disk I/O and was stopped rather than repeatedly
retried there. No independent review agent or hash check was used; the subsequent
documentation-only result update did not trigger another full validation.

## GPQA BioOrganic extension (2026-09-12 UTC)

The owner subsequently selected **batch 32 for subsequent evaluations**, including
static tasks, and authorized a separate GPQA run. Its population is
**GPQA-Diamond-BioOrganic-91**, not a strict health or clinical subset. The
64-episode panel retains all 19 Biology records (15 Molecular Biology, four
Genetics) and samples 45 of 72 Organic Chemistry records with seed 0 after sorting
Record IDs. Match the entire owner-provided ID list against the original
198-record `gpqa_diamond.csv`; missing, duplicate or differently classified IDs
stop preparation instead of silently reducing the population. Converted row
numbers are not identities. All actual IDs and source data remain private.

The executable benchmark identity is `gpqa-diamond-bioorganic`. The older unused
`gpqa-diamond-health` thinking-map entry remains readable for historical six-OOD
conditions, but is not an executable alias for this expanded population. The IID
catalog and the previous six-domain results are unchanged.

Reuse the existing single-owner actor/broker, frozen public projection and native
choice parser. Shuffle the correct option and three distractors once per Record
ID with seed 0, store the resulting gold label only in the private evaluator,
and score exact A–D accuracy. Explanations and gold-label metadata never enter the
actor; all four option texts necessarily do. Conflicting or missing final choices
are candidate failures, not invitations to select an answer from the reference.
No judge model or consultation is used. This follows the original
[GPQA evaluation approach](https://github.com/idavidrein/gpqa), with the existing
project parser and explicitly declared prompt/option-order profile.

Keep thinking off, cumulative output at 8,000 tokens including repairs, request
timeout 480 seconds and episode timeout 600 seconds. Only the existing GPU5
service is used, with at most 32 independent trajectories. Report overall accuracy
and the 19/45 stratum accuracies; do not present this sample as full-Diamond or
strict-health performance. The [original dataset card](https://huggingface.co/datasets/Idavidrein/gpqa)
describes biology, physics and chemistry rather than a medical-only benchmark.

The completed panel scored **38/64 (59.375%)**: Biology 13/19 and Organic Chemistry
25/45. All selected records were generated in one final condition; the earlier
tool-surface wiring failure produced no model responses and is not a source of
candidate selection. Fifteen responses reached the 8,000-token cap without an
unambiguous final choice and count as incorrect in the 64-record denominator.
No record exceeded that token cap or the episode deadline (maximum 228.41 seconds).
These are infrastructure/development results, not strict-health or full-Diamond
leaderboard numbers. No correctness-based regeneration was performed.

## Source and evaluator reuse

| Domain | Population / public input | Metric and reference implementation |
| --- | --- | --- |
| MuSiQue-Ans | Released v1.0 dev; question and **all** supplied paragraphs, without support labels/decomposition | Answer F1, secondary EM; reuse existing native alias normalizers, following [official evaluation](https://github.com/StonyBrookNLP/musique/blob/main/evaluate_v1.0.py), [paper](https://arxiv.org/abs/2108.00573). Not support/joint F1. |
| NQ-Open | Released dev questions; closed-book, no generated evidence dossier | Normalized alias-aware EM, secondary F1; reuse existing QA evaluator. [DPR release/evaluation](https://github.com/facebookresearch/DPR), [paper](https://arxiv.org/abs/2004.04906). |
| Omni-MATH | Full official test source, not Omni-MATH-Rule; problem only | Official equivalence prompt with an owner-requested **GPT-5.6 Luna high subagent** as post-submission judge. [Evaluation code](https://github.com/KbsdJames/Omni-MATH/tree/main/GPT_eval), [paper](https://arxiv.org/abs/2410.07985). The metric is `luna-high-equivalence-accuracy`, **not** the official GPT-4o judge score. |
| ScienceWorld | Official test variations, no simplifications; goal and observations from live reset/steps | Official final score / 100 and score-100 success. Reuse the existing private JVM worker; free-form `act(command)`, no oracle action/object combinations or gold path. [Official environment](https://github.com/allenai/ScienceWorld), [paper](https://arxiv.org/abs/2203.07540). |
| LiveCodeBench | `code_generation_lite`, frozen `release_v6` (all six release files); statement and starter interface | One generated program passes all released public + private tests; directly execute upstream `lcb_runner/evaluation/testing_util.py`. [Code](https://github.com/LiveCodeBench/LiveCodeBench), [paper](https://arxiv.org/abs/2403.07974). This is pass@1 on 64 samples, not a contamination-free claim about the base model. |
| APPS introductory | Official test split, difficulty exactly `introductory`; question and starter interface | One program passes all tests; directly execute upstream `eval/testing_util.py`. [Code](https://github.com/hendrycks/apps), [paper](https://arxiv.org/abs/2105.09938). |

`skillev_private.evaluation.ood_export` draws uniform reservoir samples before any
model evaluation and restores source order. Source locations, versions, population
sizes and selection parameters go into the private run configuration. There is no
outcome-ranked selection or replacing hard records. ScienceWorld's released test
variation population is enumerated before sampling; an old seed-42 panel is not
relabeled as seed 0.

The native code checker is mounted alone into the existing network/filesystem/
process sandbox. Only that candidate's private tests are provided to the scorer.
The APPS legacy `pyext.RuntimeModule` loader is implemented with standard-library
`ModuleType` + `exec`; its official test comparisons and four-second test timer
remain unchanged. LiveCodeBench retains its six-second test timer. Worker failure
is infrastructure failure, not a fabricated zero. No hidden test feedback reaches
the task owner. Omni reference answers and the official judge template are private
and become available only for finalized Omni-MATH answers; other independent domains may still be running.

The Luna judge is not the Git/W&B-only subagent. It reads an exported private
batch from `ood_luna_judge` and writes one explained Boolean verdict per finalized
candidate. No task trajectory receives its advice. Missing or unresolved judgements
stop scoring and can be supplied through scoring-only resume; they never become zeros.

## Running and validation

### Operational recovery after the owner budget change

`configs/evaluation/ood_step0_8000.json` is a portable runtime overlay. It clears
old decoding/budget overrides, caps each request and the entire episode at 8,000
generated tokens, sets finite request/episode deadlines, and uses presence penalty
1.5 to address the observed repeated-output stalls. Input context is not the output
budget. Decode CUDA graphs cover the actual active batch sizes; serving configuration
is recorded privately. The controlled 1,024-output-token synthetic checks measured
about 36.8 tokens/s aggregate for two concurrently decoding processes, 40.3 tokens/s
with one active process, and 156.9 tokens/s aggregate for four requests batched in
one process (about 4.3 times the original aggregate). These are infrastructure
measurements, not benchmark completion-rate or accuracy guarantees.

The owner subsequently requested batch 32 and 64. Single-service synthetic tests
with 1,024 output tokens per request measured:

| Concurrent requests | Aggregate output tokens/s | Batch wall time |
| --- | ---: | ---: |
| 16 | 636.1 | 25.76 s |
| 32 | 1,277.5 | 25.65 s |
| 64 | 2,574.1 | 25.46 s |

The seeded synthetic outputs were identical across these three batch sizes.
The final batch-64 capacity check used distinct synthetic histories of
6,425–8,826 input tokens and **8,000 output tokens per request**: all 64 finished
in 281.75 seconds, with 1,817.2 aggregate output tokens/s and no timeout or OOM.
Synthetic tests force output length; formal episodes still stop normally.
BF16 weights/KV cache, FA3 full attention, Triton linear attention, seeded PyTorch
sampling and the deterministic-inference flag remain unchanged. No quantization,
speculative decoding, seed removal or output-budget reduction was introduced.

Long-input tests initially exposed a 63-running/one-queued admission boundary.
Merely requesting 65 slots did not fix it: this installed SGLang build limits
effective requests by `max_mamba_cache_size // 5`, so 320 state slots still capped
the scheduler at 64. Raising the state pool to 336 made the effective limit 65;
the repeated long test then sustained 64 running requests with an empty queue.
The active service captures decode graphs for 1, 2, 4, 8, 16, 32 and 64 requests;
observed GPU use after qualification was about 72.3 GiB of 79.6 GiB. Failed
synthetic tests were stopped and diagnosed; no formal answers were regenerated
for throughput tuning. Capacity configuration follows the official
[SGLang tuning guidance](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning).

During the initial switch, both existing services stayed resident until a temporary
third replacement was healthy. The owner's later explicit standby-shutdown request
supersedes that residency instruction: do not automatically restart the standby.

`skillev_private.evaluation.ood_recovery plan` selects **only** over-budget,
measured-over-deadline and unfinished records from the original frozen task IDs.
It does not inspect correctness and retains wrong but completed in-budget answers.
Old missing wall-time measurements remain unknown. A replacement gets a new run
and condition ID; original candidates are never overwritten or silently imported
as a fresh arm. The subset uses the existing isolated single-owner runner.
Repeated operational recovery can use `--previous-plan` without regenerating
unaffected records. Never retry an answer solely because it failed native grading.

The preceding recovery completed 96 more episodes, bringing retained results to
321/384, before an empty ScienceWorld `act` argument reached the native worker.
The transport now rejects blank/NUL commands through the existing owner-format
repair path, preserving every model call and its token cost; it does not invent
an environment action. A frozen batch-64 recovery covered only the
remaining **63 ScienceWorld episodes** (one unfinished and 62 unstarted).
Our JVMs use two visible CPU processors each to avoid multiplying host-wide
thread pools at this concurrency. Task seeds, native horizons and the cumulative
8,000-token / 600-second episode limits are unchanged.

That real interactive recovery completed 19 episodes but hit the wall limit on
44 others, leaving **340 completed panel entries retained**. Short-request
throughput was not a reliable proxy for many-step ScienceWorld latency. Actual
consecutive prompts differed in their opening system message because both local
and episode budget counters changed on every call. In one sampled window the
median common prefix was only 1,166 characters for 38,018-character prompts;
the serving logs showed roughly 640 cached tokens despite much longer histories.

Interactive budget counters now update the **existing runtime action-surface
block**, after the episode history, rather than the opening system prefix.
The original user dialogue, archive content, native observations and hard budget
enforcement are preserved. No extra user turn or fabricated tool response is
introduced. Noninteractive QA/conversation tasks keep their existing system
controls. This input-layout change requires a new recorded implementation and
condition identity; it is not silently applied to a live run or described as a
bit-identical change. Recovery remains limited to those 44 operational failures,
not completed incorrect answers.

The corrected-layout synthetic multi-turn test completed 12 rounds at batch 32
in 47.05 seconds (768 generated tokens per trajectory), with 91.7% median
steady-state cached-token reuse. Later rounds took approximately three seconds.
At batch 64, the same fixed state pool repeatedly lost reuse: nine completed
rounds showed later-round medians of zero cached tokens and 16–21 second rounds.
That slow synthetic test was stopped; it was not reported as a completed or
qualified batch-64 multi-turn run. **Batch 32 is selected for the remaining
ScienceWorld recovery**, while the earlier one-shot batch-64 result remains valid
for its own workload. The backend, precision, model, seeds and hard limits are
unchanged; batch sizing is based on resource/latency measurements, never correctness.

The real batch-32 recovery preserved 34 additional in-budget ScienceWorld finals;
ten episodes remained unfinished. A worker close timeout exposed a blocking
ScienceWorld cleanup call inside an asynchronous method, freezing the shared
event loop and delaying unrelated model requests and episode deadlines. Cleanup
now runs in a worker thread, like the other interactive environment adapters.
The regression test requires another episode to make progress while cleanup is
waiting. The already-persisted final and native outcome survive a cleanup error;
they are not regenerated. Only the ten unfinished episodes enter the next
recovery; output budgets, deadlines, native horizons and scoring stay unchanged.

`ood_recovery aggregate --plan ...` grades the exact original panel using the
selected immutable source scopes, with a resumable private score cache. This is a
**mixed-condition operational recovery**, not a uniform fresh run or a paired
architecture comparison. Report per-domain retained/replacement source counts.
New Omni answers require new scope-bound Luna judgements; unchanged old answers
keep their original judgements. Code-scoring workers have a 300-second wall cap
without changing official per-test timeouts. A scorer timeout remains unresolved
infrastructure: fix/retry grading only, not the frozen model answer.

Prepare licensed source files and runtime paths in a private configuration, then:

```bash
CUDA_VISIBLE_DEVICES="" python -m skillev_private.evaluation.ood_export \
  --config "$PRIVATE_EXPORT_CONFIG" --output "$PRIVATE_PANEL_DIR"
CUDA_VISIBLE_DEVICES="" python scripts/run_step0_integrity_paired.py \
  --runtime-config "$PRIVATE_RUNTIME_CONFIG" --private-output "$PRIVATE_RUN_DIR" \
  --run-id "$RUN_ID" --concurrency "$QUALIFIED_CONCURRENCY" --scoring-concurrency 1
```

Use `catalog: "ood"`, the exported six sample counts and the seven-name OOD
thinking map (all false, GPQA has no sampled records). The actor's public source
must point to the deployed repository, never a private dataset directory. Keep
all prompts, answers, per-item scores, judge calls and connection details outside
Git. Preserve native reward semantics through the existing TerminalReward-compatible
environment boundary; no special success inference is added.

Scoped synthetic tests cover population sampling, answer separation, exact native
metrics, ScienceWorld commands, code-sandbox dispatch and incomplete judge output.
The full `CUDA_VISIBLE_DEVICES="" make check` runs on 22049 Linux-local storage,
not the WSL-mounted E: drive. No hash checks or retired approval scripts are used.

## MIT attribution for adapted LiveCodeBench source projection

Copyright (c) 2024 LiveCodeBench

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

The separate upstream APPS and LiveCodeBench private checkouts retain their
original licenses; no dataset or upstream repository copy is committed here.
