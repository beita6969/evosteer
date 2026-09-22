# Seven OOD trajectory fixes and fresh generation (2026-09-12)

## Scope and interpretation

A10 regenerates the **same original 448 IDs**, 64 each for MuSiQue-Ans, NQ-Open,
Omni-MATH, LiveMedBench, ScienceWorld, LiveCodeBench and APPS Introductory.
Seed 0 only. These inspected panels are **development evidence**, not untouched
final tests. There is no sample replacement, best-answer selection, reference
feedback, second solver, or training update. `idea.tex` and `AGENTS.md` are unchanged.

## Evidence from the previous trajectories

- A8 APPS already passed 39/64 (60.94%); its former 4.69% was not the latest score.
  Three outputs used the full 12,000 tokens, with one incomplete Python program.
  LCB passed 30/64 (46.875%); seven hit 12,000, three were syntactically incomplete.
  Long exploratory comments displaced executable code. Other failures included
  real quadratic-time algorithms and incorrect boundary logic, not bad transport.
- MuSiQue had two empty terminal submissions and nine terminal parse failures;
  repeated final-answer declarations and rambling exhausted some budgets.
  NQ had no empty submissions, but one 8,000-token repetition loop. Actual text
  rendering preserved newlines; it was not a double-escaped JSON prompt bug.
- Omni-MATH had twenty capped outputs and one empty submission. Its old Luna-high
  result must not be relabeled as the new medium-API protocol.
- ScienceWorld's recovered 64 episodes had 1,910 explicit unrecognized commands
  across the two native error-message variants (1,890 and 20 respectively).
  Some copied parenthesized disambiguation descriptions instead of sending the
  native choice number; others repeated unavailable-object commands. Numeric
  choices remain entirely the owner's decisions. Twenty episodes ended at a
  negative native score. The mean **raw final score was 21.84375/100**, whereas
  per-case-clipped reward averaged **0.5309375**. Neither is a peak score.
  Thirteen original deadline failures and their costs remain historical evidence.
- LiveMedBench's 64 replies all stopped normally, with no communication failure;
  their mean was only 75.3 output tokens. The copied English no-explanation prompt
  discouraged complete clinical explanations. The prior 12.5824/100 is retained,
  including negative rubric scores; shortness alone is not proof of grading error.

Licensed prompts, answers, rubrics, raw outputs and per-record diagnoses remain
private. The public note contains only aggregate observations and generic defects.

## Frozen new condition

- One adapter-free Qwen3.5-9B owner; native thinking off, skills off, concurrency
  ceiling **32**. Code receives **one uninterrupted 12,000-token** first request;
  other episodes retain 8,000 total. No chunking or reserved continuation.
- All requests explicitly state both remaining episode allowance and the actual
  per-response cap. ScienceWorld retains 256 tokens/call, 200 native interactions,
  a 600-second episode deadline and the original no-simplification condition.
- QA directly requests one final short-answer line. Code permits a concise
  algorithm/complexity explanation **outside** one complete final code block;
  the framework does not write, merge or repair programs. Medical replies may
  contain complete patient-facing explanations, without an internal reasoning
  transcript. This is a new prompt condition, not official generation equivalence.
- Code and Omni use the model card's non-thinking reasoning recommendation:
  temperature 1.0, top-p 0.95. Other tasks retain 0.7/0.8. Top-k 20, presence
  penalty 1.5 and repetition penalty 1.0 are unchanged.
  [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B).
- Predeclared generation phases: code 128, static QA/math/medical 256, then
  ScienceWorld 64. Only one phase generates at a time; completed-phase CPU/API
  grading may overlap. This avoids stacking multiple batch-32 owner queues.
- Public ScienceWorld instructions clarify direct act serialization and native
  numeric disambiguation. Both observed explicit unknown-action messages are
  recorded as rejected, preserving their original text. No oracle list, object
  chooser, automatic recovery action, or gold path is introduced.
- ScienceWorld summaries now include raw final-score mean, bounded reward mean,
  success-at-100, success-at-70, negative-score count and missing-score count.
  The learning reward contract remains unchanged.
- External judges are **gpt-5.6-luna / medium / official API**, after generation,
  using the benchmark's official prompts. Omni's scorer can judge each finalized
  candidate directly instead of first failing on a missing imported-judge file.
  False verdicts are cached just like true ones; API/format failures stop scoring
  and never trigger hidden retries or answer regeneration. Costs include usage
  and failed attempts; credentials are excluded from actors and saved artifacts.

## Upstream implementation boundaries

The existing official MuSiQue/FiD metrics and sandboxed APPS/LCB checkers are
retained; no hidden-test feedback is added to generation. The LCB public prompt
structure permits explanations followed by a complete executable program.
[LCB prompt code](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py),
[ScienceWorld API](https://github.com/allenai/ScienceWorld/blob/main/scienceworld/scienceworld.py),
[LiveMedBench generation](https://github.com/ZhilingYan/LiveMedBench/blob/main/evaluate/run_model.py).

Reference thresholds are diagnostics, not score guarantees. LCB's 65.6% reference
requires more than 62.32% to exceed 95% of that number; with 64 binary cases this
means at least 40 passes. APPS's approximate 42% reference implies 26/64. Other
references require matching evidence/retrieval, judge or success definitions.
No LiveMedBench numeric target has been supplied at launch.

## Execution and validation

Fresh generation launched at 07:10:50 UTC using the existing GPU5 service on the
approved 22049 endpoint, with no additional SGLang or training process. The private
coordinator PID is 124858. The initial targeted suite passed **187 tests**.
One subsequent static-type correction adds only a type cast to terminal reporting;
it does not change generation or scoring behavior. The final CPU `make check` on
22049 passed: **4,448 tests, 12 skipped**, Ruff formatting/lint, mypy and wheel
build/boundary checks. This was one final full validation after the type correction;
docs-only reporting does not repeat it. No independent review agent or hash check
was used. The dedicated Git/W&B agent only handled Git; no training was performed.

## Measured results and remaining failures

All six static panels completed fresh generation and native/API scoring, without
transport or grader infrastructure failures. Their original IDs and denominators
remain 64 each. These are joint prompt/decoding changes, **not a causal estimate of
the effect of transport fixes alone**.

| Benchmark / metric | Immediately previous selected run | A10 |
| --- | ---: | ---: |
| MuSiQue-Ans answer F1 | 70.24% | 54.60% |
| NQ-Open closed-book EM | 23.44% | 17.19% |
| Omni-MATH judge accuracy | 50.00% (Luna high) | 64.06% (Luna medium API) |
| LiveMedBench rubric score | 12.58/100 | 46.04/100 |
| LiveCodeBench pass@1 | 46.88% (30/64) | 53.13% (34/64) |
| APPS Introductory pass@1 | 60.94% (39/64) | 81.25% (52/64) |
| ScienceWorld raw final score | 21.84/100 | 3.19/100 |
| ScienceWorld zero-clipped final score | 53.09/100 | 34.44/100 |

Omni's old and new judge conditions are not interchangeable. APPS exceeds its
approximate diagnostic threshold; LCB is still six passes short of the 40/64
needed for the stated threshold. Other references are not matched controls, and
LiveMedBench's numerical target is still unspecified.

The QA serialization issue improved but accuracy did not: both panels now have
64 nonempty submissions and zero terminal parse failures. MuSiQue changed 12
previously EM-correct answers to wrong and five wrong answers to correct; NQ had
seven and three respectively. Mean generated tokens fell from 1,425.5 to 377.2
for MuSiQue and from 225.2 to 168.9 for NQ. The shorter-answer condition is a
plausible contributor, not an established single cause. These losses are retained,
not silently replaced with older answers.

Code generation used 165 owner calls and 576,893 output tokens for 128 trajectories.
There were 37 within-budget owner clarifications of invalid terminal formatting.
Nine LCB and one APPS trajectory exhausted 12,000 tokens without a complete final
submission; these remain failed candidates. Earlier code blocks in unfinished,
multi-draft responses were not selected or stitched together. Executable but wrong
programs remain model failures, not transport failures.

## ScienceWorld deadline recovery

The original batch-32 phase ended with **31 completed, 29 unfinished and four
unstarted** cases. Its 600-second episode deadline cancelled 27 model requests;
their completion usage is unknown and must not be reported as zero. SGLang remained
healthy. Service logs did not establish a KV-cache exhaustion or authentication
failure. All original job exits, partial calls and native observations are retained.

At 07:54:29 UTC a separate operational condition, A10r1, started **only the 33
unfinished/unstarted cases at concurrency 16** (CPU coordinator PID 343776), reusing
the same existing GPU5 service. Prompts, decoding, source IDs, the 8,000-token
episode budget, 256-token request budget, 200 native steps and 600-second deadline
are unchanged. Selection used completion/timing metadata, never native scores.
Completed wrong answers were not eligible for replacement. The resulting panel
must disclose the 31-original/33-recovery lineage rather than claim uniform batch 32.

All 33 recovered cases completed in **965 seconds**, with zero new transport
failures. The original 31 final outcomes were then scored, completing the exact
64-ID panel. Strict score-100 success is **9/64 (14.06%)**; score-at-least-70 is
**17/64 (26.56%)**. Twenty native outcomes are -100 and 35 terminate at the runner
horizon. The raw mean is **3.1875/100**, not the bounded-reward mean of 34.4375/100.

The action-format guidance did **not** solve the agent's decision failure. Selected
trajectories contain **5,713 native command rejections out of 7,372 tool calls**;
some repeat one rejected command 200 times. Inspection of the rendered requests
confirmed that the attempted action and rejection were delivered in subsequent
contexts, rather than dropped by transport. These are retained policy failures.
No automatic object search, corrective action, reference path or score-based
replacement was added to improve this result. ScienceWorld and both QA panels
regressed; the overall target is **not met**.

## Complete costs and retained artifacts

- All seven panels have 64 scored final trajectories: **448 total**, with zero
  scorer-infrastructure failures. There are **477 actual episode attempts** after
  including the 29 unfinished originals; the four originally unstarted cases do
  not count as extra attempts.
- Owner transport: **11,881 requests, 11,854 recorded responses**, 172,160,478
  reported input tokens and **1,370,461 reported output tokens**. Input usage sums
  repeated histories and is not a unique-text count. Usage for 27 cancelled calls
  remains unknown, so the known token totals are not a complete cost ceiling.
- Native tool executions across *all* attempts: **11,381**, including 4,009 from
  failed attempts. The selected final trajectories account for 7,372. Peer calls
  are zero. Ten code candidates have unresolved terminal-output failures, distinct
  from the 29 original infrastructure-interrupted ScienceWorld attempts.
- External judging: **424 resolved API calls** (64 Omni cases and 360 medical
  criteria), 1,232,766 input and 69,296 output tokens; zero failed/unknown-usage
  judging calls. The temporary credential file was removed after judging.
- Launch-to-final-generation elapsed time was **59m 45s**, including the pause to
  diagnose ScienceWorld deadlines. The private archive retains original failed
  journals as well as recovered candidates, source rows, model outputs, native
  observations, scoring ledgers and validation logs. A separate cost addendum
  explicitly distinguishes all-attempt tool calls from selected-trajectory calls.

The final personal check at 08:15 UTC found all evaluation/validation coordinators
exited, no matching project commands left, and the existing Qwen3.5-9B service
healthy on GPU5 (about 75,278 MiB reserved, 0% utilization). The original service
PIDs 102109/102336 remain running as requested; no new GPU service was launched.
All eight cards were checked before launch and at completion; no other card or
other user's process was used or modified by this evaluation.
