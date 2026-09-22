# HealthBench conversation and grader interface — 2026-09-08

Scope: the fixed 32-task HealthBench development panel, A2, Qwen3.5-9B,
single owner, no training/adapter/skills, thinking off. Target: 45/100.
A score below that target is not, by itself, evidence of an evaluator defect.

## Findings and changes

- The previous seven-IID run scored **38.4506/100**. All 32 released conversations
  survive source projection, including prior user/assistant turns. All 40 rendered
  owner requests contain the original dialogue; no messages were archived or lost.
  Inputs contain 534–1,595 tokens. Eight additional calls were owner-requested
  public history reads, not consultations or scorer-driven retries.
- Both runtime budget notices were appended as `tool` messages despite no matching
  tool call. The actual Qwen tokenizer renders these as a new user turn containing
  `<tool_response>`. Runtime counters now join the opening system message, leaving
  the real dialogue as dialogue. Qwen rejects a system message anywhere except the
  beginning, so a tail-system replacement is not compatible. Current counters can
  reduce prefix-cache reuse on subsequent calls; budget enforcement is unchanged.
- HealthBench now explicitly asks the assistant to continue the conversation using
  its preceding dialogue. This is neutral public task framing, not a medical
  checklist, rubric hint, mandatory reasoning process or answer-format restriction.
  Original system instructions, where present, are retained.
- The active IID judge actually used **temperature 0.5**, but its verifier label
  incorrectly said temperature 0. New IID records have a corrected verifier label;
  the actual temperature remains **0.5**. The separate training judge's temperature-0
  profile is not changed. Old scores and their original records are not overwritten.
- Resolve the effective IID judge profile before freezing the run, then pass its
  sampling/token controls explicitly to the transport. Preserve the original
  per-rubric calls, 2,048-token cap, thinking-off mode and retry policy.
- Previously, native per-rubric verdicts and explanations were discarded. New
  records retain them, together with actual judge request/response bodies and
  completion/usage status, **only in the private scorer journal**. Partial transport
  failure remains missing grading, never a fabricated negative verdict. No rubric,
  judge explanation or grading feedback is returned to the evaluated owner.

The scorer still follows the
[upstream HealthBench calculation](https://github.com/openai/simple-evals/blob/main/healthbench_eval.py):
earned positive and negative points divided by total positive points, then clipping
only the aggregate mean. No rubric, label, denominator, panel membership or candidate
selection rule was changed. This remains a local Qwen-judged result, not an official
GPT-judge-comparable score.

## Verification

The focused 19 tests cover public conversation role/content preservation, full
natural-language submission, answer isolation, live budgets, frozen judge settings,
complete/private rubric evidence and partial-failure cost accounting. They passed
on the approved CPU host (4.82 seconds). Local Ruff passed; local Python-based
verification timed out without producing results and is not reported as passing.

## Retrieved result of the first repair

Code revision: `a99bb9f`. Fresh run:
`health32-a2-conversation-a99bb9f-20260908T091800Z`, launched September 8 at
09:18:11 UTC, using only the existing GPU 5 service on endpoint 22048.
No other benchmark, training run or additional GPU service was launched by this task.

The earlier SSH outage delayed retrieval, not scoring. The run completed at
**09:20:04 UTC**, in **112.41 seconds**, with **32/32** candidates and scores:
**36.20596/100**, below both the previous 38.45056 and the 45 target. There were
47 owner calls, 14 history reads, one interface repair, no peer calls and no
candidate/infrastructure failures. There were 401 grader requests for 400 rubrics
(one malformed-JSON repair). Old answers and scores were not imported or overwritten.

All 47 rendered owner requests retained the source conversation, and all 401
grader requests retained the original conversation plus the submitted response.
Recomputation from saved verdicts agrees with all 32 raw scores. The 400 verdicts
contain 279 positive criteria (121 met) and 121 negative criteria (28 triggered).
These are aggregate diagnostics, not feedback delivered to the owner.

That revision's full check had one stale test expecting the old tool-role budget
notice. The subsequent `8d62eae` milestone fixed the test, without changing the
HealthBench implementation, and passed the 22049 CPU full check: 3512 tests, one
explicit CUDA skip, format/Ruff, mypy and both wheels. This is not a claim that
the newly described follow-up below has already passed its own final check.

## Follow-up: conversation/history and submission semantics

- Independent comparison with the official 5,000-case release matched the fixed
  32 public conversations and rubrics exactly. Project IDs add a `healthbench/`
  namespace to official `prompt_id`; comparison accounts for that namespace.
  Public projection is also identical, and the pinned upstream grader template
  matches the installed template. No missing-context/source-substitution defect
  was found in these checks.
- The history capability described earlier public records, but its implementation
  exposed only this episode's execution archive, not the supplied conversation.
  Saved answers sometimes treated those empty execution logs as evidence about
  the user's background or external records. History now offers a distinct
  read-only `conversation` page and explains each archive's scope. Original
  dialogue remains verbatim in every prompt; history is not an external search.
- The generic submission description asked for a final answer/code rather than
  discussion. HealthBench instead requires the next assistant turn. The public
  task and submit tool now describe a conversational reply, including an
  explanation or clarification question when the owner chooses. This adds no
  required question, medical checklist, rubric hint or extra model call.
- Some saved local-Qwen judge explanations appear to confuse negation or earlier
  assistant statements with the final response being scored. This is a judge
  reliability concern, not evidence that the transport omitted context. Do not
  flip such verdicts automatically, alter the rubric, or substitute a stronger
  judge without naming a new evaluation condition.

## Completed follow-up: 45.03364/100

Source `0937752`; run `health32-a2-conversational-history-0937752-20260908T175500Z`.
The same ordered 32-case panel was freshly generated and graded in full. Actor
seed 0, temperature 0.7 / top-p 0.8 / top-k 20, 8 calls, 8,192 tokens per call,
32,768 total output tokens, thinking off and skills off are unchanged. All 400
grader requests used the preceding run's effective temperature-0.5 / 2,048-token /
thinking-off profile. No old answer, vote, fallback or score-based retry was used.

| Run | Native local-Qwen score / 100 | Owner calls | History reads |
|---|---:|---:|---:|
| Original seven-IID round | 38.45056 | 40 | 8 |
| First conversation repair, `a99bb9f` | 36.20596 | 47 | 14 |
| Conversation/history repair, `0937752` | **45.03364** | **33** | **1** |

The new run completed **32/32 candidates and scores**, with zero candidate,
infrastructure, channel, interface-repair or peer-call failures. The sole history
read retrieved the original conversation rather than empty execution records.
All 33 owner requests retain the complete public dialogue; all 400 grader requests
retain that dialogue and the submitted reply. All 32 rubric lists and score
calculations match their sources, and all 32 submissions match the actual owner
output in the same run/arm/episode/call scope. Call IDs alone are episode-local.
The signed native scores include one negative case; only the aggregate is clipped.
The local judge marked 139/279 positive criteria met and triggered 21/121 negative
criteria. These are its verdicts, not independently certified clinical judgments.

Start: **17:54:42.774 UTC**; summary: **17:56:12.822 UTC**, **90.05 seconds**
(about 1,279 completed cases/hour for this small warm-service run). This is about
20% less elapsed time than the previous 112.41-second run, not a controlled
training-throughput claim. Actor cost: 26,191 input / 12,722 output tokens;
judge cost: 526,585 input / 40,233 output tokens, 400 requests/responses and no
schema repairs. Summed parallel scorer time is not end-to-end elapsed time.
[Aggregate-only evidence](machine-results/step0_health32_conversational_history_2026-09-08.json).

**The requested 45 threshold is met in this run, by only 0.03364 points.** This
does not establish a robust margin, performance on 128 unseen cases, matched
backbone improvement, or absence of factual errors. The panel has been inspected
for development, and the local judge has the reliability limitations noted above.
Do not rerun unchanged conditions until a more favorable score appears or relabel
this as an official GPT-judge result. The two interface changes were evaluated
together; their individual causal contributions have not been isolated.

## Follow-up validation and resources

- Main-thread implementation and self-review; no coding, review or watching agents.
- Local changed-file Ruff/format passed. Local pytest (150 seconds) and targeted
  mypy (120 seconds) timed out before producing results; neither is a pass claim.
  The affected tests were moved to the approved CPU host rather than repeatedly
  retrying the WSL-mounted filesystem: **154 passed in 21.65 seconds**.
- The single final `CUDA_VISIBLE_DEVICES="" make check` on 22049 finished at
  **17:59:35 UTC**, exit 0: **3518 passed**, one explicit CUDA-only skip,
  119 existing PEFT warnings, pytest 261.96 seconds; format/Ruff, mypy and both
  wheels/model-package boundary passed. No extra standalone full test/build/lint,
  unrelated E2E or hash check was added. Documentation-only follow-up reuses it.
- The one new HealthBench evaluation used only the existing **22048 physical
  GPU 5** SGLang service. No new service, other benchmark, training or additional
  GPU was launched. The evaluation coordinator and CPU check have ended; the
  pre-existing inference service is retained. Peak observed generation utilization
  was 100%; the service occupied about 69.8 GiB of device memory.

No coding/review subagents or hash checks were used. Licensed conversations, rubric
criteria, answers and per-task results remain in private storage, outside Git.
