# SKILLEV BayesianImprove Step-0 IID 16-Sample Diagnostic

> **Historical record (pre-v7).** This document preserves the earlier
> 16-record diagnostic and its then-current adapter behavior. It is not a score
> for the repaired `skillev-bayesian-improve-step-zero-exact-eight-32@1`
> condition. In particular, later code separates benchmark evaluation decoding
> from the training raw-softmax sampler, uses benchmark-typed retrieval and
> terminal wires, keeps controller memory outside executable actions, and keeps
> one architecture episode alive for each native interactive episode. Current
> claims must cite a fresh result whose condition receipt matches the code that
> produced it.

## Scope

This note records the repaired no-training diagnostic for the current IID list:
HotpotQA, TriviaQA, AIME 2026, HealthBench, WebShop, ALFWorld, MBPP+, and
HumanEval. It measures the optimizer-step-zero inference path, not a trained
method and not the frozen 128-sample gate.

The architecture is the project's BayesianImprove method derived from
SkillFlow and `idea.tex`, not unmodified SkillFlow. The run used the same
deterministic, evenly spaced 16-record projection of each
frozen panel as the preceding diagnostic. At 16 records, one accuracy or
success observation changes the estimate by 6.25 percentage points, so every
target comparison below is provisional. Licensed task text, answers, model
responses, and per-record traces remain in private evaluation storage and are
not part of Git.

The requested name `MBPP+ hard` needs one evaluator-level correction. The
current source and pinned EvalPlus v0.2.0 evaluator define the standard MBPP+
Base+Plus criterion, not a separately defined frozen `hard` split. This note
therefore reports the actually executable MBPP+ Base+Plus evaluation rather
than silently relabelling it as a hard subset.

## Inference condition

The historical condition was `optimizer-step-0-no-update`:

- frozen Qwen3.5-9B backbone served adapter-free by SGLang;
- the canonical two-pass reasoning/action controller;
- `SkillLibrary` initialized from the three answer-free seed skills;
- `TaskConditionedSkillRetriever` and canonical H0 assembly;
- bounded multi-turn structured-action execution;
- zero optimizer, TTB/GFlowNet, posterior, and Operator updates.

This exercises the initial controller, seed-skill retrieval, and action
interface. It does not measure improvements that can arise only after
trajectory-balance training, Bayesian posterior updates, or Operator evolution.
The scientific architecture was not changed. The repairs are confined to the
evaluation/runtime boundary needed to execute that architecture faithfully:

1. contiguous leading benchmark system instructions are content-preservingly
   merged into the controller's sole Qwen system turn;
2. the native exact-token SGLang generator can represent an adapter-free frozen
   base request without inventing a LoRA identity;
3. parse/schema failures and retrieved-skill invocations are consumed inside a
   bounded SkillFlow rollout instead of leaking to the benchmark environment;
4. only valid native `search`/`click` or `act` actions cross the WebShop and
   ALFWorld environment boundary; and
5. AIME uses an explicit boxed-integer completion surface, with an answer-blind
   projection that adds the required box only when the candidate itself already
   exposes one unambiguous integer;
6. the focused AIME rerun enables Qwen's native thinking chat template only for
   the architecture reasoning pass, gives the first solve turn the matched
   32,768-token ceiling, and returns JSON-repair turns to the ordinary
   1,024-token reasoning budget;
7. WebShop and ALFWorld require model-authored cumulative memory inside the
   structured tool action and project it back to the exact public
   `Memory`/`Thought`/`Action` transcript consumed by memory v5; and
8. the focused TriviaQA rerun adds the separate, answer-isolated detailed local
   search v2 surface rather than silently treating direct RC context as the
   requested Wikipedia synthesis track; and
9. the two long-horizon interactive bindings may use up to 49,152 H0 tokens
   under the 98,304-token service context. This preserves the public ReAct
   ledger and native history instead of turning a valid late step into an
   infrastructure failure at the generic 32,768-token H0 ceiling.

## Benchmark-visible inputs and scorers

- **HotpotQA:** question plus all ten released context passages; official-style
  normalized EM/F1.
- **TriviaQA:** the earlier eight-panel run used the released SkillFlow TriviaQA
  RC rendering and did give the model its non-empty evidence prefix. The latest
  focused rerun is intentionally a different diagnostic: it removes that RC
  preview, requires one task-local search, and supplies snippets from detailed
  Codex research dossiers plus English Wikipedia. It is compared with the
  matched 128-record detailed-search-v2 backbone track, not presented as the
  Protocol 13 direct-context score.
- **AIME 2026:** public math problem under the boxed-integer response contract;
  Qwen native thinking in the reasoning pass; exact integer scoring. The
  16-sample diagnostic is a projection of the all-30 population, not a claim to
  be the formal all-30 result.
- **HealthBench:** original released multi-turn conversation. Candidate and
  rubric grading both used local Qwen3.5-9B SGLang; the judge was not GPT or an
  OpenAI API. The metric is therefore a Qwen-local diagnostic, not an official
  GPT-4.1-comparable score.
- **WebShop:** source-proven memory-v5 prompt asset, model-authored cumulative
  memory, official native environment observations/actions, native reward and
  success.
- **ALFWorld:** task-specific source-proven memory-v5 prompt asset,
  model-authored cumulative memory, official TextWorld observations and
  admissible actions, native success.
- **MBPP+:** public MBPP function prompt and pinned EvalPlus v0.2.0 Base+Plus
  evaluator.
- **HumanEval:** public function prompt and original HumanEval tests kept behind
  the trusted scorer boundary.

## Latest focused parity repair

The latest focused rerun corrects four remaining adapter mismatches without
changing the scientific architecture, seed skills, optimizer state, or any
training code:

- **AIME:** the reasoning pass now uses the tokenizer's Qwen-native thinking
  branch with the matched 32,768-token output ceiling on the first solve turn.
  If its structured action is invalid, later repair turns retain native
  thinking but use the ordinary 1,024-token reasoning budget rather than
  repeating another 32,768-token solve. The action pass remains the
  architecture's non-thinking structured forward-policy pass, and the trusted
  scorer still accepts only the candidate's boxed integer.
- **WebShop / ALFWorld:** the step-zero bridge now exposes `memory` as a required
  model-authored tool argument. It returns the selected native action as an
  exact `Memory`/`Thought`/`Action` transcript, so the authoritative memory-v5
  outer runner carries the cumulative public-state ledger into the next native
  observation instead of dropping it at every architecture boundary. The
  evaluation-only H0 ceiling was raised to 49,152 after late-step H0 values of
  34--38k tokens proved that the generic 32,768 limit was clipping valid public
  history. WebShop's pinned official worker was also launched with its required
  pinned JDK environment; a missing `JAVA_HOME` is classified as infrastructure,
  never as a candidate zero.
- **TriviaQA:** turn one exposes only the required task-local search tool and
  turn two exposes only final completion. The request binding recognizes the
  search runner's outer-turn IDs, so retrieved snippets enter the same
  answer-free H0 and two-pass controller rather than bypassing the architecture.

For the 16 selected TriviaQA tasks, the private corpus contains 16 detailed
Codex dossiers of 2,292--2,619 characters, eight research queries and eight or
nine entities per task, 192 Wikipedia-page associations (190 unique pages),
and 2,370 indexed passages. The final private removal pass found zero literal
evaluator aliases in any model-visible title or passage. The model made exactly
one search for every record. Questions, plans, pages, labels, snippets, and
per-record outputs remain outside Git.

This adapter was checked against both
`QWEN35_9B_BACKBONE_ONLY_IID_128_FILE_AND_SCRIPT_MAP.md` and the concrete
`QWEN35_9B_TRIVIAQA_SEARCH_AUGMENTED_V2_RESULTS.md` corpus contract. It therefore
uses the detailed answer-isolated Codex/Wikipedia synthesis lane used by the
matched backbone diagnostic, rather than a generic question-only search or the
released RC preview.

The focused comparison is result-blind and panel-aware. For AIME, WebShop, and
ALFWorld, the reference is recomputed from the preserved memory-v5 backbone run
using exactly the same 16 task IDs: 75.00% AIME accuracy, 26.98% WebShop average
score / 12.50% success, and 50.00% ALFWorld success. No task identity or
per-record value enters Git. The earlier detailed-search-v2 TriviaQA artifact is
available publicly only as its aggregate, so TriviaQA is compared with that
same-condition 128-record aggregate (58.59% EM / 67.88% F1) rather than with a
different direct-context track.

## Focused rerun results

The four repaired panels produced 64/64 definitive scorer outcomes, with zero
remaining generation, environment, retrieval, or scorer infrastructure
failures. Only records whose preceding attempt had no definitive scorer result
were resumed; no successful or candidate-failure record was regenerated for an
infrastructure retry.

| Benchmark | Focused 16-sample result | Matched reference | Provisional comparison |
|---|---:|---:|---|
| TriviaQA detailed search v2 | EM 56.25%; F1 71.07% | EM 58.59%; F1 67.88% | absolute gaps 2.34 / 3.19 pp; within band |
| AIME 2026 | Accuracy 56.25% | same-ID 16: 75.00% | gap 18.75 pp; outside band |
| WebShop | Average score 23.23%; SR 6.25% | same-ID 16: 26.98%; 12.50% | gaps 3.75 / 6.25 pp; within band |
| ALFWorld | SR 25.00% | same-ID 16: 50.00% | gap 25.00 pp; outside band |

The AIME change raised accuracy from the earlier 6.25% adapter result to
56.25%. Four of the final 16 records still ended as definitive candidate parse
failures; they remain zeroes and were not retried. The first long thinking pass
ran at roughly 30 generated tokens/second on its dedicated service. Limiting
later repair reasoning to 1,024 tokens removed the accidental repeated-32k
latency without changing the answer, parser, or scorer contract.

WebShop completed at 23.23% average reward / 6.25% success with 7.88 mean native
steps. Its trace recorded 84 model-authored memory updates and 42 invalid
responses on which the outer runner correctly retained the prior ledger.
ALFWorld completed at 25.00% success with 17.62 mean native steps, 276 memory
updates, and six carried-memory invalid responses. Thus both environments now
receive cumulative ReAct memory and have complete native outcomes. WebShop is
inside the matched same-ID band; ALFWorld's remaining gap is a genuine
optimizer-step-zero policy gap rather than a missing-memory or parser-induced
zero.

TriviaQA used one task-local search on every record and the detailed
Codex/Wikipedia corpus described above. Its two headline metrics are inside the
matched search-v2 band. This directly answers the context question: the focused
run did not use the released RC context, but it did use the same detailed local
Wiki-synthesis evaluation lane as the corresponding backbone diagnostic.

The focused conclusion remains **NO-GO** because AIME and ALFWorld are outside
the strict `<7 pp` matched-reference band. This is an untrained diagnostic, so
the remaining gap is not evidence that TTB/GFlowNet, posterior calibration, or
Operator evolution has failed; none of those updates ran here.

## Earlier repaired results

All eight panels produced 16 definitive scorer outcomes; the final
infrastructure-failure count was zero.

| Benchmark | Repaired 16-sample result | Current reference | Provisional comparison |
|---|---:|---:|---|
| HotpotQA | EM 50.00%; F1 70.39% | EM 60.94%; F1 75.70% | gaps 10.94 / 5.31 pp; EM outside band |
| TriviaQA | EM 37.50%; F1 47.32% | EM 44.88%; F1 54.30% | gaps 7.38 / 6.98 pp; EM narrowly outside strict band |
| AIME 2026 | Accuracy 6.25% | 46.67% | gap 40.42 pp; outside band |
| HealthBench | Qwen-local native rubric mean 34.62% | external anchor 44.68% | diagnostic only; judge conditions are not matched |
| WebShop | Average score 19.17%; SR 0.00% | 56.93%; 32.03% | gaps 37.76 / 32.03 pp; outside band |
| ALFWorld | SR 6.25% | 48.28% | gap 42.03 pp; outside band |
| MBPP+ | Base+Plus Pass@1 50.00% | no matched target | diagnostic only |
| HumanEval | Pass@1 75.00% | 89.06% | gap 14.06 pp; outside band |

The final decision remains **NO-GO**. A 16-record projection is not a formal
gate, and several matched metrics are already outside the strict `<7 pp` band.
HealthBench and MBPP+ still lack condition-matched formal targets in the current
registry.

## What the repair changed empirically

The prior bridge reported WebShop and ALFWorld as 0 with every record ending in
`candidate-invalid`; those values were not valid estimates of method quality.
After the bounded internal action loop was connected to the native environment:

- WebShop produced 16/16 definitive native outcomes, a 19.17% average reward,
  0% success, and 8.06 mean native steps;
- ALFWorld produced 16/16 definitive native outcomes, 6.25% success, and 11.81
  mean native steps; and
- neither benchmark had an environment or generation infrastructure failure.

The remaining low interactive scores are therefore candidate/task failures on
this small untrained panel, not action-parser zeroes or missing environments.
The original AIME run had 16/16 empty parser outcomes. A separate answer-blind
shape diagnostic reproduced a valid `complete` action whose answer field was a
bare integer, which the boxed-only benchmark parser correctly rejected. The
repair makes the public completion contract explicit and performs only a
target-independent representation projection; it does not inspect, infer, or
replace the candidate answer. In the repaired run, 12/16 responses reached the
boxed parser and four remained candidate-level horizon failures; all 16 still
received definitive scores, yielding 6.25% accuracy with zero scorer
infrastructure failures.

Relative to the earlier one-turn diagnostic on the same panels, TriviaQA,
MBPP+, and HumanEval improved from 31.25%/37.95%, 25.00%, and 25.00% to
37.50%/47.32%, 50.00%, and 75.00%, respectively. HotpotQA and HealthBench moved
to 50.00%/70.39% and 34.62%. These changes are descriptive only: with 16
records and a changed execution adapter, they should not be interpreted as
trained gains. AIME moved from a parser-induced 0% to 6.25%, but remains far
outside its reference band.

## Remaining limitations of this historical run

- This is an untrained, one-seed, 16-record diagnostic, not the formal
  128-record evaluation.
- WebShop and ALFWorld in this historical run instantiated one bounded internal
  SkillFlow rollout per native outer step. The repaired condition superseding
  this record persists one architecture episode and deterministic controller
  state across the native episode.
- HealthBench self-grading with Qwen is intentionally not comparable to the
  official GPT-4.1 judge setup.
- The standard EvalPlus Base+Plus result cannot substantiate a separately named
  `MBPP+ hard` target until such a split and target are formally defined.
