# SKILLEV BayesianImprove Step-0 Exact-Eight 32-Sample Results

> **Historical composite only (superseded 2026-09-06).** All “current/latest/accepted” language
> below belongs to the historical conditions. These figures are not clean no-skill/text-only
> evidence. See the [corrected specification](STEP0_EVALUATION_INTEGRITY_SPEC.md) and
> [separate corrected results](STEP0_PAIRED_NO_SKILL_AND_TEXT_SKILL_RESULTS.md).

> **TriviaQA superseding note (2026-09-02):** The 50.00% EM / 68.84% F1
> TriviaQA row below is a historical condition result. The latest fresh
> 32-record Step-0 run, scored with the complete official alias arrays, is
> **81.25% EM / 85.83% F1** and passes the owner-set 81.00% F1 target. See
> `SKILLEV_BAYESIAN_IMPROVE_STEP0_TRIVIAQA_32_REPAIR_RESULTS.md`. Other rows in
> this historical report are unchanged.

> **AIME superseding note (2026-09-02):** The 70.00% AIME row below is a
> historical condition result. The latest fresh complete all-30 Step-0 run is
> **26/30 = 86.67%** under the repaired frozen Protocol 14-matched reasoning and
> greedy terminal-transcription condition. See
> `SKILLEV_BAYESIAN_IMPROVE_STEP0_AIME_2026_REPAIR_RESULTS.md`. Other rows in
> this historical report are unchanged.

## Authority and scope

This report is the current answer-free result for
`skillev-bayesian-improve-step-zero-exact-eight-32@1`. The numerical rollout
was evaluated from code revision `4ec3c24`. A later identity-only correction
replaced the misleading legacy receipt labels
`skillflow-seeded-controller-step-zero-exact-eight-32@2` and
`skillflow-seeded-controller-step-zero@3`; it did not change prompts, model
outputs, environments, or scores and did not regenerate any candidate. The
owner-authoritative IID catalog is exactly HotpotQA, TriviaQA, AIME 2026,
HealthBench, WebShop, ALFWorld, MBPP+, and HumanEval. SpreadsheetBench,
SWE-bench, and AppWorld are not members of this catalog.
Its promotion target is
`qwen35-current-eight-backbone-strict-improvement-2026-08-31@1`: every
headline Step-0 metric must be **strictly greater than** the corresponding
latest completed Qwen3.5-9B backbone-only score. Equality is a failure.

Seven benchmarks use a result-blind, evenly spaced 32-record projection of the
corresponding frozen 128-record panel. AIME 2026 uses its complete released
30-record population. The run therefore contains 254 planned records in one
seed/configuration. Questions, choices, answers, rubrics, tests, task IDs,
candidate responses, and per-record results remain outside Git.

The requested label `MBPP+ hard` needs an evaluator-level correction: EvalPlus
v0.2.0 defines Base and Plus tests, but no official standalone Hard split. This
report uses the standard Base+Plus Pass@1 contract and does not silently invent
a different population.

## Step-0 architecture identity

The evaluated method is `skillev-bayesian-improve-step-zero@1`. It is the
project's BayesianImprove architecture derived from SkillFlow and `idea.tex`,
not unmodified SkillFlow. The architecture contains the seeded controller,
trajectory-balance/GFlowNet path, Beta–Bernoulli/LCB calibration, and
Operator-driven skill evolution. At Step-0, the inference controller is active
while the trainable/update components are present but deliberately untrained
and inactive:

- frozen adapter-free Qwen3.5-9B for both reasoning and terminal/action
  generation;
- zero optimizer steps and no active forward/backward adapter, TTB update,
  posterior calibration, Operator update, or evolved skill;
- answer-free exact-eight seed library with deterministic typed top-1 retrieval,
  a fixed instruction budget, and valid abstention;
- `seeded-step-zero@1` H0 with the benchmark-native source, stable root query,
  selected compatible guidance, and current action surface only;
- frozen deterministic reasoning outside the trainable action-edge probability;
- benchmark-matched evaluation decoding, kept physically separate from the
  training raw-softmax generator;
- benchmark-native typed terminal wires and syntax constraints, with Python
  source and natural-language answers no longer forced through JSON strings;
- one persistent architecture episode per WebShop or ALFWorld native episode;
  deterministic controller memory is derived only from public state and stays
  outside the executable action.

## Implemented repair map

| Audit item | Implemented behavior |
|---|---|
| Evaluation/training sampler collision | Dedicated evaluation generator requires an explicit benchmark profile; training raw-softmax remains a separate API. |
| Unpaired comparisons | Exact task-ID joins against the frozen backbone panel; paired deltas, 10,000-resample paired bootstrap intervals, binary outcome cells, and exact McNemar tests. |
| Stale condition claims | The historical 16-record report is explicitly superseded; this result names its condition and evaluated code revision. |
| Wildcard retrieval | Typed benchmark/tool applicability, deterministic ranking, top-1 budget, and abstention replace unconditional all-skill injection. |
| Duplicate skill exposure | Step-0 uses full-inline/no-invoke, so an injected skill cannot also consume an invocation turn. |
| Training-state H0 | A dedicated Step-0 profile excludes posterior/training diagnostics and irrelevant tools. |
| Constant partition query | Each task has a stable answer-free root query; HotpotQA separates the question from all ten source passages, and interactive steps reuse one root query. |
| Unscored stochastic reasoning | Step-0 reasoning is adapter-free, frozen, greedy, and explicitly outside the action probability. |
| Memory inside action | WebShop/ALFWorld executable actions contain only native arguments; deterministic public-state memory cannot invalidate an otherwise legal action. |
| Fragile universal JSON completion | Grammar-constrained native action JSON plus dedicated short-answer, integer, natural-language, and Python-source terminal paths. |
| Stateless interactive bridge | Retrieval and snapshot pinning occur once per native episode; observations update one typed episode state until terminal cleanup. |

## Benchmark-native conditions

- **HotpotQA:** the question and all ten released passages are model-visible;
  official-style normalized EM/F1 is retained.
- **TriviaQA:** exactly one task-local search uses the answer-isolated detailed
  Codex/Wikipedia search-v2 corpus, followed by a short-answer terminal. The
  private 32-record corpus has 16 reused and 16 newly constructed research
  plans, 4,398 sanitized passages, 384 Wikipedia associations, and 382 unique
  pages. No evaluator alias is model-visible.
- **AIME 2026:** Qwen-native thinking, a grammar-constrained integer in
  `[0,999]`, answer-blind boxed projection, and official integer exact match.
- **HealthBench:** direct natural-language terminal followed by the pinned
  `simple-evals` rubric semantics. Both candidate and judge are local
  Qwen3.5-9B SGLang; no OpenAI API or GPT judge is used. This is a Qwen-local
  diagnostic, not an official GPT-4.1-comparable score.
- **WebShop:** official native environment, current-surface search/click
  actions, deterministic public constraint ledger, native reward and success.
- **ALFWorld:** official TextWorld environment, exact current admissible
  actions, typed public subgoal state, and native success.
- **MBPP+:** exact Python-source terminal and pinned EvalPlus v0.2.0 Base+Plus
  evaluator.
- **HumanEval:** exact Python-source terminal and original HumanEval tests, not
  HumanEval+.

## Results

All 254 records have definitive benchmark outcomes and the final unresolved
infrastructure-failure count is zero.

| Benchmark | N | Step-0 metric |
|---|---:|---:|
| HotpotQA | 32 | EM 65.63%; F1 82.84% |
| TriviaQA detailed search v2 | 32 | EM 50.00%; F1 68.84% |
| AIME 2026 | 30 | Accuracy 70.00% |
| HealthBench Full, Qwen-local judge | 32 | Native rubric mean 54.10% |
| WebShop | 32 | Average score 38.88%; SR 15.63% |
| ALFWorld | 32 | SR 31.25% |
| MBPP+ v0.2.0 | 32 | Base+Plus Pass@1 90.63% |
| HumanEval | 32 | Pass@1 93.75% |

## Exact paired backbone comparison

The non-regression rule is strict: `Step-0 - backbone > -7.00 pp`. Confidence
intervals describe uncertainty but do not replace that predeclared point-estimate
gate.

| Benchmark / metric | Paired backbone | Step-0 | Delta | Paired bootstrap 95% CI | `<7 pp` non-regression |
|---|---:|---:|---:|---:|---|
| HotpotQA EM | 68.75% | 65.63% | -3.13 pp | [-9.38, 0.00] | PASS |
| HotpotQA F1 | 85.99% | 82.84% | -3.15 pp | [-9.42, 0.00] | PASS |
| TriviaQA EM | 43.75% | 50.00% | +6.25 pp | [-9.38, 21.88] | PASS |
| TriviaQA F1 | 61.99% | 68.84% | +6.85 pp | [-5.21, 20.09] | PASS |
| AIME 2026 accuracy | 73.33% | 70.00% | -3.33 pp | [-20.00, 13.33] | PASS |
| HealthBench native rubric mean | 51.75% | 54.10% | +2.35 pp | [-5.78, 11.74] | PASS |
| WebShop average score | 39.68% | 38.88% | -0.80 pp | [-9.12, 7.37] | PASS |
| WebShop success rate | 15.63% | 15.63% | 0.00 pp | [0.00, 0.00] | PASS |
| ALFWorld success rate | 56.25% | 31.25% | -25.00 pp | [-46.88, -3.13] | **FAIL** |
| MBPP+ Base+Plus Pass@1 | 68.75% | 90.63% | +21.88 pp | [6.25, 37.50] | PASS |
| HumanEval Pass@1 | 90.63% | 93.75% | +3.13 pp | [-6.25, 12.50] | PASS |

The exact binary pairing diagnostics are:

| Metric | Both pass | Backbone only | Step-0 only | Both fail | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| HotpotQA EM | 21 | 1 | 0 | 10 | 1.0000 |
| TriviaQA EM | 12 | 2 | 4 | 14 | 0.6875 |
| AIME accuracy | 18 | 4 | 3 | 5 | 1.0000 |
| WebShop success | 5 | 0 | 0 | 27 | 1.0000 |
| ALFWorld success | 7 | 11 | 3 | 11 | 0.0574 |
| MBPP+ Pass@1 | 21 | 1 | 8 | 2 | 0.0391 |
| HumanEval Pass@1 | 28 | 1 | 2 | 1 | 1.0000 |

Binary rows additionally retain all four paired outcome cells and exact
two-sided McNemar statistics in private aggregate evidence. They are not used
to select or rerun candidates.

## Owner-authoritative strict improvement targets

The Step-0 promotion gate is not the earlier seven-point non-regression gate
and is not the paper-reference gate used to assess backbone-only parity. Its
thresholds are the owner's latest completed Qwen3.5-9B backbone-only results:
128 records per benchmark except the complete 30-record AIME population. Those
backbone results were completed at different times, rather than in one
synchronized run. The requested 32-record Step-0 diagnostic is compared with
those aggregate thresholds without claiming that the populations form a
paired statistical estimate.

The rule for every row is `Step-0 > backbone`; equality does not pass.

| Benchmark / metric | Strictly greater than | Step-0 | Delta | Status |
|---|---:|---:|---:|---|
| HotpotQA EM | 62.50% | 65.63% | +3.13 pp | PASS |
| HotpotQA F1 | 78.81% | 82.84% | +4.03 pp | PASS |
| TriviaQA EM | 51.56% | 50.00% | -1.56 pp | **FAIL** |
| TriviaQA F1 | 59.17% | 68.84% | +9.67 pp | PASS |
| AIME 2026 accuracy | 86.67% | 70.00% | -16.67 pp | **FAIL** |
| HealthBench Qwen-local native rubric mean | 53.17% | 54.10% | +0.93 pp | PASS |
| WebShop average score | 48.60% | 38.88% | -9.72 pp | **FAIL** |
| WebShop success rate | 23.44% | 15.63% | -7.82 pp | **FAIL** |
| ALFWorld success rate | 61.72% | 31.25% | -30.47 pp | **FAIL** |
| MBPP+ Base+Plus Pass@1 | 67.19% | 90.63% | +23.44 pp | PASS |
| HumanEval Pass@1 | 92.19% | 93.75% | +1.56 pp | PASS |

For AIME, **80.00%** remains the separate owner-defined minimum for the
backbone-only benchmark. Because Step-0 must improve on the current 86.67%
backbone result, its architecture target is strictly above 86.67%, not 80.00%.
HealthBench is compared only under the same local Qwen3.5-9B judge contract;
this does not make it comparable to an official GPT-judge result. The requested
`MBPP+ hard` label continues to use the truthful EvalPlus Base+Plus contract,
because EvalPlus v0.2.0 has no official standalone Hard split.

## Failure and retry accounting

The final outcome inventory is 254/254 definitive. HotpotQA, TriviaQA, AIME,
and HumanEval each produced an extractable candidate for every record;
HealthBench produced 32 non-empty candidates and 32 rubric scores. MBPP+
contains 29 successes and three definitive candidate/test failures. WebShop
contains five successes and 32 aggregate native invalid-action events;
ALFWorld contains ten successes and 108 native invalid-action events. Those
candidate and environment outcomes were retained rather than retried.

Two retry classes were used:

1. the first WebShop launch lacked its pinned Java environment, so all 32
   records failed before a model action; the corrected run evaluated exactly
   those 32 records; and
2. the first HealthBench grader rejected a semantically identical service
   whose public route name did not match its pinned profile; the retry used the
   matching adapter-free Qwen route and graded only the 32 already generated
   candidates. It required one grader JSON-format repair and zero transport
   retries.

The paired TriviaQA backbone run also had two pre-generation setup failures
(a path-type mismatch and an endpoint suffix mismatch). Its corrected attempt
generated the exact 32-record search-v2 panel once and completed with zero
infrastructure or candidate failures.

Only explicitly typed infrastructure failures were eligible for retry.
Successful samples and candidate failures were immutable, and no record was
regenerated or scored twice in the final aggregate. The initial WebShop launch
lacked its required pinned Java environment and produced only infrastructure
failures before any model action; its replacement run retried exactly that
failed set under the same frozen condition.

## Decision

**Strict Step-0 improvement: NO-GO.** Six of eleven headline metric rows are
strictly above the owner-supplied current backbone scores. TriviaQA EM, AIME
accuracy, both WebShop metrics, and ALFWorld success do not pass. In particular,
AIME is 16.67 points below its Step-0 improvement threshold even though the
separate backbone-level minimum remains 80.00%.

**Paired architecture non-regression: NO-GO.** Ten of eleven exact-panel metric
rows pass the older seven-point diagnostic rule, but ALFWorld regresses by
25.00 points. The 32-record intervals are wide, so positive point estimates on
TriviaQA, HealthBench, MBPP+, and HumanEval do not establish a trained-method
gain.

**Formal training: NO-GO.** Besides the failed score gates, the repaired frozen
Step-0 lane does not itself prove that the general trainable rollout engine has
adopted the same deterministic, adapter-free authority for every unscored
reasoning pass. That probability contract must remain a formal-training
precondition rather than being inferred from this evaluation adapter.

This is an optimizer-step-zero diagnostic of the BayesianImprove architecture.
It cannot establish learned gains from TTB/GFlowNet updates,
Beta–Bernoulli/LCB calibration, or Operator-driven skill evolution because
those update mechanisms were not active in this no-training run.
