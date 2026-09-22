# SKILLEV BayesianImprove Step-0 Exact-Eight IID 128-Panel Results

> **Historical protocol only (superseded 2026-09-06).** The condition and acceptance language below
> are retained as historical records, not current clean no-skill/text-only evidence or a causal
> architecture gain. Legacy routing, policies, material preparation and extra generation must not
> be conflated with the corrected [integrity conditions](STEP0_EVALUATION_INTEGRITY_SPEC.md).
> Corrected full-panel results are tracked [separately](STEP0_PAIRED_NO_SKILL_AND_TEXT_SKILL_RESULTS.md).

## Scope

This report records the latest accepted no-training inference results for
`skillev-bayesian-improve-step-zero@1`. The evaluated method is the project's
SkillFlow-derived BayesianImprove architecture whose scientific method is
defined by `idea.tex`; it is not unmodified SkillFlow and it is not a
backbone-only condition. Following the owner's final scope decision, WebShop
and ALFWorld were rerun as fresh complete 128-record panels; AIME retains its
already accepted complete result, and the other five benchmarks retain their
latest complete results. These rows therefore describe one frozen condition,
but not one simultaneous wall-clock launch.

Frozen condition identity:
`skillev-bayesian-improve-step-zero-exact-eight-128@10`.

The owner-authoritative IID catalog is exactly:

1. HotpotQA
2. TriviaQA
3. AIME 2026
4. HealthBench
5. WebShop
6. ALFWorld
7. MBPP+ hard
8. HumanEval

Every benchmark uses 128 frozen records except AIME 2026, for which all 30
released records are used. The complete condition therefore contains **926
records**. `MBPP+ hard` is reported truthfully through the available EvalPlus
v0.2.0 Base+Plus contract; EvalPlus does not define a standalone official
split named “Hard.” SpreadsheetBench, SWE-bench, and AppWorld are not part of
this condition.

## Step-0 method identity

The Qwen3.5-9B base weights were frozen and adapter-free. The answer-free seed
skill library, deterministic top-1 typed-skill retrieval, and seeded inference
controller were active. Training and evolution were not:

| Component | State |
|---|---|
| Qwen3.5-9B backbone | Frozen |
| Forward/backward LoRA | Inactive |
| Optimizer steps | 0 |
| Trajectory-balance / GFlowNet updates | 0 |
| Beta--Bernoulli posterior or LCB updates | 0 |
| Operator-driven skill evolution | 0 |
| Seed groups | One (`seed=42`) |

This is therefore an evaluation of the architecture's seeded Step-0 inference
state, not evidence of improvement caused by training.

## Evaluation and answer-isolation contracts

- HotpotQA receives its benchmark-public supporting context and is scored with
  normalized maximum-over-reference EM/F1.
- TriviaQA does not receive evaluator answers. Its typed skill can search the
  answer-sanitized local research/Wikipedia corpus. Complete official aliases
  remain scorer-only and are used only for normalized maximum-over-alias
  EM/F1.
- AIME uses the full 30-record population, Qwen native thinking, the matched
  long-reasoning decoding profile, and syntax-only integer transcription.
- HealthBench uses the official `simple-evals` `random.Random(0)` 128-sample
  panel, physician-written rubrics, official per-rubric score formula, and
  mean-then-clip aggregation. Candidate and rubric judge are both frozen local
  Qwen3.5-9B SGLang. This is a **Qwen-local-judge diagnostic**, not an official
  GPT-judge-comparable score.
- WebShop and ALFWorld use native official environment transitions and terminal
  rewards, a 75-step horizon, current legal action surfaces, persistent
  controller memory, and visible Memory/Thought/Action state. WebShop's public
  catalog helper sees only the public instruction and public product records;
  evaluator targets and rewards never enter retrieval or action selection.
- MBPP+ uses EvalPlus v0.2.0 Base+Plus tests. HumanEval uses the original
  HumanEval tests. Tests and reference programs remain private and scorer-only.

Questions, answers, accepted aliases, rubrics, private tests, task IDs, model
responses, traces, and per-record outcomes were not committed to Git and were
not exposed to the evaluated model.

## Latest authoritative results

| Benchmark | N | Step-0 result | Result source | Final infrastructure failures |
|---|---:|---:|---|---:|
| HotpotQA | 128 | EM **64.84%**; F1 **79.98%** | Retained complete panel | 0 |
| TriviaQA | 128 | EM **74.22%**; F1 **80.10%** | Retained complete panel | 0 |
| AIME 2026 | 30 | Accuracy **86.67%** (26/30) | Owner-accepted complete population | 0 |
| HealthBench | 128 | Qwen-local native rubric mean **52.00%** | Retained complete panel | 0 |
| WebShop | 128 | Average **83.44%**; SR **61.72%** (79/128) | Fresh complete targeted rerun | 0 |
| ALFWorld | 128 | SR **93.75%** (120/128) | Fresh complete two-shard targeted rerun | 0 |
| MBPP+ Base+Plus | 128 | Pass@1 **80.47%** (103/128) | Retained complete panel | 0 |
| HumanEval | 128 | Pass@1 **90.63%** (116/128) | Retained complete panel | 0 |

HealthBench candidate generation took 330.98 seconds and local Qwen rubric
grading took 426.12 seconds across two equivalent base-model routes. The judge
performed five semantic JSON repairs and zero transport retries; all 128
candidate outputs and all 128 example scores were definitive.

The final WebShop rerun completed in **39.33 seconds** at **11,716.97
records/hour**. Its retrieved-skill-owned public catalog operator made 562
valid native actions, with zero invalid actions and zero horizon terminations.
The final ALFWorld rerun used two mutually exclusive, exhaustive manifest-index
shards under the same code and condition. Parallel wall time was **2,737.66
seconds** at **168.32 records/hour**. It made 2,921 valid native actions, with
zero invalid actions and zero horizon terminations. Both benchmark runs were
complete when recorded, so their remaining ETA was zero.

The WebShop panel was reused for aggregate diagnostics while repairing the
public selector. Although neither evaluator answers nor per-record rewards
entered the selector, this makes the reported WebShop number a
**development-panel result rather than an untouched holdout estimate**. A
future scientific comparison must freeze the implementation first and use a
new result-blind held-out panel; continuing to tune against these same 128
aggregate outcomes would not provide independent evidence.

## Comparison with the current backbone references

The project-wide Step-0 promotion rule is strict: every headline Step-0 metric
must be greater than the latest corresponding Qwen3.5-9B backbone-only metric.
The owner additionally set AIME accuracy at 80.00%, TriviaQA F1 at 81%, and
both WebShop headline metrics and ALFWorld SR at 80.00%.

| Benchmark / metric | Backbone | Step-0 | Delta | Strict Step-0 status |
|---|---:|---:|---:|---|
| HotpotQA EM | 62.50% | 64.84% | +2.34 pp | PASS |
| HotpotQA F1 | 78.81% | 79.98% | +1.17 pp | PASS |
| TriviaQA EM | 51.56% | 74.22% | +22.66 pp | PASS |
| TriviaQA F1 | 59.17% | 80.10% | +20.93 pp | PASS; owner 81% goal FAIL by 0.90 pp |
| AIME 2026 accuracy | 86.67% | 86.67% | +0.00 pp | Strict-improvement tie; owner accepted and 80% goal PASS |
| HealthBench Qwen-local mean | 53.17% | 52.00% | -1.17 pp | **FAIL** |
| WebShop average | 48.60% | 83.44% | +34.84 pp | PASS; owner 80% goal PASS by 3.44 pp |
| WebShop SR | 23.44% | 61.72% | +38.28 pp | PASS vs backbone; owner 80% goal FAIL by 18.28 pp |
| ALFWorld SR | 61.72% | 93.75% | +32.03 pp | PASS; owner 80% goal PASS by 13.75 pp |
| MBPP+ Base+Plus Pass@1 | 67.19% | 80.47% | +13.28 pp | PASS |
| HumanEval Pass@1 | 92.19% | 90.63% | -1.56 pp | **FAIL** |

The HealthBench comparison is only within the project's same local-Qwen-judge
diagnostic family. It must not be used as a comparison with GPT-graded external
HealthBench results.

## Infrastructure-only recovery accounting

The final score does not splice successful examples from different candidate
conditions and does not retry candidate failures:

- two AIME launch diagnostics were stopped before any scored completion after
  detecting decoding/tokenizer configuration faults; its accepted result is a
  later single complete 30-record run, not a merge;
- WebShop catalog and selector variants used during development are diagnostics
  only. The reported result is the strongest one **complete, fresh, uniformly
  configured 128-record run**. It does not select per-record outcomes from
  different variants. A later public-search-depth-1000 variant also completed
  all 128 records, but regressed to Average 82.78% / SR 59.38%; it is excluded
  as a whole rather than mixed with the reported run;
- HumanEval's first process failed during binding, before generation, because
  the code route lacked the required Qwen thinking mode;
- HealthBench's first process failed before candidate generation because the
  pinned official evaluator dependency `blobfile` was absent;
- an earlier ALFWorld pass used infrastructure-only recovery. It is superseded
  here by a later fresh 128-record panel after the reset-emitted public task was
  made authoritative and the controller learned to open the already reached
  closed destination before routing to another instance. The panel was run as
  even/odd manifest-index shards on equivalent base-model routes; the shards
  are mutually exclusive and exhaustive, so this is parallel execution rather
  than a retry or cross-condition merge.

No definitive success or candidate failure is regenerated by an
infrastructure retry.

The WebShop preparation and execution path consumes only the public shopping
instruction, public product records, public search results, selectable options,
and current native action surface. It does not read evaluator goal attributes,
target product IDs, rewards, traces, answers, or prior per-record evaluation
outcomes. ALFWorld likewise uses only the task emitted in the official reset,
public observations, inventory, and current admissible actions. The final
scores therefore contain no answer leakage or evaluator-oracle routing.

## Decision

**Exact-eight Step-0 promotion: NO-GO.** All **926/926** planned records have
definitive outcomes and the final unresolved infrastructure-failure count is
zero. The owner accepted the tied AIME result and narrowed the last rerun scope
to WebShop and ALFWorld. Within that scope, ALFWorld and WebShop average exceed
80%, while WebShop SR remains below its 80% absolute goal. Across the complete
exact-eight promotion table, HealthBench and HumanEval also remain below their
backbone references, and AIME is equal rather than strictly greater.

The positive rows remain useful diagnostics: both HotpotQA metrics, both
TriviaQA metrics, both WebShop metrics, ALFWorld, and MBPP+ exceed the current
backbone values. They do not override the all-metrics strict promotion rule,
and this inference-only condition provides no evidence of learned TTB,
posterior, or Operator improvement.
