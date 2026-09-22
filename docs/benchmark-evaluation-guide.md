# SkillFlow-BayesianImprove benchmark evaluation guide

This is the engineering authority for the benchmark layer. `idea.tex` remains the immutable method
authority. The owner's current six-domain list below supersedes the historical
eight-domain Protocol 13 catalog projection. Protocol 10/12/13 historical documents
and configurations are not an alternative current IID list.

## 1. Current IID catalog

The current project IID catalog is authoritative over the historical Protocol 10 benchmark list.
Historical artifacts retain their original identity and are never relabelled; in particular, an
old Protocol 10 gate or aggregate cannot be reported as covering the current catalog merely by
renaming rows. A future formal training launch must first rebuild its population and evaluator
package for this catalog.

The required primary suite is:

```text
HotpotQA, TriviaQA, AIME2026, HealthBench, ALFWorld, MBPP+
```

The September 15 OOD catalog is **MuSiQue-Ans, NQ-Open, Math-Hard (Level 5),
GPQA-Diamond-BioOrganic-91, ScienceWorld, APPS Introductory**. For the current OOD
Step-0 condition, native thinking is **on for Math-Hard and GPQA BioOrganic** and
off for the other four domains. This supersedes older all-off and seven-domain
statements below. HumanEval, Omni-MATH,
LiveMedBench and LiveCodeBench results remain historical, not current rows.
For the current IID condition, native thinking is **on for AIME2026 and
HealthBench** and off for HotpotQA, TriviaQA, ALFWorld and MBPP+. The authorized
September 16 backbone-only evaluation uses all 30 unique AIME2026 records and
64 records for each other current benchmark, without duplicating AIME questions.
IID uses the frozen formal-training-matched read-only evaluation interface; OOD
keeps its separately frozen 64-record evaluation architecture. Neither performs
training, posterior updates or skill evolution.

See [the current source catalog and installation guide](current-datasets.md).
Existing B28/seven-domain training schedules below describe historical runs;
this acquisition update does not launch or silently mutate a training schedule.

Backbone-only reporting uses at most 128 frozen, result-blind records per benchmark. AIME 2026 has
only 30 released records and therefore uses all 30. The historical machine identity **MBPP+** means the official MBPP+ v0.2.0
tasks scored against both the base and augmented Plus tests; the active machine identity does not
claim that EvalPlus publishes a “Hard” population. Its 128-task panel is selected with `Random(0)` from rows
sorted by canonical task ID before model generation. HumanEval uses a frozen 128-task seed-42 panel
and its official execution scorer. A separate result-blind hardness population has
not been defined here: the current name is **MBPP+**, not MBPP+ hard. Retain
`mbpp-plus` for Base+Plus results; do not relabel them as an official Hard split. New runs use one seed;
historical seed-42 panels retain their original identity rather than being retroactively called seed 0.

SWE-bench was removed from the current IID catalog by the owner because its official Docker
evaluation is too slow. Existing SWE-bench results remain historical evidence and are not inserted
into current-catalog aggregates. AppWorld is likewise diagnostic-only and is absent from Protocol
12 training, final evaluation, and aggregates. SpreadsheetBench and WebShop are
outside the current seven IID. WebShop artifacts remain historical, not part of a
new seven-IID evaluation. The current training entry uses the explicitly named
seven-domain condition below and rejects WebShop before session hydration.
Eight-domain builders/readers remain historical acquisition tools, not the
population authority for a new training launch.

## 2. TerminalReward contract

The owner-authorized [Step81 format-zero review](training_format_review_step81.md)
is an opt-in **training-only** scoring segment. Its approved rewards are not
native benchmark passes; retain original native reward/success beside adjusted
training reward/success. It does not change IID/OOD evaluators.

Each trusted evaluator produces a canonical answer-free `TerminalReward` boundary:

- `value` is TTB reward `R(tau)` in `[0, 1]`;
- `success` is the explicit binary `Y` consumed by Beta--Bernoulli calibration;
- all native metrics required by `evaluation.tex` are retained in private evaluator evidence;
- native private payload is log-only and never model-visible;
- reward shifting by `epsilon_min` happens only inside TTB construction.

Protocol 10 separates native metric, TTB reward projection, and posterior success projection. A
continuous score is never implicitly converted to Bernoulli success. The implementation must add a
closed multi-field success rule for HealthBench rather than flattening its conjunction into an
undocumented scalar threshold.

A completed legal candidate failure is valid zero-reward data. Environment, Docker, grader,
snapshot, parser, or evaluator infrastructure failure aborts the uncommitted step and cannot be
reported as reward zero.

## 3. Population isolation

Every domain has disjoint training, validation, and final-evaluation populations. Final evaluation
is read-only: it does not update weights, posteriors, detector state, thresholds, or the skill
library. Canonical source IDs and normalized public content must not overlap; benchmark-specific
structural checks are additionally required for code, spreadsheets, and interactive scenarios.

Questions, choices, answers, tests, rubrics, hidden state, complete outputs, and per-item results
remain in private storage and never enter Git. Answers and verifier truth are presented only to the
trusted evaluator after generation and never enter the evaluated model's context.

Historical eight-domain Protocol 13 training has 250 questions per domain, 2,000 question occurrences total.
Each of 250 steps takes one question from every domain and four trajectories per question
(batch 32; 8,000 trajectories). Selection uses seed 0 and shuffled cycles for small populations.
This training rule does not change validation/final population sizes. Domain sampling is not
adapted from early model performance. Do not apply those eight-domain counts to
the current seven-domain suite or silently invent new domain weights. The current
HealthBench-only repair round did not launch or change formal training.

The historical owner-requested training migration used
`seven-domain-rotating-eighth-8x4-seed0@1`: each batch takes one source occurrence
from each of the seven listed domains plus an eighth occurrence rotating through
that same domain order, then four independent rollouts per occurrence (B=32).
Source lanes retained their existing seed-zero order; exhausted lanes cycled with
new occurrence identities. The current entry can read seven-only sources or
exclude WebShop from historical source files; it does not require WebShop assets.
This is a new population condition, not an equal-population speed comparison or
a continuation of the old eight-domain checkpoint. Four steps are a correctness
and recovery test, not W=50 natural-evolution or steady-throughput acceptance.
Training uses three distinct physical GPUs: one inference card and two actual
gradient workers, including the coordinator. A two-card / one-gradient fallback
is not permitted by the current owner instruction.

The current formal condition supersedes that rotating-eighth schedule:
`seven-domain-balanced-7x4-seed0@2`, **B=28**, one source occurrence per listed
domain and four independent rollouts per occurrence. There is no eighth question.
250 steps contain 1,750 question occurrences and 7,000 trajectories. The current
declared limits in the September 10 formal instance are eight agent turns for static
domains and twenty-five for ALFWorld;
these are not counts of successful environment actions. Native thinking is off
for HotpotQA, MBPP+, HumanEval and ALFWorld, and on for HealthBench, TriviaQA and
AIME2026; the reasoning phase remains present. The earlier eight/twenty and
two-thinking-domain instance is historical, not the latest effective configuration.
The September 10 transport-repair candidate is documented in
[the native tool repair note](training_protocol_latency_repair_2026-09-10.md).
New protocol or learning conditions
start with a new run identity. Historical results retain their original budgets.

## 4. Metrics and projections

| Benchmark | Public metrics | TTB reward | Posterior success |
|---|---|---|---|
| HotpotQA | F1, EM | F1 | EM |
| TriviaQA | F1, EM | F1 | EM |
| AIME 2026 | accuracy | correct | correct |
| HealthBench, Protocol 10 | Qwen local-judge native rubric score, project binary SR | per-record-clipped rubric score | score >= 0.60 and no negative rubric triggered |
| ALFWorld | seen SR, unseen SR | success | success |
| MBPP+ | pass@1 over base and Plus tests | all tests pass | all tests pass |
| HumanEval | pass@1 | all tests pass | all tests pass |

ALFWorld seen and unseen results are reported separately. HealthBench uses the full 5,000-example
release as primary; Hard and Consensus are excluded from this primary protocol. The 128-sample backbone-only diagnostic applies its frozen
result-blind selection on top of those source populations and must not be described as the full
primary score. The corrected backbone-only diagnostic uses direct answers, the official Random(0)
panel and simple-evals scoring/aggregation, but the owner-selected grader is also Qwen3.5-9B; its
native rubric mean must not be called an official GPT-4.1-comparable score.

Keep HealthBench per-record native scores, including negative values. Publication clips the mean
of those raw scores, not each record before averaging; only the separate TTB reward is clipped per
record. QA uses the benchmark-specific official normalizer: TriviaQA replaces punctuation with
spaces whereas HotpotQA removes ASCII punctuation. Do not substitute substring correctness or
select the most favorable answer span using a reference. See the
[TriviaQA scorer](https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py),
[HotpotQA scorer](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py), and
[HealthBench scorer](https://github.com/openai/simple-evals/blob/main/healthbench_eval.py).

The current A2 IID HealthBench judge uses temperature **0.5**, a 2,048-token per-rubric
cap and thinking off. Freeze its effective profile, not only a mutable configuration
path; retain rubric verdicts and actual grader I/O only in the private scorer domain.
The earlier IID verifier label incorrectly said temperature 0 despite actual 0.5
requests. Correcting that label does not change or retrospectively regrade old scores.
The separate temperature-0 training grader is a different condition. See the
[HealthBench interface diagnosis](step0_healthbench_context_diagnosis_2026-09-08.md).

The September 12 owner update sets **future external model judges** to
`gpt-5.6-luna`, reasoning effort `medium`, via the official OpenAI API. It does
not retroactively relabel local-Qwen, GPT-4.1, GPT-5.1-low or Luna-high scores,
replace native/rule metrics, or make the judge an answer agent. Keep credentials
only in the trusted scorer's private environment (`OPENAI_API_KEY`), never in
actor configuration or Git. Use a new declared judge condition and retain raw
outputs, actual usage, failures and the complete planned denominator; see the
[external-judge implementation note](ood-official-scorers.md#future-external-judges-luna-medium-via-openai-api-owner-update).

## 5. Method comparisons

The first required conditions are exact SkillFlow baseline, BayesianImprove full, and
BayesianImprove no-calibration. They share the same frozen tasks, order, seed, generation service,
teacher-forced scorer, evaluators, budget, model/tokenizer, and initial skill library. Method type is
a closed identity, not a collection of boolean flags. The historical `SKILLFLOW_DISABLED` wire value
is not an exact SkillFlow baseline identity.

One-seed results are reported only as registered primary-seed evidence. Missing or failed conditions
remain visible and are not imputed or rerun until favorable.

For architecture ablations, compare backbone, architecture without skills, and architecture with
skills explicitly. The two architecture conditions share tools, public observations, communication,
action parser, evaluator and budgets; only skill context differs. A skill is advisory, not a switch
for a hand-written task solver. A two-pass reasoner/action model is not evidence of independently
collaborating agents. Report actual model calls and tool calls rather than claiming multi-agent
superiority by construction.

The active architecture and evaluation conditions now use one Qwen3.5-9B answer owner.
The added `solver` and `researcher` consultants are retired: no advertised consultant tools,
advice-model requests, or consultant replies are part of a new run. The live controller and
trusted generation broker reject historical `multi-agent` arms rather than silently relabelling
them. Historical configurations and results remain readable under their original identities.
Skills, native tools, the forward/backward training policies, and permitted native thinking remain
separate from agent count. Upstream SkillFlow's frozen Executor belongs to its explicitly identified
baseline architecture, not an exemption for these two removed consultants. A HealthBench rubric
judge remains an isolated post-generation scorer, never a second answer owner or selector.

Inspecting development-panel outcomes to revise prompts, skill policies or selectors makes that
panel development evidence, even when individual answers never enter prompts. Establish unseen
generalization only with a separate untouched evaluation population. Do not reuse cached baseline
candidates as a fallback. Under the owner's latest rule, voting, label-swapped candidate selection,
token-cost-targeted alternative derivations and baseline fallback are prohibited in the active
evaluation, not merely something that can be fixed by reporting extra cost. Score the evaluated
Qwen3.5-9B policy's unique final answer; a rubric judge only scores after generation is finished.

## 6. Formal reporting

### September 8 engineering corrections and requested small panel

- The current owner instruction prefers native thinking for all seven IID domains.
  Both the default thinking map and the seven-domain launcher now enable it. The
  latest requested follow-up reruns MBPP first, then the other five previously-off
  domains; AIME's already-on historical run is not silently rerun or merged.
  The launcher's newly declared sampling profiles follow the
  [Qwen model card's thinking recommendations](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices),
  with coding/general profiles distinguished before generation. Non-AIME length
  pauses use the same token-stream continuation within unchanged episode budgets.
  HealthBench's evaluated owner thinking flag is separate from its frozen rubric
  judge configuration. Report actual effective settings, not just an arm's label.
- The single-main-agent topology above is unchanged. The requested new A2 round
  excludes WebShop and uses the first 32 entries of each existing frozen population,
  or all 30 AIME2026 entries. The 222-case declaration is explicit, not a different
  canary population or a result-ranked sample. This matches the current seven-domain
  IID list, not the historical eight-domain catalog.
- Python transport accepts bare source and fenced carriers. Following the
  [LCB instruct-model convention](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/utils/extraction_utils.py),
  the last code block (or last explicit code declaration) is the submission;
  Markdown labels are presentation metadata. The framework neither concatenates
  blocks nor uses syntax, execution or hidden tests to choose among them. Unlike
  upstream's line-only splitter, literal backticks inside Python are preserved and
  an unfinished final block cannot silently fall back to an earlier draft. This
  September 12 correction replaces the restrictive multiple-block rejection and
  must be recorded as a new condition, not applied invisibly to historical scores.
- QA owner parsing preserves an explicit short-answer field's line boundary instead
  of merging a following explanation into its value. Conflicting declarations still
  require clarification; unmarked prose, names and numeric spellings are not rewritten.
  Balanced Markdown emphasis around a QA final label or the entire final field is
  transport formatting, not part of the answer. A real bold final field previously
  caused the whole explanation to be submitted. Owner and trusted-broker projection
  now share this decoding; quoted code and clinical responses are not shortened.
  The version-5 owner wire also accepts Markdown `Short answer:` and `Answer:`
  aliases. A final declaration takes precedence, then `Short answer:`, then
  `Answer:`; the first nonempty field line is submitted. Identical repeated
  declarations are harmless, while conflicting declarations at the same priority
  remain unresolved. Without a top-level field, the whole reply is preserved;
  an `Answer:` example inside a code fence is not a submission. These rules are
  answer-blind and versioned, not a search for the span favored by the scorer.
  The affected OOD prompts describe optional bare/labelled answers and the code
  final-block wire before generation, without imposing reasoning-length limits.
- HotpotQA's public task instruction describes its answer semantics: a span from the
  supplied passages, or yes/no. This follows the dataset's
  [public task definition, §2](https://aclanthology.org/D18-1259.pdf), not private labels
  or a score threshold. It does not prescribe reasoning steps, choose a passage,
  rewrite the owner's answer, or alter the native scorer. Prompt revisions evaluated
  on the inspected panel remain development evidence, even when its numeric goal is met.
- AIME explicit final fields end before following explanatory prose. Repeated final
  declarations must agree. Intermediate reasoning numbers never replace an owner final.
  Native thinking has the same 81,920-token episode allowance and eight-call ceiling,
  with chunks capped at 32,768 and a 4,096-token final reserve. Length-paused thinking
  continues the exact original prompt plus actual output tokens. It is not an independent
  derivation. Every chunk and its usage is recorded; an unclosed final remains a failure.
  Necessary preceding reasoning is preserved verbatim for an interface repair within
  the declared context capacity, rather than silently discarded as old chat history.
- Clean MBPP scoring freezes `configs/evaluation/mbpp_evalplus_native.yaml`: EvalPlus
  revision `26d6d00`, native 4-second minimum and 4x reference-time factor, 4 GiB memory,
  native 60-second per-lane cap, 900-second outer worker allowance, concurrency eight.
  The time defaults follow the fixed version's
  [official configuration](https://github.com/evalplus/evalplus/blob/26d6d00/evalplus/config.py).
  Base and Plus run independently; primary pass@1 remains Base AND Plus. Native verdicts,
  syntax diagnostics and worker failures stay in the private scorer. An unexecuted lane
  is recorded as `not-run`, not as a native test verdict. Custom limits need a separate
  named condition. Historical training's three-field worker protocol remains compatible.
  A real scorer stall was traced to an EvalPlus child dying with a shared-result lock
  held. The named `single-writer-raw@1` transport avoids that abandoned lock without
  changing tests, time limits, memory or native verdict logic. Old frozen profiles
  retain synchronized transport. An infrastructure interruption has no native verdict;
  it is not a failing answer. Compare transports on unchanged saved candidates and
  record any recovery separately, never trigger an actor retry from scorer feedback.
- MBPP public prompt carriers must exactly match the standard dataset's public `prompt`
  field, preserving interfaces and any released examples. Canonical solutions and extra
  tests remain private. The existing MBPP+ development subset is not a verified hard split.
- Export effective per-benchmark thinking and budget configuration. For skills-enabled
  conditions, distinguish automatic initial retrieval from owner-requested discovery and
  retain the actually loaded snapshot and rendered bodies. Skills-off still loads none.

The read-only MBPP time-limit comparison uses unchanged saved candidates, not model
regeneration. It does not overwrite historical scores or feed test results to the owner.

Report every native metric, evaluated count, candidate-failure count, infrastructure-failure count,
parse-error count, wall time, token/model-call cost, and frozen model, tokenizer, evaluator, method,
and population identities. Private per-item data and absolute private paths are excluded.

Legacy approval/receipt/seal gates are retired. Run evaluation only in an owner-requested round;
check actual task/evaluator wiring and report incomplete or failed work honestly. The owner-requested
Step-0 integrity plan explicitly permits a nonfinal service canary, followed by frozen A1/A2 full
paired evaluation. Offline tests are not benchmark scores or a performance guarantee.

### September 16 current OOD development targets

`configs/evaluation/current_ood_targets.json` records the owner's current targets.
Both promotion from 32 to 64 samples and acceptance of a complete 64-sample panel
require the unrounded primary score to be **strictly greater than 90% of target**;
equality is not sufficient. The resulting thresholds are MuSiQue F1 67.5,
NQ EM 44.28, Math-Hard accuracy 73.26, GPQA Diamond BioOrganic accuracy 63.0,
ScienceWorld native final mean 50.553, and APPS Intro pass@1 80.46.
These are development targets, not claims of matched reproduction of other papers.
ScienceWorld uses its signed native final score, never the clipped learning reward.

The owner permits another whole sample panel for development target search. Freeze
its membership before generation, retain every attempt and report the selection
procedure. Never splice successful cases from different panels or present the
best panel as an unbiased, previously unseen generalization result. Missing native
scores and infrastructure failures cannot be silently removed from the denominator.

The new `same-owner-token-stream-bounded-thinking@1` condition affects Math-Hard
and GPQA only. Within the unchanged 8,000-token allowance, an unfinished thinking
channel at 6,000 tokens receives native newline/end/separator framing, then the same
owner generates its final answer. Each framing token is generated, persisted and
charged as a real call/token; only channel syntax is constrained, never answer
content. Natural earlier completion uses no forced framing. This is a separately
named decoding condition, not a retroactive repair of earlier failed submissions.
APPS retains one unchunked 12,000-token response allowance; the other four OOD
domains remain thinking-off. No scorer feedback enters any owner's context.

GPQA choice verifier `@ood2` additionally accepts balanced Markdown on explicit
final fields and a single labelled option such as `C. <option text>`. It does not
match the option text against the reference or select from an alternative list;
conflicting final declarations remain invalid. Legacy `@ood1` scores remain
readable under their original version and are never overwritten by this change.
The owner wire also accepts a plain `Final answer:` line immediately before its
single `submit_answer` tool call when every declared value literally agrees with
the answer argument. It neither chooses between conflicting channels nor merges
multiple calls. This avoids a redundant owner repair observed in five GPQA traces.
