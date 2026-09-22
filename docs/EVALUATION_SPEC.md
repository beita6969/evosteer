# Protocol 10 engineering map

`evaluation.tex` freezes the benchmark science. This document maps it to engineering work; it
does not change the method in `idea.tex`. The machine-readable protocol is
`configs/evaluation/protocol_v10.yaml`; public source and split choices are frozen in
`configs/evaluation/protocol_v10_sources.yaml`.

## Current admission state

Protocol 10 is scientifically frozen but **not executable**. The typed population-level reader is
implemented in `src/skillev/experiments/protocol_v10.py`; the private population loader, isolation
checks, exact 512-per-domain selection, global SkillFlow-style shuffle, and repeated-episode routing
are implemented in `skillev_private.benchmarks.protocol_v10_population`. The private historical
Protocol 10 catalog and 4,608-episode selection have been materialized on the approved server and
remain outside Git. Protocol 11 is a superseded historical ten-IID catalog; Protocol 12 is a
superseded exact-nine catalog; Protocol 13 is the current exact-eight catalog. The 18-benchmark reader is
retained only for historical `@9`
artifacts. Until the private datasets, evaluators, and canonical runtime consume `@10`, every new
formal training or evaluation run must fail admission. Existing `run_training.py` is a historical
SkillFlow baseline entrypoint and is not a BayesianImprove formal entrypoint.

The owner-final Protocol 13 private training data is an exact 2,000-question artifact (eight
domains times 250) with a uniform answer-isolated input/output envelope; see
[`PROTOCOL13_FORMAL_TRAINING_DATASET.md`](PROTOCOL13_FORMAL_TRAINING_DATASET.md). This data milestone
does not by itself open the formal execution gate. Runtime expands each balanced eight-question
step into four trajectories per question for a 32-trajectory effective batch.

## Frozen decisions

- one seed: `0`;
- current project IID reporting catalog: HotpotQA, TriviaQA, AIME 2026,
  HealthBench, WebShop, ALFWorld, MBPP+, and HumanEval;
- SWE-bench is no longer current IID; its older results remain historical only. AppWorld is
  diagnostic-only, and SpreadsheetBench was removed by the owner; none enters Protocol 13;
- the older Protocol 10 suite's `mbpp-plus-fixed-100` identity remains historical. Current
  **MBPP+** means a frozen 128-task sample from MBPP+ v0.2.0 evaluated against the official base
  and augmented tests; its active machine identity does not invent an EvalPlus “Hard” split;
- HealthBench primary population: full 5,000-example 2025 release; Hard and Consensus are excluded
  from the primary suite;
- pre-run amendment `protocol-v10-healthbench-qwen-judge@1` fixes the judge to the
  Qwen3.5-9B base model through SGLang at temperature 0.5, seed unset, and thinking disabled; it
  preserves the official rubric template and score formula but is not an official
  GPT-4.1-comparable score;
- the separate owner-selected backbone diagnostic in
  [`HEALTHBENCH_OFFICIAL_PARITY_PROTOCOL.md`](HEALTHBENCH_OFFICIAL_PARITY_PROTOCOL.md)
  uses Qwen3.5-9B for both the direct-answer candidate and local rubric grader while retaining
  the official Random(0) sample, rubric prompts, score formula, and aggregate-then-clip order;
  it is diagnostic-only and is not an official GPT-4.1-comparable result;
- historical SpreadsheetBench definitions remain readable only for their original protocols and
  cannot enter the current IID catalog, aggregate, or training population;
- EvalPlus has no official “Hard” population. The current Protocol 13 diagnostic therefore uses
  the truthful `mbpp-plus` machine identity and a result-blind 128-task `Random(0)` sample from
  canonical-ID-sorted MBPP+ v0.2.0, scored against both base and Plus tests;
- required conditions begin with exact SkillFlow baseline, BayesianImprove full, and
  BayesianImprove no-calibration;
- the current Protocol 13 training mixture has exactly 250 questions per benchmark and 2,000
  questions total; undersized source populations are repeated in deterministic seed-0 shuffled
  cycles, with a new episode ID for each occurrence;
- every optimizer step consumes exactly one question from each of the eight domains and generates
  four trajectories per question, for an effective batch of 32 and 8,000 trajectories over 250
  steps; checkpoints are saved every 10 steps, for 25 cadence checkpoints;
- persisted questions use a step-major dataset-balanced order. Domain position within a step and
  source order within a domain are deterministic seed-0-derived shuffles, and the exact order is
  shared across compared methods.

## Population model implemented in Batch 1

The active Protocol 10 reader uses tagged population records containing:

```text
benchmark
population_id
role: training | validation | final-evaluation
source_version
sampling
evaluator_identity
```

One benchmark may have multiple final populations, as required by ALFWorld and AppWorld. The
historical Protocol 10 reader also closes its three method identities, legacy 512-per-benchmark mix,
read-only final-evaluation boundary, population isolation settings, and infrastructure-failure
policy. Active Protocol 10 readers reject `@9`; historical readers may continue reading historical
artifacts only. The current Protocol 13 schedule is the 250-question-per-domain schedule above.

## TerminalReward mapping typed in Batch 1 and implemented by evaluators in Batch 3

Every trusted evaluator must return three distinct objects:

1. the complete native metric set;
2. `TerminalReward.value = R(tau)` in `[0, 1]` for TTB;
3. an explicit boolean posterior outcome `Y` for the Beta--Bernoulli update.

`PosteriorSuccessProjection` now provides the closed single-field and HealthBench-conjunction
projections. `TerminalReward.success_rule=trusted-native-projection` carries the resulting trusted
boolean when it cannot be reconstructed from scalar reward alone. Batch 3 must invoke that projection
inside each trusted evaluator. Private rubric details, answers, tests, and failure traces cannot
cross into model-visible state.

| Benchmark | Native reporting | `R` | `Y` |
|---|---|---|---|
| HotpotQA | F1, EM | F1 | EM |
| TriviaQA | F1, EM | F1 | EM |
| AIME 2026 | accuracy | correct | correct |
| HealthBench, Qwen3.5-9B judge (Protocol 10 only) | local-judge rubric score, negative-rubric count | clipped score | score >= 0.60 and no negative rubric triggered |
| WebShop | average score, SR | native score | native success |
| ALFWorld | seen SR, unseen SR | episode success | episode success |
| MBPP+ | pass@1 over base and Plus tests | all tests pass | all tests pass |
| HumanEval | pass@1 | all tests pass | all tests pass |

Infrastructure failure is typed and aborts the uncommitted step. A completed candidate failure is
valid zero-reward data. AppWorld remains available only in explicitly diagnostic artifacts.

## Data boundary required in Batch 2

Materialization must produce physically separate training, validation, and final-evaluation stores.
It checks canonical source IDs and normalized public content; code also checks signatures,
spreadsheets check workbook/formula structure, and interactive environments check goal/scenario
IDs. Any overlap blocks freezing. Final answers and verifier state remain private and never enter
the policy, skills, prompts, public logs, or Git.

The exact private ID lists and repeated episode IDs are Batch 2 products. Every benchmark must
contribute exactly 250 questions to the step-major balanced schedule. They must conform to the
already-frozen population identities and cannot redefine benchmark versions, final populations,
or projections. Repetition is confined to undersized training source populations and never copies
training items into validation or final evaluation.

## Runtime boundary required before formal execution

The canonical formal entrypoint must dispatch a closed method identity and route BayesianImprove
through `SKILLEVApplication`. All methods share the same task order, evaluators, generation service,
teacher-forced policy scoring, budgets, and initial state. Serving log probabilities are never TTB
inputs, and final evaluation is read-only for posterior and skill state.

The existing `@9` source protocol, acquisition lock, and historical artifacts remain readable only
for their original experiments. They must not be silently rewritten as Protocol 10 evidence.

## Primary sources

- [HotpotQA](https://github.com/hotpotqa/hotpot)
- [TriviaQA](https://github.com/mandarjoshi90/triviaqa)
- [AIME 2026](https://github.com/EnvCommons/AIME2026)
- [OpenAI HealthBench](https://openai.com/index/healthbench/) and
  [official simple-evals implementation](https://github.com/openai/simple-evals/blob/main/healthbench_eval.py)
- [WebShop](https://github.com/princeton-nlp/WebShop)
- [ALFWorld](https://github.com/alfworld/alfworld)
- [EvalPlus](https://github.com/evalplus/evalplus)
