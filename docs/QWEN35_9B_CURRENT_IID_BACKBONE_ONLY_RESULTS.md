# Qwen3.5-9B Protocol 12 Exact-Nine IID Backbone-Only Results

> **Historical Protocol 12 record.** This report preserves the former exact-nine catalog and is
> not the current IID authority. Protocol 13 removes SpreadsheetBench and contains exactly
> HotpotQA, TriviaQA, AIME 2026, HealthBench, WebShop, ALFWorld, MBPP+, and HumanEval. Its AIME 2026
> owner-defined minimum goal is 80.00%; the older 46.67% comparison retained below is historical.

This answer-free report superseded the former ten-row table when Protocol 12 was active:
SWE-bench was historical, AppWorld was diagnostic-only, and neither entered this aggregate. The
then-active rows were HotpotQA, TriviaQA, AIME 2026, HealthBench, WebShop, ALFWorld,
SpreadsheetBench, MBPP+, and HumanEval.

## Evaluation contract

- Frozen adapter-free Qwen3.5-9B base model served by SGLang at a 98,304-token context length.
- One result-blind 128-record panel per benchmark; AIME 2026 uses all 30 released records.
- Exactly 1,054 planned final records in one seed/configuration.
- Candidate failures are retained. Only typed infrastructure failures are resumable.
- HealthBench uses direct natural-language candidates and Qwen3.5-9B local rubric grading with the
  pinned simple-evals prompts, formula, Random(0) panel, and aggregate-then-clip order. It is not an
  official GPT-4.1-comparable score.
- MBPP+ is the truthful EvalPlus v0.2.0 identity, not a fabricated “Hard” split. Its headline is
  pass@1 against both base and Plus tests.
- HumanEval's existing 128 raw generations were replay-checked under `python-source@2`; all 128
  parser outputs were identical, so no model request was repeated.

## Results

| Benchmark | N | Backbone-only metric | Final infrastructure failures | Target status |
|---|---:|---:|---:|---|
| HotpotQA | 128 | EM 62.50%; F1 78.22% | 0 | PASS (`<7 pp`) |
| TriviaQA | 128 | EM 50.78%; F1 58.39% | 0 | PASS (`<7 pp`) |
| AIME 2026 | 30 | Accuracy 86.67% | 0 | FAIL (40.00 pp) |
| HealthBench Full, Qwen local judge | 128 | Native rubric mean 53.47% | 0 | target undefined; diagnostic-only |
| WebShop, v3 native ReAct | 128 | Average score 33.21%; SR 15.62% | 0 | FAIL (23.72 / 16.41 pp) |
| ALFWorld, v3 native ReAct | 128 | SR 35.16% | 0 | FAIL (13.12 pp) |
| SpreadsheetBench V1 Verified | 128 | Pass@1 1.56% | 0 | target undefined; diagnostic-only |
| MBPP+ v0.2.0 | 128 | Base+Plus Pass@1 66.41% | 0 | target undefined; diagnostic-only |
| HumanEval | 128 | Pass@1 92.19% | 0 | PASS (`<7 pp`, reconstructed target) |

HealthBench's bootstrap standard deviation is 2.67 percentage points. Its candidate and score
journals each contain 128 records; the completed grading covered 1,497 rubric items. A run-level
thread-context failure and one malformed-grader infrastructure case were retried without
regenerating completed candidates or regrading completed examples.

### Admitted reference comparison

Only same-metric rows with a configured reference are compared numerically. Reconstructed
populations remain approximate even when their numerical gap is below seven percentage points.

| Benchmark / metric | Reference | Observed | Absolute gap | Admission |
|---|---:|---:|---:|---|
| HotpotQA EM | 60.94% | 62.50% | 1.56 pp | PASS |
| HotpotQA F1 | 75.70% | 78.22% | 2.52 pp | PASS |
| TriviaQA EM | 44.88% | 50.78% | 5.90 pp | PASS |
| TriviaQA F1 | 54.30% | 58.39% | 4.09 pp | PASS |
| AIME 2026 accuracy | 46.67% | 86.67% | 40.00 pp | FAIL |
| WebShop average score | 56.93% | 33.21% | 23.72 pp | FAIL |
| WebShop success rate | 32.03% | 15.62% | 16.41 pp | FAIL |
| ALFWorld success rate | 48.28% | 35.16% | 13.12 pp | FAIL |
| HumanEval pass@1 | 89.06% | 92.19% | 3.13 pp | PASS, approximate population |

HealthBench, SpreadsheetBench, and MBPP+ remain target-undefined diagnostics. A local-Qwen
HealthBench grader result cannot be admitted against the GPT-4.1-judge reference contract.

## Outcome and failure taxonomy

- **HotpotQA:** 128 definitive verdicts; full distractor context was model-visible; no unresolved
  infrastructure failure.
- **TriviaQA:** 128 definitive verdicts under the frozen direct/reference condition; no retrieval
  result was relabelled as direct evidence.
- **AIME 2026:** 30 definitive verdicts and no unresolved infrastructure failure. Twenty-six
  answers were correct, 28 reached the integer scorer, and two candidate-invalid outputs were
  retained as failures. The one generation infrastructure failure was retried without repeating
  the other 29 generations.
- **HealthBench:** 128 candidates and 128 local-Qwen rubric verdicts; final unresolved
  infrastructure failures 0. The score remains diagnostic because grader identity differs from the
  official reference.
- **WebShop:** 128 definitive verdicts and no unresolved infrastructure failure. Official terminal
  rate was 50.78%, horizon termination 42.19%, candidate-invalid termination 7.03%, partial-reward
  rate 33.59%, mean steps 7.30, and parser-invalid action rate 0.96%.
- **ALFWorld:** 128 definitive verdicts and no unresolved infrastructure failure. Official terminal
  rate was 96.88%, candidate-invalid termination 3.12%, horizon termination 0%, mean steps 10.15,
  and parser-invalid action rate 0.31%.
- **SpreadsheetBench:** 128 definitive records, 123 candidate failures, 108 action parse errors,
  and 125 scorer invocations. Four infrastructure failures were resumed after bounding initial
  workbook previews and making ShareStore cleanup races non-fatal to already-scored episodes.
- **MBPP+:** 85 scored successes, 35 scored failures, 8 candidate-invalid outputs, and no generation
  or scorer infrastructure failure. The official EvalPlus full evaluator was used through a
  non-selected-task-safe carrier.
- **HumanEval:** 125 extracted and 3 empty candidates under parser v2; official pass@1 scoring
  reached 125 candidates; no unresolved infrastructure failure.

## Protocol 12 and SKILLEV repair status

Implemented code-level repairs include:

- an explicit exact-nine catalog and 4,608-episode/288-step Protocol 12 projection;
- typed execution conditions, runtime-profile admission, mutually exclusive conserving receipts,
  strict target admission, aggregate recomputation, and fail-closed rendering;
- independent direct, local-judge, benchmark-native, Spreadsheet, EvalPlus, and HumanEval contracts;
- exact tokenizer-based context accounting and explicit condition/initial-context profiles;
- Spreadsheet action format v2 with optional/default/nullable arguments, bounded typed previews,
  pagination controls, execution, and official OJ submission;
- current v3 WebShop/ALFWorld train-only ReAct demonstrations and remaining-step state;
- expanded rollout/dataflow stages, task-specific decoding evidence, stricter curve acceptance,
  and typed training diagnostic receipts;
- infrastructure-only resume behavior for direct generations, HealthBench grades, and spreadsheet
  sessions.

### Runtime and artifact binding

| Benchmark(s) | Committed entrypoint | Runtime contract | Artifact action |
|---|---|---|---|
| HotpotQA / TriviaQA / AIME | `run_qwen35_direct_reference.py` | direct profiles from Protocol 12 | fresh generation; infrastructure-only AIME resume |
| HealthBench | `run_qwen35_healthbench_official.py` | direct candidate plus local-Qwen diagnostic grader | fresh candidates and rubric grades; missing-only resume |
| WebShop / ALFWorld | `run_qwen35_direct_interactive.py` | native v3 ReAct profiles and official text environments | fresh current-condition run |
| SpreadsheetBench | Protocol benchmark runtime and official-OJ worker | action surface v2 and bounded public workbook view | fresh run; infrastructure-only session resume |
| MBPP+ | `run_qwen35_mbpp_plus.py` | EvalPlus v0.2.0 Base+Plus | fresh generation and official scorer run |
| HumanEval | `run_qwen35_direct_reference.py` | deterministic direct-code profile and official pass@1 | raw generation reused after parser-v2 replay equivalence |

All CUDA model requests used the same adapter-free served identity and explicit 98,304-token
context contract. Prompt, decoding, parser, environment, scorer, and grader profile names are
admitted by the Protocol 12 registries; rubric, tests, answers, and per-task results remain private.

The real 16-trajectory no-update and one-step controls were not executed because the formal gate is
closed; the validation-only legacy scripts are not accepted as evidence. No formal training was
started.

## Final gate

**Backbone baseline readiness: NO-GO.**

**SKILLEV formal training: NO-GO.**

All 1,054 planned final records now have definitive outcomes and final unresolved infrastructure
failures are zero. The gate remains closed because AIME, WebShop, and ALFWorld fail the strict
seven-point comparison, HealthBench, SpreadsheetBench, and MBPP+ lack same-condition reference
targets, and the real no-update/one-step training controls have not executed. Diagnostic closeness
cannot upgrade those rows into formal evidence.
