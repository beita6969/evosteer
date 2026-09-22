# Qwen3.5-9B Protocol 14 Target Evidence

Protocol 14 freezes matched, answer-free reference targets before the fresh candidate run. Seven
benchmarks use a frozen 128-record panel; AIME 2026 uses all 30 records. The public evidence files
contain only aggregate receipts and provenance—no task IDs, questions, answers, rubrics, tests,
model responses, or trajectories.

## Frozen matched references

| Benchmark | Condition | Reference scope | Frozen target(s) | Formal gate |
|---|---|---:|---|---|
| HotpotQA | `hotpotqa-source-parity-128@1` | exact 128 | EM 61.7188%; F1 78.5316% | required |
| TriviaQA | `triviaqa-source-parity-128@1` | exact 128 | EM 49.2188%; F1 56.0379% | required |
| AIME 2026 | `aime2026-source-integer-final-all30@1` | all 30 | accuracy 93.3333% | required |
| HealthBench | `healthbench-qwen-local-random0-128@1` | exact 128, seed 0 | overall score 52.9815% | diagnostic only |
| WebShop | `webshop-skillflow-action-only-128@1` | exact 128 | average score 32.0350%; success 12.5000% | required |
| ALFWorld | `alfworld-skillflow-action-only-128@1` | exact 128 | success 3.9063% | required |
| MBPP+ | `mbpp-plus-evalplus-v020-nonthinking-random0-128@1` | full 378, projected to exact 128 | pass@1 67.1875% | required |
| HumanEval | `humaneval-source-parity-128@1` | exact 128 | pass@1 92.1875% | required |

The MBPP+ full-378 reference was 64.8148% Base+Plus pass@1. The 67.1875% target is the
preselected `Random(0)` 128-panel projection from those same 378 generations. The published 71.8%
EvalPlus composite remains a non-target external anchor; it is not substituted for standalone
MBPP+.

## HealthBench boundary

HealthBench candidates and per-rubric grades both use the frozen Qwen3.5-9B SGLang base route.
The judge label is **Qwen-local-judge diagnostic**, and its receipt declares
`official_gpt_comparable: false`. Consequently, its matched 128-panel reference is frozen for
diagnostic comparison but `required_for_formal_gate` is false. The public 44.68 HealthBench-500
anchor is retained only as a different-condition external reference and is not promoted to the
Protocol 14 target.

## Independence and provenance

1. Source contracts, panels, model revision, prompts, decoding, parsers, environments, scorers,
   grader profile, and one-seed aggregation were frozen before reference execution.
2. Reference attempts use dedicated IDs and private run directories. Infrastructure-only repairs
   preserved completed generations and did not alter prompts, panels, scoring rules, or semantic
   decoding parameters.
3. The target registry and public reference package were frozen before the candidate attempt. The
   candidate must use a separate attempt ID and an execution-equal condition.
4. MBPP+ uses EvalPlus 0.2.0 over all 378 tasks before projecting the fixed 128 panel. HumanEval
   remains the original 128-task scorer; HumanEval+ is only a separate cross-check.
5. Adapter policy is forbidden throughout; the served route is the frozen Qwen3.5-9B backbone.

Machine-readable evidence:

- `configs/evaluation/qwen35_protocol14_iid_targets.yaml`
- `docs/machine-results/qwen35_protocol14_reference_targets_20260830.json`
- `configs/evaluation/protocol_v14_conditions.yaml`
- `configs/evaluation/protocol_v14_healthbench_grader.yaml`
