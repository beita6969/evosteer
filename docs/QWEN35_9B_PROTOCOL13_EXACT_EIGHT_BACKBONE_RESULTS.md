# Qwen3.5-9B Protocol 13 Exact-Eight IID Backbone-Only Results

This is the fresh post-repair Protocol 13 run. Seven benchmarks use a frozen 128-record panel;
AIME 2026 uses all 30, for 926 planned records. The committed aggregate contains no task,
answer, rubric, test, model response, trace, or per-record payload.

| Benchmark | N | Outcomes (success/failure/definitive/invalid/infra) | Metrics | Numeric / Scientific / Formal |
|---|---:|---:|---|---|
| HotpotQA | 128 | 80/48/128/0/0 | EM 62.50% (target 60.94%, +1.56 pp); F1 78.38% (target 75.70%, +2.68 pp) | pass / protocol-incomplete / blocked |
| TriviaQA | 128 | 66/61/127/1/0 | EM 51.56% (target 44.88%, +6.68 pp); F1 59.17% (target 54.30%, +4.87 pp) | pass / protocol-incomplete / blocked |
| AIME 2026 | 30 | 22/0/22/8/0 | accuracy 73.33% (target 46.67%, +26.66 pp) | fail / protocol-incomplete / blocked |
| HealthBench | 128 | n/a/n/a/128/0/0 | native rubric mean 53.26% | not evaluated / external anchor / blocked |
| WebShop | 128 | 17/110/127/1/0 | average score 35.38% (target 56.93%, -21.55 pp); success rate 13.28% (target 32.03%, -18.75 pp) | fail / protocol-incomplete / blocked |
| ALFWorld | 128 | 50/75/125/3/0 | success rate 39.06% (target 48.28%, -9.22 pp) | fail / protocol-incomplete / blocked |
| MBPP+ | 128 | 92/23/115/13/0 | Base+Plus pass@1 71.88% | not evaluated / external anchor / blocked |
| HumanEval | 128 | 118/7/125/3/0 | pass@1 92.19% (target 89.06%, +3.13 pp) | pass / protocol-incomplete / blocked |

## Interpretation

- The formal result is **blocked**, not “seven benchmarks passed.” Paper-reported values with
  incomplete execution evidence remain non-formal, external aggregate anchors are not silently
  treated as matched targets, and undefined/mismatched references fail closed.
- AIME's 22 correct, strictly parsed answers do not produce a numeric pass because eight planned
  records were candidate-invalid; definitive coverage is part of the numeric contract.
- HealthBench uses the owner-required Qwen3.5-9B SGLang self-judge. It is a local diagnostic and
  must never be described as comparable to the GPT-4.1-graded 44.68 external anchor.
- MBPP+ is the frozen result-blind `Random(0)` 128-record panel scored by pinned EvalPlus Base and
  Plus tests. The pinned evaluator emits `pass`/`fail`; the adapter handles those native statuses
  directly and keeps generation separate from scoring.

## Interactive ReAct memory

WebShop and ALFWorld use a multi-turn ReAct transcript. After every action, the runner stores the
pre-action public state, chosen action, returned public state, and returned available/admissible
action surface. The next prompt receives a contiguous suffix of those transitions plus the task;
it never skips a newer oversized turn in order to retain an older turn. Demonstrations are
source-proven successful train-environment replays whose actions are valid on the observed action
surface. This is the required execution pattern for later runs, rather than a stateless or
single-turn “naked” policy.

The persistent returned-state memory materially improved ALFWorld versus the earlier naked run
(3.91% to 39.06% success), while the exact matched target remains unmet. WebShop also remains
below both reference metrics; no target-based parser or action correction was introduced.

Machine-readable evidence: [`qwen35_protocol13_exact_eight_128_20260830.json`](machine-results/qwen35_protocol13_exact_eight_128_20260830.json).
