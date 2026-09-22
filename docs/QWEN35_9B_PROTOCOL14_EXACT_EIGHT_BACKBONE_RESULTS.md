# Qwen3.5-9B Protocol 14 Corrected Exact-Eight IID Backbone-Only Results

Seven benchmarks use frozen 128-record panels; AIME 2026 uses all 30 (926 planned records total). Formal parity uses strict absolute gap `<7pp`; relative error is reported separately. HealthBench is a Qwen3.5-9B SGLang local-judge diagnostic and is excluded from the formal gate.

## Formal parity results

| Benchmark | N | Valid / model-invalid / infra | Metrics | Parity / Floor / Formal |
|---|---:|---:|---|---|
| hotpotqa | 128 | 127 / 1 / 0 | em=62.50% (target 61.72%, gap 0.78pp, rel 1.27%)<br>f1=78.81% (target 78.53%, gap 0.28pp, rel 0.35%) | pass / pass / pass |
| triviaqa | 128 | 127 / 1 / 0 | em=51.56% (target 49.22%, gap 2.34pp, rel 4.76%)<br>f1=59.17% (target 56.04%, gap 3.14pp, rel 5.60%) | pass / pass / pass |
| aime-2026 | 30 | 27 / 3 / 0 | accuracy=86.67% (target 93.33%, gap 6.67pp, rel 7.14%) | pass / pass / pass |
| webshop | 128 | 124 / 4 / 0 | average_score=32.03% (target 32.03%, gap 0.00pp, rel 0.00%)<br>success_rate=12.50% (target 12.50%, gap 0.00pp, rel 0.00%) | pass / pass / pass |
| alfworld | 128 | 128 / 0 / 0 | success_rate=3.91% (target 3.91%, gap 0.00pp, rel 0.00%) | pass / pass / pass |
| mbpp-plus | 128 | 119 / 9 / 0 | pass_at_1=67.19% (target 67.19%, gap 0.00pp, rel 0.00%) | pass / pass / pass |
| humaneval | 128 | 125 / 3 / 0 | pass_at_1=92.19% (target 92.19%, gap 0.00pp, rel 0.00%) | pass / pass / pass |

## Diagnostic results

| Benchmark | N | Valid / model-invalid / infra | Metrics | Diagnostic / Formal |
|---|---:|---:|---|---|
| healthbench | 128 | 128 / 0 / 0 | overall_score=53.17% (diagnostic target 52.98%, gap 0.19pp, rel 0.36%) | pass / diagnostic-only |

The HealthBench score is graded directly by the same adapter-free Qwen3.5-9B SGLang model family, never by OpenAI or GPT. Its matched local reference gap is diagnostic only and cannot grant formal admission.
Published external aggregates and diagnostic judges are never substituted for standalone matched panel targets. HumanEval and HumanEval+ remain separate.
This public document contains no task, answer, rubric, test, prompt, response, or per-record payload.
