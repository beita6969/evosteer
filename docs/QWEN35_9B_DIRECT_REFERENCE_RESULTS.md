# Qwen3.5-9B Direct Reference Results

> Generated from the executable protocol and public aggregate JSON; no private predictions or targets are read by this renderer.

> **Catalog note:** this table is a historical direct-reference snapshot under
> the superseded seven-IID catalog, with the repaired SWE-bench component
> substituted in place. The current authoritative project IID catalog is
> HotpotQA, TriviaQA, AIME 2026, HealthBench, WebShop, ALFWorld,
> SpreadsheetBench, SWE-bench, and HumanEval. Therefore the 798-record scope and
> legacy aggregate rows below are not a current all-IID result.

This is the adapter-free Paper Direct-Qwen track. It does not use LoRA, skills, retrieval, the posterior, the Z head, or the structured action codec.

Requested role: **iid**; planned records: **798**. AIME uses its complete 30-record panel and every other selected benchmark uses 128.

## Component metrics

The numeric rule is strict: `absolute gap < 7.0 percentage points`. The full-precision value drives the status; the two-decimal value is display only.

| Benchmark | Metric | Reference | Observed (full) | Observed (display) | Gap (pp) | Threshold | Numeric | Scientific comparability | Planned/scored | Infra (gen/scorer/env) | Prompt | Decoding | Population | Seed aggregation |
|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---|---|---|---|
| hotpotqa | em | 60.94% | 63.28125% | 63.28% | 2.34125 | `<7.0 pp` | PASS | exact-paper; comparison-semantics-unknown; scientific NO-GO | 128/128 | 0/0/0 | `hotpotqa-full-context@1` | `qwen35-nonthinking-general@1` | `skillflow-released-iid-v3` | comparison-semantics-unknown; seeds=42 |
| hotpotqa | f1 | 75.70% | 78.97818687133945% | 78.98% | 3.27818687133945 | `<7.0 pp` | PASS | exact-paper; comparison-semantics-unknown; scientific NO-GO | 128/128 | 0/0/0 | `hotpotqa-full-context@1` | `qwen35-nonthinking-general@1` | `skillflow-released-iid-v3` | comparison-semantics-unknown; seeds=42 |
| triviaqa | em | 44.88% | 52.34375% | 52.34% | 7.46375 | `<7.0 pp` | FAIL | exact-paper; comparison-semantics-unknown; scientific NO-GO | 128/128 | 0/0/0 | `short-answer-with-context@1` | `qwen35-nonthinking-general@1` | `skillflow-released-iid-v3` | comparison-semantics-unknown; seeds=42 |
| triviaqa | f1 | 54.30% | 59.813988095238095% | 59.81% | 5.513988095238095 | `<7.0 pp` | PASS | exact-paper; comparison-semantics-unknown; scientific NO-GO | 128/128 | 0/0/0 | `short-answer-with-context@1` | `qwen35-nonthinking-general@1` | `skillflow-released-iid-v3` | comparison-semantics-unknown; seeds=42 |
| aime-2026 | accuracy | 46.67% | 86.66666666666667% | 86.67% | 39.99666666666667 | `<7.0 pp` | FAIL | exact-paper; comparison-semantics-unknown; scientific NO-GO | 30/30 | 0/0/0 | `integer-cot@1` | `qwen35-thinking-math@1` | `skillflow-released-iid-v3-full` | comparison-semantics-unknown; seeds=42 |
| medqa | accuracy | 69.53% | 80.46875% | 80.47% | 10.93875 | `<7.0 pp` | FAIL | exact-paper; comparison-semantics-unknown; scientific NO-GO | 128/128 | 0/0/0 | `multiple-choice-label@1` | `qwen35-thinking-general@1` | `skillflow-released-iid-v3` | comparison-semantics-unknown; seeds=42 |
| webshop | average_score | 56.93% | 38.06361607142858% | 38.06% | 18.86638392857142 | `<7.0 pp` | FAIL | approximate-only; comparison-semantics-unknown; scientific NO-GO | 128/128 | 0/0/0 | `webshop-native-react@2` | `qwen35-nonthinking-general@1` | `official-valid-seed42-reconstruction` | comparison-semantics-unknown; seeds=42 |
| webshop | success_rate | 32.03% | 14.0625% | 14.06% | 17.9675 | `<7.0 pp` | FAIL | approximate-only; comparison-semantics-unknown; scientific NO-GO | 128/128 | 0/0/0 | `webshop-native-react@2` | `qwen35-nonthinking-general@1` | `official-valid-seed42-reconstruction` | comparison-semantics-unknown; seeds=42 |
| alfworld | success_rate | 48.28% | 29.6875% | 29.69% | 18.5925 | `<7.0 pp` | FAIL | approximate-only; comparison-semantics-unknown; scientific NO-GO | 128/128 | 0/0/0 | `alfworld-native-react@2` | `qwen35-nonthinking-general@1` | `official-valid-seen-unseen-seed42-reconstruction` | comparison-semantics-unknown; seeds=42 |
| swe-bench | resolved | 17.19% | 12.5% | 12.50% | 4.69 | `<7.0 pp` | PASS | exact-paper; comparison-semantics-unknown; scientific NO-GO | 128/128 | 0/0/0 | `qwen-direct-readonly-repository-agent@1` | `qwen35-repository-agent-supervisor@1` | `skillflow-released-iid-v3-verified` | comparison-semantics-unknown; seeds=42 |

## Coverage and telemetry

Submission is an output-contract event, not benchmark success. A missing official verdict remains infrastructure and is never converted to a zero.

| Benchmark | Final | Candidate responses | Definitive verdicts | Submission | Scorer/terminal reach | Valid native action | Parse invalid | Env invalid | Partial reward | Infra total | Availability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hotpotqa | 128/128 | 128 | 128 | 99.22% | 99.22% | N/A | N/A | N/A | N/A | 0 | complete |
| triviaqa | 128/128 | 128 | 128 | 99.22% | 99.22% | N/A | N/A | N/A | N/A | 0 | complete |
| aime-2026 | 30/30 | 30 | 30 | 90.00% | 90.00% | N/A | N/A | N/A | N/A | 0 | complete |
| medqa | 128/128 | 128 | 128 | 86.72% | 86.72% | N/A | N/A | N/A | N/A | 0 | complete |
| webshop | 128/128 | 128 | 128 | 100.00% | 63.28% | N/A | 0.00% | N/A | 47.66% | 0 | complete |
| alfworld | 128/128 | 128 | 128 | 97.66% | 89.06% | N/A | 1.09% | N/A | 0.00% | 0 | complete |
| swe-bench | 128/128 | 128 | 128 | 42.97% | 100.00% | N/A | N/A | N/A | N/A | 0 | complete |

## Legacy published aggregate rows

These rows retain the old component definitions and are shown only for
historical continuity. They must not be presented as an aggregate over the
current nine-item IID catalog.

| Aggregate | Observed (full) | Observed (display) | Reference | Gap (pp) |
|---|---:|---:|---:|---:|
| avg-iid-answer-em | 57.8125% | 57.81% | 52.91% | 4.9025 |
| avg-iid-answer-f1 | 69.39608748328877% | 69.40% | 65.00% | 4.39608748328877 |
| avg-iid-acc-pass | 43.57483878968253666666666667% | 43.57% | 45.11% | 1.53516121031746333333333333 |

## Gates

- Scope numeric parity: **NO-GO**
- Scope scientific parity: **NO-GO**
- Engineering validation: **PASSED**
- Formal training: **NOT-EVALUATED**

Unknown paper seed/protocol semantics remain scientific NO-GO even when a numeric single-run score falls inside the strict `<7.0` percentage-point band.

Engineering receipt command: `CUDA_VISIBLE_DEVICES="" TMPDIR=<linux-local-temp> make check`.
