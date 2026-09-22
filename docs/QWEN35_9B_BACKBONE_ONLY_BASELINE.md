# Corrected Qwen3.5-9B sampled backbone-only baseline on Protocol 10

> This is the Protocol 10 structured-agent control. The separate adapter-free
> Paper Direct-Qwen reproduction is reported in
> [`QWEN35_9B_DIRECT_REFERENCE_RESULTS.md`](QWEN35_9B_DIRECT_REFERENCE_RESULTS.md);
> the two tracks have different interfaces and must not be conflated.
> It also predates the current IID catalog. The current nine-benchmark
> adapter-free evaluation is reported in
> [`QWEN35_9B_CURRENT_IID_BACKBONE_ONLY_RESULTS.md`](QWEN35_9B_CURRENT_IID_BACKBONE_ONLY_RESULTS.md).

This report **supersedes the all-zero report produced at commit `79645df`**. That result was
dominated by a shared prompt/action-interface failure and must not be interpreted as a model
capability result.

## Evaluation condition

- **Model:** frozen Qwen3.5-9B backbone
- **Adaptation:** none (no LoRA adapter, trainable state, or learned skill library)
- **Protocol:** `skillev-benchmark-protocol@10`, final-evaluation populations, seed 0
- **Sampling:** deterministic, evenly spaced samples of at most 128 records per benchmark; complete
  populations are used when smaller (AIME 2026: 30; MBPP+ fixed-100: 100)
- **Total:** 1,026 records across nine benchmarks
- **Rollout contract:** canonical Qwen chat encoding, `structured-action-json@2`, JSON-root generation
  boundary, benchmark-visible action surfaces, public completion sentinels, and preregistered
  benchmark-specific turn/token profiles
- **Isolation:** evaluator answers, rubrics, tests, goal predicates, and verifier truth were absent
  from model context

## Backbone-only results

Native metrics follow [`benchmark-evaluation-guide.md`](benchmark-evaluation-guide.md). Mean reward
and success rate are the corresponding Protocol 10 `TerminalReward` projections.

| Benchmark | Final population | N | Public native metric(s) | Mean reward | Success rate |
|---|---|---:|---:|---:|---:|
| HotpotQA | hotpotqa-v1.1-dev-distractor | 128 | F1 66.24%; EM 49.22% | 66.24% | 49.22% |
| TriviaQA | triviaqa-v1.0-unfiltered-dev | 128 | F1 48.40%; EM 42.19% | 48.40% | 42.19% |
| AIME 2026 | aime-2026-all-30 | 30 | accuracy 0.00% | 0.00% | 0.00% |
| HealthBench | healthbench-full-5000 | 128 | Qwen3.5-9B-judge native rubric mean 0.3108; project binary posterior SR 10.16% | 31.15% | 10.16% |
| WebShop | webshop-goals-test-v1 | 128 | average score 0.0028; SR 0.00% | 0.28% | 0.00% |
| ALFWorld | alfworld-valid-seen | 95 | SR 8.42% | 8.42% | 8.42% |
| ALFWorld | alfworld-valid-unseen | 33 | SR 3.03% | 3.03% | 3.03% |
| SpreadsheetBench V1 Verified | spreadsheetbench-v1-verified-400 | 128 | Pass@1 1.56% | 1.56% | 1.56% |
| AppWorld | appworld-test-normal | 37 | TGC 21.22%; SGC not reported | 21.22% | 0.00% |
| AppWorld | appworld-test-challenge | 91 | TGC 15.45%; SGC not reported | 15.45% | 0.00% |
| MBPP+ fixed-100 | mbpp-plus-fixed-100-v1 | 100 | Pass@1 20.00% | 20.00% | 20.00% |

AppWorld SGC is intentionally absent because an evenly spaced task sample does not preserve every
complete scenario group. HealthBench's local-judge native rubric score can be negative for an
individual record; Protocol 10 preserves that native value and clips only the reward projection,
so the mean native score and mean reward need not be identical. This historical lane is not an
official GPT-4.1-judge HealthBench score.

## Aggregate execution statistics

- Record-weighted mean reward: **23.34%**
- Record-weighted success rate: **15.69%**
- Benchmark-macro mean reward: **21.31%**
- Benchmark-macro success rate: **14.46%**
- Scorer invocations / submission rate: **657 / 64.04%**
- Candidate failures (scored candidates with zero reward): **253**
- Posterior failures (scored candidates with `success=false`): **496**
- Partial rewards (positive reward with `success=false`): **243**
- Infrastructure failures: **0**
- Model requests / request errors: **11,268 / 0**
- Mean episode steps: **5.4912**

### Action telemetry

Rates use the 5,634 action attempts as their denominator.

| Outcome | Count | Rate |
|---|---:|---:|
| JSON syntax error | 940 | 16.68% |
| Action schema error | 1,382 | 24.53% |
| Valid action | 2,098 | 37.24% |
| Unsupported resource | 10 | 0.18% |
| Unsupported tool | 55 | 0.98% |
| Invalid arguments | 152 | 2.70% |
| Invalid completion | 997 | 17.70% |
| Unavailable skill | 0 | 0.00% |

Of the valid actions, 1,455 reached an environment, 643 were explicit completions, and 14 produced
environment terminals. There were 369 horizon/no-submission trajectories. These are reported
separately and are not mislabeled as candidate failures.

## Interpretation boundary

The corrected run demonstrates that the shared benchmark interface is reachable: direct-answer
tasks produce nonzero scores, interactive tools reach their environments, and every benchmark has
scorer-reached records. AIME remains at zero on this fixed 30-item population, but 22 candidates
reached its scorer; this is therefore a result under the stated model/rollout condition rather than
the former shared-interface failure.

This is a structured-agent backbone-only baseline, not a one-shot QA evaluation. Results are
sampled estimates except for the two smaller complete populations. Comparisons should reuse the
same deterministic selection, evaluator routes, model/tokenizer identity, action protocol, and
budget profiles. No private item, answer, model output, or per-item result is included here.
