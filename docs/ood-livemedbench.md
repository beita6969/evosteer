# LiveMedBench replaces the GPQA OOD slot

The owner's 2026-09-12 instruction replaces GPQA with **LiveMedBench**. The active
OOD catalog is MuSiQue-Ans, NQ-Open, Omni-MATH, LiveMedBench, ScienceWorld,
LiveCodeBench and APPS Introductory. IID is unchanged. Historical GPQA scores and
source contracts remain readable, but GPQA cannot enter a new active OOD panel.
`AGENTS.md` currently names ThReadMed-QA; the newer conversation instruction takes
precedence. Neither `AGENTS.md` nor `idea.tex` is edited.

The subsequent owner-authorized seven-panel rerun uses a new complete-response
prompt condition; see [trajectory fixes](ood-trajectoryfix-20260912.md). The
first run below retains its original prompt, score and provenance.

## First-run frozen source and generation condition

- **64 samples, seed 0**, uniform reservoir sampling in the released file order;
  exact selected case IDs are saved privately before generation.
- Use one `LiveMedBench_v202601.json` snapshot, not the live repository's combined
  JSON files. This is the official Quickstart snapshot, containing 2,756 cases and
  16,702 criteria. It is not a claim that every case postdates Qwen's training.
  [Official repository](https://github.com/ZhilingYan/LiveMedBench/),
  [dataset](https://huggingface.co/datasets/JuelieYann/LiveMedBench),
  [paper](https://arxiv.org/html/2602.10367v1).
- Fresh frozen Qwen3.5-9B Step-0 output; one owner per trajectory, skills off,
  native thinking off, batch concurrency 32, one uninterrupted maximum of 8,000
  completion tokens (including any budgeted clarification), seed 0. No consultants,
  candidate selection, old answer reuse or outcome-dependent sample changes.
- Reuse the current actor/broker isolation, native XML submission interface and
  explicit final-answer ownership checks. Only `narrative` and `core_request`
  reach the actor. The official language-dependent user instruction is reused;
  the project's frozen budget and submission interface are additional framing.
- Noncode decoding remains temperature 0.7, top-p 0.8, top-k 20, presence penalty
  1.5, repetition penalty 1.0. These are our declared conditions, not a claim to
  reproduce every model configuration in the paper.

## Official scoring logic, explicitly different judge

`livemedbench_upstream.py` copies the official single-criterion prompt and the two
point-calculation functions, with the upstream MIT license. Only type spelling
and formatting are adapted. The old model SDK runner and dataset-curation agents
are not imported. [Official evaluator](https://github.com/ZhilingYan/LiveMedBench/blob/main/evaluate/evaluate_model.py),
[metric calculation](https://github.com/ZhilingYan/LiveMedBench/blob/main/evaluate/metric_calc.py).

The judge is **gpt-5.6-luna / medium / OpenAI Chat Completions**, one request per
criterion, maximum 8,000 completion tokens per request, timeout 120 seconds,
SDK retries disabled and `store=false`. Grading starts only after the complete
generation phase. The judge never writes back to the evaluated trajectory.
Our metric is `luna-medium-api-rubric-score`; it is **not** the paper's original
GPT-4.1-2025-04-14 score and is not multiple-choice accuracy.

```text
raw_case_score = sum(points * criterion_met) / sum(positive_points)
reported_score = clip(mean(raw_case_score over all 64 cases), 0, 1)
terminal_reward_projection = clip(raw_case_score, 0, 1)  # separate per-case value
```

The README describes case-wise clipping, but the actual `metric_calc.py` preserves
raw negative case scores and clips the final aggregate. This implementation uses
the actual code path and retains the raw scores. Negative criteria are triggered
when the prohibited behavior is present; their negative weight is applied once.
No official binary success threshold is invented.
The released snapshot also contains negative-only cases: as in the official
`calculate_model_scores`, a zero positive denominator yields a raw score of zero.
Such selected cases stay in the fixed panel and their rubric verdicts are retained;
they are not dropped or replaced.

The thin adapter deliberately does **not** copy upstream silent omissions:
missing candidates, malformed rubrics, missing verdicts, transport errors and
truncated/invalid judge replies cannot silently become zero or disappear from
the denominator. Empty owner submissions are explicit candidate failures at zero.
Every requested criterion retains its raw reply, finish reason, usage and status.
Saved true and false verdicts both survive an exact-submission resume. A failed
request stops the condition rather than triggering an undisclosed retry.

## Entrypoints and private artifacts

The existing `skillev_private.evaluation.ood_export` entrypoint accepts a source
with benchmark `livemedbench`, one `paths` entry, and `snapshot: v202601`.
Its output feeds the existing `scripts/run_step0_integrity_paired.py` runner with
`catalog: ood`, `evaluation_sample_counts: {livemedbench: 64}`, the updated seven-name
thinking map, owner concurrency 32 and scorer concurrency 4.

Private scorer settings contain `profile`, `judge_model`, `reasoning_effort` and
`cache_directory`. Credentials are supplied only to the trusted grader environment;
the actor's cleared environment, private filesystem and network namespace expose
neither credentials nor the grading data. Do not place credentials or dataset
contents in a runtime config intended for Git.

Record actual candidate and rubric counts, native aggregate, raw mean, negative
criteria, truncations, parser/transport errors, model/tool calls, token usage,
elapsed time and the frozen service settings. Retain unsuccessful records and
failed judging costs. Do not compare the resulting rubric percentage directly
with the retired GPQA choice accuracy or rerun the other six OODs for this change.

## Completed first 64-case run (2026-09-12 UTC)

| Item | Observed result |
|---|---:|
| Luna-medium native rubric mean | **12.5824 / 100** |
| Fresh owner submissions / planned | 64 / 64 |
| Scored cases / planned | 64 / 64 |
| Resolved criterion calls / planned | 360 / 360 |
| Positive criteria met | 49 / 238 |
| Negative criteria triggered | 20 / 122 |
| Raw negative-score cases | 12 |
| Selected zero-positive-denominator cases (retained) | 1 |
| Owner calls / tool calls / consultant calls | 64 / 0 / 0 |
| Empty responses / communication repairs / truncated outputs | 0 / 0 / 0 |
| Owner input / completion tokens | 47,733 / 4,819 |
| Judge input / completion tokens | 245,331 / 44,987 |
| Judge reasoning tokens (included in completion count) | 13,071 |

All 64 owner generations reported `finish_reason=stop`; the longest consumed
368 completion tokens, well below the declared 8,000 cap. Mean response length
was 75.30 tokens / 203.06 characters. Short responses and unmet rubric criteria
are observations, not a justification to repair answers, suppress negative
scores or regenerate this panel. There were zero failed judge requests and no
missing usage reports. The separate synthetic API connectivity check used one
call, 409 input tokens and 39 completion tokens; it is not one of the 64 cases.

Generation took approximately 14.37 seconds (4.45 cases/second), judging about
261.92 seconds (0.244 cases/second), and launch-to-exit wall time was 282.37 seconds.
Only the pre-existing GPU5 inference service was reused; no new SGLang service or
formal training was started. Other six OOD results were not regenerated or edited.
This is a low measured rubric score, not evidence of meeting a literature threshold
or equivalence to the original GPT-4.1 judge. All licensed inputs, outputs, criterion
verdicts and service details are retained privately rather than committed.

### Verification and remaining limits

- The original upstream `calculate_model_scores` was also executed after grading:
  all 64 per-case values and the complete-panel aggregate matched this adapter.
- Local changed-file Ruff checks passed. Remote CPU targeted tests passed:
  14 LiveMedBench tests and 38 public-input/noninterference tests. An initial
  actor test caught the missing public task-family registration, fixed before
  any benchmark generation. A selected negative-only case exposed the official
  zero-denominator behavior, which was retained rather than filtered away.
- Final `CUDA_VISIBLE_DEVICES="" make check` passed on the current integration
  baseline: **4,442 passed, 12 skipped**; format, lint, mypy, both wheels and the
  model-wheel/private-package boundary check also passed. Earlier checks caught
  a tuple type annotation and missing synthetic catalog fixtures; both were
  corrected without changing the running generation/scoring snapshot or answers.
- Complete checks were concentrated at integration/finalization, not repeated
  for the final documentation-only notes. No hash checks or independent review
  agents were used; the mandated Git-only agent does not answer or judge cases.
- This is one small, explicitly configured sample evaluation. Judge substitution,
  short-answer behavior and possible training-data exposure limit external
  comparisons. No formal training or W&B training logging was performed.
