# HealthBench Full 2025 official-parity backbone diagnostic

This is a separate evaluation lane from the Protocol 10 structured-agent
`TerminalReward` evaluator. Its purpose is to measure the frozen Qwen3.5-9B
backbone while changing as little as possible from OpenAI's standard
HealthBench Full evaluation. Two grader backends are explicit and must never
be conflated: the OpenAI GPT-4.1 reference backend and the owner-selected local
Qwen3.5-9B diagnostic backend.

## Frozen 128-sample contract

- **Population:** HealthBench Full 2025-05-07; Hard, Consensus, and Professional
  are excluded.
- **Selection:** the official `HealthBenchEval(num_examples=128)` path, which
  applies `random.Random(0).sample(examples, 128)` to the released 5,000 rows.
- **Candidate input:** the official `You are a helpful assistant.` system
  message followed by the unmodified released multi-turn conversation.
- **Candidate output:** one direct natural-language completion. There is no
  agent controller, reasoning/action split, skill retrieval, JSON schema, or
  candidate parser.
- **Candidate decoding:** temperature 0.5 and 2,048 maximum output tokens, as in
  the official GPT-4.1 `ChatCompletionSampler`. Qwen thinking is disabled so
  the served completion is the direct answer rather than a model-specific
  hidden reasoning channel. No other sampling override is sent.
- **Candidate identity:** frozen adapter-free Qwen3.5-9B. This is the sole model
  substitution relative to the standard evaluator.
- **Rubric grader:** `gpt-4.1-2025-04-14` through the official
  `ChatCompletionSampler`, including its helpful-assistant system message,
  temperature 0.5 default, 2,048-token budget, parser, and retry behavior.
- **Scoring:** the pinned official `HealthBenchEval.grade_sample()` and
  `_compute_clipped_stats()` implementations. The headline score is the mean
  of raw per-example rubric scores, clipped only after aggregation; the
  official bootstrap standard deviation is also reported.
- **Length adjustment:** disabled, matching HealthBench Full rather than
  HealthBench Professional.

## Owner-selected local Qwen grader variant

When `--grader-backend qwen-sglang` is selected, the grader model is also the
frozen adapter-free Qwen3.5-9B base route. The grader otherwise retains the
official helpful-assistant system message, temperature 0.5, 2,048-token
budget, rubric prompt, permissive JSON parser, retry loop, rubric formula, and
aggregate-then-clip calculation. Qwen thinking is disabled for the same
model-specific transport reason as candidate generation.

The first grader attempt has no response schema, matching simple-evals. If the
official parser rejects that output and requests the same rubric again, the
Qwen transport constrains the retry to a JSON object whose only required field
is the boolean `criteria_met` accepted by the official parser. `explanation`
remains optional, matching the parser rather than making it a new scoring
requirement. This transport repair prevents a Qwen-only malformed-output loop
while preserving the official first attempt, retry decision, and semantic
acceptance contract.

Equivalent SGLang replicas may be supplied by repeating
`--grader-endpoint-base`; requests are round-robin distributed without
changing prompts or decoding. Results from this variant must be named
**HealthBench Full / 128-sample / Qwen3.5-9B local-judge diagnostic** and are
not official GPT-4.1-comparable scores.

The 128-row result is an **official-evaluator 128-sample diagnostic**, not the
official 5,000-row headline evaluation. Per-example prompts, rubrics, answers,
grader outputs, and verdicts remain in the private runtime directory and never
enter Git.

## Resume and isolation

`scripts/run_qwen35_healthbench_official.py` writes separate private append-only
candidate and score journals. A completed candidate is never regenerated when
grading is resumed. The candidate phase receives only the released prompt;
rubrics remain in the evaluator-owned row until the selected isolated grading phase.
Engineering checkpointing does not change official sampling, prompts,
decoding, grading, or aggregation.

The historical Protocol 10 lane remains useful for TTB training diagnostics,
but its three metrics must be named explicitly:

1. HealthBench Full / 128-sample / Qwen3.5-9B-judge native rubric mean;
2. HealthBench per-record-clipped TTB reward;
3. project-specific binary posterior success rate.

None of those historical metrics is an official GPT-4.1-judge HealthBench
score.
