# Qwen3.5-9B HealthBench Full 128-sample local-judge result

## Result

| Population | N | Released rubric items | Native rubric mean | Bootstrap standard deviation |
|---|---:|---:|---:|---:|
| HealthBench Full 2025-05-07 | 128 | 1,497 | **52.40%** | **2.42 percentage points** |

The unrounded aggregate is `0.5239648130934516`; the official aggregation
routine's 1,000-resample bootstrap standard deviation is
`0.0241554881104185`. All 128 sampled examples produced a candidate and a
definitive rubric score. This result is named:

> **HealthBench Full / 128-sample / Qwen3.5-9B local-judge diagnostic**

It is not an official GPT-4.1-comparable HealthBench score.

## Candidate contract

- frozen adapter-free Qwen3.5-9B base model;
- official `HealthBenchEval(num_examples=128)` selection, i.e.
  `random.Random(0).sample` over the released 5,000-example Full population;
- official `You are a helpful assistant.` system message plus the original
  multi-turn conversation;
- one direct natural-language completion, with no skill, agent controller,
  reasoning/action split, candidate JSON, or answer parser;
- temperature 0.5 and 2,048 output tokens; Qwen-specific thinking disabled.

All 128 candidates were non-empty, unique by released prompt ID, and exactly
matched the official Random(0) panel. No candidate was regenerated during
grading.

## Qwen local-judge contract

The same frozen adapter-free Qwen3.5-9B base model performed rubric grading on
two equivalent SGLang replicas. Apart from grader identity and the
Qwen-specific non-thinking transport, grading retained the pinned OpenAI
simple-evals behavior:

- official HealthBench rubric prompt and released physician rubrics;
- helpful-assistant system message, temperature 0.5, and 2,048-token budget;
- one independent `criteria_met` decision per rubric;
- official permissive parser and parser-triggered retry loop;
- official `calculate_score()` per-example formula;
- no length adjustment;
- raw per-example scores averaged first, then the aggregate clipped to
  `[0, 1]`;
- official bootstrap calculation.

One uncommitted example repeatedly produced malformed unconstrained JSON. Its
infrastructure-only resume preserved all 127 completed scores. On an official
parser-triggered retry, the Qwen transport constrained output to a JSON object
whose only required field is the boolean `criteria_met`; the official parser
and scoring formula remained authoritative. No successfully scored example
was repeated.

Per-example conversations, rubrics, candidates, grader outputs, and verdicts
remain in the private evaluation directory and are not stored in Git.
