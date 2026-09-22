# SKILLEV BayesianImprove Step-0 AIME 2026 Repair Results

> **Historical inference condition (superseded 2026-09-06).** The thinking/terminal-transcription
> protocol below is not the corrected single-final, literal thinking-off condition. It does not
> establish a requirement that AIME disable thinking-off. Retain its numbers only with their
> original conditions; [corrected results](STEP0_PAIRED_NO_SKILL_AND_TEXT_SKILL_RESULTS.md) are separate.

## Scope

This report supersedes the AIME 2026 row in the earlier Step-0 exact-eight
32-sample reports. The evaluated method is
`skillev-bayesian-improve-step-zero@1`: the project's SkillFlow-plus-`idea.tex`
BayesianImprove architecture, not unmodified SkillFlow. At Step-0 the
answer-free seeded inference controller and typed skill retrieval are active;
optimizer, adapters, TTB updates, Bayesian posterior/calibration updates, and
Operator skill evolution remain inactive.

AIME 2026 has only **30 released records**. The final run therefore uses the
complete population once; it does not duplicate two records to manufacture a
nominal 32. The owner-authoritative IID catalog remains exactly HotpotQA,
TriviaQA, AIME 2026, HealthBench, WebShop, ALFWorld, MBPP+ hard (truthfully
implemented as EvalPlus Base+Plus), and HumanEval.

## Root cause and repair

The original Step-0 AIME result was 21/30 (70.00%). The repair found three
condition-level problems rather than changing the architecture or exposing
answers:

1. The Step-0 adapter capped AIME reasoning below the public Qwen reference
   condition. AIME now receives the same 81,920-token reasoning ceiling as the
   Protocol 14 backbone condition.
2. The initial controller and reasoning suffix prescribed extra behavioral
   constraints. The seeded Step-0 controller now states only the phase and
   wire. Task strategy comes from the single typed retrieved skill. This rule
   is shared by all eight IID routes; syntax constraints and current native
   action surfaces remain wire/environment requirements rather than hidden
   solution strategies.
3. The terminal pass sometimes recomputed a solved problem and changed the
   model's own final integer. In the immediately preceding complete diagnostic,
   26/30 reasoning traces contained a boxed integer and five terminal outputs
   disagreed with that last boxed value. Aggregate rescoring showed that all
   five disagreements harmed rather than improved accuracy. The final terminal
   wire therefore uses frozen greedy Qwen only to transcribe the reasoning
   pass's last explicit integer. It receives no task statement and is explicitly
   forbidden from solving the problem again.

The reasoning authority is frozen, adapter-free Qwen3.5-9B with the public
Protocol 14 AIME decoding contract: native thinking, sampling temperature 1.0,
top-p 0.95, top-k 20, presence penalty 1.5, 81,920 output tokens, and seed 42.
Realized reasoning remains outside the TTB action-edge probability. The
terminal transcription pass is non-thinking greedy decoding under the
`[0,999]` syntax-only grammar.

The seeded H0 boundary also rejects structured answer, gold, oracle, scorer,
rubric, reward, solution, demonstration, and ad-hoc strategy fields. It does
not scan or censor legitimate public question text. No evaluator answer,
label, score, previous candidate, or per-record outcome entered model context.

## Final fresh all-30 result

| Population | Correct | Accuracy | Owner minimum | Status |
|---:|---:|---:|---:|---|
| 30 | **26** | **86.67%** | 80.00% | **PASS** (+6.67 pp) |

- 30/30 definitive outcomes.
- Zero candidate parse failures and zero scorer/infrastructure failures.
- One seed group (`seed=42`), one frozen condition, and no candidate-failure
  retries or result splicing.
- Frozen Qwen3.5-9B, no LoRA/adapter, and zero optimizer, TTB, posterior, or
  Operator updates.
- Wall time: 3,223.92 seconds; throughput: 33.50 records/hour. The long tail
  came from the public 81,920-token reasoning allowance rather than a stalled
  request.

This was an adaptive engineering repair, not an untouched confirmatory
estimate: earlier complete runs were used only to identify general prompt,
budget, and terminal-wiring faults. The reported number is a new complete
all-30 run under the final frozen code and condition; no successful records
from earlier runs were reused.

## Gate interpretation

The current task's explicit AIME gate is **at least 80.00%**, so 86.67% passes.
The separate global architecture-promotion rule requires Step-0 to be
**strictly greater than** the latest 26/30 (86.67%) backbone aggregate. The
final Step-0 run ties that count and therefore does **not** pass the stricter
promotion rule. These two decisions must not be conflated.

Questions, answers, task IDs, candidate responses, and per-record results
remain in the private evaluation directory and are not committed to Git.
