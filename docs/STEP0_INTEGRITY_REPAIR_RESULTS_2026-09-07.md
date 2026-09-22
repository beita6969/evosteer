# Step-0 integrity repair — incomplete A2 attempt, not an all-goals result

Run: `full-a2-integrity-repair-7926eb9-20260907T073806Z`. Actor/parser implementation:
`7926eb9`. Post-run communication reporter: `871bf92` (no regeneration or regrading).
[Aggregate-only machine record](machine-results/step0_a2_integrity_repair_2026-09-07.json).

**926/926 candidates retained; 925/926 definitive scores.** The run started at
2026-09-07 07:38:09 UTC and ended with a grading failure after about **81.91 minutes**.
One HealthBench rubric produced malformed JSON in both its original call and its one
permitted semantic repair. The attempted second repair was refused before another model
call. All paid-for failed-invocation costs are retained. No further retry was started.

## Frozen condition and denominators

Only `A2-no-thinking-integrity-repair@5`: frozen Qwen3.5-9B, no adapter route, optimizer
step zero, native thinking off, optional advice-only peers, skills/demonstrations/handwritten
policies off, and seed 0. Temperature 0.7, top-p 0.8, top-k 20, min-p/presence penalty 0,
repetition penalty 1. The complete actual tokenizer/template and public inputs are frozen
privately; safe descriptors, parser and native verifier versions are in the export.

Generation concurrency 96, scoring concurrency 8, three existing inference replicas.
WebShop horizon 10 and ALFWorld horizon 20 are unchanged. Ordinary static tasks retain
8 calls / 32,768 output tokens; AIME retains 8 calls / 81,920 output tokens; interactive
tasks retain 8 calls per turn / 160 total calls / 163,840 output tokens. The actual
per-request and context budgets are exported, not inferred from observed usage.

All answers were retained before scoring. There was no vote, selector, fallback answer,
score-guided regeneration, skill injection, automatic catalogue action or semantic action
pruning. These panels are **development-exposed**, not untouched holdouts. No A1 or A3
was run; there is no matched control, paired delta/CI or causal improvement claim.

## Native scores (percent; missing is not zero)

| Benchmark | Scored / planned | Native primary | Secondary / denominator | Development goal |
|---|---:|---:|---|---|
| HotpotQA | 128/128 | Answer F1 **76.40** | Answer EM 59.38 | >75: met |
| TriviaQA released RC | 128/128 | Answer F1 **82.57** | Answer EM 78.91 | >75: met |
| AIME2026 | 30/30 | Accuracy **70.00** | 21/30 correct; one empty submission counted as candidate failure | ≥80: not met |
| HealthBench | **127/128** | **Not available** | One missing native grade; no partial mean | >53.17: unresolved, not a matched-control comparison |
| WebShop | 128/128 | Native reward **22.41** | SR 8.59 (11/128) | ≥80: not met |
| ALFWorld | 128/128 | Native SR **48.44** | 62/128 | ≥80: not met |
| MBPP+ hard (owner-facing name) | 128/128 | Base AND Plus **71.09** | 91/128; Base 85.16, Plus 71.09 | >70: met |
| HumanEval original | 128/128 | Pass@1 **85.16** | 109/128 | ≥118/128: not met in this attempt |

MBPP uses the actual EvalPlus MBPP v0.2 population, not an invented hard split.
Its 37 candidate failures comprise 19 Base-test and 18 Plus-test failures. HumanEval's
19 candidate failures comprise 15 test, one syntax and three runtime failures; they are
not grader outages. ALFWorld seen: **46/97 (47.42%)**; unseen: **16/31 (51.61%)**.
HealthBench remains the explicitly labelled **local Qwen rubric** track, not the official
external judge. The historical acceptance of HumanEval 118/128 is not copied into this run.

## Communication, actual peer use and the real-run F04 correction

**Actual peer calls: zero across all 926 episodes.** The topology exposes optional peers,
but this run only observed owner inference. Peer execution is therefore **not observed**;
the separate synthetic isolated-broker tests establish interface behavior, not a measured
multi-agent benefit on this panel. Peers were not forced to improve the label or score.

All **3,925** model transports have a recorded completion. All **2,741** accepted native
actions have acknowledgements (WebShop 857; ALFWorld 1,884). No stale public surface,
missing acknowledgement, route/feedback mismatch or missing required observed boundary
was reported. This does **not** mean every proposed action was valid or every final resolved.

The original `7926eb9` reporter missed unresolved terminal/native-output failures when
forming its overall status. Its original values remain in `frozen_communication`. The
separate `871bf92` analysis of the same events now reports:

| Benchmark | Corrected post-run status | Unresolved outputs |
|---|---|---:|
| AIME2026 | `unresolved-output-failure` | 1 terminal parse failure |
| WebShop | `unresolved-output-failure` | 136 rejected action decisions |
| ALFWorld | `unresolved-output-failure` | 96 rejected action decisions |

These are **attempt counts, not failed-episode counts**. A later accepted output resolves
earlier failed attempts; an earlier success cannot hide a later failure. HotpotQA, MBPP+
and HumanEval each have one resolved terminal repair. TriviaQA and HealthBench owner
communication is complete; the latter's separate private-grader failure still prevents a mean.

All WebShop rejection categories, including later resolved ones: 101 expected-one-native-call,
96 not-on-current-public-surface, eight unavailable-tool, one ambiguous-action-fields.
ALFWorld: 212 not-on-current-public-surface, 84 expected-one-native-call, eight invalid-tool-call.
These observations do not justify inventing task recipes or automatically selecting a legal
action for the owner. Unresolved outputs remain a limitation, not an integrity PASS.

## Actual cost and elapsed-time limitations

- Owner/model: **3,925 calls**, 11,583,006 input tokens and 1,061,799 output tokens.
- **485 budgeted interface-repair calls**, already included above; they are not free
  serialization. Zero peer calls, skill calls/bodies, handwritten decisions, demonstrations,
  automatic catalogue actions or policy-removed legal actions.
- Local grading: **1,499 requests/responses**, 2,532,755 input and 164,877 output tokens;
  two semantic repairs across the attempt, zero unknown-usage calls.
- Included failed grading invocation: nine requests/responses, 10,483 input and 4,674
  output tokens, one semantic repair. It is not discarded because it produced no score.
- The last AIME/interface-repair tails dominated elapsed time. Completion rates dropped
  below 40 records/hour and were explicitly escalated as needing human decision. The
  original budgets were retained; no long-tail task was cut or replaced to improve speed.

Three inference services remain resident by owner request. Two services were restored once
after simultaneous, unexplained SIGTERM before this attempt; no further restart was made.
Intermittent SSH banner timeouts recovered. Shared-card free memory briefly fell below the
1.5 GiB watch threshold; this was reported and no new GPU work was admitted. No other user's
process was modified. The evaluation coordinator and CPU validation jobs have ended.

## Verification and remaining decision

The [implementation record](STEP0_INTEGRITY_REPAIR_IMPLEMENTATION.md) lists the affected
test batches and the 15-case real native canary. Initial final CPU validation passed 2,841
tests; after the observed F04 correction, 26 targeted tests and two-file mypy passed, followed
by **2,845 tests**, Ruff format/lint, 498-file mypy, both wheel builds and the model-wheel
private-boundary check on the Linux CPU host with CUDA disabled. No independent review
agents or digest checks were used. Unchanged suites were not repeated for these documents
or exports. Only aggregate records are public; licensed per-item material remains private.

**This is not full task/goal acceptance.** It lacks one HealthBench grade, observed peer
execution and several performance goals, and retains unresolved model-control outputs.
Further grading beyond the frozen repair allowance requires an explicitly approved new
scoring condition, separately versioned and never represented as the original complete run.
Existing candidates and definitive scores must remain unchanged.
