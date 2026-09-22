# A2 AIME2026 native-thinking development result

The owner requested a new thinking-on round after accepting the earlier thinking-off
result. This is a separate **complete fixed30** run, not a rerun of selected failures and
not an eight-IID aggregate. [Machine-readable aggregate](machine-results/step0_a2_owner_thinking_aime30_2026-09-07.json).

| Item | Observed result |
| --- | --- |
| Evaluated source | `6c930c4` |
| Condition | `A2-owner-native-aime-thinking@1` |
| Accuracy | **25/30 = 83.33%**; development score threshold80 reached |
| Complete owner finals | 27;25 correct and2 incorrect |
| Budget-exhausted candidate failures | 3, included as zero in the30-task denominator |
| Failed channels | 2 unfinished reasoning;1 malformed channel; all3 ended by length |
| Infrastructure failures / missing scores | 0 / 0 |
| Communication status | `unresolved-output-failure`, **not a clean communication pass** |
| Time to complete summary | 3,225.58472 seconds = **53.76 minutes** |
| Actual calls / input tokens / output tokens | 30 /31,873 /916,715 |
| Actual peer / tool / grader model calls | 0 /0 /0 |

The evaluated model was adapter-free Qwen3.5-9B with zero optimizer steps and no skills.
All30 persisted source records and actual sampling profiles have native thinking enabled.
The arm's legacy/global `native_thinking: false` field is **not** the resolved benchmark
setting: the explicit benchmark map overrides it consistently at template, actor and broker.
Each task had the same seed0, temperature0.7, top-p0.8, top-k20, context98,304, output
allowance81,920 and at most8 calls. Each of the3 unsuccessful channel completions used
the entire81,920-token output allowance. No reasoning draft was promoted to a final.

No voting, alternate-candidate selector, baseline fallback, cross-run answer reuse, extra
serialization generation or selective regeneration occurred. Peers were available but not
invoked; this run does **not** demonstrate an observed multi-agent cooperation gain.
The prior thinking-off23/30 remains its own historical result. Neither condition is a matched
backbone control, and the two answers are not combined or described as a causal architecture gain.

## Execution and validation

- A separate three-task synthetic native-channel canary completed3/3 with4 real calls;
  one budgeted interface repair resolved. It is not benchmark performance evidence.
- The evaluated source had already passed the one22049 CPU `CUDA_VISIBLE_DEVICES="" make check`
  recorded in the [source integration note](STEP0_SINGLE_OWNER_SOURCE_IMPLEMENTATION_2026-09-07.md):
  3,031 tests, full Ruff/mypy and both wheels. No unchanged full suite was rerun for this report.
- The formal30-task coordinator (PID38649) exited after30 candidates and30 scores were written.
  Main-thread checks confirmed the three existing22048 GPU4/5/7 services still healthy afterward.
  No new inference daemon, training process or additional GPU was used. The canary PID25476 ended.
- Main-thread monitoring raised the below40-completions/hour alert during startup/long-tail phases.
  Budgets were not shortened and active requests were not migrated or replayed to fill idle cards.
- Source/grade extraction was read-only and this release is aggregate-only. No question, answer,
  per-item score, username, private path or key is published. No hash checks or review agents were used.

The AIME score target is met, but the3 output failures remain visible. ALFWorld/WebShop repairs
and their newly declared evaluation conditions are separate work, not results of this run.
