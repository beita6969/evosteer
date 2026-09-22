# A2 standalone-call repair: complete development round

Run `full-a2-standalone-native-5fa9b0e-20260907T105437Z`, actor/coordinator revision
`5fa9b0e`, condition `A2-no-thinking-integrity-repair@8`.
The [aggregate-only export](machine-results/step0_a2_standalone_native_full_2026-09-07.json)
contains the exact controls, costs, denominators and communication counts.

**926/926 candidates and definitive scores; no missing grade. The all-eight goal is
not met.** This is one A2 development-exposed condition, not an unseen test result,
a matched backbone comparison or a causal improvement estimate. Earlier high scores
are not substituted into this table.

## Native scores

Values below are on the 0–100 scale; the machine export retains native fractions.

| Benchmark | Count | Native metric | Score | Current goal | Met |
|---|---:|---|---:|---:|---|
| HotpotQA | 128 | Answer F1 | 75.8113 | >75 | Yes |
| TriviaQA | 128 | Answer F1 | 82.2926 | >75 | Yes |
| AIME2026 | 30 | Accuracy | 53.3333 (16/30) | >=80 | No |
| HealthBench | 128 | Local-Qwen rubric mean | 52.6488 | >53.17 | No |
| WebShop | 128 | Native reward | 24.1406 | >=80 | No |
| ALFWorld | 128 | Native success rate | 39.8438 (51/128) | >=80 | No |
| MBPP+ | 128 | Base AND Plus pass@1 | 72.6563 (93/128) | >70 | Yes |
| HumanEval | 128 | Pass@1 | 86.7188 (111/128) | >=92.1875 | No |

- HotpotQA exact match: 58.5938; TriviaQA exact match: 78.1250.
- WebShop success: 12/128 (9.3750), separately from its native partial-credit reward.
- ALFWorld: seen 40/97 (41.2371), unseen 11/31 (35.4839), no missing split metadata.
- MBPP+ Base: 109/128 (85.1563); Plus: 93/128 (72.6563). There is no invented official
  “hard” split or score-selected subset.
- AIME retains one empty-submission candidate failure, scored as a native failure,
  not omitted from the 30-task denominator. All other AIME results are definitive.
- HumanEval has 14 native test failures and three runtime errors, with no syntax or
  infrastructure failure. The owner's earlier acceptance of 118/128 is preserved as
  history, but that older score does not replace the new condition's 111/128.
- HealthBench uses the declared local Qwen3.5-9B grader, not the official external
  judge. Every one of this round's 128 answers has a grade. Earlier incomplete runs
  remain unchanged; this is neither an extra retry nor a substitute missing grade.

## Integrity, communication and cost

Qwen3.5-9B, no adapter, optimizer steps zero, native thinking off, one seed zero.
The owner supplies the answer or action; no vote, selector, fallback, cross-run answer
reuse, injected skill, demonstration, handwritten decision, semantic action pruning
or automatic catalog action was used. The frozen sampling and per-episode budgets
were not reduced. No A1 or A3 condition was generated.

All **4,053 model requests completed**, using **11,433,940 input** and **878,747 output
tokens**. All **2,900 native actions were acknowledged**. The 459 budgeted interface
repair calls are included in model cost, not labelled free serialization.

There is no observed route mismatch, missing native acknowledgement, feedback mismatch,
surface mismatch or unobserved required transport boundary. Nevertheless, communication
is **not wholly successful**:

| Domain | Unresolved output failures | Later resolved failures |
|---|---:|---:|
| AIME terminal submission | 1 | 4 |
| WebShop action attempts | 72 | 102 |
| ALFWorld action attempts | 120 | 181 |

These are attempts, not counts of failed episodes. HotpotQA, TriviaQA, MBPP+ and
HumanEval each resolved one terminal-payload failure. Native task failure is reported
separately from transport failure. A valid but incorrect model decision is not, by
itself, proof of a broker bug.

**Actual peer model calls: zero in all 926 episodes.** Peers were available, but the
policy never requested one. Real peer execution is `not-observed`; the synthetic
actor/broker peer tests do not establish real multi-agent performance benefit.

All 926 scores have cost records. Grading made 1,500 local-Qwen requests, consuming
2,587,106 input and 163,793 output tokens, with three permitted semantic repairs,
zero failed invocations and zero unknown-usage calls. Summed parallel scorer time
was 1,612.34 seconds; it is not additional wall time to add to the run duration.

## Execution and validation

Start: 2026-09-07 10:54:40 UTC. Summary written after **3,695.66 seconds**, or
**61.59 minutes**, about **902 records/hour end to end**. This is evaluation throughput,
not a training steps/hour measurement. The main thread escalated the late AIME tail
below 40 completed records/hour as “needs human decision”, retained the original
budget, and waited for normal completion. Cumulative-rate seconds-level ETAs were
explicitly rejected as unsuitable for the long tail.

Generation concurrency was 64; scoring concurrency eight. Actor and grader traffic
used the existing physical GPUs 1/3. The existing GPU7 service stayed resident without
new run traffic: it had only 724 MiB free at launch. Its memory later recovered, but
the frozen run was not rerouted. All three services passed the main-thread post-run
PID, command, CUDA mapping and health checks. Bulk generation and grading used both
routed replicas; the final AIME tail occupied one. No inference daemon, fourth GPU
or training process was added. Both new canaries and the full coordinator ended.

All code was written by the main thread; the only subagent watched GPUs. The affected
tests passed first as a 55-test submission-channel set, then as a 200-test native-call,
prompt and actual-broker set after the newly observed serialization defect was fixed.
Local Ruff and targeted mypy passed. The two changed-source final milestones ran on
the approved CPU host with `CUDA_VISIBLE_DEVICES="" make check`: **2,873** and then
**2,890** tests passed, including complete Ruff/mypy and both wheel builds. Each had
85 existing warnings. No unchanged-source full suite was repeated for result-only
documentation; no independent review agent, digest check or retired gate was used.

The fixes establish narrower submission/serialization correctness, not all-goal
acceptance. Remaining output failures, zero actual collaboration, development exposure
and five unmet goal criteria remain explicit risks. No goal-completion claim is made.
