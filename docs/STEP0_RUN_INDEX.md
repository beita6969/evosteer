# Step-0 run index — separate conditions, never a best-of table

## Latest owner scope — seven-IID summary, 2026-09-08

The owner withdrew further WebShop work and requested the
[current seven-benchmark summary](STEP0_SEVEN_IID_CURRENT_RESULTS_2026-09-08.md).
It lists the latest available observation for each domain with source, thinking,
sample count and actual peer use; it is **not one complete current single-owner
condition**. ALFWorld remains 126/128 in the recorded snapshot. The just-started
`@6` mixed round was stopped with 44 unscored ALFWorld candidates and no WebShop
candidate; its answers are not reused. Older entries below retain their historical
scope and are not authorization to launch another eight-IID round.

Historical full-panel scope: **A2 only**, seven domains at 128 tasks plus all 30 AIME2026 tasks.
The frozen development panels have prior exposure. A complete single-arm run is not a paired
backbone comparison, and a focused rerun is not a new complete eight-IID result.

## Completed, separately reported A2 rounds

All four entries below are preserved separately in the
[aggregate-only round export](machine-results/step0_a2_separate_rounds_2026-09-07.json).

| Revision / condition | Scope | Scored / planned | Interpretation |
|---|---|---:|---|
| `3f37ec0`, native API, presence penalty 0 | Eight IID | 926 / 926 | Complete round; four current goal criteria met, not all eight. |
| `9a6c4c1`, literal transport, presence penalty 0 | WebShop + ALFWorld | 256 / 256 | Focused interface development; not a replacement eight-IID table. |
| `9a6c4c1`, uniform presence penalty 1.5 | Eight IID | 926 / 926 | Separately declared sampling experiment; not promoted or described as a bugfix-only ablation. |
| `7fda9d4`, authoritative ALFWorld reset task | ALFWorld | 128 / 128 | Focused reset/interface round. |

The owner's acceptance of HumanEval **118/128** remains an inclusive development criterion.
The separate 117/128 sampler result does not erase that acceptance, and scores from these
different rounds are not spliced into a single fictitious all-goals result.

The earlier retained A2 result and its separately labelled post-run diagnostics are in the
[2026-09-06 retained-results export](machine-results/step0_a2_iid_retained_results_2026-09-06.json).
That export must be read with its actor/coordinator revisions and limitations, not relabelled
as a fresh run of the latest communication implementation.

An additional retained full round, `3ab098e` (submission feedback, 926/926), is
documented in [the repair history](STEP0_COMMUNICATION_DEFECTS_AND_REPAIRS.md).
It is a separate earlier condition, not a missing cell to fill into any table above.

## Incomplete historical paired attempts and diagnostic canaries

| Record | Retained candidates / planned | Scored | Status |
|---|---:|---:|---|
| [`3f903e3`](machine-results/step0_paired_2026-09-06_incomplete.json) | 559 / 1,852 | 0 | Incomplete infrastructure failure; unknown interrupted-call cost is not zero. |
| [`7f80f26`](machine-results/step0_paired_2026-09-06_literal_calls_incomplete.json) | 866 / 1,852 | 0 | Stopped for literal-call interface repair. |
| [`b140e33`](machine-results/step0_paired_2026-09-06_named_arguments_incomplete.json) | 66 / 1,852 | 0 | Stopped for operation-header/named-argument repair. |

The [four-candidate field canary](machine-results/step0_integrity_2026-09-06.json),
[four-candidate literal-call canary](machine-results/step0_integrity_literal_calls_2026-09-06.json)
and [24-candidate expanded native canary](machine-results/step0_integrity_expanded_native_2026-09-06.json)
are communication diagnostics, not full-panel quality estimates. Their older communication
summaries predate trusted event-origin coverage and must not be upgraded retrospectively.

The `7069132` full A2 attempt retained all 926 candidates but only 925 definitive scores.
Its missing HealthBench grade remains missing after the allowed diagnostic retry failed;
there is no complete HealthBench mean, substituted answer or successful-score retry loop.
Its failed-invocation costs and limitations remain separately documented in
[the repair history](STEP0_COMMUNICATION_DEFECTS_AND_REPAIRS.md).
The earlier `17b918b` plain-control attempt likewise retained 926 candidates and
925 scores; its missing HealthBench score and failed retry remain incomplete.

## Withdrawn evidence and current work

Historical AIME alternative-derivation/top-2 composites and label-swapped/fallback selections
remain withdrawn as standalone-policy evidence; see the
[historical trained report](SKILLEV_BAYESIAN_IMPROVE_TRAINED16_IID_128_RESULTS.md).

## Current frozen integrity repair

The [integrity/communication repair](STEP0_INTEGRITY_REPAIR_IMPLEMENTATION.md) is frozen
at actor revision `7926eb9`, with separately labelled post-run reporting correction
`871bf92`. All entries below are single-arm A2, not paired comparisons.

| Run ID | Scope | Status |
|---|---|---|
| `canary-a2-integrity-repair-7926eb9-20260907T071727Z` | 15 nonfinal canary tasks | Incomplete: 9 candidates, no scores; six native resets failed from missing Java environment configuration. Resume refused the changed frozen environment without further generation. |
| `canary-a2-integrity-repair-7926eb9-20260907T073158Z` | 15 nonfinal canary tasks | Complete: 15 candidates and scores; 159 native actions acknowledged, ten budgeted repairs resolved; actual peer calls zero. |
| `full-a2-integrity-repair-7926eb9-20260907T073806Z` | Eight IID, 926 planned | Incomplete: 926 candidates, 925 scores; one HealthBench grade missing after its allowed repair. No additional retry, zero substitution or partial Health mean. |

The [attempt report](STEP0_INTEGRITY_REPAIR_RESULTS_2026-09-07.md) and its
[aggregate-only export](machine-results/step0_a2_integrity_repair_2026-09-07.json) include
the seven complete native benchmark means, all eight denominators, actual peer use,
failures, cost, parser/verifier versions and development exposure. Original communication
values and corrected post-run diagnostics are both retained; no native score was changed
or regenerated. The attempt does not replace an earlier complete condition, and its zero
actual peer calls do not establish real multi-agent improvement. Canary single-task scores
remain private.

The script name `run_step0_integrity_paired.py` is historical: it executes the explicitly
configured one or two arms and labels a one-arm report as single-arm. It never creates an A1
control or A3 skill run implicitly. No current paired delta or causal improvement is claimed.

## Signature and code-fence follow-up (`db680d9`)

The [separate aggregate-only export](machine-results/step0_a2_signature_fence_repair_2026-09-07.json)
records condition `A2-no-thinking-integrity-repair@6` without replacing any earlier run.

| Run ID | Scope | Status |
|---|---|---|
| `canary-a2-signature-fence-db680d9-20260907T094538Z` | 6 WebShop + 1 MBPP+ + 1 HumanEval nonfinal tasks | 8/8 complete; all 53 native actions acknowledged; nine budgeted repairs resolved. Single-task scores private. |
| `focused-a2-signature-fence-db680d9-20260907T094943Z` | WebShop, MBPP+ and HumanEval, all 128 fixed tasks each | 384/384 complete in 24.24 minutes. WebShop reward 21.8229, MBPP+ Base AND Plus 91/128, HumanEval 110/128. Not an eight-IID result. |

The focused round has no transport loss or missing native acknowledgement, but 128 unresolved
WebShop decision attempts remain; these are attempts, not a count of failed episodes.
All four terminal-payload failures were followed by an owner submission within the shared
budget. Actual peer calls remain zero, so no observed multi-agent performance benefit is
claimed. Neither unchanged domains nor the missing HealthBench grade were regenerated;
no old answer, metric or best-of score fills a cell in this focused result.

## Submission-channel and standalone-call follow-up

These are independent A2 conditions, not replacements for earlier scores.

| Run ID | Scope | Status |
|---|---|---|
| `canary-a2-submission-channel-1119025-20260907T103540Z` | 15 disjoint/native-train canary tasks | 15/15 scored, but eight standalone-call submissions remained unresolved; no full round of this condition launched. |
| `canary-a2-standalone-native-5fa9b0e-20260907T104849Z` | The same 15 canary tasks, newly generated | 15/15 complete in 137.84 seconds; all 144 native actions acknowledged and all nine budgeted repairs resolved. |
| `full-a2-standalone-native-5fa9b0e-20260907T105437Z` | Eight IID, 926 planned | 926/926 candidates and scores, complete in 61.59 minutes; three of eight goal criteria met, no answer or score reuse. |

The aggregate-only [submission-channel canary](machine-results/step0_a2_submission_channel_canary_2026-09-07.json)
and [standalone-call canary](machine-results/step0_a2_standalone_native_canary_2026-09-07.json)
record both outcomes. The latter parser fixes the eight observed false rejections without
changing historical native execution or grades. Both conditions used existing GPUs 1/3
for model traffic and kept the existing GPU7 service resident without new requests because
of low remaining memory. Per-episode budgets and sampling were not reduced. Canary scores
remain private; zero actual peer calls are explicitly reported in both conditions.

The [full result and limitations](STEP0_STANDALONE_NATIVE_RESULTS_2026-09-07.md) and
[full aggregate export](machine-results/step0_a2_standalone_native_full_2026-09-07.json)
retain every native denominator and the unresolved output failures. All model calls
completed and all native executions were acknowledged, but actual peer calls remain
zero and the all-eight goal remains unmet. No earlier incomplete HealthBench result
was retried or replaced.

## Model-native tool-template canary

| Run ID | Scope | Status |
|---|---|---|
| `canary-a2-native-tools-f79a70b-20260907T124441Z` | 15 disjoint/native-train tasks, A2 only | 15/15 complete in 172.07 seconds; 161 completed model calls, all 145 native actions acknowledged, one actual optional peer call, no unresolved communication failure. |
| `canary-a2-native-tools-b31c562-20260907T174551Z` | Same 15 canary tasks, newly generated under `@10` | 15/15 complete in 284.15 seconds on GPU7; 163 completed calls, 146 native acknowledgements, one actual peer call, all twelve repairs resolved. |
| `full-a2-native-tools-b31c562-20260907T175910Z` | Eight IID, 926 planned, A2 `@10` only | Complete 926/926 in 59.67 minutes; two of eight goals met, 50 actual peer calls, unresolved model-control outputs retained. |
| `canary-a2-native-tools-5a1517a-20260907T184053Z` | Same 15 canary tasks, newly generated under feedback condition `@11` | 15/15 complete in 366.98 seconds; 165 calls and 144 native acknowledgements. Eight unresolved prose-only action attempts in one WebShop episode remain, not an all-clear result. |
| `full-a2-native-tools-5a1517a-20260907T185427Z` | Eight IID, 926 planned, A2 `@11` only | Complete 926/926 in 59.45 minutes; two of eight goals met, 50 actual peer calls, all 2,742 native executions acknowledged; unresolved outputs retained. |
| `focused-aime-a2-native-tools-4dd26d3-20260907T193224Z` | Complete fixed AIME2026 cohort, all 30, A2 `@12` only | Complete 30/30 in 53.43 minutes; 23/30 correct (76.67%), one empty submission, 39 completed calls, zero actual peers; still below80; not an eight-IID result. |

The [native-tool protocol note](STEP0_NATIVE_TOOL_PROTOCOL_2026-09-07.md) and
[aggregate-only canary export](machine-results/step0_a2_native_tools_canary_2026-09-07.json)
retain the `@9` condition. Three repaired action attempts exposed an ambiguity in
native function argument descriptions, clarified separately in `@10` without
automatically rewriting model targets. The [separate `@10` canary export](machine-results/step0_a2_native_tool_arguments_canary_2026-09-07.json)
records its actual execution. After the owner approved replacing unavailable GPUs1/3
with GPUs4/5, the main thread restored three healthy replicas and started the fresh
full round above. Historical scores are not used as results for either new condition,
and the all-eight goal remains unmet by the completed `@10` and preceding full rounds.

The completed [`@10` full aggregate](machine-results/step0_a2_native_tool_arguments_full_2026-09-07.json)
now establishes its own result: two of eight goals met, not all-goal acceptance.

The [separate feedback-canary record](machine-results/step0_a2_native_feedback_canary_2026-09-07.json)
keeps the unresolved model-origin failures visible. Condition `@11` clarifies command
availability and incomplete native-call feedback, and records backend stop causes;
it does not change budgets or select answers. The two complete panels above remain
separate; no best-of score or candidate substitution is permitted.

The completed [`@11` full aggregate](machine-results/step0_a2_native_feedback_full_2026-09-07.json)
records 926 scores, 4,268 completed model calls and 635 budgeted repairs. It does not
include the later `@12` scalar fix. The focused AIME30 round evaluates only that
affected complete cohort; its result cannot fill an `@11` table cell. Its earlier
zero-call helper failure is documented in the protocol note, not hidden as a model
retry. All-eight acceptance remains unmet.

The [completed scalar-repair AIME30 aggregate](machine-results/step0_a2_native_scalar_aime30_2026-09-07.json)
retains the full denominator and remaining output failure. One previously rejected
equivalent-scalar response was accepted from the identical freshly generated owner
text, without another model call. This does not authorize replacing any old score or
choosing a later answer from the remaining ambiguous response.
