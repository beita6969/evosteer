# Native tool transport and training-quality repair

## Observed failure, not a claimed training improvement

The formal run stopped at its complete Step 10 checkpoint. Its original fixed
quality check reported regression; no Step 11 was started. The same fixed panel's
action-structure validity fell from 0.99065 to 0.40244, first-turn validity from
0.96429 to 0.39286, mean reward from 0.75660 to 0.57003, and success from 0.75 to
0.50. Changing training batches alone would not establish regression; this fixed
panel is the relevant comparison. All original results/checkpoints remain intact.

Actual request inspection found a concrete conflict: the installed Qwen native
tool template allows natural-language commentary before a function call, while
the project's v1 parser rejects any response not starting with `<tool_call>`.
Many later outputs contain a complete explicit submission but are rejected solely
because of that commentary. Others use a JSON function call or its arguments.
Repeated rejection adds more reasoning/action calls and longer histories. There
are also genuinely malformed, truncated and off-task generations; fixing transport
does not establish that those outputs are correct.

One inspected off-task generation received the correct complete code task and
owner reasoning, with zero reported cached prompt tokens. Missing task context is
therefore not an explanation for that particular failure. This does not prove the
cause of every semantic error or independently qualify the trained policy.

## Explicit new carrier condition

`native-single-tool-call@2` is supported throughout configuration, H0, phase
rendering, live execution, persisted-action interpretation and diagnostics.

- Decode one complete native function call, allowing surrounding commentary.
- Decode JSON calls with `name` and `arguments`/`parameters`, including the native
  JSON wrapper and a single explicit JSON fence with surrounding commentary.
- Decode a bare argument object only when its field names uniquely identify a
  declared public function. No function is chosen using an answer, scorer or model.
- Preserve argument content and the entire original sampled response/token span.
  Rollout, sealed scoring and provisional F/B scoring retain identical action IDs.
- Reject multiple/incomplete carriers, duplicate parameters and calls inside an
  unfinished reasoning channel. A nested incomplete submission previously swallowed
  by the old greedy parser is explicitly covered by a synthetic regression test.
- Keep v1 interpretation unchanged for existing evidence. New sampling conditions
  are explicit; historical failures are not regraded or relabelled as successes.

The new phase message removes the commentary prohibition rather than adding more
format restrictions. It does not claim that the backward policy can see the current
reasoning; the hindsight condition remains answer/action/current-reasoning isolated.

The phase-cross diagnostic had a separate wiring defect: it always used the JSON
parser and JSON-root stop rule, even for native-tool histories. It now derives both
from the persisted phase contract. This correction is diagnostic only, not evidence
that the committed training parse-error counters were all mistaken.

The cooperative pause callback now rechecks operator intent after an awaited quality
probe. A stop arriving during the probe cannot inadvertently admit another batch.

The real baseline probe exposed another persistence defect after its final action:
a valid skill invocation executed, but `TrajectoryRecord` used a separate legacy
regular expression and rejected the complete record. Native-v2 record admission
and invocation-to-execution evidence now use the same persisted-contract codec as
live dispatch. Visible-skill admission and exact action-derived credit remain
mandatory; missing or fabricated credit is not accepted. Five full-episode cases
cover native/commentary/JSON/bare/fenced skill calls through dispatch, serialization
and evidence links. The related engine/evidence subset passed 44 tests, with mypy
passing for the three changed source modules.

## Latency scope

The new `protocol13_native_carrier_latency.yaml` candidate admits 12 concurrent
requests to the existing 12-slot server, instead of limiting admission to eight.
The final profile also raises subprocess-grader slots from four to eight. During
the real baseline collection, all four HealthBench worker processes held those
slots while awaiting HTTP grading; a ready MBPP CPU grader waited at least 173
seconds. Eight slots allow four network-waiting graders and four CPU graders to
coexist, still bounded by the existing terminal and inference limits. This is not
an increase in task or judge-call budgets. A targeted concurrency test reproduces
that specific resource dependency without model requests.

Other execution settings remain unchanged, including the qualified scorer,
activation storage, within-step overlap, canonical batch reduction and frozen Z
feature cache. This profile does not reduce B28, trajectory turns or token budgets.
It is not a claimed measured speedup or a silent replacement of a live profile.

The original Step 9 spent 2,370.05 of 2,574.87 seconds in collection, followed by
a 202.22-second gradient tail; Step 10 spent 1,403.72 of 1,466.73 seconds in
collection, with a 59.14-second gradient tail. Checkpoint saving took only 2.61
and 3.87 seconds respectively. Reducing routine persistence is not the primary
remedy for this run.

Long-prefix activation offload remains a known secondary bottleneck. Pinned offload
is not bundled into this parser fix or promoted without its separate numerical and
performance qualification. A lower TTB loss alone is not a quality criterion: the
loss divides the residual by the realized horizon before squaring, and error-induced
longer trajectories change that normalization.

The first three committed batches each had 28 positive TTB residuals. Under the
unchanged paper equation, the direct forward-log-probability coefficient has that
same positive sign, including on successful trajectories; Z and the backward policy
must also learn the balance. This is not evidence of an implementation sign error,
nor proof that the initial updates improve task competence. From Step 1 to Step 10,
mean squared raw residual rose from 2.935 to 4.019 while mean horizon rose from
2.25 to 7.07 and reported TTB loss fell from 1.792 to 0.384. These are different
training batches; the fixed quality panel, not the loss trend, establishes regression.
The repair does not undo optimizer updates previously made using transport-rejected
trajectories and must not be used to silently waive the checkpoint quality pause.

## Validation status

Synthetic tests cover carrier semantics, preserved code strings, visible skill
admission, ambiguous signatures, original-token and F/B-prefix preservation,
configuration recovery, diagnostic stop/codec wiring, and pause-during-probe behavior.
Real saved-action comparisons are read-only transport diagnostics, not new benchmark
scores; neither their candidates nor private test outcomes feed the actor.

On the original saved Step 9 actions, the first development decoder recognized
140/172 instead of 41/172; on Step 10, 170/198 instead of 76/198. The latter includes 95 recovered
carriers and one old false acceptance rejected because an unclosed outer parameter
had swallowed a second submission. These counts do not regrade either batch.

The first development probe runs from its own frozen source. It exposed a further
carrier omission: commentary followed by a single JSON fence. That case and its
conflicting-carrier/string-content boundaries were subsequently covered by 42
passing parser tests. The later source is separately frozen; the first probe is
not retrospectively presented as evidence from that later source. Unstructured
prose still counts as a parse failure, not a renamed schema failure.

An actual ALFWorld reasoning request contained the task, current admissible commands,
and the public meaning of using a lamp while holding an object. The observed failure
to follow that meaning is not a demonstrated missing-context defect. Long repeated
generations still need to be distinguished from environment latency and transport
rejection; this fix does not establish that the trained policy has recovered.

The parser milestone CPU `make check` on the approved Linux host passed **4,072 tests**, with
only the two explicitly opt-in CUDA tests skipped, plus formatting, lint, mypy,
both wheel builds and the public-wheel boundary check. The actual deployed Qwen
tokenizer/template CPU cases ran, including both carrier versions. Earlier full
checks covered earlier source snapshots; they are not substituted for this later
result. Unchanged CUDA gradient/Adam qualification was not repeated. No independent
review/coding agent or manual hash verification was used. The subsequently observed
grader-slot contention required only a concurrency-limit configuration change and
seven passing workflow/profile tests, not another repetition of unchanged numerical
tests or wheel builds. The later real persistence failure justified one final
expanded check after its repair. Running probes retained their frozen settings.
That final CPU check passed **4,078 tests, two CUDA opt-ins skipped**, formatting,
lint, mypy, both wheels and the public-wheel boundary check (292.32 seconds for
pytest). Earlier integrated snapshots passed 4,055, 4,063 and 4,072 tests as the
real tokenizer, additional carrier cases and persistence fix were introduced.
The local/development subsets included the carrier/phase, actual tokenizer,
workflow/profile and record/invocation-evidence tests; successful unchanged CUDA
and numerical validation was not duplicated. No new GPU training was performed.

## Complete fixed-panel diagnostics

These are four distinct sources per domain, 28 total, **not seven 32-sample IID
evaluations** and not optimizer steps. All use saved Qwen3.5-9B policies, one owner,
the unchanged training budgets/thinking map and native evaluators. The two native-v2
development sources remain separate: the Step-10 probe predates the JSON-commentary
fence addition; the fresh initial-policy probe includes it. They are not a claimed
strict single-axis causal experiment or a chosen best-of-several score.

| Complete fixed panel | Action structure valid | First turn valid | Mean reward | Success | ALFWorld |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Step-10 policy, original v1 interface | 66/164 (40.24%) | 11/28 | 0.570030 | 14/28 | 2/4 |
| Step-10 policy, first v2 development interface | 120/129 (93.02%) | 23/28 | 0.628789 | 16/28 | 0/4 |
| Initial policy, later v2 interface, recovered | 93/99 (93.94%) | 26/28 | 0.738695 | 19/28 | 2/4 |

**HealthBench is also below the required success floor.** The Step-10 v2 panel
has mean native rubric score **0.401525 and 0/4 successes**; the recovered initial
policy panel has **0.420863 and 0/4 successes**. Neither satisfies the owner's
at-least-1/4 HealthBench admission requirement. Its frozen binary success rule is
per-case native score >= 0.60 **and** zero triggered negative rubrics. A positive
rubric mean or a higher whole-panel reward cannot substitute for that conjunction.
The previous summary emphasized ALFWorld but omitted this separate HealthBench
failure; the recovered initial panel is not an admitted replacement T0. No threshold,
rubric, verdict, candidate, or panel membership is changed to make it pass.

Subsequent work identified the missing public tool catalog in the reasoning
phase. The separately declared HealthBench-only repair reached **1/4** on those
same four sources; see [the complete repair result](healthbench_success_floor_2026-09-10.md).
It does not relabel or patch the complete B28 results in the table above.

The baseline originally finished all generation but stopped at 27 admitted records
on the skill-persistence error. A separate CPU-only recovery used its exact saved
198 actor and 58 judge responses, with network generation forbidden. It completed
28 records in 12.19 seconds, with zero new model requests, adapter operations,
optimizer updates or posterior updates. All 27 previously complete candidates,
reasoning, public observations and scalar native scores remained unchanged. MBPP
runtime/reference timing diagnostics changed during CPU execution; both original
and recovered evidence are retained rather than overwriting old results.

The original baseline generation/collection attempt took 860.88 seconds (14.35
minutes), then needed the separately reported repair/recovery gap. The Step-10 v2
collection took 2,737.75 seconds (45.63 minutes). They partially shared the existing
inference service; neither number is an end-to-end formal steps/hour measurement.
Both retained four subprocess-grader slots; the final eight-slot profile is covered
by the targeted contention test, not falsely presented as already timed by them.

The four Step-10 ALFWorld episodes all reached 25 turns. They generated 119,217
reasoning tokens, with three reasoning length stops; environment execution totaled
6.68 seconds. The slowest episode alone generated 56,668 reasoning tokens and spent
2,175.23 seconds in reasoning phases, 553.90 seconds in action phases and 1.03 seconds
in environment execution (phase timings include queueing). `thinking-off` does not
remove the architecture's explicit reasoning phase: its 8,192-token per-turn
allowance remains. These data do not support blaming ordinary persistence or a
missing environment response for that long tail.

**Not qualified for resumed 250-step training or a 72-hour completion claim.**
The interface failure rate materially improved, but HealthBench's success floor,
ALFWorld and policy-quality failures remain. No quality threshold was relaxed and
no Step 11 or replacement formal run was started. Restoring training needs an
explicit new-condition decision and short-training quality/capacity evidence;
an old loss decrease is not that evidence.

No method equation, model, adapter optimizer, reward definition, seed, or skill-evolution
rule was changed. No consultant, vote, alternative-answer selection, grammar mask,
hidden-test repair or benchmark-answer feedback was introduced.

The later Step-20 pause has a separate
[action-handoff repair and complete fixed-panel result](step20_action_handoff_repair_2026-09-10.md).
Its new interface reaches the structure floor, but HealthBench remains below
the owner's success requirement; it does not authorize a quality-warning bypass.

## Step-16 saved-batch diagnosis

The catalog-from-Step-2 run's original Step-16 totals are correct: reward
**16.2 / 28 = 0.5785714286**, success **15 / 28 = 0.5357142857**, and TTB loss
**0.7388599834**. These are trajectories sampled with the **Step-15 policy**, before
the Step-16 update. B28 contains seven source questions with four trajectories each,
not 28 independent questions; adjacent batches change questions and do not establish
a same-task policy regression. Source-level results remain private.

Read-only diagnosis covered all 222 saved owner requests, 111 submitted action
spans, 111 scored edges and 32 frozen-base rubric requests. Public questions,
dialogue, current reasoning and latest interactive observations were present in
the actual model inputs; owner responses matched persisted actions. The sampled
policy, scorer adapter versions, edge sums, shifted rewards, residuals and batch
loss were consistent. Rubric coverage, submitted-answer routing, score arithmetic
and the existing success conjunction also matched. This checks saved mathematics,
not a new GPU gradient-equivalence experiment.

The failed saved Python submissions also fail their existing public example:
no fence loss, hidden-test timeout or scorer-side code modification is needed to
explain them. Importantly, this run explicitly configured MBPP as **thinking-off**
with a **1,024-token reasoning-phase limit**; it must not be described as an
MBPP thinking-on result. That configured limit was reached, but changing the budget
would be a new condition, not a demonstrated correction to these historical scores.
Interactive failures retained nonterminal environment feedback and reached the
declared horizon; repeated generation, rather than simulator execution, dominated
the long tail.

No new reproducible pipeline defect was found in this batch. Its two invalid
actions were genuinely incomplete or used an undeclared function; the existing v3
handoff improvement does not retroactively make them valid. No candidate, reward,
threshold, training condition or checkpoint was changed, and training was not
resumed. Validation used the saved evidence and four CPU-bounded public-example
executions, with no new model requests. This documentation-only update reuses the
already successful 4,166-test full check of the unchanged implementation; repeating
the complete suite, GPU tests or hash checks would add no relevant coverage.
