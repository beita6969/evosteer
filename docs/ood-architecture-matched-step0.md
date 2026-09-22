# Architecture-matched OOD Step-0

OOD stays outside training. The evaluator constructs only the single Qwen3.5-9B
owner, read-only policy/library bindings, native environment bridge and isolated
scorers. It does not construct a trainer or write a training evidence store.
Expanded controls and the summary separately report evaluation-time
`training_updates=0`, `posterior_updates=0`, `skill_evolution=0` and
`training_evidence_writes=0`. These are not the selected checkpoint's historical
optimizer step or library's historical evolution counts.

## Freeze once; replace snapshots, not the evaluator

`scripts/prepare_architecture_matched_ood.py freeze` takes an explicitly prepared
OOD runtime config. It copies the already selected source files, official code
checkers and judge template into a **private** architecture directory. It does
not sample tasks, inspect scores, discover checkpoints, or generate answers.

```bash
python scripts/prepare_architecture_matched_ood.py freeze \
  --base "$PRIVATE/base-runtime.json" \
  --destination "$PRIVATE/architecture-X" --architecture-id ood-X
python scripts/prepare_architecture_matched_ood.py select \
  --architecture "$PRIVATE/architecture-X/architecture-private.json" \
  --arm-id step0-policy-only --output "$PRIVATE/step0-selection.json"
python scripts/run_step0_integrity_paired.py \
  --runtime-factory skillev_private.evaluation.architecture_matched:create_runtime \
  --runtime-config "$PRIVATE/step0-selection.json" \
  --private-output "$PRIVATE/step0-run" --run-id ood-X-step0 \
  --concurrency 32 --scoring-concurrency 4
```

For a later checkpoint, `select --snapshot` reads an explicit private JSON with
`policy_id`, `optimizer_steps`, and `policy` (the existing completed-checkpoint
binding: `checkpoint_directory`, `adapter_name`). Pass the Step-0 run's
`architecture-controls-private.json` as `--reference-controls`. The adapter must
already be loaded on the explicitly configured service; the evaluator never
updates the service, restores training state, or falls back to the base model.

The factory compares **actual expanded controls before generation**, including
ordered public inputs, tokenizer, base model, decoding, budgets, tool/action
surface, submission/parser, native environment and scorer targets. Only selected
policy/library state may differ. Run-local judge cache locations and the list of
registered adapter paths are storage/snapshot details, not exceptions for judge
or decoding changes. Exact named adapter routing is independently checked by the
existing policy binding. A changed serving backend, LoRA capability, public
semantics, scorer, timeout or input fails the architecture comparison.

Retain the frozen source code/environment alongside the private architecture;
do not run a future checkpoint on whatever source revision happens to be latest.
Within-run refresh still checks that actual controls did not change.

## Weight and library interpretation

The policy-only baseline has skills off. Its matched trained control also has
skills off. Switching skills on is a **different architecture condition**.
A library-aware architecture starts with an explicitly selected initial library,
then permits `library_id` / `library` snapshot substitution while preserving the
same discovery/retrieval interface. Use the four declared weight/library cells
to separate effects; a combined weight-and-library change is never called a
weight-only gain. No consultant, second solver, voting or answer merger is added.

## This development rerun

The requested seven domains are MuSiQue-Ans, NQ-Open **closed-book**, full
Omni-MATH, **LiveMedBench**, ScienceWorld, LiveCodeBench and APPS Introductory.
LiveMedBench follows the latest dialogue instruction; a conflicting catalog
entry elsewhere is not silently used to switch medical datasets.

This round uses the same repeatedly inspected 448 source IDs, not an unseen final
holdout: single owner, shared task semantics v5, native thinking off, skills off,
seed 0, batch 32, Qwen XML tool wire, no native continuation chunks or finalization
reserve. Code first-response and episode limits are both 12,000 output tokens;
other episode totals are 8,000. ScienceWorld uses `public-state@1`, 200 native
actions, and the repaired **raw terminal score** as its primary metric. Its
zero-clipped learning projection and success thresholds are separate metrics.
Omni/LiveMedBench use their retained official-style protocols with the terminal
GPT-5.6 Luna medium judge, fresh run-local judging and complete denominators.

The GPU7 service is made LoRA-capable **before** this baseline; Step-0 still sends
no adapter route. This freezes a serving condition that can later accept the
trained forward adapter without changing the surrounding evaluator.

No score threshold is used to select a checkpoint, sample, candidate, or retry.

### A16 results and incomplete ScienceWorld generation

Fresh A16 generation produced all 384 candidates in the six noninteractive
domains. Scoring those frozen candidates completed without regeneration:

| Benchmark (64 each) | Primary score |
| --- | ---: |
| MuSiQue-Ans, supplied candidate paragraphs | 65.76 F1 |
| NQ-Open, closed-book | 18.75 EM (12/64) |
| Omni-MATH, Luna medium equivalence | 50.00 (32/64) |
| LiveMedBench, Luna medium rubric | 41.70/100 |
| LiveCodeBench | 57.81 pass@1 (37/64) |
| APPS Introductory | 76.56 pass@1 (49/64) |

A16 ScienceWorld is **incomplete: 52/64 scored, 12 without a sealed owner
submission**. Do not present its observed-only mean as a 64-task score. All 448
attempts, 436 sealed candidates and 12 cancelled model transports remain private
and retained. Those cancelled requests have unknown serving usage, not zero cost.
The scoring-only recovery evaluated existing candidates and did not generate any.

The traced defect was concrete: an owner saying it was done could only be asked
for another environment command, not end the episode. A partial native state did
not become terminal merely because the owner believed it had finished. Some
episodes cycled through interface repairs and commands until the 600-second outer
timeout cancelled an in-flight model request, interrupting both usage accounting
and normal finalization. Native negative scores remain genuine environment
outcomes; this diagnosis does not reclassify all low scores as communication bugs.

### A17: authorized ScienceWorld-only interface rerun

The opt-in `scienceworld_termination_profile` is
`owner-finish-cooperative-deadline@1`. The legacy condition remains available for
historical interpretation and is not silently changed. The new condition:

- Advertises zero-argument `finish()` alongside `act(command)`, also accepting
  the entire literal reply `Finish`. It does not infer stopping from prose,
  choose between conflicting commands, or claim success from an owner statement.
- Stops on that owner's explicit submission without an additional simulator
  action. The scorer uses the actual current native score, including partial or
  negative scores, not a synthetic completion reward.
- Preserves **8,000 total output tokens, 200 native actions, batch 32, seed 0,
  thinking off, public-state@1 and the 600-second admission deadline**. No new
  serving request or native action is admitted after that deadline.
- Lets an already admitted request finish within its existing 480-second request
  timeout; the hard actor watchdog includes a separate 510-second drain/cleanup
  allowance. Returned tokens are charged even when the next action is discarded
  at the deadline. This is not additional action or solving time.
- Records a broker-declined request as not admitted, rather than charging a
  nonexistent serving call or labelling its usage unknown. Genuine failed
  transports remain unknown and are not replayed.
- Captures the current native outcome before closing even on exceptional exits,
  bounded to ten seconds. Missing outcome remains incomplete; no candidate or
  zero score is fabricated. Capturing state does not execute another action.

All original 64 ScienceWorld IDs are rerun with fresh owner generations, not just
the timed-out cases. The six A16 domains above are retained, not regenerated.
Consequently this is a **six-domain A16 plus ScienceWorld A17 development report**,
not a claim that seven domains finished a single fresh A17 run. Future checkpoint
comparisons must use the corresponding frozen per-domain architecture and source
snapshot, including this stopping/drain condition for A17 ScienceWorld.

Validation for this fix: 92 affected tests plus one additional broker-admission
integration test passed on the approved Linux CPU host. These include real
isolated-actor execution with synthetic model/environment responses, owner finish
at partial score, repair without semantic guessing, deadline crossing with exact
usage, no post-deadline action, legacy behavior and missing native outcomes.
Remote CPU `CUDA_VISIBLE_DEVICES="" make check` passed: **4,570 tests, 12 skipped**
(319.48 seconds for pytest), formatting, Ruff, mypy, both wheels and the model
wheel boundary check. The 12 skips are the existing private-tokenizer/CUDA-only
qualifications; no new GPU validation or judge rerun was hidden in these checks.
No independent
review agent or hash checks are used; frozen-output official-checker tests are
not repeated because the scorer did not change.

### A17 completed results: transport repaired, performance target not reached

ScienceWorld completed **64 fresh generations and 64 native scores**, no resumes
or candidate replacements. Wall time was about **8 minutes** (generation rate
485 records/hour). Its results must not be substituted into the incomplete A16
run identity:

| ScienceWorld A17 metric | Result |
| --- | ---: |
| **Raw native final score, primary** | **3.171875/100** |
| Zero-clipped learning projection, auxiliary only | 50.046875/100 |
| Native score 100 | 29/64, 45.3125% |
| Native score at least 70 | 32/64, 50.00% |
| Negative native terminal scores | 30/64, all -100 |

This is **not** a successful performance-target reproduction. The auxiliary
50.05 must not be presented as the primary score. The 30 genuine negative
terminals account for the entire 46.875-point clipping difference. Transition
records show 24 followed `focus` commands; the other six followed `move`, a
numeric disambiguation response, or `look`. This categorizes the last command,
not proof of the underlying decision error, and supplies no corrective actions
to the evaluated owner.

There were **zero failed model transports, zero missing environment
acknowledgements and zero missing native scores**. Two trajectories exhausted
8,000 output tokens during their next reply, without completing another action;
their previous acknowledged actions and actual environment scores were retained.
Accordingly the communication summary still reports two unresolved output
decisions, not a blanket communication pass. These are not unrecorded serving
usage or dropped episodes. No episode hit the wall deadline in this new run;
cross-deadline drain is covered by the synthetic integration tests, not claimed
as an exercised live recovery in A17.

The two explicit owner finishes stopped at **75 and 82 native points**, with no
subsequent model call or extra native action. Neither was counted as full success.
All other stops were 60 simulator terminals and two output-budget stops.

Cost: **1,322 owner calls**, **11,663,407 input tokens**, **147,821 output tokens**,
1,318 acknowledged native actions and two finish operations. Every trajectory
stayed within 8,000 output tokens / 400 total model calls / 200 native actions;
observed maxima were 8,000 / 63 / 62. All training/posterior/skill-evolution and
training-evidence counters remain zero. No external judge is used for ScienceWorld.

Source change: `56427b6`. The run reused the existing GPU7 SGLang service, did not
create another service, and exited normally. Before launching, an unrelated GPU7
process had already departed; it was never stopped or modified by this work.
The inference service remains resident as requested. Raw records, frozen source,
source population, controls and costs are retained in private local archives,
not Git. Only this result note changed after full validation, so no duplicate
full check was run for the documentation update.
