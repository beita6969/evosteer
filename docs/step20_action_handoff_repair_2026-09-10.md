# Step-20 action handoff repair

## Observed pause, not a waived warning

The saved Step-20 training transaction has loss 0.462697, reward 0.705215,
and 19/28 successes. Its separate fixed development panel has 20/28 successes.
First-action structure validity was 26/28 at both Step 15 and Step 20, below
the unchanged 95% floor (which requires at least 27/28). The consecutive-warning
pause is correct. The complete Step-20 checkpoint and historical decisions
remain untouched; this repair does not authorize training continuation.

## Diagnosis from saved requests

All 56 first-action requests from these two complete panels were read back from
the durable journals and decoded with the deployed Qwen tokenizer. The failing
requests contain their actual public task, declared functions, and current owner
reasoning. This is not missing HealthBench dialogue or missing environment
context. No grader answers are introduced by the repair.

The four invalid first actions separate into:

- One complete native call following a non-JSON braced annotation. Carrier v2
  routes every leading `{` to JSON decoding before considering the native call;
  this incorrectly rejects the otherwise explicit submission.
- One skill identifier emitted as a function name, rather than as the argument
  of `read_skill`. Rejecting that undeclared function is correct.
- Two prose-only responses rather than executable calls. One describes an
  environment command; the other gives a conversational answer. These are not
  valid calls and must not be relabelled as valid to clear the pause.

The common action reminder is underspecified about this boundary. It also offers
final-answer submission on environment-only surfaces where that function does
not exist. Actual function schemas are present, but the immediate R-to-A handoff
does not identify their concrete names/parameter roles or distinguish skill IDs
from callable functions.

## Explicit new interface condition

`native-single-tool-call@3` is opt-in; v1/v2 retain their historical meanings.

- Generate the action handoff from the actual public function catalog. Only
  available functions appear. Explain that intended actions are executed through
  their parameter fields and that skill IDs are `read_skill` argument values,
  not additional functions. Skill reading remains optional.
- Use the same instruction in static and typed interactive prompts and in both
  forward and backward scoring renderers. Backward context still excludes the
  current reasoning/action. Reasoning remains a separate, non-dispatch phase.
- Recognize ordinary non-JSON braced commentary preceding one explicit call.
  Incomplete JSON objects, conflicting calls, undeclared functions, unclosed
  reasoning and prose-only replies remain failures. Never repair answer content.
- Carry the new codec through persisted skill invocation evidence. Complete
  sampled action text and token IDs remain the scored span; no sampling mask,
  forced prefix, voting, alternative-answer selection, or hidden-test retry.

Read-only interpretation of the *saved* actions changes one Step-15 action only:
26 to 27 first valid actions and 123 to 124 valid actions out of 126. Step 20
stays 26/28 and 91/94. This diagnostic does not overwrite either historical
result and cannot clear the Step-20 pause; new generation is necessary.

## Latency diagnosis

The slowest saved Step-15 trajectory is ALFWorld at 1,291 seconds; its four
episodes consume all 100 available turns and 61,715 reasoning tokens. At Step 20
the longest trajectories are two AIME-labelled tasks at about 641/646 seconds.
The HealthBench interface repair tail takes about 450 seconds and includes two
invalid actions, one ending at the output cap. These are rollout/reasoning and
interface-repair costs, not evidence that ordinary checkpoint I/O is the primary
bottleneck. No task budget or reasoning mode is reduced by this repair.

## Validation and isolated measurement

110 targeted tests passed across the new handoff, carrier parsing, complete
raw-token/scoring-prefix agreement, real Qwen tokenizer, skill persistence,
formal configuration propagation, and fixed-panel accounting. An initial
temporary-directory ownership issue prevented 40 fixture setups; those 40 alone
were rerun under a new private Linux-local temporary root and passed. The 70
already-passing cases were not rerun at that stage.

A complete read-only 28-source probe used the exact saved Step-20 forward
adapter and skill library, original source order, seed coordinates, budgets,
reasoning modes, and native scorer rules. New interface condition and execution
placement are recorded explicitly. Existing 22048 GPU 6 serves the owner; the
existing GPU 0/7 services served the frozen base grader. There were no optimizer,
posterior or library updates and no new model servers. All 28 first-action
sampling parameter sets, including their seeds, match the original Step-20
panel. Only the explicitly declared interface/input and grader placement differ.

| Complete panel | Original Step 20 | New interface, same Step-20 weights |
| --- | ---: | ---: |
| First valid actions | 26/28 (92.86%) | 27/28 (96.43%) |
| All valid actions | 91/94 (96.81%) | 94/96 (97.92%) |
| Admitted actions | 91/94 | 94/96 |
| Terminal successes | 20/28 | 18/28 |
| Mean reward | 0.754995 | 0.723240 |
| HealthBench successes | 1/4 | 0/4 |

The new collection took **731.63 seconds (12.19 minutes)**, with 96 agent turns,
192 owner model calls and 58 frozen-grader calls. New per-domain success counts
are HotpotQA 3, TriviaQA 1, AIME-labelled development sources 4, HealthBench 0,
ALFWorld 2, MBPP+ 4 and HumanEval 4. This is not a speedup measurement: the sampled
trajectories differ, and grader placement differs. Formal training throughput
remains zero while paused; its historical 4.50 steps/hour does not describe this
read-only probe or imply that its ETA has resumed.

The two remaining invalid actions are real model emissions: prose without a
call and an unfinished function carrier, both ending with model stop rather
than a repaired/accepted submission. They remain errors. The structure floor is
met, but aggregate success did not improve and the owner's HealthBench >=1/4
requirement is **not met**. No second seed, failed-source replacement, additional
answer selection or selective regrading was used to conceal this result.

Additional read-only HealthBench checks used all four saved answers: all eight
actor contexts contain the complete public dialogue, and all 58 grader calls
contain the complete dialogue and the exact owner-submitted answer. Each rubric
was graded once, all responses completed normally, the frozen base route was
used, and the weighted scores/negative-criterion counts reproduce the stored
success flags exactly. No private rubric text occurs in those actor contexts.
Thus the remaining 0/4 is not explained by omitted dialogue, candidate handoff,
score arithmetic, a truncated grader reply or accidentally grading a different
answer. It is not legitimate to fix it by lowering the success threshold or
removing negative criteria. Clinical answer quality and occasional malformed
model emissions remain unresolved, not proven to be further parser defects.

The complete CPU `make check` on 22049 passed: **4,166 tests, two explicit CUDA
skips**, ruff formatting/lint, mypy and both wheel builds/model-wheel boundary
check. Pre-test attempts exposed an incomplete legacy-source transfer and stale
lint import-classification results; completing the transfer and disabling that
cache resolved them without rewriting unrelated imports. No extra E2E or GPU
gradient comparison was run for this interface change. No independent review
agent or manual hash checks were used. The authorized Git-only agent does not
modify implementation files.

The finite probe client exited and unloaded its temporary adapter. Existing
inference/standby services are retained. Historical quality warnings and the
complete Step-20 pause are preserved. A new-input-condition probe is not a
replacement for the historical Step-20 quality record or an authorization to
bypass the pause. **Do not report the full quality repair as complete.**

This is a training-development panel, not the seven IID evaluation score table.
In particular, its historical AIME training-source provenance must not be
reported as official AIME2026 IID performance.
