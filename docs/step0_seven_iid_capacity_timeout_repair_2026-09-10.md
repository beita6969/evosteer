# Step-0: replica capacity and episode cancellation

This follows the saved Step-16 diagnosis; it does not revise that batch's
reward, success, policy, or checkpoint. This work does not resume formal training.

## Scope

The current catalog is **HotpotQA, TriviaQA, AIME2026, HealthBench, ALFWorld,
MBPP+, HumanEval**. WebShop is excluded. A full panel means six sets of 128
and all 30 available AIME2026 questions: **798**, not 926 or duplicated AIME.
The recent fixed-32 development runs do not establish full-panel acceptance.

The current target configuration now describes one untrained Qwen3.5-9B owner,
skills off and native thinking on, rather than the retired multi-agent/eight-domain
condition. HotpotQA and MBPP+ retain the later 80-point requirements. HealthBench's
45-point diagnostic floor does not replace strict improvement over the historical
53.17 local-Qwen reference; that reference is not a matched official-GPT comparison.
No scoring formula or historical result changes with this configuration correction.

## Engineering corrections

- **Context-aware replica dispatch.** The available services have different
  context capacities. Routing previously considered only in-flight counts.
  It now checks actual input tokens plus the unchanged output reservation, then
  balances among eligible replicas. Long continuations are not sent to a smaller
  service, truncated, or assigned a reduced budget. This is not evidence that
  heterogeneous routing caused an older single-service benchmark failure.
- **Cancellable evaluation HTTP.** The former executor caught cancellation but
  waited for its blocking HTTP worker. An episode deadline could therefore wait
  until the much longer request timeout. Request-scoped asynchronous HTTP closes
  the socket on cancellation; an explicit abort names only that request and has
  a three-second cleanup allowance. There is no replica-wide abort, adapter
  mutation or change to the training rollout lease/transaction implementation.
- **Honest timeout accounting.** A cancelled owner call is journaled with unknown
  usage and no automatic replay. It is not a submitted candidate, native failure
  score, or zero-cost request. ALFWorld's 20-minute limit still permits bounded
  request/actor cleanup and does not reset between actions.
- **Light imports retained.** The official aiohttp package is an explicit
  dependency, loaded only when HTTP is actually used. Model-facing public imports
  still work without network/model dependencies installed.

The old cancellation delay was reproduced with two CPU HTTP fixtures. Relevant
tests cover real socket cancellation, scoped abort failure, queued-call isolation,
transport timeout, unchanged payloads, broker provenance and context routing.
The live synthetic cancellation request ended after about 1.02 seconds and 47
generated tokens rather than exhausting its 8,192-token allowance. An initial
diagnostic falsely treated stale serving metrics as ongoing work; subsequent
native request statistics and empty service loads confirmed cleanup. That failed
diagnostic log is retained, not represented as a passing test.

## Verification and evaluation status

Development used focused CPU tests and local Ruff. The capacity milestone's
full CPU check passed 4,177 tests with two CUDA-only skips. The cancellation batch
passed 68 focused tests after correcting a test's missing-candidate expectation.
The subsequent full check found an eager dependency import; after making it lazy,
27 import-boundary/HTTP tests passed. The three existing serving replicas then
passed four synthetic cold/warm numerical cases (24 calls): identical generated
tokens, maximum selected-logprob difference 0, and at least 5,504 reused input
tokens. The numerical part took 13.44 seconds. These are transport/cache checks,
not benchmark scores or evidence that all acceptance goals have been met.

The transport snapshot's final `CUDA_VISIBLE_DEVICES="" make check` on the approved Linux CPU host
passed **4,183 tests**, with two CUDA-only skips (pytest: 319.62 seconds). Ruff,
formatting, mypy, both wheels and the public-wheel import boundary also passed.
No extra complete check is needed for this subsequent documentation-only update.

A fresh, development-exposed 798-episode evaluation was started, with one A2 arm,
native thinking on, seed 0, unchanged per-task token/call allowances and no reuse
of prior answers. It reuses the existing inference services on physical GPUs
0, 6 and 7 of the approved 22048 endpoint. Two independent benchmark coordinators
run at a time, each with 12 episodes in flight; MBPP+ and HotpotQA start first.
Native scoring concurrency is eight. The frozen sources, generated answers,
scorer targets and detailed logs remain private. No full-panel scores are yet
asserted by this implementation note.

The first deployment attempt exited during isolated-actor import: its thin
diagnostic environment borrowed PyYAML from an unmounted environment. All seven
private journals contained **zero model starts and zero candidates**. It was not
scored as zero and its records were retained. A separate runtime environment now
contains official PyYAML 6.0.3 locally; the isolated actor imports successfully
while the private evaluator remains unavailable. A fresh `r2` run uses the same
panel, seed and decoding conditions, without relaxing filesystem isolation.
This deployment-only correction does not require repeating the unchanged full
source test suite.

The first MBPP+ scoring attempt then exposed the analogous companion-environment
dependency in its separate EvalPlus sandbox. It stopped before writing any score;
all 128 generated answers survived. A CPU-only same-run scoring resume used the
scorer's intended companion environment, with the same frozen native profile.
It completed all 128 scores in 54.13 seconds. The journal still contains exactly
128 owner model starts and 128 submitted candidates: there was no regeneration,
answer replacement or test-directed repair. Both failed deployment logs remain.

### Full-panel results available at 19:34 UTC

These are newly generated, development-exposed A2 results, not the formal Step-16
policy or a matched comparison against the historical backbone reference.

| Benchmark | Count | Native primary score |
|---|---:|---:|
| HotpotQA | 128 | F1 76.7479; EM 59.3750 |
| TriviaQA | 128 | F1 75.8296; EM 69.5313 |
| MBPP+ | 128 | Base AND Plus 78.9063 (101/128); Base 91.4063 |
| HealthBench | 128 | Local-Qwen rubric mean 42.7153 |

These results **do not meet all targets**. HumanEval still has one generation
tail, ALFWorld is in progress, and AIME2026 is queued. No missing result is imputed.

Read-only diagnostics verified that all 128 HotpotQA rendered requests contain
their complete question and all ten released passages. Every HealthBench initial
conversation is present in order, with one to nine source messages per case.
Recomputing the 1,497 retained HealthBench rubric verdicts reproduces every stored
score. Its separate per-record-clipped TTB mean is 45.4976 and its project binary
success count is 28/128; neither replaces the 42.7153 native benchmark score.

MBPP+'s six execution failures are not missing scorer dependencies after recovery:
three submitted modules use a different function name from their public example,
and three contain faulty code that fails during their own top-level example tests.
The other failures are five Base and sixteen Plus test failures. Candidate syntax
is valid and no outer scorer timeout occurred. These are not grounds to rewrite
the submitted programs. HumanEval's remaining trace shows repeated reasoning and
a successful public-history read, not a parser-rejection loop. Its problematic
public example was compared directly with the upstream release and was not
introduced by local context transmission. The frozen task was not edited.

Code is written and reviewed by the main thread. No review/coding subagents or
hash checks were used. Repeated WSL tests/type checking that stalled without a
result were not counted as passes; verification moved to the approved Linux CPU
host. No new model service or formal training process was launched for this work.

### Later completion and ALFWorld stop (20:25 UTC)

HumanEval completed **124/128 = 96.875%**, with three native test failures and
one empty submission after output-budget exhaustion. Its score does not mean
every output channel succeeded. AIME2026 has 24/30 generated candidates and no
completed native aggregate yet; its observed rate is about 35.5 records/hour,
with a rough ten-minute remaining estimate, subject to the final tails.

ALFWorld ended **incomplete** under the existing fail-fast policy: 59 native
episodes completed successfully, seven exceeded the 20-minute trajectory limit,
and 62 had not started. There is no full-128 native score. In particular, 59/59
must not be represented as 100% benchmark success. The cancelled calls have
unknown usage, are not automatically replayed, and were not assigned fabricated
native outcomes. The ALFWorld coordinator has exited; AIME continues separately.

Read-only inspection confirmed that those ALFWorld episodes received the
authoritative live reset instruction rather than the stale crowd annotation.
The inspected carrier rejects were ordinary prose, not complete native calls
lost by parsing. Some owner outputs nevertheless substitute different simulator
object types, confuse appliance operations, or assert completion while the
environment remains active. This is not evidence of missing task context or a
license to execute prose, extract actions from unfinished thinking, or change
the scorer.

### Opt-in public semantic clarification

`public-task-semantics@4` extends the existing ALFWorld public interface catalog
with object-category identity and the distinct clean/heat/cool state transitions.
These meanings are specified by the public
[ALFWorld domain](https://github.com/alfworld/alfworld/blob/master/alfworld/data/alfred.pddl),
not by task answers or scoring feedback. The text supplies no episode IDs,
locations, demonstrations, desired action sequence, or alternative solver.
Versions 1–3 and all other domains' semantic text remain unchanged. Selecting v4
also preserves the existing typed ALFWorld history in training context assembly.

The change is opt-in: it has not altered the running evaluation, formal training,
saved checkpoints, or timeout accounting. No improved ALFWorld score is claimed.
Changing the current whole-panel stop to isolate a timed-out episode was offered
to the owner; it has not been enabled while awaiting that decision.

The related two test files passed **31 cases** on the approved CPU host. The
first invocation passed 28 but could not set up three cases because an inherited
temporary directory belonged to another user; only those three setup failures
were rerun in a new private temporary directory, and all passed. No shared
directory was modified. A WSL attempt that timed out with no result is not a pass.
The v4 milestone's complete CPU `make check` passed **4,186 tests**, with two
CUDA-only skips (pytest: 315.98 seconds), plus formatting, lint, mypy, both wheels
and the public-wheel boundary. The unchanged transport/cache numerical checks
were not repeated. Main-thread self-review covered this small versioned change;
no coding/review subagent or manual hash check was used.

### AIME completion and separate ALFWorld v4 rerun

AIME2026 finished at 20:58 UTC: **27/30 = 90%**, with all 30 candidates scored,
zero candidate/infrastructure failures, and complete-with-repairs communication.
There were 54 owner calls: 40 complete response chunks, 12 unfinished reasoning
chunks and two unfinished final chunks. Same-policy continuation and interface
repair produced all final submissions within the unchanged declared budget.
This new 27/30 is not the withdrawn historical top-two/composite 27/30: it uses
fresh owner answers on all 30 questions, with no LoRA substitution, voting,
candidate selector or baseline fallback. Wall time was about 76.64 minutes,
23.49 episodes/hour. The long-tail rate fell below the 24/hour planning line
and was reported for human decision; no budget was silently reduced.

The six complete static-domain native results are therefore HotpotQA F1 76.7479,
TriviaQA F1 75.8296, AIME2026 accuracy 90, HealthBench local-Qwen rubric mean
42.7153, MBPP+ Base AND Plus 78.9063 and HumanEval pass@1 96.875. HotpotQA,
HealthBench and MBPP+ still do not meet their current full-panel targets.
The preceding incomplete ALFWorld run is not filled in or combined with them.

At 21:00 UTC a separate **ALFWorld-only 128-episode** run started from the
qualified v4 source, using a fresh run ID and no reused candidate answers.
It selects the explicit v4 public semantic catalog; it retains one untrained,
adapter-free owner, skills off, thinking on, seed 0, 12 concurrent independent
episodes, the same native horizon and token/call allowances, and the 1,200-second
episode timeout with the existing whole-panel stop policy. A change to timeout
isolation remains unapproved and is not part of this run. The other six domains
are not rerun. It reuses only the existing services on approved 22048 physical
GPUs 0, 6 and 7; no model service or formal training process was created.
No v4 score is claimed at launch. This deployment/configuration-only step reuses
the already completed source validation, rather than repeating its full check.
