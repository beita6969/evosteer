# Native interaction: measured slow tail and repair

## Scope

This is the A2-only, frozen-base Qwen3.5-9B development evaluation of the
complete fixed WebShop128 and ALFWorld128 panels. It is not a new eight-IID
aggregate. AIME thinking-on is not regenerated. Only the two already allocated
evaluation GPUs on endpoint `185.212.56.211:22048` are used; physical GPU4 belongs
to the separate training session. The completed full CPU validation ran on
endpoint `185.212.56.211:22049`. Following the owner's subsequent instruction,
further CPU checks and completed-record scoring use `185.212.56.211:22048`
with CUDA hidden, bounded CPU parallelism and separate temporary/output paths;
the serving environment is not modified. An already successful check is not
repeated merely to change hosts.

## Measured bottleneck, not inferred from GPU utilization alone

Read-only observations during the @3 tail, starting at 01:23 UTC:

| Measurement | Observed |
| --- | --- |
| Tokenization of an actual 29,298-token input | median 0.0459 s, three CPU encodes |
| JSON serialization of that input | about 0.00051 s |
| Serving queues on both replicas | zero |
| Active sequences per replica | two or three, later two |
| Decode speed per sequence | approximately 26 tokens/s |
| Decode CUDA Graph | disabled |
| Prefix cache | disabled; no cached prompt tokens |
| GPU utilization | approximately 93–99% |
| GPU memory-controller utilization | approximately 18–20% |

Long inputs are not free, but measured prefill is around one second for a
32k-token input. The dominant late work is generation: some decisions repeatedly
emit about 1,250, 2,380, or as many as 8,192 tokens. At the observed decode speed,
one full 8,192-token call alone takes roughly five minutes. High GPU utilization
therefore does not demonstrate good per-sequence efficiency.

These measurements rule out tokenizer/JSON work and a serving queue as the
primary cause of this particular tail. They do not claim that queues were always
empty during the earlier, larger batches. Enabling graphs is a concrete
performance hypothesis until the equal-work serving comparison is completed.

The old cumulative-rate ETA is misleading after most short episodes finish.
Report recent completion rate and the remaining episodes' progress instead;
when no episode completes in the window, a completion-rate ETA is unavailable.
An active sequence producing long output is not the same as a dead worker.

The fixed-work synthetic comparison is now complete: 32,768 input tokens and
512 generated tokens per request, with identical sampling parameters. The old
eager replica and graph/cache replica ran on the two allocated H800s:

| Phase | Eager tokens/s | Graph/cache tokens/s | Throughput ratio |
| --- | ---: | ---: | ---: |
| Cold single request | 24.55 | 25.76 | 1.049 |
| Warm single request | 24.55 | 27.52 | 1.121 |
| Four concurrent requests, aggregate | 83.16 | 108.51 | 1.305 |

This is twelve synthetic requests, not benchmark evidence or an end-to-end
speedup guarantee. Single-request output token sequences matched; not all
four-request outputs matched, so no bitwise-equivalence claim is made. The
probe initially failed before admission on a tokenizer return-type mismatch,
then between phases on stale scheduler gauges; completed phases were retained
and only unexecuted phases resumed. HTTP completion, not asynchronously sampled
running-request gauges, now delimits these private probe phases.

## Correctness defects behind wasted calls and failed episodes

- Complete ALFWorld commands sometimes occupy the native function-name slot.
  They are now accepted only verbatim, without invented arguments, and only
  when they match the current public action surface and revision. Multiple
  submissions and action examples remain non-executable.
- The eight-call per-action cap closed whole episodes while most of the
  declared 160-call allowance remained. @4 uses the shared allowance without
  this smaller cap. Whole-episode calls, output tokens, context, and native
  action horizons do not increase. Local exhaustion is also labelled separately
  from whole-episode exhaustion.
- WebShop's flat text renderer discarded native selected-option markers. @4
  uses the released `text_rich` renderer, not framework-derived product advice.
  A CPU check of the actual released renderer confirmed that selection survives
  in rich text and disappears in flat text.
- The HTML search button was advertised as `click[search]`, although the native
  environment unconditionally rejects that click. The actual `search[query]`
  API remains available; other native buttons are unchanged.
- Horizon reporting uses acknowledged native steps without changing reward or
  success. A real terminal purchase is not reclassified merely because it
  coincides with the step limit.
- Late output inspection found repeated confusion between actions unavailable
  **now** and impossible actions. The released ALFWorld `alfred.pddl` confirms
  that taking requires empty hands and access at the current location; placing
  requires a compatible accessible receptacle; state-changing operations require
  an eligible held object and the corresponding local appliance. These public
  preconditions are now in the shared API reference, including the peer copy.
  No target object, location, hidden state, or task-solving sequence is supplied.

Recorded calls and environment acknowledgements show that these are not simply
lost environment replies. Model errors also remain: no purchase, wrong options,
wrong objects, and repetitive discussion are not automatically repaired into
successful decisions by the framework.

## Serving and scheduling changes

`configs/serving/qwen35_9b_native_evaluation.yaml` separates the evaluation
deployment from training defaults. It enables the supported full decode graph,
limits capture to the actual 48-request serving range, and enables native
prefix reuse. Prefill graphs remain disabled for compatibility. Model weights,
precision, deterministic request seeds, sampling, context length, and episode
allowances remain unchanged. Resolved graph/cache settings enter the frozen
execution controls.

Dynamic remaining-budget counters are required runtime facts near the prompt
tail, not changing text at the beginning of every request. Stable submission
rules remain in the single system message. This preserves conversation and peer
message bodies while allowing a stable prefix to be cached.

The next full round starts ALFWorld first based only on its longer declared
action horizon, rather than starting all shopping tasks first. No item is
ordered or selected using its answer, previous reward, or a difficulty label.
The same 128 cases per benchmark are generated afresh, with one seed and no
answer reuse, voting, selector, fallback, or hidden environment policy.

The owner explicitly requested an immediate switch and a fresh complete round.
The @3 coordinator was therefore stopped at 01:50 UTC with 253/256 completed
records; those records and the original frozen source are preserved unchanged.
Service changes are rolling, retaining the other replica and a temporary
reservation context while each replacement loads. Fixed-work speed probes use
synthetic text, not licensed benchmark inputs or evaluated answers. None of the
old answers is reused in @4.

At 02:00 UTC both replacement services were personally verified healthy, with
full decode graphs and native radix caching enabled. Temporary reservation
contexts had stopped. Before fresh benchmark generation, the owner requested
verification of an owner-only topology; **the new 128+128 run has not started**.
The prepared multi-agent profile must not be launched unchanged under that rule.

### Topology verification requested before launch

`idea.tex` specifies the agent's history, forward/backward policies, diagnostics
and skill evolution; it does not require two consulting agents. The active
evaluation's `solver` and `researcher` are explicitly introduced by
`agent_communication.py` and the `multi-agent` arm configuration, not by the
scientific method. The [official SkillFlow architecture](https://github.com/beita6969/SkillFlow#architecture)
does include one Supervisor and a frozen Executor. Its
[Executor implementation](https://github.com/beita6969/SkillFlow/blob/74be52bb6bd9f0e9e68dacb72636b75649197983/src/executor/m_exec.py)
is an LLM execution interface, not two named consultants. Removing the extra
consultants must not be misreported as proving that upstream has no Executor.
No topology code or historical result has yet been relabelled in this verification.

## Completed-record scoring already checked

The read-only 250-record snapshot took about 3.06 seconds and made no model
calls or environment steps:

- WebShop: all 128 complete; mean native reward **41.3039/100**, with **27/128**
  full successes. This does not meet the goal.
- ALFWorld: **101/122** successes, or **82.7869%** of completed records. This is
  an incomplete, completion-conditioned result, not a full-panel goal pass.
- All **75** recorded purchases reproduce their original native reward under
  the released scoring function. Among 48 nonperfect purchases, 39 have an
  option deficit; 32 required options but selected none. Categories overlap.

Licensed inputs, model outputs, item identifiers, targets and per-item scoring
details remain private. This repair does not alter already recorded answers or
claim that a low score by itself proves an implementation defect.

After stopping @3, a separate read-only snapshot scored all 253 completed
records in 3.09 seconds, with no model calls or environment steps. WebShop is
unchanged; ALFWorld is **101/125 = 80.8%**, with three unfinished cases. This is
still not a complete 128-case ALFWorld result and is not mixed into @4.

## Validation status

The first combined CPU check passed Ruff and mypy, then found five tests whose
message-placement assumptions needed attention. Stable submission instructions
were restored to the system message; tests now check intact peer delivery and
runtime-counter roles rather than a fixed last-message position. The affected
73 tests subsequently passed. The combined check at `4ea2660` passed Ruff,
mypy, **3260 tests** (one explicitly CUDA-dependent test skipped), both wheels,
and the model-wheel boundary check. Pytest took 169.49 seconds. The subsequent
public-precondition clarification passed its four affected tests, followed by
the final CPU check at `09809e4`: Ruff, mypy, **3261 passed, one CUDA-dependent
test skipped**, both wheels, and the model-wheel boundary check. Pytest took
167.63 seconds. The GPU equal-work results and their limitations are above.

No independent coding or review agents were used. The authorized watcher only
observes resources. No hash checks were performed; unchanged successful local
tests are not repeatedly run between patches.

## Owner-only @5: implemented and launched

The subsequent owner-approved repair is `d2c4af9`. It removes the live consultant
implementation, consultant tool advertisements and consultation-budget rules.
The later output inspection below found one residual peer reference in the
generation layer's shared-budget notice; that wording was not yet removed in @5.
Both the controller and trusted runtime reject an old multi-agent arm before
generation; the broker accepts only owner model requests. Historical arm and
message parsing remains available for old records. The active plain/native
profiles use one owner and new condition identities. Skills, native tools,
training policies, permitted native thinking and isolated grading are retained.

The fresh A2@5 WebShop128 + ALFWorld128 round started at **02:23 UTC on September
8**, with 64 workers and the two graph/cache serving replicas on physical GPUs
5 and 7. GPU4 remains assigned to the independent training session. Every case
starts afresh; no old answers, voting, selectors or baseline fallbacks are used.
Budgets and the complete official catalogue are unchanged. This is still
development-panel evidence, not an eight-IID aggregate or a completed goal pass.

The deployed source is the complete Git archive of `d2c4af9`, not the shared
working directory containing another session's unfinished skill-library edits.
The affected-scope checks passed **314 tests**. The final CUDA-hidden CPU
`make check` passed Ruff, mypy, **3289 tests with one explicit CUDA-dependent
skip**, both wheels and the model-wheel boundary check; pytest took 173.29 s.
A local pytest attempt stalled while importing dependencies and was stopped
before collection; those tests ran on the approved CPU host instead. Failed
old peer-expectation and collection checks were corrected without repeating
the already successful affected tests. Documentation-only updates do not
trigger another full validation.

## Inspection of the running owner-only round

The main thread inspected raw outputs, native acknowledgements, prompt rendering
and private scoring records on the CPU of endpoint `185.212.56.211:22048`.
The running @5 source, answers, budgets and serving replicas were not changed.

At 02:55 UTC, 39 started episodes contained at least ten consecutive identical
owner outputs at the same public state revision. The longest run was 142.
Together, 2,938 adjacent duplicate calls produced 356,171 output tokens. Typical
failures were declaring completion without an action, substituting the wrong
object, or confusing transformation with placement. These are not successful
environment outcomes merely because the model says it is done.

Six inspected failure prompts retained the task, latest observation, active
status, repair feedback and budget after the actual Qwen chat template was
tokenized. Counts matched the recorded input lengths; successive prompts were
not identical. This rules out omitted template feedback for those samples,
not every possible serving or model defect. No missing action acknowledgements,
surface mismatches or feedback mismatches appeared in the 166-record completed
snapshot. Its remaining output/control failures are reported separately rather
than called a clean communication pass.

That read-only snapshot scored ALFWorld at **103/108 successes** and WebShop at
**61.4901/100 over 58 cases**, with **17 full successes**. These are
completion-conditioned partial results, not final 128-case scores. The official
WebShop reward function reproduced all **53 purchases**. Of 36 nonperfect
purchases, 26 lost option credit, including 21 with no options selected;
16 lost attribute credit and five lost product-type credit. Categories overlap.
The scoring and component checks made no model calls or environment steps.

Two narrowly scoped source repairs follow this inspection, for a future frozen
deployment rather than an in-flight switch:

- The generation-layer budget notice no longer implies that peers exist. The
  regression checks all rendered message roles with budget counters enabled,
  rather than inspecting only the system prompt.
- One trajectory spent 157 calls emitting a complete household command inside
  a closed native envelope whose function header lacked `>`. A narrow syntax
  accommodation retains that exact argument-free command, including the
  observed empty parameter closing tag. Current-menu, state-revision and
  environment checks still apply. Competing calls, examples, missing envelope
  endings and argument-bearing malformed calls remain unexecuted. No command,
  argument, target, success or answer is invented.

The ordinary no-action repetition loops and WebShop decision errors are not
claimed fixed by these changes. Licensed item contents and individual evidence
remain in private diagnostic files only.

The offline syntax check recognized the original literal current-menu command
in all 157 recorded malformed calls. This established no new task successes:
neither the environment nor the model was called and no original result changed.

Validation used an isolated CPU check environment on endpoint 22048 and the
committed base `16022bc` plus the five owned source/test changes. **165 targeted
tests passed**. Full validation covered Ruff (**991 files**), mypy (**519
sources**), **3354 passing tests and one explicit CUDA-dependent skip**, both
wheels and the model-wheel boundary. The first full command stopped on a
missing check-environment dependency; its remaining stages resumed after the
declared dependencies were installed. The full pytest pass took 225.24 s with
3353 successes and one environment-only Unix-socket-path failure. A short
private temporary directory fixed that failure; its sole test passed in 0.35 s.
Already successful Ruff stages and the other 3353 tests were not repeated.
Serving packages, services and the live evaluation source were not modified.

### Follow-up: literal shopping commands in function headers

The 213-record snapshot exposed the same 157-call header failure in a shopping
episode. Its complete `click[...]` command was present in the current public
menu, but the earlier accommodation recognized only household commands.
The parser now transports a literal `click[...]` or `search[...]` function name
as its explicitly written tool and argument, with or without the observed
missing header delimiter. It never replaces a target or discards conflicting
parameters. Existing environment, current-menu and state-revision checks remain.
Synthetic tests cover navigation labels containing `>`, conflicting arguments,
stale state, wrong environments and competing or example calls.

The two affected test modules passed **92 tests** on the isolated, CUDA-hidden
22048 CPU environment. A read-only check recognized the unchanged current-menu
command in all **314** malformed calls across the two observed trajectories.
This made no model calls or environment steps and changed no scores. The live
@5 round still uses its original frozen source, not these subsequent repairs.

At 213 completed records, ALFWorld had **103/120 successes** and WebShop averaged
**50.4096/100 over 93 cases**, with 23 full successes. These are partial,
completion-conditioned scores. Even if all 35 remaining shopping cases scored
one, the full WebShop mean could reach only **63.9695/100**, not the target 80.
No action acknowledgement, public-surface or feedback mismatch was observed in
that snapshot; unresolved control/output failures are still reported as failures.

The follow-up's push-time `CUDA_VISIBLE_DEVICES="" make check` completed
successfully on the same isolated 22048 CPU snapshot: Ruff covered 991 files,
mypy covered 519 sources, **3373 tests passed with one explicit CUDA-dependent
skip** in 223.30 s, and both wheels plus the model-wheel boundary check passed.
This new check covers a changed action-parser state; earlier successful tests
were not rerun merely for the documentation update. The main thread reviewed
the change; no coding/review subagent or hash verification was used.

## Repair @6: repeated drafts and reset sampling streams

The @5 tail was still at 254/256 at 04:55 UTC, with no new completion in
the preceding 396 seconds. Its completed WebShop panel scored **37.4591/100**
over all 128 cases, with 23 full successes. ALFWorld was **103/126** with two
unfinished cases. The latter is not a final 128-case score. The rolling
completion ETA had underestimated long responses that consumed the unchanged
8,192-token call limit. The low-rate tail was escalated to the owner; no
unfinished result was silently scored zero or replaced.

Inspection identified two additional runtime problems:

- Rejected, unexecuted drafts and their feedback accumulated in active history.
  One actual failure input had 29,759 tokens and about 91% of its text characters
  were earlier model discussion. FIFO packing could then remove earlier real
  observations before these repeated drafts. @6 retains only the latest
  unresolved draft/feedback pair in active input; earlier pairs stay verbatim
  in the existing paged public archive. Accepted actions and native results
  are retained. A factual notice exposes the archive's availability. Repair
  feedback is addressed to the owner as a user-role runtime notice, not a
  fabricated return from a tool that never executed.
- Every stochastic request explicitly reset SGLang's stateless sampling seed
  to the experiment seed. @6 advances one deterministic, episode-local call
  stream on every admitted call, independently of its text, outcome or score.
  There is still one experiment seed, no selected retry and no answer choice.
  The trusted broker derives the same counter-based seed and checks it against
  the request. Greedy generation is unchanged. The policy is recorded in
  frozen controls and request diagnostics.

A synthetic stress test used the actual Qwen tokenizer and identical 130-entry
archives. Active input fell from **32,275 to 2,320 tokens**, template encodes
from nine to one, and CPU packing from **0.556 to 0.0044 seconds**. The earlier
environment fact remained visible with @6; the full archive was unchanged.
These are synthetic context-packing measurements, not whole-run speedups.

Eight short, non-benchmark SGLang requests on the existing GPU5 replica checked
the sampling behavior. Four identical requests resetting seed zero produced
one distinct output; four requests advancing the single stream produced three.
All eight calls completed in about 2.49 seconds of serving-call time. No
licensed prompt, answer, score, additional GPU or persistent service was used.
This establishes the sampling behavior, not improved benchmark accuracy.

Validation on isolated, CUDA-hidden endpoint 22048 CPU covered **245 targeted
tests**, then the full check: Ruff on 993 files, mypy on 520 sources, and
3379 passing tests plus one explicit CUDA-dependent skip. Two old assertions
expected repair feedback to have tool rather than user role; after updating
those assertions, only those two tests were rerun and passed. Both wheels and
the model-wheel boundary check then passed. Thus 3381 tests passed across the
full pass and its targeted completion; the original full command itself had
stopped on the two outdated assertions. Successful stages were not repeated.
The main thread implemented and reviewed this batch without coding/review
subagents or hash checks. The test source was isolated from another session's
concurrent training changes. @5 remains frozen; @6 is a separate full 128+128
development round with unchanged model, one owner, skills-off state, action,
call, token and context allowances. Score and end-to-end speed improvement
remain to be measured on that complete round.

### Subsequent owner scope change

The owner then withdrew further WebShop work and requested an existing-results
summary for the other seven benchmarks. The newly started @6 mixed round was
stopped at 04:59:56 UTC, preserving 44 ALFWorld candidates, no WebShop candidate
and no scores. It is incomplete, contributes no headline score and cannot
establish end-to-end speedup. The older @5 ALFWorld tail and both serving
replicas remain; no answer is copied between rounds. See the
[seven-IID summary](STEP0_SEVEN_IID_CURRENT_RESULTS_2026-09-08.md) for the current
scope and the older configurations' explicit limitations.
