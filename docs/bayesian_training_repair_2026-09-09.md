# Training repair: implementation progress

The old run remains paused at its complete Step17 checkpoint. No new 250-step
run has started. This is a repair progress note, not method acceptance.

The sections below are chronological evidence. Earlier pending statements and
test counts describe their then-current snapshot; later sections supersede them.

## Initial implementation snapshot (historical)

- Added committed-event metrics with explicit trajectory/action denominators,
  raw residual squared versus TTB loss, native metric separation and exact event
  deduplication. Reconciled all 17 historical commits: 476 trajectories and
  3,621 actions. Historical rewards and posteriors are unchanged.
- Added effective scientific/execution condition projections, an isolated
  collect-only training API, direct TTB coefficient diagnostics, fixed-panel
  quality rules and cooperative checkpoint-pause integration. The collect-only
  API has component coverage; real E0/T0 pairing is still outstanding.
- Added an aggregate-only local-login W&B uploader. It retains uncertain uploads
  instead of resending them, and rejects source changes. No new training run is
  currently supplying live data. Admission/execution counts remain unknown when
  the historical source lacks explicit evidence; structural validity is not
  substituted for them.
- I0 used the existing inference service for seven paired fixed action inputs:
  base versus saved initial F produced identical output token IDs and stops in
  all seven pairs. The 152 initial LoRA-B tensors contain no nonzero elements.
  Pair execution took 45.29 seconds. These historical-reasoning-conditioned action
  inputs do not replace native E0 or on-policy T0. The temporary adapter was removed.

An input-condition decision is outstanding: the existing TriviaQA training
source builder explicitly requires empty context (closed-book), whereas the
clean IID exporter requires released reading-comprehension context. This is a
declared-input mismatch, not evidence of transport corruption. Do not invent
context, borrow final-evaluation material, or call the scores same-condition.

Phase changes, real paired bridge experiments, full update
attribution, quality-panel runs covering Steps12–17, and new-condition recovery
and evolution qualification are **not yet completed**. No grammar, SFT, KL,
learning-rate, sampling or horizon change has been used to conceal these gaps.

Verification: 42 affected tests passed after fixing isolated test staging and
updating the formal-entry fixture for the new metadata. The final CPU-only
`make check` on 22049 passed format, lint, mypy, **3,760 tests** (two explicit
CUDA opt-ins skipped), both wheels and the public-wheel boundary check. Unchanged
Step16-to-17 GPU gradient/Adam replay was not repeated. No independent review
agent or new hash checks were used.

## Action interface and execution metrics follow-up

- H0 action examples and surface guidance now come from `ActionContract`, which
  freezes its public surface rather than retaining mutable nested dictionaries.
  Tool admission, completion carrier validation and H0-only skill admission use
  this shared contract. Existing wire wrappers, feedback, native validators and
  raw full-vocabulary sampling are preserved; this is not constrained decoding.
- Execution emits explicit admission, environment-call and terminal-submission
  assessment fields. These are event-side diagnostics, not model-visible history
  or reward. A normal environment return is never labeled task progress/success.
- Committed metrics join assessments to the original action IDs and observation;
  abandoned-attempt edges do not enter the count. Partial coverage is reported,
  and unavailable admission/execution totals remain null rather than using a
  smaller denominator. W&B receives only the resulting batch scalar counts.
- Targeted CPU verification: 75 tests passed, including native completion errors,
  hidden skills, tool failures, immutable public examples, complete/partial metric
  coverage and export restart. Mypy passed for six changed source modules. Local
  WSL mypy timed out; verification used the isolated 22049 CPU checkout instead.

This does not establish that weak-format model outputs have improved, nor does
it implement new mechanical feedback or the optional grammar strategy. Those
behavior changes still require the declared zero-update comparison and new run.

Follow-up integration check: CPU-only `make check` on 22049 passed format,
lint, type checks, **3,775 tests** (two explicitly opt-in CUDA tests skipped),
both wheels and the public-wheel check; pytest took 342.21 seconds. No new GPU
qualification or training was launched for this behavior-preserving extraction.
Unchanged numerical kernels were not retested; no hash checks or review agents
were added. The old complete Step17 checkpoint and existing inference/standby
services were personally rechecked at 19:16 UTC.

## Update-attribution tools

The private `ttb_update_attribution` diagnostic now observes parameter-gradient
hooks on the production single-edge scorer, leaves autograd gradients unchanged,
and runs one full Adam update on a newly loaded diagnostic replica. It reports
F/B/Z norms, actual update norms, relative parameter changes, K/T/B coefficients
and four structure/success categories. Z remains trajectory-level rather than
being assigned to a fabricated action category. Canonical and category-regrouped
reconstructions must meet tolerances supplied before the run. Optional fixed
synthetic actions provide before/after NLL, not generated legality or accuracy.

A separate `phase_cross_diagnostic` issues six requests per fixed public history:
R0 and Rk are each sampled natively, then A0 and Ak are sampled for both histories.
All four actions and their original token identities are retained. There is no
execution, evaluator, answer selection or training-batch conversion. Thinking is
confined to the declared reasoning phase. A request failure stops the diagnostic
without resampling and releases its episode lease.

Targeted CPU tests cover independent source-state preservation, existing Adam
moments, mixed valid/invalid steps with different K/T, full-batch rejection and
six-request phase provenance/cleanup. These tools are not themselves evidence
that the real model has regained protocol competence.

Attribution milestone verification: targeted tests passed after correcting the
synthetic multi-turn budget fixture; private-module mypy passed for both tools.
CPU-only `make check` on 22049 passed **3,780 tests** (two CUDA opt-ins skipped),
format, lint, mypy, both wheels and the public-wheel check; pytest took 341.57s.
No production scoring implementation or optimizer update was altered. Real
full-B28 attribution and real generated-protocol qualification remain separate
from these CPU checks.

## Fixed-panel collection integration

The isolated production collector can now take a declared `FixedQualityPanel`
and emit a quality probe only after the entire collection and budget settlement.
It checks the ordered task slots against native evaluator source provenance,
counts distinct sources rather than rollout repetitions, and excludes declared
training/final-evaluation populations before issuing requests. Panel membership
and scientific condition remain fixed; no diagnostic optimizer or posterior is
constructed. A partial or mismatched collection cannot become quality evidence.

The probe records structure and first-turn validity, horizon, native per-domain
scores and explicit execution assessment coverage. It reuses the existing
exact-token/observation assessment join rather than fabricating commit events.
Missing execution observations stay null; zero tool calls do not become 100%
execution success. Full diagnostic results and per-domain probe values remain
private, separate from the aggregate training W&B stream.

Targeted verification passed 16 tests and mypy for three affected modules. The
first test invocation lacked the isolated pytest temporary root; the resulting
setup errors were fixed in the invocation, without changing shared directories.
The runner does not choose production panel questions or quality thresholds:
those remain declared inputs, and real-model panel qualification is outstanding.

### Real phase-cross result (diagnostic, not benchmark)

Seven fixed first-turn public histories, selected by canonical order from the
saved complete batch, were evaluated with saved initial F and saved F16. Each
reasoning was freshly generated, and all four R/A combinations were retained.
Raw sampling and the declared thinking split/budgets were unchanged. Across the
seven domains, structure-valid counts were:

| Reasoning | Initial action policy | F16 action policy |
|---|---:|---:|
| Initial | 5/7 | 0/7 |
| F16 | 4/7 | 0/7 |

All fourteen F16 action responses were parse errors. Thus initial reasoning did
not restore the trained action policy's formatability on this small controlled
panel. This is not proof of one exclusive root cause, and the initial policy
itself was not fully format-reliable. No actions were executed, task rewards
computed, optimizer/posterior updated, or production cursor advanced. This is
not native E0/T0 or an independent task-quality evaluation.

The existing inference service handled 42 requests in 733.46 seconds (about
3.44 requests/minute). One earlier client startup failed before generation due
to an unnecessary benchmark-dependency import; removing that import allowed the
diagnostic to run without changing model execution. Temporary adapters were
unloaded and the client exited. At 19:48 UTC the existing inference/standby
services and complete paused Step17 checkpoint were personally confirmed intact.
Private token outputs remain outside Git and W&B.

Panel integration milestone: CPU-only `make check` on 22049 passed format,
lint, type checks, **3,783 tests** (two explicit CUDA opt-ins skipped), both
wheels and the public-wheel check. Pytest took 336.30 seconds. The earlier
attribution milestone and this panel milestone tested different source states;
unchanged GPU gradient replay was not repeated. No independent review agent or
new hash checks were used. New formal training remains paused pending the
zero-update bridge, chosen protocol condition and real quality qualification.

## Native-source bridge and real T0 (20:10 UTC)

The real zero-update production-path T0 completed all 28 trajectories from the
first seven original training sources (four rollouts each, not 28 independent
questions). It reproduced the historical Step1 aggregate: 7 successes, reward
mean 0.3371062271; 153 actions comprised 97 structure-valid, 52 parse errors and
4 schema errors. First-turn validity was 17/28. Collection took 678.998 seconds
(11.32 minutes). There were no optimizer or posterior updates. Its temporary
adapter was unloaded. These are diagnostic results, not a new formal W&B step.
All action assessments were available: 91 admitted actions, 71 executed commands
and 20 accepted submissions; command return success does not mean task success.

The native-source bridge preserves the original public question/context/messages
and trusted scorer targets. Training-source TriviaQA is explicitly labelled as
such, not misrepresented as the released RC IID condition. A separate native
reference-budget attempt exposed two real deployment issues before completion:
AIME needs the reference context capacity, and the current ALFWorld overlay must
explicitly override the legacy per-turn continuation allowance. A separate
98304-context service was started on the approved 22048 GPU6; GPU0/GPU7 services
were not changed. An initial service startup failed for missing CUDA_HOME in the
copied environment; restoring the existing compiler environment resolved it.

The reference attempt then exposed a trusted EvalPlus wrapper that existed on
the host but not inside its sandbox. The fix discovers the actual interpreter
and wrapper-declared Python dependency paths, clears inherited coordinator
PYTHONPATH, and mounts only those dependencies. The actor gets no scorer data or
broad private parent mount. After fixing this, the exact saved MBPP candidate
received a real Base/Plus pass in an isolated EvalPlus process, with no new model
requests. The original infrastructure-failure record is retained separately;
it was never a negative reward. New slots stopped at the failure while already
issued requests drained. This incomplete native panel is not a completed E0
comparison or permission to begin formal training.

Verification: the public-source milestone completed CPU-only `make check` on
22049 (3,791 passed, two CUDA opt-ins skipped, 341.82 seconds; lint, types, both
wheels and public-wheel check passed). Subsequent targeted checks passed seven
native-source bridge cases, 23 budget/continuation cases, 44 real sandbox/broker
cases and four private quality-binding cases. The latest five main-agent changed
modules passed lint and mypy. These later changes and the parallel SkillFlow
implementation still require the next combined milestone check; the earlier
full result must not be advertised as validation of the entire latest tree.

## Actual quality collection wiring

Formal training now accepts a paired `--quality-policy` / `--quality-panel`.
The private panel binding contains original source records, final-evaluation
exclusions and fixed sampling coordinates; training sources are also excluded.
Distinct source questions determine sample size. Exact records/targets and
membership are frozen privately and checked on resume, never published.

At committed boundaries the asynchronous quality coordinator invokes the real
read-only collector using the current forward policy, current skill library,
production wire/budgets and native evaluators. It uses independent task/budget
state and request storage, shared actor/grader limits within the probe, and never
updates production posterior/optimizer. Step0 and declared cadence probes are
actually collected rather than requiring somebody to manufacture metric files.
A completed probe is recovered without generation; an interrupted unresolved
attempt pauses rather than silently sampling a replacement. The last planned
step also records quality status separately from training completion. Source
selection, thresholds and real candidate qualification remain pre-run inputs;
no passing panel or new formal run is claimed here.

A user-requested GPT-6 Astra/high implementation agent is integrating native
Qwen tool carriers, phase-specific contexts, catalog/read and versioned numerical
candidates from the separate SkillFlow repair plan. It is not a task Agent and
has not started GPU work. Formal launch waits for coordinated integration and
real candidate qualification; the complete old Step17 remains preserved.

### E0 recovered to full coverage (20:41 UTC)

The native-source diagnostic completed all 28 fixed slots. Recovery reused all
nine previously generated candidates and the eight completed native scores;
only the remaining slots generated new outputs. The missing MBPP score was
recovered from its original candidate. No training update or posterior event
was created, and nothing was uploaded as a formal W&B step.

| Domain (one source question, four rollouts) | Native E0 result |
|---|---:|
| HotpotQA | answer F1 1.0; EM 4/4 |
| TriviaQA, original training public input | answer F1 0.0; EM 0/4 |
| AIME2026 | integer accuracy 4/4 |
| HealthBench | mean local rubric score 0.2820513 |
| ALFWorld | native success 4/4 |
| MBPP+ | native Base+Plus pass 4/4 |
| HumanEval | native pass 4/4 |

These heterogeneous native metrics are not averaged into an overall accuracy.
E0 differs from T0 in native owner/wire, thinking, sampling, skill exposure and
reference budgets (including ALFWorld 100 environment steps and AIME's extended
native response budget). The same source inputs and base weights do not make
this a single-axis protocol ablation. The next native training-wire candidate
must retain the separately declared training 8/20-turn condition. This small
panel is not a held-out IID score or evidence of learned improvement.

The two executing diagnostic segments took 485.019 and 602.478 seconds
(18.12 minutes combined), excluding the repair/restart gap and earlier separate
failed conditions. The final tail consisted of AIME decoding, not an environment
hang: serving showed two live requests, zero queued requests and about 62 total
tokens/second before the final completions. Full wall-clock and failed attempts
remain in private logs; no formal steps/hour claim is made.

### Combined repair milestone and native training-wire probe

The first integrated check found three concrete integration failures (an overly
broad probe exception boundary, report schema registration and a public-import
boundary). They were repaired with targeted regression checks before proceeding.
The second CPU-only `make check` passed: **3,833 tests passed, three opt-in tests
skipped**, full formatting/lint and mypy passed, both wheels built and the public
wheel check passed. Pytest took 344.35 seconds. The skips do not constitute CUDA
or deployed-tokenizer qualification; the real tokenizer was checked separately.
No hash checks or independent review agents were used.

After this milestone a new collect-only native training-wire B28 probe started
on the existing inference service (22048, physical GPU6). It uses the frozen
initial forward policy, original first seven sources/four rollouts, mixed
thinking, 8/20 turns and the original per-phase budgets. The explicit candidate
changes are phase-specific context and native single-tool-call carrier; skills
remain full-inline and sampling remains raw categorical. It does not update
optimizer or posterior, is not a formal W&B run, and has not yet passed real
collection or gradient qualification. The old Step17 remains preserved.

### Native T0 result and task-quality hold

The collect-only native candidate subsequently completed **28/28** in
570.07 seconds: **8/28 binary successes**, mean native reward **0.42125**,
107 valid actions, two schema-invalid actions and one parse error (110 total).
First-turn validity was 26/28. Relative to the earlier legacy T0, action validity
improved from 63.4% to 97.3%; this is not evidence of learned task improvement.

| Domain | Native T0 result (one source, four rollouts) |
| --- | --- |
| HotpotQA | Mean answer F1 0.50; exact match 0/4 |
| TriviaQA | Mean answer F1 0.25; exact match 0/4 |
| Historical AIME training source | Accuracy 0/4; not an AIME2026 IID measurement |
| HealthBench | Mean rubric reward 0.44872; frozen binary success 1/4 |
| ALFWorld | Success 0/4; all reached 20 turns |
| MBPP+ | Base-plus pass 3/4 |
| HumanEval | Pass 4/4 |

The three invalid actions were an unavailable skill identifier, a skill name
used as a function name, and a natural-language action response reaching the
2,048-token budget. They remain in the batch and scoring evidence. No transport
corruption or infrastructure retry is inferred from these model outputs.

Native context assembly was found to omit domain-specific public answer
instructions while replacing the old JSON-wire guidance. An explicit versioned
semantic/wire separation is being implemented. Formal250 remains on hold for
task-quality repair and matched comparisons; neither parser validity nor a
numerical replay pass alone qualifies the new protocol. Historical 32-question
F1/rubric scores are not interchangeable with this seven-source binary count.
The earlier reference ALFWorld candidates used 12, 16, 18 and 21 tool calls;
therefore the native candidate's failures cannot all be attributed to a 20-turn
limit. Other reference conditions differ and still require separate comparison.

A complete saved-B28 sealed/provisional numerical diagnostic is running on
22048 physical GPU7, without generation, production optimizer or posterior
updates. Its first attempt failed on a missing training-dependency path before
gradient work; the qualified existing dependency path was restored for the
second attempt. This is a single-owner numerical check, not a distributed or
formal-training result. Original failed-attempt evidence is retained privately.

The subsequent CPU-only check including wire-aware update-attribution categories
passed: 3,833 tests, three opt-in skips, full lint/mypy and both wheel builds
(343.95 seconds for pytest). This check precedes the public-action-semantics
repair and must not be reported as validation of that later change.

### Public-action-semantics repair: matched T0 completed

Explicit formal `@5` / rollout `@8` restores the domain semantic instructions in
native R/A contexts, while keeping legacy context interpretation unchanged.
Targeted checks passed (75 integrated tests including the deployed tokenizer,
YAML roundtrip, 25 final action-contract tests, scoped Ruff and mypy).

The new frozen, collect-only B28 completed in **667.76 seconds** with **11/28
successes (39.29%)**, mean reward **0.49679**, and **100/105 structurally valid
actions** (five schema-invalid, no parse errors). This improves the preceding
native candidate's 8/28 but does not meet the requested task-quality recovery.
Budgets, thinking, raw sampling, source slots, initial weights and full-inline
skills were unchanged; no training or posterior update was applied.

| Domain | Native @4 | Restored semantics @5 |
| --- | --- | --- |
| HotpotQA | F1 0.50, EM 0/4 | F1 0.50, EM 0/4 |
| TriviaQA (closed-book training source) | F1 0.25, EM 0/4 | F1 0.125, EM 0/4 |
| Historical AIME training source | 0/4 | 1/4 |
| HealthBench | Reward 0.44872, success 1/4 | Reward 0.35256, success 1/4 |
| ALFWorld | 0/4 | 1/4 |
| MBPP+ | 3/4 | 4/4 |
| HumanEval | 4/4 | 4/4 |

These are not independent 32-question IID estimates. Remaining training/evaluation
public-guidance differences are being consolidated behind a separate opt-in
condition; no question-specific instruction or scoring relaxation is introduced.
Formal250 remains paused.

The preceding native @4 saved-B28 numerical replay also completed in 861.65
seconds: all 110 edges, F/B/Z full gradients and independent Adam parameters,
updates and moments matched in the tested single-owner sealed/provisional paths
(observed differences zero). No production learning occurred. Its owner exited
and restored GPU7 SGLang, whose health endpoint was checked; the @5 T0 client
also exited and unloaded its temporary adapter. GPU0's existing standby and GPU6's
inference service remain resident.

The @5 integrated CPU check exposed a launcher temporary-directory ownership
error and an obsolete test that replaced only the derived instruction field.
The launcher now uses its own temporary directory; the test updates the explicit
instruction source and projection together (15 scoped tests passed). The repaired
full check passed: 3,851 tests, four opt-in skips, lint/mypy and both wheel builds
(pytest 298.20 seconds), without modifying either GPU snapshot. Deployed-tokenizer
tests were exercised in the separate targeted run, not these four skipped cases.

### Shared public task guidance: final matched T0 result

The separately frozen formal `@6` / rollout `@9` candidate uses
`public-task-semantics@1` in both training and explicitly selected evaluation
arms. It shares public domain instructions, not answers or task-specific rules;
closed-book Trivia remains closed-book. Original budgets, skills, thinking and
raw categorical sampling remain unchanged.

Its complete B28 took **569.02 seconds**, with **12/28 successes (42.86%)**,
mean reward **0.52747**, and **89/89 structurally valid actions**. No production
optimizer/posterior updates occurred. Per-domain outcomes: HumanEval 4/4, MBPP+
4/4, ALFWorld 2/4 (turn counts 20, 20, 10, 15), historical AIME 1/4, HealthBench
1/4 (mean rubric reward 0.44231), HotpotQA EM 0/4/F1 0.50, and TriviaQA EM
0/4/F1 0.0. All results, including the Trivia regression and failed ALFWorld
trajectories, are retained; no best-of selection or score-rule change was used.

This improves the original native T0's 8/28 but **does not qualify task-quality
recovery or a formal250 restart**. Seven distinct training questions repeated
four times are not the historical 32-question per-domain panels; QA F1 and
HealthBench rubric averages are not binary success rates. Historical AIME/ALF
budgets and evaluation sampling also differ. Remaining thinking/skill/input/budget
comparisons need separately declared conditions, not a promise that the first
batch must hit a chosen percentage.

Final shared-guidance CPU `make check`: **3,872 passed, five opt-in skips**,
Ruff/mypy, both wheels and public-wheel check passed; pytest took 298.21 seconds.
The targeted deployed-tokenizer tests passed separately. No hash checks,
independent review agents, repeated unchanged kernel trials or formal W&B points
were added. Both finite T0 clients and the numerical client exited; the existing
GPU0/GPU6 services and restored GPU7 standby remain resident on 22048. This
9.48-minute collection time is not full training throughput or a 250-step ETA.

### Approved fixed source holdout (new data condition)

The owner approved a fixed holdout on 2026-09-09. The private split is now
materialized as `seven-domain-holdout4-seed0-cycle250@1`: one seed (0), four
independent canonical source questions per domain, 28 sources total. Selection
uses sorted source identities, not answers, rewards or previous success rates.
All occurrences of a held-out source stay out of training, including population
aliases and repeated code questions. Remaining original lane order is cycled to
250 question occurrences per domain: **1,750 occurrences, 7,000 trajectories,
B28, 250 steps**. No eighth domain/question was added.

Retained unique training sources are 246 each for HotpotQA, TriviaQA, historical
AIME, HealthBench and ALFWorld, 214 for MBPP+, and 28 for HumanEval. The four
HumanEval holdouts remove 32 original occurrences; four MBPP+ holdouts remove
six. Repetition is recorded, not represented as independent evidence. Training,
holdout records and their private source mapping were written to new files;
old sources and the paused Step17 checkpoint were not overwritten.

The trusted-side exclusion check covered 798 final-evaluation source coordinates,
public-question normalization for the differing QA/math positional namespaces,
and actual ALFWorld game routes. No overlap was found. The holdout is a
**development/quality-monitoring panel**, not a new untouched final benchmark.
Historical AIME training data remains historical, and closed-book TriviaQA is
not relabeled as RC IID. This is a new data condition, not equivalent continuation
of the old Step17 data schedule. No training was launched by splitting the data.

### Skills-off single-axis diagnostic (not a formal configuration)

The already-running skills-off diagnostic completed in **475.66 seconds**:
**9/28 successes**, reward **0.44139**, **97/98 valid actions**. The one parse
failure occurred in ALFWorld. HumanEval and MBPP+ each scored 4/4; ALFWorld 1/4;
other domains had no binary successes (HotpotQA F1 0.50, HealthBench mean rubric
reward 0.33974). Its matched full-inline baseline was 12/28 and reward 0.52747;
these seven-source observations do not establish a population effect and do not
justify disabling the formal skill library. Every diagnostic outcome is retained.
The client exited and its temporary adapter was unloaded. No optimizer/posterior
updates or formal W&B steps were produced. Existing inference/standby services
on 22048 GPU0/GPU6/GPU7 remained resident; no new GPU process was needed for the
holdout preparation. Neither collection timing is full training throughput.

The quality collector now uses `(benchmark, canonical source ID)` for both
independence counts and exclusion checks; population labels remain provenance.
This closes the previous discrepancy with the split planner's source grouping.
The planned launch bindings retain static8/ALF20, five-domain thinking-off,
HealthBench/AIME thinking-on and checkpoint-every10; they are staged, not launched.

Actual ALFWorld reset evidence is now captured from the existing verified reset
return (no additional environment call), then carried in private terminal reward
metadata. Static tasks explicitly record no environmental reset. Same-source
contrast grouping compares the original public initial-state projection and
observed reset, not a seed-only surrogate. Authoring receives short evidence
kinds and trajectory references, not the large projection or hidden targets.
Historical records without reset evidence remain unqualified for these contrasts;
no reward values, F/B inputs or old posterior evidence are rewritten.

New private bindings carry the inline data declaration into the existing
`EffectiveRunCondition` together with all 7,000 actual selected source/occurrence
coordinates. The frozen candidate loader was exercised against the staged
bindings without creating a model, service or optimizer. Resume compares this
section against existing process declarations and rejects adding/removing a data
condition, changing its declaration, or changing actual source order. Legacy
`None` adds no new scientific field; an allowed horizon transition cannot bypass
the new-data restriction. No new digest or parallel state counter was introduced.

The owner reaffirmed that **formal training is skill-on**. The staged candidate
was read back as `skill_exposure: full-inline`; the skills-off branch is a finite
zero-update diagnostic only and must not be promoted to the formal recipe.
Skill-on retains retrieval/exposure, actual invocation evidence, calibration and
library evolution. It neither forces a skill call nor credits mere exposure as
an invocation. No formal optimizer update has yet used the new holdout condition.

Integrated holdout/reset/data-binding milestone: **3,909 tests passed, five
opt-in skips**, Ruff/format/mypy, both wheels and the public-wheel check passed
on 22049 CPU (`CUDA_VISIBLE_DEVICES=""`); pytest took 288.56 seconds. Targeted
split (8), reset (42 plus one serialization), and binding/resume (28) checks
preceded that single full check. The parent WSL quality test invocation was
stopped during dependency import after E-drive I/O stalls, before any test ran;
those cases passed in the remote integrated suite instead. No repeated unchanged
GPU/kernel checks, independent review agent or digest checks were run. This
milestone validates code paths, not real-model quality retention or formal250
completion. The finite CPU verification process exited successfully.

### Next qualification: frozen quality rules and exact checkpoint handoffs

Before running the fixed source panel, its retention policy was frozen: 28
independent sources (four/domain), a probe every five committed steps, protocol
structure/first-turn/admission floors of 0.95, and a maximum 0.25 absolute drop
in each domain's declared native metric relative to T0. Regression must persist
for two due probes to trigger the existing save-and-pause mechanism; missing
evidence or a protocol baseline below its floor is not a pass. Native floors
are zero: this is a regression guard, **not absolute task-quality qualification**.
Small four-source domain panels cannot establish a population accuracy guarantee;
no thresholds will be tuned after inspecting this panel's outputs.

The optional `--pause-at-step` is an execution handoff, not a shortened dataset
or evolution plan. At its committed boundary the normal due quality probe runs,
then the existing complete checkpoint pause saves the state. It enables exact
2+2 process handoffs and a 25-step retention checkpoint while retaining the
250-step source schedule. Explicit operator stop requests retain their original
immediate boundary behavior. No next batch is shortened or partially committed.

Quality reports now include equally weighted canonical-source means, source SD
and conditional SE (not an action/rollout-level confidence interval). A one-source
or incomplete group has unknown SD/SE; mixed-domain pooled SE is not fabricated.
Valid terminal records are counted once per trajectory from explicit accepted
submission OR environment terminal assessment, independent of task success.
Missing complete execution evidence stays unknown. Historical training tuple
counts remain readable rather than being silently reinterpreted.
A horizon-limited unsuccessful rollout remains a complete, valid **training**
artifact even when it lacks native task-terminal evidence. The new terminal
metric is descriptive; it is not a sample filter or permission to omit its TTB
or posterior contribution.

### Skill-on confirmation and completed thinking-only diagnostic

The staged formal configuration was read back again with `skill_exposure:
full-inline`, the original five thinking-off domains and HealthBench/AIME
thinking-on. No formal condition was changed to skills-off. The completed
same-source, skill-on thinking-six-on diagnostic retained HumanEval off and
changed only thinking in the other domains: 28 trajectories, 12 successes,
mean reward 0.509615, 765.53 seconds, 96/105 valid actions and 27/28 valid first
actions. It made zero optimizer or posterior updates; its client exited and
reported its temporary adapter unloaded. This is collection latency, not
formal committed training throughput.

The five-off skill-on reference had 12/28 successes, mean reward 0.527473,
569.02 seconds and 89/89 valid actions. Six-on therefore provides no aggregate
quality or latency justification to replace the current recipe on this small
panel. It did record one actual invocation of the planning skill on a failed
TriviaQA trajectory; the reference recorded none. These are only seven source
questions with four samples each, not 28 independent questions or an IID score.
Exposure is not invocation credit, and one failed invocation is not evidence
that calibration or skill evolution has been validated.

The handoff integration check exposed four failures in an old fake application
that omitted its policy snapshot identity. The fixture now supplies initial or
restored identity like the real training loop; no production fallback identity
was introduced. The affected entry/handoff tests passed (28 tests).
The final integrated CPU check then passed: 3,930 tests, five opt-in skips
(290.45 seconds), Ruff/format/mypy, both wheel builds and the public-wheel
check. No additional GPU qualification, independent review agent or digest
check was run for this fixture repair. The verification process exited; no
new resident service or formal training job was started in this follow-up.

### Fixed-source panel startup repair

The first held-out T0 attempt stopped before any model generation: the declared
sampling/ordered-source IDs were accepted by the panel binding, but the legacy
sampling coordinate required digest-shaped strings. Zero trajectories or updates
were produced, and the temporary adapter was unloaded. That failure is retained.

The coordinate now accepts nonempty declared IDs while preserving its historical
serialized field names, old coordinates and seed derivation. No fake digest,
new hash check, source selection or threshold change was introduced. The real
collector test now exercises declared panel IDs instead of borrowing hashed
fixture IDs; 43 affected coordinate, rollout and panel tests passed, with Ruff
and formatting. A fresh frozen source directory is used for the retry; the
failed attempt and running service code were not hot-patched.

The completed skill-on shared-semantics B28 also received a CPU-only comparison
against its original durable HTTP requests: all 89 served action input sequences
equaled the sealed scorer's forward input; provisional F/B inputs equaled the
sealed plan, and all original action IDs were preserved. This covered 1,446,032
F/B prefix tokens and took 4.52 seconds, without generation or updates. It is
exact-input evidence for the new public semantics, not a fresh GPU-gradient
or task-quality claim; unchanged numerical kernels were not rerun.

The sampling-ID repair's integrated CPU check passed: 3,939 tests, five opt-in
skips (293.84 seconds), Ruff/format/mypy, both wheels and the public-wheel check.
This rerun followed an actual core-coordinate change; unchanged GPU arithmetic
was not repeated. The CPU verification process exited.

### Completed fixed-source skill-on baseline

The repaired collector completed the preselected 28-source panel in 548.40
seconds: 15/28 terminal successes, mean projected reward 0.566128, 104/105
structure-valid and admitted actions, 27/28 first-turn-valid trajectories.
All 28 had complete assessment evidence; 24 had an accepted native terminal
record. The frozen baseline decision was `verified`, without missing metrics
or altered thresholds. The diagnostic client exited and unloaded its temporary
adapter, with zero optimizer/posterior updates.

| Domain | Successes / 4 independent sources | Mean reward |
| --- | ---: | ---: |
| HotpotQA | 4 | 1.000000 |
| TriviaQA (closed-book training inputs) | 1 | 0.416667 |
| Historical AIME training source | 1 | 0.250000 |
| HealthBench | 1 | 0.296231 |
| ALFWorld | 0 | 0.000000 |
| MBPP+ | 4 | 1.000000 |
| HumanEval | 4 | 1.000000 |

ALFWorld's 80 actions were all structurally valid and executed, but all four
episodes exhausted 20 turns without native terminal success. HealthBench had
the single parse error. There were zero actual skill invocations. Consequently
this is a usable frozen retention/protocol baseline, **not evidence of adequate
ALFWorld capability, skill learning, or natural evolution**. The zero ALFWorld
success baseline cannot reveal a further success-rate drop below zero; other
protocol metrics remain monitored. No new success floor is selected after seeing
these outputs. These distinct held-out sources also cannot be presented as a
same-question improvement over the original seven-question B28 diagnostic.

Execution bindings for subsequent qualification are staged to retain the tested
native inference service on physical GPU6, with GPU0/GPU7 as gradient participants.
This is only a staged mapping: the two owned standby services have not yet been
converted to training workers. Formal250 and new-condition updates remain unstarted.

### Skill-on reaffirmation and real lifecycle launch status

The staged formal configuration was read directly again: `skill_exposure:
full-inline`, with only HealthBench/AIME native thinking enabled. Skills-off
remains an isolated diagnostic; exposure never counts as an invocation.

The first real native lifecycle attempt stopped during model initialization:
its private launcher omitted the existing FLA dependency directory. The installed
FLA 0.5.2 import was confirmed on CPU and the launcher path corrected, retaining
the existing kernel requirement. No author request or optimizer update occurred.
A second attempt passed that dependency preflight but stopped at the released-GPU
availability check (30% utilization despite low memory occupancy). No workload
was started on that busy device; the owned standby restoration was initiated.
These are launch failures, not successful real-author or formal-training evidence.
The completed fixed-source panel and old committed training state remain intact.

### Owner-required T0 success floors and budget candidate

After seeing the fixed panel, the owner required at least 2/4 terminal successes
for HotpotQA, TriviaQA, AIME, MBPP+ and HumanEval; HealthBench and ALFWorld require
at least 1/4. This is a newly requested admission rule, not a rule preregistered
before the old panel. A separate policy version re-evaluated the original complete
probe without new generation: TriviaQA, AIME and ALFWorld fail. The earlier policy,
probe, native labels and decisions are preserved; formal training remains stopped.

`QualityRule.baseline_minimum` now distinguishes a stricter initial success floor
from the existing later retention thresholds. Missing success metrics cannot be
substituted with F1 or reward. Older persisted policies retain their interpreted
defaults without file rewriting. Gate/persistence/panel tests: 25 passed; mypy for
both source modules passed. An initial WSL test invocation was stopped before
collection while blocked in the E-drive filesystem; the successful run used the
isolated CPU checkout. No successful identical test was rerun.

The owner also explicitly authorized changing thinking, reasoning budgets and
ALFWorld horizon. The old panel's AIME reasoning reached the 1024-token limit in
4/4 calls, TriviaQA in 2/4; all four ALFWorld episodes exhausted 20 turns, with
substantial repeated actions in two episodes. The next declared combined candidate
keeps skill-on, the same source membership, seed zero, raw sampling, native reward,
static eight-turn cap, 65536 input cap and 2048 action cap, and changes only:

- TriviaQA: thinking on, reasoning cap 4096.
- AIME: thinking remains on, reasoning cap 16384.
- ALFWorld: thinking on, at most 50 turns, within its existing native environment cap.

Other domains retain their original thinking and 1024 reasoning cap. Per-domain
budgets are wired through both lazy/hydrated sessions and formal quality collection;
the global reservation/service envelope grows without granting every domain the
larger cap. The explicit candidate identity records these controls. Configuration,
hydration and quality-binding tests: 28 passed; mypy for four source modules passed.
This reused panel is now an adaptive development/retention panel, not untouched
final evaluation. No result or speed benefit is claimed before execution.

### Real Generate contract failure repaired, not retroactively accepted

The real native lifecycle attempt reached the frozen-base author, whose response
omitted the requirement kind. The prompt required a strategy, but the generation
schema permitted omission; historical decoding correctly interpreted it as immutable,
and Generate correctly rejected it. The rejected request and draft are unchanged.
`skill-generate@7` now requires an explicit `evolvable-strategy` kind and forbids
nonempty replacements in the Generate response schema and prompt. Other operations
and historical decoding retain their prior semantics; immutable validation is not
relaxed. This requires a new author condition/request, not a retry presented as
recovery. Authoring targeted tests and type/lint checks passed. A fresh controlled
real-model lifecycle attempt is separate from formal updates and natural adoption.

Budget/start-floor milestone: CPU-only `make check` on 22049 passed 3,961 tests
(five opt-in/private-fixture skips), formatting, lint, mypy, both wheels and the
public-wheel boundary check; pytest took 292.69 seconds. This frozen test snapshot
contains Generate@7. A subsequent real Generate@7 response correctly failed the
unchanged applicability validator, so the real lifecycle is still unqualified;
its original failed draft is preserved and a further structural-contract fix is
being implemented separately. This is not a successful mutation/recovery claim.

The declared budget candidate began collect-only T0 on the existing GPU6 inference
service. At 23:19 UTC the live client had completed 11/28 trajectories in 183 seconds,
with 69 completed model requests and no optimizer/posterior updates. The preceding
150-second interval added 11 complete trajectories; this early mixed-domain rate
cannot predict the ALFWorld tail or formal steps/hour. GPU7's owned standby was
restored after the failed controlled lifecycle. Formal250 remains unstarted until
the new success floors and remaining compatibility checks pass.

### Author structure/escaping and current budget-panel evidence

The next author repair fixes the entire Generate structural projection: exact
applicability (including empty tools), one draft, ordered required IDs, strategy
kind and no revision metadata. Prompt and response schema use the same projection.
Real CPU compilation with the installed xgrammar also exposed an existing
`minLength` string translation that rejected legitimate JSON escapes. Free prose
now uses its standard JSON-string grammar; unchanged decoders/validators still
reject empty or whitespace-only content. Generate@9, Retain-compress@9, Refine@7
and Split@8 explicitly identify this new author condition. Old rejected journals
are preserved; this is not recovery by replacement or a relaxed semantic validator.

Targeted CPU tests: 51 passed. Actual installed grammar tests accepted newline,
quotes, backslashes, Chinese and JSON-looking prose in UTF-8/ASCII serialization,
rejected all 16 structural negative cases and rejected unescaped raw newlines.
The combined CPU-only `make check` passed 3,980 tests (five skips), lint/format,
mypy, both wheels and public-wheel boundary checks; pytest took 291.27 seconds.
The preceding combined staging attempt had four failures because an updated
schema test file was omitted from the overlay. After copying that already-fixed
file, its five targeted tests and the final combined check passed. No production
validator was weakened to fix those staging failures. No hash checks or independent
review agent were used, and unchanged GPU math benchmarks were not repeated.

At 23:44 UTC the same budget-candidate T0 client remained active: 24/28 complete,
450 completed model requests, about 28 minutes elapsed. Completed native success
counts were HotpotQA 4/4, TriviaQA 2/4, AIME 4/4, HealthBench 1/4, MBPP+ 4/4 and
HumanEval 4/4. All six meet the owner-required floors; four ALFWorld episodes were
still at turns 39–48, so the complete panel had not passed. AIME's four reasoning
outputs used 6,741–16,384 tokens; two still reached the declared limit. These are
adaptive-panel observations, not independent final-evaluation estimates. Formal250
remains unstarted; this longer candidate is not a claim of meeting 4.2 steps/hour.

### Budget candidate 1 completed; quality, not speed, remains the blocker

The owner accepted slower execution if the quality requirements pass; 4.2
steps/hour remains a reported target, not a reason to bypass or stop qualification.
Candidate 1 finished all 28 fixed-source trajectories in 2,209.76 seconds:
19 successes, mean reward 0.685176, no optimizer/posterior update. ALFWorld remained
0/4 with four 50-turn episodes. Its 200 actions included 46 parse errors and one
schema error; an additional HealthBench action had a parse error. The unchanged
quality policy rejected the ALF success floor and panel structure/admission floors.
The complete original responses, native labels and rejected decision are retained.

A new explicitly declared candidate changes only ALFWorld's reasoning allowance
from 1024 to 4096, retaining thinking-on and 50 turns. Other domains, fixed panel,
seed zero, raw sampling and skill-on remain unchanged. At 23:55 UTC its new client
was collecting on the existing GPU6 service: zero complete trajectories and 17
completed model requests in 33 seconds. This is not a measured throughput or a
passed quality gate; formal250 is still unstarted.

Generate@9 meanwhile passed the controlled real-weight lifecycle: the existing
owned GPU0 service performed one real frozen-base author call, and a finite GPU7
fixture performed mutation, new-process continuation and a third-process state
comparison. New skill exposure and two scripted native-wire invocations produced
two posterior updates with total flow mass two; Z Adam state was cleared at mutation.
The final process verified parameters, named optimizer state, library, posterior
and cursors without model requests or another update. All three stage exits were
zero. The GPU7 owner then restored its standby, personally checked healthy (HTTP
200); fixture processes exited. These three isolated fixture updates are not
formal training, natural W50 triggering, on-policy adoption or benchmark gains.

### Owner override: ALFWorld is strictly capped at 25 turns

The owner superseded the 50-turn allowance. The in-progress second candidate was
stopped through its verified client PID, without stopping the inference service.
Cleanup finished with 16 completed records and all 73 dispatched model responses
preserved; its furthest ALFWorld episode had reached turn five. No optimizer or
posterior update occurred, and its temporary adapter was unloaded. This is an
owner-cancelled partial condition, not a complete quality result or four failures.

A separate candidate now uses ALFWorld at most 25 turns with thinking-on/R4096;
static domains remain at most eight turns and retain their preceding budgets.
The private formal launcher also refuses a new configuration above the 25-turn
ALFWorld cap. At 00:06 UTC on September 10 the new collect-only client was live
on the existing GPU6 service, with 15 completed model requests and zero finished
trajectories in its first 33 seconds. Original 50-turn evidence is not relabelled.
The 250-step formal run remains gated on the complete new-condition result.

### M07 complete-batch attribution runner and live bounded qualification

The isolated attribution API now accepts read-only before/after observers and
per-artifact progress callbacks. A private runner preserves the original full B28,
per-edge scores, four structure/success groups (including empty groups), separate
Z gradients and one complete Adam update. It saves both model/optimizer boundaries
and compares ten predeclared public synthetic native-action probes using original
tokens, legality, fixed-reference NLL and categorical KL. This is not production
training or a task-performance evaluation.

The first actual-input CPU preflight exposed an incorrect runner assumption:
manifest task IDs are unique rollout slots, not source-question IDs. The private
check now groups by the existing benchmark/population/source identity; its new
regression test and the unchanged full 28-trajectory input passed. No trajectory
was filtered or reassigned. Six earlier targeted tests, mypy and ruff passed;
the combined CPU-only `make check` passed 3,980 tests (five skips), lint, typecheck,
both wheels and wheel-boundary checks; pytest took 287.77 seconds. No hash check,
review agent or unrelated GPU equivalence rerun was added.

At 00:22 UTC on September 10, finite M07 execution was live on the owned GPU7,
with 61/110 edges completed, approximately 0.236 edges/second and a rough remaining
math estimate of 208 seconds (excluding final probes/save). Its initial ten
synthetic generated actions were all structurally valid; the post-update result
was still pending. The launcher drained only its owned standby and will restore
that service after the diagnostic. GPU6 continued the separate strict-25 T0:
24/28 trajectories and 214 model requests completed, no update or error reported.
The six completed domains met their success floors; ALFWorld was still pending.
Neither diagnostic is uploaded as formal-training W&B data. Formal250 remains
unstarted until the actual qualification results are available.

### M07 real full-B28 result

The finite historical-input run completed successfully in 507.21 seconds with
all 28 trajectories and 110 edges, one isolated Adam update and no production or
posterior update. Canonical gradient reconstruction was exactly equal; category
reassociation relative L2 errors were 9.18e-8 (F), 8.40e-8 (B) and zero (Z), below
the unchanged 1e-3 bounds. Original-order math and category attribution are not
claims that category-specific Adam updates can be added.

All 28 trajectory coefficients were positive. The evidence comprised three
structurally invalid/failed edges, 99 valid/failed edges and eight valid/successful
edges; invalid/successful had no evidence. Component gradient norms were
F=0.63620, B=1.61509 and Z=38.46794. Relative parameter changes from the complete
Adam update were F=0.014924, B=0.014934 and Z=0.010909. These are descriptive,
not causal estimates of any category's quality effect.

The ten predeclared synthetic protocol generations remained 10/10 structurally
valid before and after. Mean fixed-action NLL increased from 0.890745 to 0.896478;
post-update categorical KL ranged from 4.06e-7 to 3.53e-4. Thus this single update
did not collapse this small protocol panel, but did not demonstrate task-quality
improvement or long-run stability. Actual before/after parameters and named Adam
state, exact sampled tokens, per-class measurements and original batch evidence
remain private. At 00:29 UTC the diagnostic and owner PIDs were gone and the
restored GPU7 standby was personally confirmed live with HTTP 200. The strict-25
T0 on GPU6 was not restarted or changed by this diagnostic.

### Timed parallel ALF candidate and communication-repair handoff

The collect-only bridge now persists its existing answer-free per-trajectory
phase/queue/cache/environment progress and monotonic collection time, including
failed cleanup. Missing server measurements remain null. Six affected CPU tests,
mypy and ruff passed; the combined 22049 CPU-only `make check` exited zero before
the separate typed-history repair. Unchanged gradient kernels were not retested.

At the owner's request, an independent GPU7 inference service ran the fixed panel
with ALFWorld thinking off, reasoning cap 8192 and at most 25 turns, while the
previous thinking-on/R4096 candidate continued on GPU6. Startup took 34.01 seconds.
Neither attempt performed optimizer or posterior updates. Both were then stopped
at the owner's request to use the other session's communication repair. The
thinking-on attempt retained 27 completed trajectories; the thinking-off monitor
had recorded 21 before cancellation. These are incomplete, owner-cancelled
conditions, not full-panel quality results or replacement negative labels.

The thinking-off client's cancellation cleanup stalled despite zero queued or
running service requests. After more than fourteen minutes without progress,
only its verified owned client and environment child were terminated; request
journals and original artifacts were retained. At 01:06 UTC on September 10,
those PIDs, the temporary inference service and its owner were gone. GPU7's
original standby role had been restored and personally checked HTTP 200. The
existing GPU6 service remained healthy and was not restarted.

The other session's separately declared `public-task-semantics@2` ALF4 diagnostic
is reused rather than launched twice. It retains four fixed sources, initial F,
full-inline skills, thinking off, R8192 and 25 turns, with explicitly separate B4
sampling coordinates. At 01:06 UTC two trajectories had completed: one native
success in seven turns and one failure at 25 turns. The other two were still
running. This partial ALF-only evidence is not the full B28 T0 gate, not a formal
training metric and not uploaded as formal W&B data. See the separate ALFWorld
communication-repair note for that session's completed checks and final results.

### Owner-selected T0 and formal250 launch (September 10, 01:46 UTC)

The owner explicitly selected the completed six-domain four-source groups from
candidate 3 plus the other session's latest complete ALF4 under
`public-task-semantics@3`. Whole domain groups were retained, not per-question
best results. Original artifacts, native labels and sampling coordinates remain
private and unchanged. This is an owner-selected composite baseline, not one
joint same-condition B28 sample, untouched IID measurement or training evidence.
No replacement model requests, optimizer updates or posterior updates were made
when assembling it.

T0 has 21/28 native successes and mean reward 0.7566043992. Domain success counts
are HotpotQA 4, TriviaQA 2, AIME training sources 4, HealthBench 1, ALFWorld 2,
MBPP+ 4 and HumanEval 4. The latest ALF4 took 426.00 seconds. All unchanged
source-holdout quality floors pass. The actual new initial model accepted this
exact imported probe; the baseline was not regenerated or relabeled.

The composite/import interfaces passed 57 directed CPU tests. A single combined
CPU-only `make check` on 22049's Linux local storage passed: 4,016 tests, seven
explicit optional skips, lint/type checks and both wheel builds. Unchanged GPU
kernels were not retested; no manual hash checks or independent review agents
were used.

Formal250 began at 01:46:36 UTC on 22048: GPU6 inference, GPU0 coordinator plus
gradients, GPU7 second gradient worker. Only the two owned standby services were
drained; foreign processes were preserved. Before training, residual usage on
GPU0/GPU7 was below 25%. Owner PID 63376 launched torchrun PID 1583 and ranks
2075/2076; inference parent 64305/scheduler 65398 stayed resident. The live entry
confirmed B28, 250 steps, two participating gradient workers and canonical
within-step accumulation. Static tasks allow eight turns; ALFWorld at most 25.
The approved per-domain thinking/budget settings and full-inline skills remain.

The first planned stop is after a complete step-2 checkpoint for the required
same-run new-process recovery check, then continuation of the same 250-step
plan, not a replacement four-step experiment. Ordinary checkpoint cadence is
ten steps. At 01:47 UTC the first batch was collecting; no training loss/reward
or measured throughput was available yet. Committed-only aggregate export and
local W&B synchronization are being attached; T0 is not a fabricated optimizer
step. The 4.2 steps/hour and 72-hour figures remain planning targets, not observed
performance guarantees.

### First formal commit and live W&B acknowledgment repair

The first complete B28 transaction committed at 01:54:06 UTC: loss 1.79247530,
mean reward 0.70970696 and 16/28 terminal successes. All four training ALFWorld
rollouts succeeded. This is a different question set from the selected T0 panel;
the two aggregate success fractions are not a paired quality comparison. There
were 63 actions, 62 structurally valid, zero parse errors and one schema error.
The transaction took 447.65 seconds; rollout span was 441.01 seconds and gradient
tail 4.45 seconds, with 27 contributions completed before the last artifact.
The single warmup step is not a steady-state throughput qualification.

Live synchronization exposed an actual uploader bug: the installed W&B SDK
keeps an explicitly numbered `log(..., step=...)` open unless `commit=True` is
specified. The first uploader consequently timed out awaiting visibility; its
normal finish flushed the original point. The repaired uploader now explicitly
commits each point before awaiting acknowledgment. Four directed CPU tests pass,
including a lost-acknowledgment resume that does not resubmit the original point;
ruff and formatting pass. Training processes and their frozen code were not
changed. The existing W&B run was resumed with its original durable state, not
replaced. A direct API read at 02:08 UTC confirmed exactly the original step-1
loss/reward point. The new local uploader PID is 335767; aggregate mirror 332761
and remote committed-metrics exporter 5750 remain separate CPU processes.
The uploader-only follow-up also passed the combined CPU-only `make check` on
22049: 4,017 passed, seven optional skips, lint/type checks and both wheels.
The frozen, running training checkout was not hot-patched for this local-only
publishing repair.

### Complete step 2 and same-condition new-process continuation

Step 2 committed at 02:32:11 UTC with loss 0.66124367, reward 0.69937206 and
18/28 successes. Its 129 actions included 124 structurally valid actions, five
parse errors and no schema errors. It took 2,284.25 seconds; these two different
question batches do not establish a learning-quality trend. The average of the
first two complete steps is about 2.64 steps/hour, below the 4.2 target. Holding
that rate would imply roughly 94 hours for the remaining steps before additional
quality/recovery costs; this is a provisional extrapolation, not steady-state
qualification. The owner explicitly retained the current configuration.

The planned step-2 pause saved a complete checkpoint and exited zero. Owned GPU0
and GPU7 standby services were restored and personally checked healthy, then
exchanged for new training processes without touching foreign workloads. Owner
44680/torchrun 53754 launched new ranks 57294 and 57297. The same frozen training
checkout resumed at optimizer step 2 at 02:35:14 UTC, retained the imported T0,
and began batch 3. Recorded session preparation took 2.50 seconds and model/
resume preparation 8.85 seconds. Two further actual commits are still needed to
finish the same-run 2+2 recovery qualification. No training condition changed.

A subsequent W&B API response returned the original step-1 row twice with
identical internal step, timestamp, source commit and every metric; step 2 was
already present once. Acknowledgment now accepts only fully identical repeated
history rows, reports their count, and still rejects any differing field for an
optimizer step. No remote history was deleted and no original point was resent.
Five directed uploader tests and ruff passed. This is reporting recovery only,
not a change to optimizer/posterior evidence or the active training checkout.
The full-row acknowledgment repair subsequently passed the combined CPU-only
check: 4,018 tests, seven optional skips, lint/type checks and both wheels.
Uploader PID 342772 resumed the same W&B run and acknowledged the newly committed
step 3 without resending steps 1 or 2. Step 3 completed in 868.94 seconds with
loss 1.92708878, reward 0.50000000 and 13/28 successes. It is process-local warmup
step 1 after recovery, not a steady-state observation. At 02:54 UTC batch 4 was
live at 16/28 artifacts and 15/28 gradient contributions; the full 2+2 recovery
qualification and later held-out quality checks are still pending.

### Batch-4 generation timeout: three commits preserved, no partial update

At 03:03 UTC the resumed training process failed on the last unfinished AIME
reasoning request in batch 4. The formal rollout HTTP timeout was 600 seconds;
this request declared up to 16,384 reasoning tokens. The serving log confirmed a
client disconnect followed by request abort. The durable client journal has
1,012 completed responses and one unresolved dispatch, not an original complete
response that can simply be replayed. No replacement answer was generated.

Steps 1–3 remain committed, including a complete step-3 checkpoint. Batch 4's
27 completed artifacts remain in the private in-flight store; no optimizer or
posterior update was made from that incomplete batch. W&B contains unique
committed steps 1–3 only, not a fabricated timeout failure label. The proposed
recovery is an explicitly recorded retry of only the aborted request with its
original policy, input and seed, plus a longer transport timeout; owner approval
is pending. The timeout must not be resolved by deleting its journal entry or
claiming that a newly generated response was recovered from the original call.

At 03:10 UTC training owner/rank processes were gone. The existing GPU6 inference
service and the automatically restored owned GPU0/GPU7 standbys (parents 65328
and 65431) were personally checked HTTP 200. Local mirror 332761 and W&B uploader
342772 remained live. No additional GPU job was launched after the failure;
training ETA is suspended pending recovery authorization. The same-condition
2+2 qualification has not completed: the second process committed step 3, not 4.

### Owner-authorized recovery of the single aborted batch-4 request

The owner explicitly approved raising only the rollout transport timeout from
600 to 1,800 seconds and regenerating the confirmed aborted request once with
its exact original input, route, policy and sampling seed. The journal now
provides an operator-only single-use authorization: it retains the original
dispatch in a separate archive, rejects changed inputs and does not renew a
consumed authorization after another failure. This is explicitly regeneration,
not recovery of a missing original response. No automatic retry was enabled.

Thirty-eight directed CPU tests passed, including exhausted authorization,
changed-payload rejection and completed-response reuse; targeted mypy passed.
The first test invocation encountered an inherited shared temporary directory
owned by another user. Validation was rerun with a private temporary directory;
no shared directory was modified. A combined CPU-only check is in progress.

A fresh direct read confirmed checkpoint 3, exactly 27 saved batch-4 artifacts
and one unresolved AIME reasoning dispatch. GPU6 inference and owned GPU0/GPU7
standby services returned HTTP 200; other GPU workloads were not touched.
Recovery will use a separate frozen checkout, not hot-patch a running worker.
The actual process split is now 2+1+recovery, not a completed clean 2+2 test.

The repair passed the combined CPU-only `make check` on 22049: 4,021 passed,
seven optional skips, lint/type checks and both wheels. The private journal was
backed up with SQLite before installing the single-use authorization. No original
request row or completed artifact was removed. Unrelated GPU/kernel comparisons
were not repeated; no hash checks or independent review agents were used.

At 03:30:38 UTC the authorized recovery started at optimizer step 3, B28. Owner
9981/torchrun 14811 launched ranks 15097 and 15099, plus CPU metrics exporter
14813. GPU6 continues inference; GPU0 coordinates and computes gradients;
GPU7 computes gradients. Both owned standby services drained successfully;
foreign processes remained untouched. All eight GPUs were checked before and
after launch. The collector immediately restored 27 artifacts and began
recomputing their temporary gradients while the remaining AIME request ran.
A complete step-4 checkpoint is required before judging this recovery successful.
Local W&B uploader 342772 and aggregate mirror 332761 remain on the same run.

The authorized reasoning retry completed successfully in 473.04 seconds with
16,384 generated tokens and a length stop; its journal authorization is now
consumed/COMPLETED while the original dispatch remains archived. It reused
1,152 of 1,154 input tokens in the serving cache. This measures the new attempt,
not the lost original response. Its first action then used 1,908 tokens in
58.97 seconds and was classified as natural-language/non-JSON output. That is a
real sampled action error, not an infrastructure failure or permission to retry.
At 03:42 UTC the last AIME trajectory was on reasoning turn 2 of the unchanged
maximum 8, while the other 27 gradient contributions were complete. There was
still no step-4 commit. Completing a request is not completing its trajectory.

### Step-4 checkpoint saved; adapter retirement stalled

The final AIME trajectory eventually made an accepted action on turn 7. All 28
artifacts, gradient contributions and canonical merges completed. Checkpoint 4
was saved with the original two source events in `checkpoint-published` state.
However, retiring the old serving adapter repeatedly timed out; attempted swap
rollback also timed out. Training owner 9981 exited nonzero at 03:56 UTC. This is
a publication failure after saved learning, not permission to apply Adam again.
The request journal contains 1,026 completed calls and no unresolved dispatch.
W&B still has only three completed training transactions.

The actor loaded and validated the new adapter successfully before the old
adapter's unload stalled. Installed SGLang code waits for LoRA usage counters on
unload and has an abort-echo cleanup path without the normal release call; this
is a plausible consequence of the earlier abort, not a measured live counter.
No counters were forced to zero and no third-party package was hot-patched.
With zero running/queued generations and saved checkpoint 4, only the owned
GPU6 actor was restarted using its original arguments and environment. The first
restart guard caught delayed CUDA memory teardown and stopped before loading;
a fresh availability check then permitted the replacement actor PID 2966.
Its HTTP health returned 200 at 04:04 UTC. GPU0/GPU7 owned standby services were
also healthy. Foreign processes were preserved. Recovery must now reconcile
checkpoint 4's pending adapter/event publication, not collect batch 4 again.

At 04:06 UTC fresh-process recovery loaded checkpoint 4 and reconciled its pending
adapter and original source-event publication. The journal is now `committed`;
no batch-4 recollection or optimizer replay occurred. Source event time remains
03:53:09 UTC, whereas successful publication/reconciliation happened at 04:06;
these are distinct timestamps. Step 4 has loss 0.64555187, reward 0.73809524,
20/28 successes, 137 actions, 17 parse errors and one schema error. This does
not establish improvement on a fixed IID question set.

Owner 18152/torchrun 20883 continued at optimizer step 4 into batch 5, with CPU
metrics exporter 20885. The timeout repair remains in the frozen training
checkout; original T0, policy condition and every-five-step held-out quality
policy remain unchanged. The next explicit owner pause is step 25. W&B's same
run now contains unique steps 1–4 (the previously documented exact repeated
step-1 history row remains). The three-step ordinary timings exclude this
failed/recovered batch and must not be used to hide its end-to-end cost.
At 04:07 UTC all eight GPUs were checked again: this run uses GPU6 actor
2966/scheduler 5653 and GPU0/GPU7 training ranks 21061/21062. Batch 5 had five
artifacts and five gradient contributions. Including the persisted run clock,
failure and recovery time, four published steps represent roughly 1.7 steps/hour;
a naive remaining-246-step extrapolation is about 145 hours, not a steady-state
forecast. The 72-hour target is therefore not supported by the current evidence.
Meeting that deadline would require an explicit resource/budget decision; the
owner's instruction to retain the current scientific condition is respected.

### Step 5 completed and declared quality probe started

At 04:25 UTC direct process/journal/metrics inspection confirmed the next complete
transaction without another adapter-unload error. Step 5 recorded loss
0.55645276, reward 0.70578231 and 19/28 successes; its 122 sampled actions included
17 parse errors and no schema errors. All 28 artifacts, gradient contributions
and canonical merges completed. Step wall time was 1,131.22 seconds (18.85
minutes), approximately 3.18 steps/hour for this one step, not steady-state
qualification. Its ordinary-step extrapolation for 245 remaining steps is about
77 hours before quality probes, initialization and failures; the lower persisted
end-to-end rate still includes earlier outages. The 4.2 target remains unmet,
and meeting 72 hours would require an owner resource/budget decision rather than
a silent condition change.

The unchanged every-five-step policy then began the step-5 held-out quality
collection; its attempt is explicitly `collecting`, not a completed score. The
existing actor had seven running requests and none queued during the direct
check. Existing owner 18152, ranks 21061/21062 and actor 2966/5653 remained live;
no new GPU process or changed GPU role was introduced in this monitoring turn.
A direct W&B API read confirmed unique optimizer steps 1–5 and the original
step-5 metrics. No source-code edits, tests, builds or hash checks were needed
for this monitoring-only interval; the restricted Luna agent continued publishing
real committed data through the existing uploader and mirror.

### First held-out quality warning; frozen continuation rule retained

The completed step-5 held-out probe recorded 17/28 successes and mean reward
0.68379125, versus 21/28 and 0.75660440 in the owner-selected composite T0.
First-turn structure validity was 75.00% versus 96.43%; overall structure
validity/admission was 87.23% versus 99.07%. ALFWorld had 0/4 successes versus
2/4 at T0. Other domain successes were HotpotQA 4, TriviaQA 1, AIME 4,
HealthBench 0, MBPP+ 4 and HumanEval 4. HealthBench's continuous reward increased
from 0.29623 to 0.36987 despite its terminal-success count decreasing; these are
not interchangeable metrics. Selected/composite T0 is not an untouched IID
baseline, and this small comparison alone does not establish causal regression.

The saved decision is explicitly `warning / continue`, with the three panel
structure/admission rules and ALFWorld success rule triggered; no unavailable
metric was hidden. This is the first observation under the predeclared
consecutive-failure rule, not a passed quality check. Thresholds, horizon,
thinking configuration and samples were not altered to remove the warning.
The existing live owner/ranks began batch 6 after the probe completed. Probe
collection itself spanned about 1,268.59 seconds (21.14 minutes), separately from
training-step wall time, so throughput estimates must include this recurring
cost as well as prior failures/recovery. No new GPU process, source-code change,
test, build or hash check was performed during this monitoring interval.

### Step 6 complete; AIME generation was the last dependency

Direct inspection at 05:17 UTC confirmed six complete transactions and live
batch 7. Step 6 recorded loss 1.21346861, reward 0.52161654 and 14/28 successes.
Its 124 actions included 36 parse errors (29.03%) and no schema errors. The last
unfinished trajectory was AIME: repeated natural-language/non-JSON actions kept
it in the unchanged multi-turn loop before all artifacts arrived. These sampled
failures were retained, not classified as infrastructure retries or omitted.
The final artifact's gradients were then computed and canonically merged; adapter
publication completed without repeating the earlier unload failure.

Step wall time was 1,770.97 seconds (29.52 minutes), about 2.03 steps/hour for
this observation. The process-local warmup/steady-state reporter still correctly
says `insufficient-steady-state-evidence`; no fastest-step selection or claimed
4.2-step/hour qualification was made. Current losses/rewards come from different
training questions and do not by themselves prove an IID learning trend. All
existing owner, training and serving processes remained live; no new GPU job,
configuration change, code change or redundant validation was introduced.

### W&B curve visibility and single-writer quality summaries — 2026-09-10

A direct W&B history read confirmed committed steps 1–6 already contain
`train/batch_success_fraction`, `train/reward_mean`, and `train/ttb_loss`.
An independent report now exposes these three curves against `optimizer_step`,
filtered to the current run; existing layouts/history were not overwritten:
[training curves](https://wandb.ai/lanlangcll-university-of-illinois-urbana-champaign/skillev123/reports/Formal250-t0v3-training-curves--VmlldzoxNzkwNTI2OQ==).

The active SDK writer had overwritten separately published quality summary keys
when the next training point arrived. Completed aggregate probes are now loaded
by that same writer, with source-condition and committed-step matching. Quality
probes remain separate from optimizer history; no task text or individual result
is uploaded. A local CPU mirror replacement includes completed probe/decision
aggregates. Seven targeted uploader tests passed, including retention across a
resumed writer and the next training point. No training/serving process or
scientific condition changed. Full CPU checks run on the approved Linux host,
not the WSL data drive.
The final CPU `make check` passed: 4,023 tests passed, seven declared opt-in
skips; formatting, lint, mypy and both wheel builds also passed. No GPU test,
independent review agent, hash check or redundant kernel qualification was run.
A subsequent uploader restart exposed the concrete cause of partial ACK scans:
W&B 0.30's cached unfiltered history contained only step 1, while the server
held steps 1–6. A direct `scan_history(use_cache=False)` returned all six. ACK
reads now bypass that cache and reject incomplete optimizer-step coverage before
any upload. The attempted old-step write was rejected by W&B; original history
and the pending record were preserved. This repair affects telemetry only.
After the actual history-cache failure was fixed, eight targeted tests and the
final Linux CPU `make check` passed: 4,024 passed, seven opt-in skips, with
format/lint/mypy/build checks successful. Direct uncached W&B verification found
steps 1–6 exactly once and the completed step-5 quality summary restored. Only
local CPU telemetry processes were replaced; the three-GPU training continued.

### Step 7 and live W&B ACK indexing — 2026-09-10

Step 7 completed in 2,249.56 seconds (37.49 minutes): loss 0.19587411,
reward 0.69418960, 18/28 successes, 126 actions and 76 parse errors (60.32%),
with no schema errors. Batch 8 started normally. Step 7's late AIME trajectory
spent 1,676 seconds across its first seven reasoning requests; their measured
prefill span totaled only 3.15 seconds, with 148,352/198,576 input tokens cached.
The post-prefill span is not mislabeled as a directly measured decode timer.
These data identify a real long generation/invalid-action tail, not a disabled
cache or a dead training worker; no sampling/budget change was made.

W&B step 7 was uploaded once and the step-5 quality summary survived. However,
the new summary briefly preceded the queryable history index, tripping the
new conservative completeness check. The uploader now waits within the existing
ACK deadline for complete uncached reads; it never repeats the upload to repair
visibility, and conflicts still fail immediately. Training and its three GPU
roles were unaffected. This follow-up uses targeted uploader tests rather than
repeating unrelated kernel or full training checks.

### Step 8 committed; ACK wait exercised — 2026-09-10 06:26 UTC

Direct remote inspection confirmed eight committed updates and batch 9 running.
Step 8 took 1,866.77 seconds (31.11 minutes), with loss 0.18350096, reward
0.62857143 and 16/28 successes. Its 192 actions included 134 parse errors
(69.79%) and one schema error. A late ALFWorld trajectory generated 25 turns
and approximately 1.46 million scored edge tokens; after all artifacts arrived,
its owner still had long-prefix provisional gradients to finish. The batch
remained intact and completed without a training/process restart.

A direct uncached W&B read confirmed unique optimizer points 1–8, no conflicting
rows, and the step-5 quality warning still present after the new point. The API
also repeated six completely identical earlier rows; these are deduplicated,
not reported as additional optimizer updates. The repaired uploader remained
live after the step-8 acknowledgment. No new GPU or resident CPU process,
configuration change, or test run was introduced during this monitoring period.

The latest two post-warmup steps average about 34.30 minutes; two observations
still do not qualify the existing steady-throughput gate. Including recorded
run overhead, eight updates in about 4.67 hours imply roughly 1.71 steps/hour
and a naive remaining estimate of 141 hours, not a guarantee. Reaching a hard
72-hour target requires an explicit owner decision; budgets are not silently
reduced. Rising format failures remain a quality concern, not evidence that
lower TTB loss means improved evaluation. The next fixed quality probe is due
at step 10 under the existing consecutive-warning stop rule.

### Step 9: capacity shortfall now measured — 2026-09-10 07:08 UTC

Nine complete updates are committed and batch 10 is running. Step 9 took
2,574.87 seconds (42.91 minutes): loss 0.46026591, reward 0.50850340,
11/28 successes, 172 actions and 131 parse errors (76.16%), with no schema
errors. The late ALFWorld/AIME requests continued advancing through their
original horizons; no partial batch, resampling or process restart was used.

The existing three-post-warmup-step forecast now reports `capacity-shortfall`,
not insufficient data: 2,230.40 seconds/step (37.17 minutes), approximately
1.61 steps/hour, 154.66 projected total hours and roughly 149 hours remaining.
This is a limited-sample forecast, not a guarantee. A hard 72-hour completion
requires an owner decision; no hidden budget or sampling changes were made.

Direct W&B inspection confirmed unique committed coordinates 1–9 without
conflicting history rows, step-9 loss/reward/success, and the retained step-5
quality warning. Local telemetry and all training/serving processes remained
live. No new GPU task, resident process, code change or test was introduced.
The fixed quality check after step 10 remains scheduled; persistent quality
regression must follow its existing pause rule rather than being waived.

### Step 10 saved; fixed quality regression pause — 2026-09-10 08:05 UTC

Step 10 committed in 1,466.73 seconds (24.45 minutes), with loss 0.38384461,
reward 0.45660881 and 12/28 successes. Its 198 actions included 122 parse
errors and no schema errors. Both cadence-step-10 and the separate paused-step-10
checkpoint contain COMPLETE, optimizer state, policy files and runtime state;
the latter records optimizer step 10. The application exited normally after
saving at 07:59 UTC. No step 11 was started.

The complete fixed 28-source quality probe triggered the declared consecutive
regression pause:

| Aggregate | Owner-selected composite T0 | Step 5 | Step 10 |
| --- | ---: | ---: | ---: |
| Success | 21/28 | 17/28 | 14/28 |
| Mean reward | 0.756604 | 0.683791 | 0.570030 |
| First-turn structure valid | 96.43% | 75.00% | 39.29% |
| All-action structure valid | 99.07% | 87.23% | 40.24% |

Step-10 decision is `regressed / pause-after-commit`, with no missing metrics.
Triggered rules are first-turn structure validity, action structure validity,
admitted fraction, and AIME accuracy. This selected composite T0 is not an
untouched IID baseline. Actual authoring/evolution or causal benefit is not
claimed from these ten updates. No thresholds were relaxed, samples replaced,
or repeated probe selected to escape the stop rule.

The private wrapper labels the owner `failed / AssertionError` because its
post-exit assertion expected the planned step-25 pause, while the application
correctly paused at step 10; training and final metric export both exited 0.
This wrapper classification is not a CUDA crash or lost checkpoint. The wrapper
restored its previously owned SGLang standby services on physical GPU 0 and 7
(parent PIDs 8820 and 8827, schedulers 10610 and 10643). Direct health requests
confirmed both standbys and the unchanged GPU-6 actor healthy. Training workers
are gone; no unrelated process was touched.

Direct W&B inspection confirmed training step 10 and the complete step-10
quality aggregates/decision uploaded. CPU mirror/uploader remained live. The
latest four post-warmup steps imply about 1.77 steps/hour (roughly 136 hours for
240 further updates after resumption), below target; while paused there is no
completion ETA. Owner input is required before overriding or changing the
quality policy or beginning a repaired training condition. The 250-step goal
is not complete. No algorithm edit or new test run was made in this interval.

### Owner changed the next run to a fresh start — 2026-09-10

The latest owner instruction supersedes checkpoint continuation: wait for the
other local session's protocol repair and qualification to finish, then start
the 250-step run from Step 0. Preserve the paused step-10 run and its W&B history;
do not import its learned optimizer/posterior/library state or completed rollout
artifacts into the new run. The new protocol requires its own recorded condition
and applicable T0 evidence, not relabeled old quality scores. At the latest direct
check, the peer's isolated real-model probe was still collecting; no new formal
training process had been started by this session.

### Owner-authorized fresh carrier-v2 run — 2026-09-10 09:31 UTC

The owner explicitly requested starting now instead of waiting for the remaining
quality issues, and continuing later from a complete checkpoint after compatible
repairs. A new run starts from Step 0; the old ten-step run, optimizer/posterior
state and W&B history remain separate. It uses the peer session's merged native
carrier-v2 fixes plus a recorded, run-scoped initial-admission override. Missing
or mismatched T0 evidence still fails; every-five-step retention checks and the
consecutive-regression pause remain unchanged. The observed T0 is not relabeled
as satisfying the old floor: 19/28 successes, reward 0.738695, HealthBench 0/4.
Its complete original probe is imported without resampling or changed labels.

Seven domains, B28, seed 0, 250 steps, static eight / ALFWorld 25 turns, skill-on,
previously approved domain thinking/budgets and every-ten-step checkpoints remain.
No fourth GPU, smaller batch, grammar mask, reward change or old training artifact
is introduced. Physical GPU 6 serves inference; GPUs 0 and 7 both compute gradients,
with 0 coordinating. Only the two owned idle standbys were drained; unrelated
processes remain untouched. The new owner started at 09:31 UTC. Initial startup
is not a committed update or a throughput measurement; no reliable new ETA yet.

The private launcher now accepts a normal quality-triggered checkpoint pause
instead of incorrectly asserting a hard-coded early pause step. Later changes
are installed only after complete checkpoint drain and compatibility assessment,
not by hot-editing the running source or combining incompatible in-flight data.

The scoped quality-gate/monitor tests passed (22). One final CPU-only `make check`
on 22049's local Linux memory filesystem passed: 4,074 tests, eight skips,
format/lint/mypy and both wheel builds. The private tokenizer was not supplied
for six skips; the other two require explicit CUDA qualification. Peer real-tokenizer
coverage is recorded in its preceding repair report; unchanged CUDA numerical
checks were not repeated. The WSL targeted attempt stalled during imports and
was stopped before any result; no pass was claimed from it. No hash checks or
independent review agent were used. A new local CPU W&B mirror and uploader
were started for `bayesian250-carrier2-fresh-20260910`; the old run is untouched.

### Fresh run step 1 committed — 2026-09-10 09:50 UTC

Direct inspection confirmed optimizer step 0 at initialization, then one complete
update at 09:43:51 UTC and batch 2 continuing with both gradient workers alive.
Step 1: loss 1.69655031, reward 0.69139194, successes 17/28, 90 actions,
two parse errors and no schema errors (97.78% structurally valid). These are
training trajectories, not the independent T0 panel. No sample was filtered.

The step took 711.88 seconds (11.86 minutes): rollout 683.74 seconds, gradient
tail 25.81 seconds, durability 2.33 seconds; 27/28 gradient contributions had
finished before the final rollout. This single warmup observation corresponds
to 5.06 steps/hour, but does not pass the steady-throughput gate. A simplistic
249-step extrapolation is about 49.2 hours before quality/evolution and other
costs; it is not a validated ETA or a 72-hour guarantee. At 09:50, batch 2 had
21 artifacts and 21 gradient contributions; no restart or budget change occurred.

The new CPU W&B uploader initially rejected an unsupported private mirror scope
label before logging step 1. The mirror now uses the supported, accurate
`fixed-held-out` scope; the explicit T0 waiver remains in the run's quality
policy. Only the new run's two local telemetry processes were restarted, with
source/identity retained. The resumed uploader acknowledged step 1. Training and
old W&B state were untouched; no new GPU process or test run was needed for this
private label correction. Intermittent SSH observation timeouts were not treated
as training failures. The selected GPUs read 1,980 MHz and 29–42 C in the sampled
inspection; this alone is not a full thermal-slowdown diagnosis.

### Step 2 and uploader heartbeat repair — 2026-09-10 10:22 UTC

Two updates are committed; direct inspection found the owner and both ranks
alive with batch 3 at 25/28 artifacts and gradient contributions. Step 2 committed
at 10:16:13 UTC: loss 1.84447290, reward 0.56711146, 15/28 successes; 127 actions,
six parse errors and three schema errors (92.91% structurally valid). It took
1,942.53 seconds, including 1,783.24 seconds of rollout and 156.81 seconds of
post-rollout gradient tail. The batch stayed complete; no budget, sample or
policy change was made to shorten its long tail.

The two warmup steps average 1,327.20 seconds (22.12 minutes), about 2.71 steps/hour.
A naive remaining 248-step estimate is about 91.4 hours before future overheads;
there is still insufficient steady-state evidence. A hard 72-hour limit needs
an owner decision. The authorized ongoing run is not stopped just because this
provisional rate misses the target. Training batch scores are not the fixed
quality panel; the scheduled quality checks remain in place.

W&B showed `crashed` despite live uploader/core processes and correctly uploaded
step 1; its last heartbeat was the last metric write. A once-per-minute summary
heartbeat now identifies only the uploader and last acknowledged optimizer step.
It uses the same SDK writer, does not create history rows, and never asserts GPU
training liveness. After one controlled local uploader restart, direct API reads
confirmed `running`, advancing heartbeat and exactly steps 1–2 with unchanged
metrics. The mirror and training were not restarted. The old writer's shutdown
also logged a missing local metrics file; it predates the current healthy writer,
so it was not used to justify another restart.

The heartbeat change passed ten targeted tests. A test-format issue and generated
pytest fixtures accidentally left inside the reused CPU-check tree were corrected;
those attempts stopped at the format stage. The private temporary directory was
moved outside the source tree, preserving its prior output. The subsequent single
complete CPU-only check on 22049 passed: 4,075 tests, eight skips, lint, mypy and
both wheel builds. No CUDA numerical tests, independent review agent or hash
checks were added for the uploader-only change. New local uploader PIDs are
424878/424907; the CPU validation process has finished. No new GPU task was started.

### 2026-09-10 — owner-selected Step-2 reasoning repair continuation

The owner selected this fresh run's **Step 2**, not its later commits, as the
continuation source. Its full snapshot was copied out of rolling retention
before requesting cooperative drain. The old attempt completed Step 5 and
paused at **10:59:39 UTC**; its Step 3–5 evidence and original W&B history remain
unchanged. Those later updates are not imported into the new branch.

The peer-session repair adds the public tool catalog to the reasoning phase and
uses HealthBench reasoning budget 4096. This is an explicitly recorded **new
input condition**, not exact same-condition recovery. All other accepted domain
settings remain unchanged: static cap 8, ALFWorld cap 25; thinking on for AIME,
HealthBench and TriviaQA, off for the other four domains. Skill-on, B28, seed 0,
F/B/Z, Adam, posterior, library, detector and the fixed task cursor are retained.

Implementation adds a narrow reasoning continuation and a new-directory branch
containing the original committed event/journal prefix through Step 2. The task
and seed curriculum identity is preserved; old incomplete work and old quality
scores are not carried into the new condition. Quality collects its own baseline
at the restored Step 2, without relabeling it as T0. Aggregate export and W&B
support a declared segment beginning at optimizer Step 3; parent points are never
relabeled or uploaded as new-condition updates.

Old-condition additional complete observations (not new-branch measurements):

| Optimizer step | Loss | Mean reward | Success / 28 |
| --- | ---: | ---: | ---: |
| 3 | 1.960562 | 0.648352 | 17 |
| 4 | 0.699091 | 0.693223 | 17 |
| 5 | 0.673273 | 0.678571 | 20 |

The old process's three post-warmup steps averaged 867.96 seconds, approximately
4.15 steps/hour, slightly below the 4.2 target. That is a reference only: the new
condition and its fresh-process warmup need separate measurement. No extra GPU
kernel comparison, hash check or independent audit agent was run for this
persistence/condition-boundary change.

Validation for this integrated continuation: 75 targeted training/metrics/quality
checks and the implementation agent's 53 core/branch checks passed (fixture-only
failures corrected before qualification). One CPU `make check` on the approved
22049 Linux filesystem completed with **4,121 passed, 10 explicit optional
skips**, lint/mypy and both wheels successful. No repeated unrelated GPU tests.
The continuation owner launched at **11:08:10 UTC** on 22048, selecting GPU 6
for existing SGLang inference and GPUs 0/7 for the two participating gradient
ranks. Initial process launch is not yet a new committed update.

At **11:08:43 UTC**, the new process reported `initial_optimizer_step=2`.
The named new-condition boundary was complete; direct comparison of its saved
execution state with the imported Step-2 execution state was equal. The owner
then began the isolated new-condition quality baseline at Step 2, before any
new optimizer update. No parent training point appeared in the branch metrics.

New W&B segment:
https://wandb.ai/lanlangcll-university-of-illinois-urbana-champaign/skillev123/runs/bayesian250-catalog-step2-20260910

Runtime processes at handoff: CPU owner 12081, torchrun 21336, participating
training ranks 23179/23180 (GPU 0/7); existing GPU-6 inference scheduler 5653.
CPU aggregate exporter 21341 and local mirror/uploader 439239/439268 serve the
new segment. The old owner restored owned standbys 30116/30126 after pausing;
the new owner subsequently drained only those two idle services for training.

### Step-2 branch: first new commit verified (2026-09-10 11:44 UTC)

The branch's read-only Step-2 quality baseline completed all 28 fixed sources in
1,261.66 seconds: 18/28 success, mean reward 0.716066. HealthBench remained 0/4
and TriviaQA 1/4; the explicit initial-floor waiver permitted continuation.
`verified` here means the declared waived admission policy was evaluated, not
that the earlier per-domain startup success goals were met. ALFWorld was 2/4.
These are new-condition restored-policy diagnostics, not T0 or training labels;
they are separately identified in the new W&B run configuration.

At **11:41:18 UTC**, new-condition optimizer **Step 3** committed: loss
**1.641774**, reward **0.700549**, success **18/28**, 2 parse errors and no schema
errors. Its full checkpoint has `COMPLETE`, its transaction is `committed`, and
W&B remotely exposes exactly one training history point, at Step 3 (no relabeled
parent Step 1/2). Step 4 is collecting under the updated policy.

Step 3 took **690.83 seconds (11.51 minutes)**; rollout span 682.88 seconds,
post-rollout gradient tail 5.49 seconds, and overlap 654.27 seconds. This is the
first process-warmup update, not qualified steady throughput. The observed rate
of this single step is about 5.21 steps/hour; the formal forecast correctly
remains `insufficient-steady-state-evidence`. The 21-minute condition-baseline
cost and prior run time remain recorded rather than disappearing at restart.

### Step-2 branch: Steps 4–5 persisted (2026-09-10 12:15 UTC)

Step 4 committed at 11:52:45 UTC: loss **0.793940**, reward **0.709707**,
success **18/28**, 3 parse errors and 2 schema errors; 687.63 seconds with a
4.88-second gradient tail. Step 5 committed at 12:11:01 UTC: loss **0.908877**,
reward **0.739796**, success **21/28**, 4 parse errors and no schema errors.
Both full checkpoints are complete. A direct W&B API read found unique
new-condition history steps **[3, 4, 5]**, including all three training metrics.

Step 5 required **1,095.70 seconds (18.26 minutes)**: rollout 1,044.58 seconds,
gradient tail 48.77 seconds, and 27 complete contributions before the last
rollout. The last ALFWorld trajectory was observed progressing from turn 14 to
19 before completion; no infrastructure failure or partial-batch update was
observed. The fixed Step-5 quality collection is now running, not yet accepted.

Steps 3–4 are process warmup. Step 5 alone gives **3.29 steps/hour**, below the
4.2 target; this was escalated to the owner as needing performance/ETA attention,
not silently reconfigured. If sustained, the remaining 245 updates alone take
about **75 hours**, before quality checks and other overhead. Only one measured
post-warmup step exists, so neither the earlier 5.22 warmup rate nor this single
slower sample qualifies as steady-state evidence. Existing GPU roles remain
0/7 training and 6 inference; no additional GPU process was launched. No code
changed, so previously successful checks and GPU comparisons were not repeated.

At **12:31 UTC**, the fixed Step-5 held-out check was complete (926.46 seconds):
**20/28 success, mean reward 0.748124**. Domain success counts were HotpotQA 4,
TriviaQA 2, AIME 4, HealthBench 1, ALFWorld 1, MBPP+ 4, HumanEval 4, each out of
four independent held-out sources. Structure-valid actions were 107/108; valid
first turns 27/28. The existing retention decision was `verified / continue`,
with no triggered rule or unavailable metric. W&B remotely exposed these
separate `quality/step5` results. This satisfies the earlier per-domain count
thresholds on this particular Step-5 panel, not retrospectively at T0; it is not
training reward or independent proof of generalization. Step 6 was collecting
21/28 artifacts/contributions, without an abort. No configuration was changed.

### Step-2 branch: measured throughput warning (2026-09-10 13:02 UTC)

Steps 6 and 7 committed at 12:38:19 and 12:57:49 UTC. Step 6: loss **1.736565**,
reward **0.642857**, success **18/28**, 2 parse/10 schema errors, 710.62 seconds
and a 13.93-second gradient tail. Step 7: loss **0.333052**, reward **0.907274**,
success **23/28**, no parse/schema errors, 1,170.52 seconds and a 21.42-second
gradient tail. Both full checkpoints were directly confirmed complete; W&B's
unique committed history is **[3, 4, 5, 6, 7]** with loss/reward/success present.

The three non-warmup steps now average **3.63 steps/hour**. The existing gate
reports `needs-human-decision`, below 4.2. The owner was notified: remaining
243 updates imply about **67 hours of training**, roughly **80–84 hours** with
observed periodic quality-check costs, not a guarantee. Step 7's observed final
actor was AIME in its second reasoning turn; ALFWorld was already complete.
No budget, sampling, task or scoring setting was reduced. Step 8 was collecting
22/28 artifacts and contributions without an abort. Existing inference GPU 6
and training GPUs 0/7 remain in use; no additional GPU task or service started.

### W&B mirror read race repaired (2026-09-10 13:38 UTC)

The local uploader exited after acknowledging Step 9 when an aggregate decision
file disappeared between `exists()` and `read_text()`. Formal training stayed
alive and continued collecting Step 10. The reader now catches only the observed
`FileNotFoundError` and defers that incomplete quality pair until the next poll;
malformed JSON and conflicting provenance still fail explicitly. A regression
test covers disappearance, subsequent recovery, and malformed-data rejection.

Ruff and format passed. The local pytest process stalled in WSL filesystem I/O
before results; only that owned CPU test was terminated. The same targeted file
on 22049's Linux CPU workspace passed **12 tests in 0.33 seconds**, with CUDA
hidden. No training code, model process or GPU configuration changed, and no
unrelated full checks or kernel comparisons were repeated.

The sole uploader resumed the existing run and state without `--new`. Root's
direct API check at 13:38 UTC found `running`, a fresh uploader heartbeat and
unique history **[3, 4, 5, 6, 7, 8, 9]**; the existing mirror remained running.
No baseline or quality probe was converted into a training step.

### Step-10 cadence save and quality check (2026-09-10 14:00 UTC)

Step 10 committed at **13:47:00 UTC**: loss **0.324757**, mean reward
**0.771362**, success **20/28**, 5 parse errors and no schema errors. Root
confirmed both `step-00000010` and `cadence-step-00000010` complete. W&B exposed
unique new-condition commits **[3–10]**. The step took 1,369.74 seconds, including
1,290.71 seconds of rollout and a 75.66-second post-rollout gradient tail.

The fixed Step-10 quality panel completed in **619.04 seconds**: **19/28**
success, reward **0.743712**. Success counts by domain were HotpotQA 4, TriviaQA
2, AIME 4, HealthBench **0**, ALFWorld 2, MBPP+ 3, HumanEval 4. The prior Step-5
count thresholds were therefore not sustained for HealthBench. Existing
relative-baseline retention rules returned `verified / continue`, no triggered
rules or unavailable metrics; this must not be described as meeting every
earlier absolute startup floor. W&B quality results were directly read back.

Step 11 was collecting **22/28** artifacts/contributions with no abort. The
non-warmup training rate is **3.64 steps/hour**, below 4.2 and explicitly flagged
`needs-human-decision`. Rough remaining work is 66 training hours, approximately
**76–83 hours** including observed varying check costs. All wall time remains
recorded. Existing three-GPU roles and scientific configuration are unchanged.

### Steps 11–14: throughput gate currently passes (2026-09-10 14:39 UTC)

All four new commits were directly observed; their full save points were
complete when checked and W&B's unique history advanced through Step 14.

| Step | Loss | Mean reward | Success / 28 | Seconds | Parse / schema errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 0.432901 | 0.887755 | 23 | 375.14 | 1 / 0 |
| 12 | 1.187593 | 0.626093 | 17 | 651.75 | 1 / 0 |
| 13 | 0.753727 | 0.702421 | 14 | 352.50 | 3 / 0 |
| 14 | 0.360912 | 0.674048 | 15 | 943.10 | 8 / 2 |

The aggregate non-warmup rate is now **4.36 steps/hour**, with the existing gate
reporting `pass`. This supersedes the earlier below-target speed warning for
the current measured aggregate, not a guarantee about future tasks. Remaining
236 updates imply about 54 training hours, roughly **62–71 hours** including
observed quality-check durations. No slow observations were removed. Recent
batch success varies with the fixed source sequence; throughput passing does
not prove quality improvement. Step 15 was collecting 19/28, no abort, and its
scheduled held-out check will be assessed separately. The same three GPUs and
model processes remain; no new GPU task, configuration edit or test run.

### Step-15 quality warning (2026-09-10 15:13 UTC)

Step 15 committed at 14:47:09 UTC and its full checkpoint was confirmed complete:
loss **0.652802**, reward **0.796253**, success **22/28**, 1 parse error and no
schema error. Wall time was **666.57 seconds**, gradient tail **6.00 seconds**.
W&B had unique training steps through 15. Non-warmup rate **4.44 steps/hour**
currently passes throughput; remaining training plus observed check overhead is
roughly **62–70 hours**, not guaranteed.

The separate fixed held-out check completed in **1,294.73 seconds**: success
**18/28**, reward **0.673702**. Counts were HotpotQA 4, TriviaQA 1, AIME 4,
HealthBench 1, **ALFWorld 0**, MBPP+ 4, HumanEval 4. First-turn structure validity
was **26/28 (92.86%)**; all-action validity was 123/126. The declared retention
policy returned **`warning / continue`**, triggering first-turn structure and
ALFWorld success rules. These are genuine quality warnings, not infrastructure
errors or a passing quality result. The existing two-consecutive-failures rule
was preserved; Step 20 needs close attention. No threshold, source, score or
sampling setting was changed. Root directly read the same warning and metrics
back from W&B. Step 16 was collecting 21/28 without an abort; existing model
processes and three-GPU roles remain active.

### Step-19 grader failure, work preserved (2026-09-10 16:03 UTC)

Before the failure, Steps 16–18 committed and were directly confirmed complete:

| Step | Loss | Reward | Success / 28 | Seconds | Parse / schema errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.738860 | 0.578571 | 15 | 983.66 | 1 / 1 |
| 17 | 0.176687 | 0.864819 | 19 | 613.09 | 0 / 0 |
| 18 | 0.793487 | 0.795918 | 22 | 677.47 | 1 / 0 |

Step 19 failed during HealthBench terminal evaluation. One frozen base-Judge
rubric response hit its **1,024-token** limit; the sole declared repair also
ended with `finish_reason=length` and incomplete JSON. Both HTTP 200 responses
are durably recorded. This is an unavailable evaluation, not a negative task
label. No reward or posterior update was invented for it.

The owner exited with status 1 at **15:58:03 UTC**, after draining the batch.
Root confirmed all owner/training/exporter PIDs gone, **Step 18 complete**, and
**no Step-19 checkpoint/commit**. The in-flight store retains **27 complete
artifacts**, while all 3,865 journaled requests are completed; no ambiguous
request remains. Existing actor responses for the failed trajectory also remain
available. The stale progress sidecar's 23-ready count is not the final durable
artifact count and is not evidence that training still runs.

The owner automatically restored its authorized SGLang services on GPU 0/7:
parents **64399/64404**, schedulers **2038/2042**, ports 18472/18477. Root health
checks returned 200 for both and for existing GPU-6 inference on 18476. No other
project's processes were changed. These restored services are standby, not
active gradient workers. Training throughput is currently **zero / paused**;
the previous 4.50 steps/hour rate is historical, so ETA is suspended.

The owner was asked to authorize a larger Judge repair output allowance as an
explicitly declared scoring-execution condition. Blind restart would replay the
same truncated responses. No budget, retry allowance, rubric, success label or
quality threshold has been changed while awaiting that decision; the 27 saved
artifacts must not be silently discarded or relabeled as another condition.

### Authorized Judge repair and Step-19 recovery (2026-09-10 16:38 UTC)

The owner authorized continuing with a **4,096-token Judge repair allowance**;
first attempts remain 1,024 tokens. The explicit deployment field defaults to
1,024 for existing callers and is passed through the training session builder
to the bounded sampler. The new graded-artifact verifier identifies this repair
profile. Actor sampling, task identities, rubrics, success rules, posterior and
quality thresholds are unchanged. Historical grades are not recomputed; this
is a recorded scoring-execution change beginning with the missing Step-19
result, also applicable to future held-out repairs, not retroactive equivalence.

Targeted CPU verification on the approved Linux host: **59 tests passed**,
plus mypy on the three modified source modules and local ruff/format. No
unrelated full suite, GPU numerical replay, hash check or review agent was run.
The implementation was published in `c4bd574`.

A private recovery client initially used the API root rather than the chat
completion route and received a known HTTP 404 before generation. The route
was corrected using the existing explicit one-use aborted-request recovery;
the failed dispatch remains recorded. The subsequent real frozen-Judge call
returned valid rubric JSON, `stop`, **132 output tokens in 2.92 seconds**. Its
exact response was persisted before training resumed. The 27 completed
artifacts and previous rewards were retained; the missing artifact now brings
the saved Step-19 batch to **28/28**. No trajectory was replaced.

Root directly confirmed the new process restored **optimizer step 18**, then
committed **Step 19 at 16:37:43 UTC** with a complete checkpoint:
**loss 1.052452, reward 0.908333, success 24/28**, 1 parse error and 0 schema
errors. All domains except HealthBench had 4/4 terminal successes; HealthBench
had 0/4. These are training results, not held-out evaluation. W&B was read back
with the same unique Step-19 commit and loss/reward/success values.

Runtime on the approved 22048 endpoint: existing inference GPU 6 (parent
2966 / scheduler 5653); new GPU 0/7 gradient ranks **58276/58277** under
launcher **58027**. CPU owner **56384** and metric exporter **58031** were
started. The owner's two idle standby services were drained; foreign processes
were untouched. Step 20 is collecting. Three physical GPUs remain in use.

The pre-interruption non-warmup rate was **4.50 steps/hour**; remaining work
including observed evaluation overhead is roughly **60–69 hours**, not a
guarantee. Step 19 reused paid-for work and is a process-cold recovery step,
not a fresh throughput observation. The Step-15 quality warning remains active;
the existing consecutive-warning decision at Step 20 must not be bypassed.

### Step-20 complete; declared quality pause (2026-09-10 17:03 UTC)

Step 20 committed at **16:50:02 UTC**: loss **0.462697**, reward **0.705215**,
success **19/28**, wall **739.07 seconds**. Root confirmed the complete ordinary,
10-step cadence and final paused checkpoints. The resumed process's Steps 19
and 20 remain cold/warmup observations; the prior steady estimate is still
4.50 steps/hour rather than a replay-inflated rate.

The separate held-out Step-20 collection completed all **28 independent sources**:
reward **0.754995**, success **20/28**. Per-domain successes were HotpotQA 4,
TriviaQA 1, AIME 4, HealthBench 1, ALFWorld 2, MBPP+ 4, HumanEval 4.
First-turn structure validity was **26/28 = 92.86%**, below the declared 95%
minimum again after Step 15. The two first-turn invalid cases were in HealthBench
and ALFWorld. Overall action validity was **91/94 = 96.81%**. The native ALFWorld
metric recovered, but the repeated first-turn structure rule still correctly
returned **regressed / pause-after-commit**. This is a quality decision, not
another Judge infrastructure failure; no unavailable metric was reported.

The owner exited normally at **17:01:01 UTC**, only after saving the complete
Step-20 pause state. Root found no remaining owner, torchrun, gradient-rank or
exporter PIDs, and no Step-21 checkpoint. Original GPU-6 inference remains;
GPU-0/7 standby services were restored (parents **21638/21670**, schedulers
**23625/23616**). All three service health checks returned 200. These two
restored GPU services are standby, not continuing optimization; no foreign
process was changed.

W&B was directly read back with unique training data through Step 20, the real
quality regression and `paused-quality-regression` execution state. Current
training rate is **zero while paused**, so completion ETA is suspended. This
requires an owner decision; the prior authorization for Judge repair does not
waive the quality retention rule. No threshold was lowered, no baseline was
regraded, and the pause was not silently overridden.

### Owner-approved v3 continuation from Step 20 (2026-09-10 21:24 UTC)

The owner explicitly accepted the remaining quality shortfall and instructed
immediate continuation with `native-single-tool-call@3`, not v2 or the later
v4 environment-guidance candidate. The peer's completed v3 handoff repair is
`d1649ca`. New continuation support (`9bd7799`) permits only the declared
native v2-to-v3 field change, independently of the existing reasoning/horizon
transitions. It restores the complete saved Step-20 method state and saves a
new action-wire condition boundary before collecting Step 21. Historical
rewards, trajectory wire identities, posterior evidence and quality failures
are not rewritten. Full 250-step plan, B28, single Agent, seed, budgets, thinking
modes, skills, optimizer, and two participating gradient workers are retained.

The original complete v3 development probe at the same Step-20 weights is reused
as the new-condition baseline, without generating another panel: **18/28
successes**, reward **0.723240**, first valid actions **27/28**; HealthBench
remains **0/4**. Its original bytes/provenance remain saved. Explicit owner
initial admission accepts this shortfall; it is not a passing HealthBench result
or a retroactive replacement for the original v2 Step-20 evaluation. Future
scheduled quality measurements remain enabled. The new continuation code passed
**89 targeted CPU tests**, mypy and ruff/format. The peer repair's earlier full
check is recorded separately; no unchanged full suite or kernel replay was
repeated for this launch, and no hash checks or review agents were used.

Root checked all eight GPUs on 22048 and the three services' active request
counts before launch. The peer evaluation had stopped and owned standby services
were idle. GPU 0/7 standby parents 21638/21670 were drained by the owner, leaving
only the small existing foreign allocations; no foreign process was changed.
GPU 6 inference remains parent 2966 / scheduler 5653. New CPU owner **61279**,
launcher **63811**, gradient ranks **64015/64016**, and metric exporter **63812**
were personally confirmed. Application startup at **21:23:37 UTC** reports
`initial_optimizer_step=20`, `total_steps=250`, `batch_size=28`; Step 21 was
collecting at 21:24 with no abort. The full v3 condition checkpoint is complete.

The source run remains untouched. The new input condition has a separate
[W&B continuation run](https://wandb.ai/lanlangcll-university-of-illinois-urbana-champaign/skillev123/runs/bayesian250-action-wire3-step20-20260910?nw=nwuserlanlangcll),
starting with actual update 21; old steps 3–20 stay in the parent run. The existing
personal workspace exposes loss, mean reward and success fraction directly on
the run page. New local CPU mirror **533996** and uploader **534261/534272**
replace the old sync processes. No synthetic Step-20/21 history was created.

There is no completed v3 training step yet, so its throughput is **not measured**.
The older v2 rate of 4.50 steps/hour gives only a planning reference: remaining
230 updates plus observed evaluation overhead are roughly **60–69 hours**,
not a new-condition speed claim or a guarantee. Three physical GPUs are in use.

### First v3 update committed (2026-09-10 21:39 UTC)

Root verified **Step 21 committed at 21:37:49 UTC**, its full checkpoint is
complete, and Step 22 is collecting (11 artifacts and 11 gradient contributions
at the observation). Step 21: **loss 0.243939, reward 0.877126, success 22/28**,
**zero parse errors and zero schema errors**. These describe this training
batch, not proof of fixed-panel quality improvement.

Wall time was **849.42 seconds / 14.16 minutes**, an individual-step equivalent
of **4.24 steps/hour**. This is the first cold-process v3 update, so the monitor
correctly reports insufficient steady-state evidence and no numerical steady
ETA. The older 60–69-hour planning range remains conditional, not a v3 forecast.
Inference GPU6 and gradient GPU0/7 remain active with the same PIDs; no new GPU
process or setting change occurred during monitoring.

### Token-cap notice from Step24 (2026-09-10 22:30 UTC)

Owner requested the model be told its actual token limits from Step23 or24.
Step23 was already collecting, so it completed unchanged and paused at a full
checkpoint (22:22:57 UTC commit; normal process exit). Its loss was0.369585894,
mean reward0.75, success20/28, parse errors0 and schema errors0. Step22 had
loss0.361624683, reward0.670972644, success18/28 and took1755.22 seconds.
Steps21–23 are retained in the original v3 W&B segment without relabeling.

`6d49d12` adds opt-in `token-budget-notice@1`: persisted phase context carries
actual per-request reasoning/action limits and the common rollout/F/B renderer
adds a short runtime-enforced cap notice. Caps, tokens, thinking, turn limits,
reward and truncation/next-turn behavior are not changed. Historical notice-off
prefixes retain their identity. A single-axis complete-state continuation
records `saved_optimizer_step=23`, `effective_from_optimizer_step=24`.

Focused validation:135 CPU tests passed on22049 in6.74s;10 source files passed
mypy;13 changed files passed Ruff/format. Root inspected the combined changes
and imported the isolated deployment (only `token_budget_notice` differs from
its parent config). No independent review agent, hash checks, repeated full
suite or GPU numerical replay was run: there is no changed numerical kernel.

New owner59809, torchrun3122 and CPU exporter3123 started on22048. Inference
remains physicalGPU6; physicalGPU0/7 remain the two gradient roles. Prior
owned standby PIDs11194/11200 were idle and drained; foreign processes were
untouched. All eight GPUs and service health/empty queues were checked before
launch. The app restored Step23 at22:26:29 UTC. At22:30 it was collecting the
new-condition held-out baseline, not yet training Step24; three actual persisted
contexts confirmed phase-context@3 and per-request caps. No old probe was
relabelled as new-condition evidence.

Local CPU mirror543909 and W&B uploader544339/544353 track the new segment:
https://wandb.ai/lanlangcll-university-of-illinois-urbana-champaign/skillev123/runs/bayesian250-token-notice-step23-20260910
The first optimizer history point must be24. No optimizer point is fabricated
for restore or baseline collection. Previous21–23 observations average about
3.04steps/hour (not steady-state);227 remaining updates project roughly75h,
excluding future quality/restore overhead. This is below4.2 and remains an
owner-decision performance warning, not an automatic stop: owner directed
training to continue despite the quality/performance thresholds.

At23:00 UTC the new-condition baseline had completed with a recorded continue
decision, and Step24 was collecting with17/28 artifacts and17/28 gradient
contributions. Root inspected actual Step24 event H0 metadata: phase-context@3
contained the real per-request caps (including4096/2048 and1024/2048). This
confirms notice activation in training, not only a config edit. Gradient ranks
3463/3464 were live; GPU0/6 were actively computing andGPU7 remained the second
gradient owner. No Step24 optimizer metric had yet been emitted.

At23:17 UTC root verified the first notice-enabled update: Step24 committed
at23:16:32 UTC with COMPLETE checkpoint, wall1131.732s, loss0.606008944,
reward0.688449848, success18/28, parse errors3 and schema errors1 across106
actions. Uncached W&B history contained this single actual Step24 point with
matching metrics. Step25 was collecting15/28 artifacts/gradient contributions,
without an abort. This cold-process observation is3.18steps/hour equivalent;
226 further updates would be about71h at that single-step pace, before extra
quality/restore costs. It is not evidence that adding the prompt notice sped
up generation, nor a steady-state throughput pass. Training continues.

At23:28 UTC root verified Step25 COMPLETE and uncached W&B ACK: committed
23:24:28 UTC, wall475.752s, loss0.421740009, reward0.965225564, success28/28,
77 actions with zero parse/schema errors. The scheduled Step25 held-out quality
check was active after commit. Steps24–25 average4.48steps/hour excluding
quality overhead; both are process warmup observations, not a steady-state
pass or a causal comparison of token notices.225 more updates at this two-step
average project about50h plus quality/restore costs. All rows, including slower
Step24 and baseline collection time, remain retained.

At2026-09-11 00:07 UTC root verified Step26 COMPLETE and uncached W&B rows24–26
without duplicates. Step26 committed00:04:49 UTC, wall955.856s, loss0.174739672,
reward0.872180451, success23/28;103 actions,4 parse errors,0 schema errors.
Step27 was collecting23/28 artifacts and gradients. Step25 held-out check had
completed28/28 with success17/28, reward0.711248439, first-action validity26/28
and warning/continue; these are separate from training Step25's28/28 success.
The quality check cost about24.4min. Step26 is the first non-warmup process step
(3.77steps/hour); the monitor still reports insufficient steady-state evidence.
Using all three recent steps versus this single non-warmup step projects about
53–59.5h of remaining training. If45 subsequent scheduled quality checks each
cost24.4min, that adds about18.3h: rough remaining71–78h, not a guarantee and
excluding additional failures/evolution work. No settings or processes changed;
Luna continues Git/W&B only, current CPU writer550027/550043 and mirror543909.

At2026-09-11 00:39 UTC root verified Step27 COMPLETE and uncached W&B rows24–27
with matching unique metrics. Step27 committed00:37:25 UTC: wall1955.863s,
loss0.508405582, reward0.928571429, success26/28;47 actions,7 parse errors,
zero schema errors. Its final AIME trajectory required three reasoning/action
turns; no budget, horizon or sample was altered. Step28 was collecting23/28
artifacts/gradients, no abort; both gradient ranks and inference remained live.
The two non-warmup observations26–27 average2.47steps/hour; forecast remains
insufficient-steady-state-evidence. The monitor's remaining pure-training ETA
is90.2h. At the last quality-check duration, future checks add about18.3h, so
roughly108h remains before unforeseen costs, with substantial uncertainty.
The previous faster estimate is not retained as the current forecast. Owner's
continue instruction remains in force despite the performance warning.

## Committed telemetry expansion (2026-09-11)

Read-only `skillev-training-metrics@2` joins complete source commits to their
same-run batch/step performance rows; a separate output/store preserves legacy
`@1` values. It adds per-domain native reward/success, explicit action admission
and execution fractions, invocation/posterior counts, phase checks/triggers/no-op
and actual mutation counts, logical tokens, server-counter intervals, timings,
and declared GPU reservation hours. Missing evidence remains null. Collection
ledger tokens do not imply an actor/grader split; server-counter deltas may span
nonstep work. Process GPU hours include preparation/quality/waiting, whereas
step GPU hours do not: these overlapping scopes must not be added together.

W&B supplemental history refers to the existing source commit on a separate
`telemetry/optimizer_step` axis. Original optimizer rows are not resent;
append-only SDK indices permit old-step backfill without blocking future steps.
Only allowlisted aggregates leave the machine; no question/action/answer text.

Validation: 30 focused metrics/export/monitor tests passed on 22049 CPU, three
source modules passed mypy, and affected source/tests passed ruff. Local W&B
mapping/backfill/recovery tests passed14/14 (8.44s final run). Full training,
kernel and full-suite checks were not repeated for this read-only change;
no independent review agent or hash checks were used. The implementation agent
handled export code; the designated Luna agent handles only Git/W&B operations.

At00:59 UTC root verified the independent CPU exporter15934 and mirror567441
produced steps24–28 with unchanged original metrics. Old mirror543909 stopped;
original training exporter3123 and all training/model processes remain running.
Across these five steps actual credited skill invocations and posterior update
events were0; source-history cumulative updates remain80 across8 cells. Each
step recorded one phase check, no trigger/no-op or library mutation. Full-inline
skill exposure is not credited as an invocation. This is a coverage limitation,
not missing telemetry, and no fabricated updates were added.

Step28 committed00:57:18 UTC: loss0.558996489, reward0.903571429, success24/28,
43 actions with1 parse error and0 schema errors, wall1193.394s, rollout1163.046s,
gradient tail27.063s. At01:06 UTC Step29 was collecting21/28 artifacts and
contributions with no abort. Three non-warmup steps26–28 average2.63steps/hour,
about6.4% above the prior two-step estimate but below4.2. Remaining pure-training
ETA is84.4h; similar future quality-check costs add roughly18h. This is a
capacity-shortfall needing an owner decision; the owner's explicit continue
instruction remains in force. GPUs0/6/7 on22048 retain their existing roles;
no GPU task was launched or restarted by the telemetry update.

At01:13 UTC root independently read uncached W&B history: original optimizer
rows24–28 and supplemental telemetry rows24–28 each occur once and match the
local committed exports. The same run now receives all requested metric groups;
writer568833/568852 uses the new aggregate input, same run/state, no new training
run. Step28 action admission is42/43, execution return validity18/18; neither
is the24/28 native task success count. Its declared step reservation is0.9945
GPU-hours. The latest process-inclusive reservation is7.5638GPU-hours, a
different overlapping accounting scope. No training or scoring configuration
changed, and telemetry contains no generated answers or licensed task text.

At2026-09-11 01:27 UTC root verified Step29 COMPLETE and both unique original
and supplemental W&B rows24–29. Step29 committed01:25:47 UTC: loss0.735229323,
reward0.614935065, success16/28;113 actions,10 parse errors and1 schema error,
102 admitted,78 executed and78 execution-success returns. Unlike steps24–28,
this step contains6 actual credited skill invocations and6 posterior updates;
one phase check, no trigger/mutation. Wall1709.176s, rollout1355.385s, gradient
tail350.274s. The four non-warmup steps now average2.48steps/hour (about5.9%
below the preceding estimate), remaining pure-training ETA89.2h plus roughly18h
if future quality-check costs persist. Capacity warning does not override the
owner's continue instruction. Step30 collection subsequently reached10/28;
all existing training/model roles remain unchanged, with no new GPU processes.
This verifies new-step telemetry delivery after historical backfill, not only
static upload of old records. No code or checks were repeated for this note.

At2026-09-11 01:40 UTC root verified Step30 COMPLETE and uncached W&B original
and supplemental rows24–30, each unique. Committed01:36:09 UTC: loss1.156707566,
reward0.635834334, success17/28;71 actions,1 parse error,0 schema errors. No
credited skill calls/posterior updates this step; one phase check, no trigger
or mutation. Wall621.224s, rollout610.956s, gradient tail5.774s. The scheduled
Step30 quality attempt is collecting, not another optimizer update or a stalled
commit. Five non-warmup steps average2.80steps/hour, up12.9% from the previous
estimate but still below4.2; remaining pure-training ETA78.7h plus quality and
other costs (roughly18h if prior check durations persist). Training, serving,
CPU exporter and W&B writer roles remain unchanged; no new tasks, code changes,
or repeated checks for this note. Owner's continue instruction remains active.

At2026-09-11 02:09 UTC root verified Step30 held-out quality completion and its
W&B summary in the same writer:28/28 collected, success19/28, reward0.703392260,
first-action validity27/28, action/admission validity123/125. Domain successes
(out of4 independent sources each): Hotpot4, Trivia2, AIME4, HealthBench1,
ALFWorld0, MBPP+4, HumanEval4. Decision verified/continue with no triggered rules
or unavailable metrics; ALFWorld0/4 remains a limitation, not a domain-level
pass. Probe persisted02:05:44 UTC, about29.6min after Step30 commit. Step31 had
already resumed15/28 by02:08 without a restart. At the last training-rate
estimate, remaining work is78.7h plus about21.7h if44 further quality checks
match this duration (roughly100h, substantial uncertainty, not a guarantee).
All existing roles/PIDs remain live; no code, GPU tasks, or tests were changed.

At2026-09-11 02:20 UTC root verified Step31 COMPLETE and unique W&B original
and supplemental rows24–31. Committed02:19:30 UTC: loss0.593078486,
reward0.847893114, success22/28,73 actions with zero parse/schema errors;
no credited invocation/posterior update, one phase check, no trigger/mutation.
Wall826.067s, rollout804.659s, gradient tail17.921s. This individual step is
4.36steps/hour equivalent; six non-warmup steps average2.97steps/hour (up6.3%),
not a steady4.2 pass. Remaining pure-training ETA73.6h, roughly95h including
future checks at the latest duration, excluding unknown failures/evolution.
Step32 is collecting; no restart or configuration change followed quality30.
No new GPU task, source edit or test run; existing three-card roles continue.

At2026-09-11 02:39 UTC root verified Step32 COMPLETE and unique original plus
supplemental W&B rows24–32. Committed02:35:02 UTC: loss0.160971531,
reward0.883381924, success25/28;124 actions,4 parse errors and1 schema error.
No credited invocation/posterior update; one phase check, no trigger/mutation.
Wall932.211s, rollout853.358s, gradient tail75.288s. Seven non-warmup steps
average3.08steps/hour (up3.4% from the prior estimate); remaining pure-training
ETA70.9h, roughly93h including future checks at the last observed duration,
not a4.2-throughput pass or a72h guarantee. Step33 was collecting14/28 at02:38.
Existing owner/ranks/actor/exporter are live; storage has about3.3TiB available.
Selected GPUs0/6/7 remain allocated with about71.5/62.2/71.7GiB total device
memory occupied (including any other users), not new reservations. No task
restart, hardware adjustment, source change or repeated check was performed.

At2026-09-11 03:41 UTC root verified Step33 COMPLETE and unique original plus
supplemental W&B rows24–33. Committed03:40:47 UTC: loss0.240732617,
reward0.763540031, success21/28;77 actions,20 parse errors,0 schema errors;
no credited invocation/posterior update, one phase check, no trigger/mutation.
Wall3944.361s (65.74min), rollout3585.023s, gradient tail356.263s. The last AIME
trajectory reached the declared8-turn limit; observed earlier rounds hit
16384 reasoning/2048 action token caps and returned non-structural actions.
Requests advanced normally, then full8-turn scoring/offload completed; no
trajectory was omitted and no budget or method changed. Eight non-warmup
steps average2.37steps/hour, down22.9%; remaining pure-training ETA91.5h,
roughly113h with future checks at the latest duration. This capacity shortfall
is material; owner's explicit continue instruction remains in force. Step34
started without restart. Existing GPUs0/6/7 and all owners remain live; no new
GPU task, source change, hardware adjustment or repeated test was performed.

### 2026-09-11 04:03 UTC — Step 34 committed; Step 35 collecting

- Root read back the complete Step 34 checkpoint (04:00:15 UTC) and W&B original/telemetry histories: each contains steps 24–34 once. TTB loss 0.57148546, reward 0.65254237, terminal successes 18/28; 3 parse errors, 0 schema-invalid actions among 126 actions. Admission fraction 97.619%; no skill calls, posterior updates, or library mutation this step.
- Step wall 1,167.88 s (19.46 min), rollout 1,130.12 s, gradient tail 34.73 s. Nine non-warmup steps 26–34 average 2.435 steps/hour, up 2.63% from the preceding cumulative observation. Remaining 216 steps: approximately 88.7 training hours, or roughly 110 hours including future quality-check overhead at the last observed duration; not a guarantee. Below-target throughput remains a human-decision warning; the owner's explicit continue instruction remains in force.
- Live owner, both gradient ranks, actor scheduler, and telemetry exporter verified; Step 35 at 17/28 artifacts and contributions, no abort. Existing three-GPU allocation unchanged; no new process, restart, sampling/configuration change, or test run.

### 2026-09-11 04:20 UTC — Step 35 committed; scheduled quality collection active

- Root verified Step 35 COMPLETE (04:18:00 UTC), all 28 artifacts/contributions/canonical merges, and uncached W&B original plus telemetry ACKs for steps 24–35 exactly once. TTB loss 0.68616687, reward 0.71666667, binary successes 19/28. Among 108 actions: 8 parse errors, no schema-invalid actions, structure/admission validity 92.593%; execution returned success for 76/76 dispatched calls (not a task-success measure). No skill calls/posterior updates/library mutation this step.
- Wall 1,065.09 s (17.75 min), rollout 1,014.83 s, gradient tail 47.23 s. Ten non-warmup steps 26–35 average 2.505 steps/hour, up 2.88% from the preceding cumulative estimate; remaining 215 steps approximately 85.8 pure-training hours, roughly 107.5 hours with estimated quality-check overhead. Throughput remains below target; owner-directed continuation unchanged.
- Scheduled Step 35 quality collection was 9/28, separate from training evidence; no quality result claimed yet. Owner, both gradient ranks, actor scheduler, and telemetry exporter live. Three-GPU allocation unchanged; no new resident/GPU process or restart. No code/config edits, tests, or hash checks during this monitoring interval.

### 2026-09-11 04:42 UTC — Quality 35 finished; Step 36 resumed automatically

- The separate 28-source quality panel completed around 04:38:26 UTC, approximately 20.4 minutes after the Step 35 commit. Native aggregate reward 0.77750017; successes 20/28; first-action structure valid 26/28 and all-action structure/admission valid 106/108. Domain successes (four sources each): HotpotQA 4, TriviaQA 2, AIME 4, HealthBench 0, ALFWorld 2, MBPP+ 4, HumanEval 4. These are diagnostic results, not additional training updates.
- Persisted decision is warning/continue for first-turn structural validity. Root read back the same decision and success/action aggregates from W&B. Failures remain reported; no threshold or training condition was changed. The unchanged owner/ranks/actor are live and Step 36 is collecting 19/28 artifacts with 18/28 gradient contributions, no abort. No restart or new process was needed.

### 2026-09-11 04:52 UTC — Step 36 complete; Step 37 active

- Root verified Step 36 COMPLETE (04:50:40 UTC), unchanged live training processes, and uncached W&B original/telemetry histories containing steps 24–36 once. Loss 0.20599036, reward 0.75770308, successes 21/28. Structure/admission validity 109/119 (91.597%); 10 parse errors, 0 schema-invalid actions. No skill calls, posterior updates, or library mutation this step.
- Step wall 733.99 s (12.23 min), rollout 726.06 s, gradient tail 5.00 s. This individual step exceeded the 4.2 steps/hour target, but the complete non-warmup average remains 2.622 steps/hour (+4.65% versus the preceding cumulative estimate), not a demonstrated sustained target rate. Remaining 214 steps approximately 81.6 pure-training hours, roughly 103 hours including estimated future quality overhead. Continue per owner instruction; all slow steps remain included.
- Step 37 collection has started. Same three physical GPUs and process roles; no new resident process, restart, code/config change, tests, or hash checks.

### 2026-09-11 05:30 UTC — Step 37 complete; Step 38 collecting

- Root verified Step 37 COMPLETE (05:26:20 UTC) and uncached W&B original/telemetry ACKs for steps 24–37 once each. TTB loss 0.86731501, reward 0.63780488, success 13/28. Among 87 actions: 8 parse errors, 2 schema-invalid actions; structural validity 88.506%. No skill calls, posterior updates, or library mutation this step.
- Wall 2,140.38 s (35.67 min), rollout 2,073.55 s, gradient tail 63.11 s. The last AIME trajectory advanced through round 3 before completing; all 28 samples were retained. Twelve non-warmup steps 26–37 average 2.505 steps/hour, down approximately 4.45% from the preceding cumulative rate. Remaining 213 steps approximately 85.0 pure-training hours; approximately 99–107 hours if future quality checks each resemble the observed 20–30 minute range, excluding other unmeasured future overhead. Below-target warning remains subject to the owner's already explicit instruction to continue.
- Owner, both gradient ranks, actor scheduler, and exporter live. Step 38 has 18/28 artifacts/contributions, no abort. Same three-card allocation, no restart/new process/configuration change. Earlier 05:15 GPU check found no active hardware/software thermal-slowdown flags on the three assigned devices. No tests or hash checks were needed for this monitoring-only interval.

### 2026-09-11 05:45 UTC — Step 38 complete; five genuine skill updates

- Root verified Step 38 COMPLETE (05:42:56 UTC), unchanged live owner/ranks/actor/exporter, and uncached W&B original/telemetry ACKs for steps 24–38 once. Loss 0.89442281, reward 0.82142857, successes 23/28. Among 91 actions: 3 parse errors and 1 schema-invalid action, structural validity 95.604%. Five actual skill invocations generated five posterior update events; no library mutation. This does not establish causal skill benefit or natural evolution.
- Wall 995.53 s (16.59 min), rollout 975.81 s, gradient tail 16.75 s. Thirteen non-warmup steps 26–38 average 2.566 steps/hour (+2.42% versus the prior cumulative estimate); remaining 212 steps roughly 82.6 pure-training hours, or 97–104 hours including estimated 20–30 minute future quality checks, not a guarantee. Continue under the owner's existing below-target authorization.
- Step 39 collecting 13/28 artifacts and contributions without abort. Local mirror and sole W&B writer remain live, with 15 committed rows mirrored and completed quality panels 25/30/35. Existing three-GPU roles unchanged, no new processes, restart, source/config edits, tests, or hash checks.

### 2026-09-11 06:12 UTC — Step 39 complete; Step 40 collecting

- Root verified Step 39 COMPLETE (06:06:57 UTC) and uncached W&B original/telemetry histories containing steps 24–39 once. Loss 1.16542822, reward 0.59863946, successes 15/28; 9 parse errors and no schema-invalid actions among 106 actions (structural validity 91.509%). No skill calls, posterior updates, or mutation this step.
- Wall 1,441.23 s (24.02 min), rollout 1,386.53 s, gradient tail 51.05 s. Fourteen non-warmup steps average 2.561 steps/hour, approximately 0.19% below the preceding cumulative rate. Remaining 211 steps approximately 82.4 pure-training hours; roughly 97–104 hours including future 20–30 minute scheduled checks, excluding other overhead. Below-target continuation remains explicitly authorized by the owner.
- Existing owner/ranks/actor/exporter verified live. Step 40 at 24/28 artifacts and gradient contributions; four AIME requests still generating, no abort. Same three GPUs, no new process/restart/config edits, tests, or hash checks. The next scheduled quality panel is separate from the Step 40 training transaction.

### 2026-09-11 06:18 UTC — Step 40 checkpoint and W&B confirmed

- Root verified Step 40 COMPLETE (06:16:32 UTC), 28/28 artifacts/contributions/canonical merges, and W&B original/telemetry histories for steps 24–40 once each. Loss 0.06960188, reward 0.91829004, successes 24/28. Structural validity 40/42 (95.238%), 2 parse errors, 0 schema-invalid actions; no skill calls, posterior updates, or library mutation this step.
- Wall 574.66 s (9.58 min), rollout 565.62 s, gradient tail 4.28 s. Fifteen non-warmup steps average 2.666 steps/hour (+4.10% versus the preceding cumulative observation); remaining 210 steps about 78.8 pure-training hours. Future quality checks at the observed 20–30 minute range give roughly 93–100 hours including the current pending check, excluding other future overhead. This fast individual step does not establish sustained target throughput; all prior slow steps remain counted and owner-directed continuation unchanged.
- Scheduled quality panel 40 is collecting separately, 7/28 complete at the root check; no quality result claimed yet. Owner, both training ranks, actor scheduler and exporter live. Existing three-GPU deployment unchanged; no new process, restart, source/config modification, tests, or hash checks.

### 2026-09-11 06:55 UTC — Quality 40 warning recorded; Step 41 resumed

- Separate 28-source quality collection persisted at 06:52:53 UTC, about 36.3 minutes after the Step 40 commit. Reward 0.69828772, successes 17/28; first-turn structure valid 27/28; total structure/admission valid 103/109 (94.495%). Domain successes: HotpotQA 3/4, TriviaQA 1/4, AIME 4/4, HealthBench 0/4, ALFWorld 1/4, MBPP+ 4/4, HumanEval 4/4. Preserve these failures and do not reinterpret execution validity as answer correctness.
- Decision warning/continue triggered action-structure and admission fractions. Root confirmed W&B quality40 decision, success fractions and validity values. The unchanged training owner/ranks/actor are live; Step 41 at 12/28 artifacts and gradient contributions without abort. No restart, configuration change, new process, tests, or hash checks.
- Committed-step throughput is unchanged at 2.666 steps/hour; remaining pure-training estimate remains 78.8 hours. This longer quality panel widens a simple future-check allowance: approximately 93–104 hours with 20–36 minute checks, excluding other overhead. Below-target continuation remains the owner's explicit decision; no slow samples/checks are omitted from recorded wall times.

### 2026-09-11 07:07 UTC — Step 41 checkpoint and upload confirmed

- Root verified Step 41 COMPLETE (07:06:10 UTC) and uncached W&B original/telemetry ACKs for steps 24–41 once each. Loss 0.38045091, reward 0.73373016, successes 19/28. Structural validity 101/107 (94.393%), 5 parse errors and 1 schema-invalid action. No skill calls/posterior updates/library mutation this step.
- Wall 796.92 s (13.28 min), rollout 779.92 s, gradient tail 14.00 s. Sixteen non-warmup steps average 2.736 steps/hour (+2.63% versus the preceding cumulative rate); remaining 209 steps about 76.4 pure-training hours, roughly 90–102 hours including future 20–36 minute quality checks, excluding other overhead. Cumulative throughput remains below target and owner-authorized continuation unchanged.
- Existing owner, two gradient ranks, actor scheduler and exporter remain live. The sampled stream sidecar still showed the preceding drain stage while the authoritative complete checkpoint and committed metrics had advanced; it is not an abort/deadlock signal. Three-GPU roles unchanged; no new process, restart, configuration change, tests, or hash checks.

### 2026-09-11 07:29 UTC — Step 42 committed; Step 43 active

- Root verified Step 42 COMPLETE (07:25:53 UTC), unchanged live owner/ranks/actor/exporter, and uncached W&B original/telemetry ACKs for steps 24–42 once each. Loss 1.38651501, reward 0.65616246, successes 18/28. Structural validity 62/69 (89.855%); 7 parse errors, no schema-invalid actions; no skill calls, posterior updates, or library mutation this step.
- Wall 1,183.26 s (19.72 min), rollout 1,155.20 s, gradient tail 25.00 s. Seventeen non-warmup steps average 2.752 steps/hour (+0.60% versus the preceding cumulative rate); remaining 208 steps approximately 75.6 pure-training hours, roughly 90–101 hours with future 20–36 minute quality checks. These estimates exclude other future overhead and remain below the planning target; owner-directed continuation unchanged.
- Step 43 collecting 20/28 artifacts and gradient contributions without abort. Three-GPU allocation unchanged; no new process, restart, source/config edits, tests, or hash checks. The 07:13 device check found no active thermal-slowdown flags and about 3.3 TiB free on the run volume; a high gradient-memory sample was observed, not an OOM or a reason to disturb other processes.


### 2026-09-11 08:14 UTC — Training-host observation unavailable

- Root last verified the training owner/ranks/actor live at 08:01 UTC: Step 43 collecting 27/28, with the final ALFWorld trajectory advancing at round 20. Last complete checkpoint and W&B ACK verified are Step 42.
- Subsequent strict SSH attempts to approved endpoint 22048 time out during banner exchange. Endpoint 22049 remains reachable; a read-only CPU TCP probe from that host connects to 22048 but receives no SSH banner within eight seconds. This establishes an observation/connectivity problem, not training termination. Host/SSH-service investigation needs operator attention; avoid restarting the machine before preserving any live training state.
- The local mirror reports fetch timeout; existing mirror and sole uploader processes remain live, preserving previously downloaded records. Only one stuck local read-only SSH observer was terminated; no training, actor, gradient worker, or other user process was stopped. No new GPU or resident process was started. Current GPU execution status is unknown. No source/config edits, tests, or hash checks.
- Last committed non-warmup rate remains 2.752 steps/hour; the prior approximately 75.6-hour pure-training ETA excludes this unresolved interval and is no longer a fresh wall-clock forecast. The monitoring goal remains unfinished.
