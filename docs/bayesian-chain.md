# Bayesian training chain

`idea.tex` remains the scientific authority and is unchanged. This note fixes the
engineering semantics of the existing full application; it is not a new training
method or an evaluation/answer-selection path.

## One batch, one commit

A complete batch is sampled under one forward-policy snapshot and one active
library. Its TTB gradient computation emits paired forward/backward scores for the
same recorded action tokens and the same token-count denominator. Each training
edge records batch, trajectory, step, policy snapshot, library and adapter versions.
The complete coordinator join rejects missing/duplicate edges, mixed snapshots or
adapters, different action denominators or exact token spans (including equal-length
substitutions), and residuals built from other edge scores. Residual order must match
the sealed trajectory order; worker completion order is not statistical order.
Workers may compute disjoint contributions early, but they never normalize posterior
weights or make evolution decisions. Diagnostic/calibration projection happens only
after full-batch coverage is available.

The order is: complete batch -> TTB residual/edge scores -> diagnostics -> weighted
posterior -> confidence queries -> joint phase condition -> Phi -> coordinated
library/Z/projection/detector state -> durable checkpoint and adapter publication ->
source events -> next batch. The optimizer is applied once after projection preview;
the wider step transaction must finish before another batch can be collected.

Fresh construction and public resume emit `method_state_recorded` with the parsed
`ApplicationConfig`, extractor and authoring-template versions, forward/backward/Z
versions, library/active IDs and posterior cursor. The report names query, evolution and split confidence
controls separately through their actual config sections. Historical YAML is not an
input to this typed composition root. This telemetry is not decision evidence.

- `TerminalReward.native_payload`: private, log-only native metrics.
- `TerminalReward.value`: continuous R in [0,1]; epsilon shifting is only for TTB.
- `TerminalReward.success`: explicit binary Y for Beta updates. It is not inferred
  from R, native score, or observation status by calibration.
- `sample_log_state_weight`: cumulative single-path log importance, not an already
  conditionally aggregated F(H_t). Skill flow pools these samples with log-sum-exp
  and divides by the number of invoking trajectories, not invocation count.
- Posterior weights normalize all invoking edges in one complete batch to mean one.
  Retrieval, prompt inclusion and a proposed call do not create invocation evidence.
  Non-finite flow values fail with source coordinates, without clipping. At linear
  conversion, log normalization recovers representable subnormals; a normalized
  weight that still underflows to zero rejects the whole preview with its edge
  coordinates. It is never replaced by zero, uniform weights or a partial batch.

## State and confidence

The immutable projection snapshot owns the posterior. A bound `CalibrationEngine`
reads that snapshot for every query and cannot independently commit cells. Preview
transitions carry a base revision; stale commits and stale resets are rejected.

The private provenance journal retains complete ordered posterior batches, including
empty-invocation batches, trajectory identities, scoring policy/library, and each
update's binary outcome, weight and before/after counts. Duplicate batch or trajectory
submission rejects the whole batch, including after restore; it never renormalizes a
remaining subset. Four distinct trajectories still contribute four observations.
Restore checks the actual weighted arithmetic and cell cursor, not only event counts.
The new projection snapshot format is v7; old ID-only snapshots are not silently
upgraded. They require their original source evidence and an explicit migration, or
a fresh run. No historical run has been migrated or resumed by this change.

Features are post-hoc: namespaced task family; edge observation status; H0 token
bucket; final trajectory horizon bucket. Final horizon is not available for pre-rollout
queries. Prior settings and extractor semantics remain unchanged. Calibration and
Phi now use the same cell-query arithmetic. Ordinary queries use
`calibration.default_k`; actual action decisions explicitly use `evolution.k`;
split-mode comparisons explicitly use `evolution.split.confidence_k`. These may differ
by declared configuration, not by a hidden fallback. Queries expose both actual
event count and weighted evidence mass; mass is not an independent sample count.
Unobserved skills remain unobserved, not failures. No extra evidence threshold is added.
Resolved-state reporting also derives distinct trajectory counts from the existing
provenance; multiple invocations in one trajectory do not become independent samples.
`prequential_calibration` groups by scoring policy, library, skill and post-hoc cell.
Every prediction within a batch uses that cell's **pre-batch** alpha/beta, never the
rolling before-value after earlier same-batch invocations. It reports unweighted
invocation Brier error, probability bins, distinct trajectories and flow mass separately.
These descriptive diagnostics neither prove causal credit nor supply pre-execution features.

### Estimator and hand calculation

The state in `idea.tex` is the complete history, not a question or task family. With
fixed forward/backward parameters, library, tokenization and deterministic scoring,
an identical complete prefix determines its local ratios. Conditioning on that exact
history can therefore give the same value as its prefix sample. We do **not** merge
different histories into a coarse state or claim a multi-path conditional estimator.
Across policy versions, windows pool batch-level empirical statistics and the posterior
retains declared historical evidence; neither is advertised as one fixed-policy flow.

Example: one trajectory invokes a skill at prefix weights 2 and 6 (ratios 2 then 3);
another invokes it at weight 4. Skill marginal flow is `(2+6+4)/2 = 6`, with two invoking
trajectories as denominator. Batch posterior weights are `(0.5,1.5,1)`. If the first
trajectory succeeds and the second fails, prior `(1,1)` becomes `(3,2)`. Tests use these
values directly; no coarse question-level aggregation or shard-local normalization.

## Phase and skill changes

The joint condition still uses windowed **delta squared**, not (delta/T) squared,
and a strictly decreasing invocation-entropy history. Windows never cross library
segments. No forced periodic evolution or threshold relaxation is introduced.

Method semantics **v4 explicitly changes** the former permanent no-op closure.
`DETECTED` -> `NO_OP_WAITING` records phase ID, batch, step and reason, within its
library segment. It consumes that projection evidence cutoff, not the entire library.
Re-evaluation requires `max(2 * residual_window, entropy_window + consecutive_drops)`
new observed committed batches, all strictly after the cutoff. Only then are the
unchanged joint criteria evaluated again. This is a fresh comparison, not immediate
re-authoring on overlapping old evidence. Cursor and cutoff are durable; duplicate
diagnostics are rejected after restore as before. No-op resets neither posterior,
diagnostics, library nor Z. A real mutation still opens a new library segment.
With zero invocation coverage, a zero-entropy history cannot satisfy strict decline.
`phase_detection_recorded` explains the unmet condition after the step commits; no
special Generate bootstrap, fabricated invocation or threshold change is introduced.
Closure slots explicitly report `slot-does-not-allow-phase-detection` without advancing
the detector; every observed diagnostic is matched to the just-installed batch and step.
Reports include invocation fraction, per-skill counts, entropy evidence, residual
windows and uncovered high-importance edges. The latter are labeled batch telemetry,
not the Operator's phase-window candidate population. Zero/one-skill entropy cannot
strictly decline forever; no cold-start exception or forced invocation is introduced.

The short Protocol 13 debug default remains nonformal. Beyond its eight-step default,
an explicit `--run-plan` is required. The supplied `production-run-plan.json` declares
249 search steps, one closure step and at most two mutations, compatible with W=50.
All declared cycles receive phase checkpoints. `--phi-calls-per-cycle` explicitly
sets the per-cycle envelope (default 64); its token envelope uses the configured
authoring bounds, and total budget scales by cycle capacity. `resolved-run-plan.json`
records the parsed application, schedule, per-cycle and total budgets before training.
The cycle-limit error remains, so a plan is never silently expanded. This
configuration enables multiple cycles; it does not guarantee that either will occur.

Phi obtains diagnostics, cells, event provenance and authoring projections from one
immutable `freeze_for_phase()` read. Its library and evidence cutoff must match the
phase; later projection commits cannot change this decision's captured inputs. The
posterior and authoring mappings are read-only, not live references to a calculator.

Retain/compress, refine and split already produce previously unseen skill IDs;
generate also creates a new ID. Their cells start at the configured prior. No parent
evidence is copied to a modified skill or duplicated across split children. Unchanged
IDs retain their cells. Pruned/replaced documents and evidence remain historical,
while active retrieval excludes them. The retriever reads the current immutable
library on every call, so there is no independent retrieval cache to invalidate.
Authoring receives structured, answer-free training evidence and declared schema/tool
authority, not evaluator payloads, hidden tests or file access.

Refine evidence carries the actual selected weak cell's post-hoc coordinates as well
as its key, LCB, k and event IDs, through to the authoring request. A legacy decision
without those coordinates can be inspected but not silently authored with guessed context.
The wire schema registry is version 4.1 for this additional evidence field.
Requirement kinds now distinguish `immutable-constraint` from `evolvable-strategy`.
Historical untyped requirements remain immutable with their original wire identity;
they are not retroactively reclassified. New advisory seeds have new IDs and explicitly
tag their strategies. Retain preserves all structured requirements. Refine/Split preserve
immutable requirements and applicability/tool/schema contracts, but may correct or drop
strategies. Each changed/removed strategy must be named in a new/changed strategy's
`replaces` list with an evidence-grounded `change_reason`; this is retained in the new
document and its original proposal/authoring journal. Authors cannot promote a new
strategy into immutable authority. Context patches and child modality strategies remain
evolvable. All products receive new skill IDs and no inherited posterior.
This **draft-acceptance change** is declared in semantics v4 and authoring template
versions. Text checks do not prove semantic equivalence
of free-form instructions or measured skill quality; no evaluator-based voting is added.
Draft validation lives in `evolution/authoring_validation.py`, separate from model calls.

Z resets only after a prepared real mutation. Its gradients and Adam state are
cleared; forward/backward LoRA optimizer state is retained. The action is effective
for collection at the **next optimizer step**, not retroactively for its trigger batch.
The cycle event links the triggering windows, action evidence, library transition,
reset seed/version, and training-step cursor.
Both sealed-batch and streaming gradient workers receive the coordinator's exact
forward/backward/Z lineage along with synchronized tensors. Inferring worker Z
identity from `optimizer_step - 1` is insufficient after a library reset or resume.
Synchronizing metadata invalidates policy-episode caches, not frozen Z features or
LoRA optimizer state.

## Failure and recovery

Any failed wider transaction stops that application instance; retry requires a fresh
restore. Authoring failure is not a successful/no-op phase. The existing journal
remains the recovery authority. All public construction paths install it by default;
omitting a custom journal does not disable transactions. Fresh construction persists
`initial-step-00000000`, including empty optimizer/posterior state and initial cursors,
so failure during the very first update has a full rollback point:

- Before checkpoint publication is recorded: restore the preceding committed step,
  preserve any orphan snapshots/staging directories for that failed step in an
  uncommitted archive, then retry from the saved task cursor. A saved phase cannot
  silently disappear if the retry's detector no longer produces that phase.
- After checkpoint publication is recorded: restore that complete model/optimizer,
  posterior, diagnostic/detector, library and cursor state; match it to the journal;
  restore the matching server adapter; publish only missing source events.

This also handles the filesystem-checkpoint / journal-write interruption gap.
No model update, posterior event or library mutation is re-applied when finishing a
durable transaction. Adapter publication is reconciled through the transaction's
revision, not guessed from whichever model is currently served.
The public `SKILLEVApplication.resume()` always reconciles its installed journal,
including local-only execution without a server adapter. A trained checkpoint missing
its original transaction record is rejected rather than interpreted as a fresh run.
Resume telemetry is emitted only after transaction reconciliation.

The step journal also owns private `authoring-step-*` records: original frozen phase,
proposal, request, received draft and measured call usage. Rollback reuses received
drafts and their original accounting without a second model call or duplicate budget
events. Unknown/failed responses stop for explicit resolution, never random re-authoring
or a fake no-op. Keep these records with the checkpoint root when recovering; copying
only LoRA files or an isolated snapshot is not a complete recovery bundle.

Optimizer checkpoints bind parameter names, shapes, dtypes, groups and ordering to
saved state IDs. Restore checks this binding and Adam moment shape/finiteness **before**
installing model or optimizer state. Legacy optimizer files without the semantic layout,
or application configurations before semantics v4, require an explicit migration using
original evidence; they are not silently upgraded. No historical run was migrated here.

Validation uses synthetic fixtures, including weighted/empty/duplicate evidence,
full versus shard normalization, confidence, no-op restore, skill identity,
Z/LoRA optimizer separation, first-step rollback, saved-draft reuse, missing-journal
rejection, named optimizer restore, and interruption at each commit phase. The opt-in
`tests/application/test_bayesian_cuda.py` runs three 32-trajectory synthetic batches
through the existing application with real Qwen3.5 BF16 scoring/backpropagation.
Its scripted calls, labels, author output and existing test thresholds are explicitly
fixtures. It is neither natural on-policy evolution nor a formal three-role run;
the adapter transport there is a test double. Production reward/threshold settings
are untouched. This is engineering validation, not benchmark or quality evidence.

`test_multicycle_loop` covers two controlled mutations, both phase checkpoints and
new-library steps, while `test_fresh_process_resume` compares an entirely new CPU
process/model's next update against uninterrupted execution. These are synthetic
component/transaction results, not real-policy natural evolution. Four real batches
with W=50 cannot reach the 100-batch residual eligibility point. Real-model authoring,
real adapter publication/resume and production-window natural evolution must be
reported separately, including non-trigger reasons rather than relaxed thresholds.

## Shared terminal scoring and product identity (September 2026 repair)

New MBPP training and clean IID both use the trusted-side `MBPPScorerProfile` in
`skillev_private.benchmarks.mbpp_scoring`, fixed EvalPlus source `26d6d00`, native
4s minimum / 4x reference timing, and independent complete Base/Plus lanes.
The dependency and isolated worker now agree on that source, not just its time
limits. Reference: [fixed official implementation](https://github.com/evalplus/evalplus/blob/26d6d00/evalplus/eval/__init__.py).
Request/verdict v2 requires the complete profile; absence, an incompatible reply,
or reference/infrastructure failure cannot become a Bernoulli failure. Native
candidate failures/timeouts remain real negative labels. Train and IID share the
same decoder and retain native lane status. Synthetic independent-process tests
cover passing, Base/Plus-only failure, a timing-sensitive candidate, process timeout,
and reference failure. They are scorer integration checks, not benchmark scores.

Protocol 13 requires `--mbpp-source-root`; `--mbpp-profile` optionally selects an
explicit custom timing condition. The resolved profile and public task-feature
mapping are copied into snapshot identity v7 and checked against the actual session
factory at application assembly/resume. Identity v6 remains readable with unknown
historical conditions; it is not silently promoted to the new condition. Old rewards,
weights and posterior cells are never rewritten. Continuing under changed scoring
or task features requires a separately declared run/condition, not exact resume.

The private evidence journal now also retains answer-free benchmark/source-question
coordinates and H0 exposure metadata. Reports distinguish matched/retrieved/visible
skills, invoking edges and trajectories, updated cells, distinct source questions,
and unknown historical coordinates. Grouped pre-batch Brier errors and weight
concentration are descriptive, not independent-sample counts, causal effects or
95% intervals. Phase detections/no-ops are counted from committed source events;
actual library mutations and proposal counts come from the committed run cursor.
Uncovered importance telemetry includes public task-family/context coordinates.

The shared `benchmark-public-task@1` mapping uses namespaced public task family and
split-independent `<benchmark>:task` context for new training and IID. It does not
supply tools or read answers/horizon/outcome. ALFWorld subtype remains unspecified
unless explicitly supplied as public metadata; no private game-path inference.
Explicit historical context overrides are exported, not hidden aliases. Retrieval
reports family/context/exclusion/tool mismatches using the very same predicate that
selects active documents; applicability is not widened to make a result look better.
Actual prepared-body vs packed-context exposure and explicit calls remain separate.

IID reads only forward weights and an immutable active library. Coordinator-only
provenance records checkpoint origin, actual mutation/proposal/posterior counts and
library versions; none enters the actor. A nonzero source step is not proof of a
mutation. Summary axes distinguish C0 (Step-0/initial), C1 (trained/same initial),
C2 (Step-0/frozen checkpoint library), and C3 (trained/corresponding checkpoint
library). Cross-checkpoint pairs are not silently labeled C3. Paired comparisons
still enforce fixed non-intervention controls: C0→C1 weights, C0→C2 and C1→C3 library,
C0→C3 combined. No new IID round, checkpoint selection or expected improvement is
implied by this reporting support.

Real verification remains separately named: complete on-policy optimizer/posterior/
adapter transactions; real author plus new-library continuation; fresh-process real
service restore; production-window natural evolution. The currently monitored
four-step run is frozen on its original source and legacy scoring condition, not
this repair. Four steps at W=50 cannot verify natural evolution. Non-triggering and
non-improvement are valid reported outcomes, not reasons to change thresholds.
