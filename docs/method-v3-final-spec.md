# SKILLEV Protocol v3 final engineering specification

This document is the current cross-layer engineering authority. `idea.tex`
remains the immutable scientific authority. Phase goals are historical
implementation records; their superseded execution rules are listed in
[`protocol-v3-errata.md`](protocol-v3-errata.md). Protocol v3 names the method,
while exact runtime/checkpoint wire records use version `@6`.

## 1. Formula identities

- History is `H_t = H_{t-1} ⊕ (r_t, a_t, o_t^{exec})`.
- Forward and hindsight policies score the same recorded action token span and
  the same `K_t`.
- `P̂_F` and `P̂_B` are natural-log per-token means.
- `Δ = log Z(q) + ΣP̂_F − β log(R + ε_min) − ΣP̂_B`.
- Training loss is `(Δ/T)²`; phase stagnation uses raw `Δ²`.
- `log I(t) = P̂_F(t) − P̂_B(t)`, and state-flow accumulation stays in log
  space without clipping in the full method.
- Bayesian cells use the exact four coordinates from the contracts and
  uncapped mean-normalized flow weights.
- A phase requires both residual stagnation and strictly decreasing skill-use
  entropy.
- Φ is the closed nonempty set Retain/Compress, Refine, Split, Prune, and
  Generate. Every proposal is executed or the attempt fails.

## 2. Literal full-core configuration

- rollout sampler: raw categorical softmax;
- policy temperature/top-p identity: `1/1`;
- importance clip: absent;
- flow-weight cap: absent;
- gradient clip: absent;
- AdamW weight decay: `0.0`;
- posterior decision: confidence bounds with configured `k`;
- phase rule: residual AND entropy;
- one preregistered experimental seed.

No method control is inferred from missing attributes, `None` fallbacks,
exception handling, or result-dependent behavior.

## 3. Scientific source model

The four scientific source events are:

```text
LIBRARY_INITIALIZED
TRAINING_STEP_COMMITTED
EVOLUTION_PHASE_OPENED
EVOLUTION_CYCLE_COMMITTED
```

`TRAINING_STEP_COMMITTED` contains the exact admitted records, edge scores,
TTB stats, posterior batch, and report. Derived diagnostics and posterior
states are not events and are not recovery inputs. `RUN_ATTEMPT_FAILED` may
carry a public failure summary but has no scientific or recovery authority.

Every source training and evolution-cycle payload carries the run cursor after
the represented transition. A successful publication additionally binds the
exact input, builder kind, public method identity, full ordered source-log set,
and tagged success outcome. The source logs are the only audit inputs; a
failed or incomplete child remains quarantined rather than becoming an audit
bundle.

## 4. Failure semantics

- Fixed B tasks are planned before collection.
- Each task is attempted exactly once.
- Reward-zero and invalid agent actions remain data.
- Any infrastructure failure ends the current attempt.
- There is no replacement task, retry, repair, fallback reader, or partial
  success.
- Projection state is previewed before the source append and published only
  after that append succeeds.
- A verified phase must yield a complete nonempty decision. Budget is
  preflighted for the entire decision before authoring.
- Authoring, mutation, reset, library apply, checkpoint, or cycle-source
  failure ends the attempt; active code does not rollback or replay.

## 5. Active package graph

```text
contracts → policy → scoring → rollout → training
                                  ↓
                       diagnostics + calibration
                                  ↓
                              evolution
                                  ↓
                         application builder
```

`SKILLEVApplication` constructs one explicit object graph. Training and
evolution drivers are real production entrypoints, not script-only assembly.
The active graph never imports `skillev.audit`.

## 6. Exact finite run plan and evolution coordinates

An attempt freezes positive `phase_search_steps`, positive `closure_steps`, a
maximum cycle count, and at least one required post-cycle closure step before
task execution. The detector is consumed only in the phase-search slots. A
phase in the last search slot is resolved in that same child, and the remaining
closure slot(s) then commit ordinary training under the new library and
reset-Z lineage. No task or budget is appended after observing a phase.

Protocol-v3 Context is `TrajectoryRecord.task_family`. Consequently, Split
partitions the complete finite expansion of `SkillApplicability.task_families`
against the attempt's frozen task-family universe. Every source family needs a
qualified posterior mode and must fall wholly on one side of the credible-
interval cutpoint. The two children are nonempty, disjoint, and exhaustive;
they preserve the separate rollout `context_id` axis
(`applicability.contexts`), required tools, and excluded contexts exactly.

Generate uses `|log I|`, the absolute log-density ratio. A zero-coverage edge
is eligible only when it meets both the explicit positive absolute floor and
the strict window-relative upper-tail percentile. Equal values do not raise
one another's percentile. Terminal submission edges are excluded from the
population; tool edges remain eligible.

The public application identity fixes these method semantics exactly:

```text
Context feature          namespaced-task-family@1
Calibration outcome      trajectory-terminal-success@1
Flow normalization       per-batch-invoking-edge-mean@1
Skill credit             explicit-skill-action-in-h0@1
Phase detection scope    phase-search-slots-only@1
Initial library          non-empty-seeded-library@1
State-flow estimator     single-observed-history-prefix@1
```

Thus every invocation on a trajectory shares its terminal success label;
non-invoking edges do not enter the flow-weight denominator; only an admitted
structured skill action earns skill credit; closure slots train normally but
cannot open another phase; and the method is seeded skill-library
self-evolution rather than zero-library bootstrap. A materialized state-flow
value is one observed history-prefix sample, not a claim that equivalent
states were integrated out.

## 7. Ablation graph

Six alternate implementations live only in `skillev.experiments.arms`:

- no Bayesian calibration;
- unit flow weights;
- capped flow weights;
- clipped importance;
- posterior-mean decisions;
- residual-only phase detection.

Each arm has an explicit builder. There is no arm-name switch or target-axis
branch in the full core. The no-Bayesian arm additionally owns its projection,
training source, event envelope, decision, executor, cycle result, and outer
loop; it never emits a full posterior placeholder or full evolution cycle. All
builders take the same closed application-input envelope; arm-result comparison
requires exact equality of the ordered training task IDs and resource identity.
The capped-flow and clipped-importance builders obtain their fixed bounds from
their tagged protocol variants rather than caller-supplied runtime controls.

Every rollout action has one sampled model-content token span. That exact span
is authoritative for `K_t` and for both forward and hindsight teacher-forced
scoring. Its deterministic tokenizer decode is the action text used by JSON
parsing, environment execution, public history, and artifact serialization.
Artifact admission checks `decode(recorded_action_token_ids) == action_text`;
it never re-encodes text to replace the sampled segmentation. Tokenizer decode
may be many-to-one, so inverse encode/decode identity is not a protocol rule.

## 8. Checkpoint construction boundary

`RuntimeSnapshot@6` stores one `RuntimeExecutionState@6` using fixed artifact
paths. It binds the builder, configuration, method, protocol/freeze, run-plan,
initial-library, ordered-task, and public-identity hashes to the exact run
cursor, task cursor, immutable library, typed projection state, and typed
detector state. A snapshot is written only for a fully committed training
boundary. Construction from a checkpoint takes one exact directory and a
fresh application graph. It does not scan event logs, locate “latest”, accept
old schemas, expose a live task rewind mutator, or recover a half-completed
phase. Live snapshot code never cleans or prunes; those operations belong to
explicit offline maintenance.

## 9. Offline audit and formal benchmark boundary

`skillev.audit.audit_published_attempt` selects the sole audit kernel from the
published identity. It independently recomputes diagnostics, calibration,
phase detection, ordered Phi proposals, action/document provenance, library
lineage, Z reset/closure, budget settlements, and the final tagged summary.
The seven arms have typed audit paths; no-Bayesian uses its own hashed source
stream and each full-shaped single-axis arm uses its preregistered alternate
kernel.

The diagnostic value called `sample_log_state_weight` is the single-trajectory
estimator of equation (10): model-visible text histories effectively do not
collide, so the conditional expectation at an observed `H_t` is represented by
the one sampled path through that history. Offline audit verifies every
scientific transformation downstream of the materialized per-edge forward and
backward log-probabilities. It does not rerun the model to rederive those
log-probabilities; their provenance is instead bound to the exact checkpoint,
implementation-build identity, execution-hardware identity, tokenizer, and
recorded prefix/action-token identities.

Formal benchmark training is built only by the private fixed benchmark worker
from a private frozen curriculum. Public `ExactCompletionCase` inputs are
correctness and real-backbone vertical fixtures, not admissible primary
benchmark environments.

## 10. Validation gates

Before primary benchmark execution:

1. related behavior tests and one full `make check` pass in the Linux/WSL
   execution environment required by the POSIX sandbox and atomic-publication
   implementation;
2. public and private wheels build and the public-wheel boundary passes;
3. **Gate 4b (tiny backbone)** performs three training steps, fires the real
   detector, completes exactly one Φ cycle and library mutation, resets Z
   deterministically, and continues under the new library.  This is the
   end-to-end detector→Φ→mutation→reset→continue proof;
4. **Gate 4a (real Qwen3.5-9B)** separately proves real generation, exact
   preservation of the sampled action content span with a nonempty
   deterministic text projection,
   teacher-forced finite forward/backward scores with forward-adapter,
   backward-adapter, and Z-head gradients, one optimizer step, bit-identical
   checkpoint save/load, model/tokenizer/source/hardware identity, and
   adapter-disabled structured `generate_base` authoring for Retain, Refine,
   Split, and Generate.  It accepts and classifies whatever rollout action the
   fixed protocol seed emits: no exact action predicate and no seed search;
   the `@7` gate binds the executing source tree/commit and frozen-base bytes,
   requires four unique settled authoring calls, and passes only when all four
   structured authoring results validate.  Private diagnostics retain only
   cause chains and token/output hashes, never generated text;
5. **Gate 4c `@6` (real Qwen3.5-9B production shape)** proves the fixed
   near-cap synthetic streaming TTB optimizer step plus stateless and cached
   15-turn rollouts. The real rollouts must preserve each generator content
   span in the trajectory, restore their artifacts exactly, agree across cache
   modes, and stay within the frozen input and memory caps;
6. every arm's single-axis identity is tested;
7. Protocol-v3 JSON and freeze contain zero prior results.

Retain/Compress measures the complete canonical skill content that is shown in
rollout `H_0` (title, summary, instructions, applicability, and requirements),
not the instructions field in isolation.  Applicability and requirement
identities remain fixed and the authored representation must use strictly fewer
tokens under the pinned tokenizer.  If a source is already at the minimum legal
representation, Retain is infeasible before generation and the closed attempt
does not emit a Retain proposal or shift content into an uncounted field.  If
no other frozen threshold admits a proposal, the phase commits the explicit
verified no-op described in `docs/IDEA_IMPLEMENTATION_SPEC.md`; a failed
authoring call is still an infrastructure failure and never becomes a no-op.

Every authored Retain, Refine, Split, or Generate document must additionally
fit the same 3,400-token cap for its complete canonical `H_0` skill block at
every reachable position 1--5.  The counted block includes `[position]`, skill
ID, version, content hash, the `content:` label, the complete model-visible
skill content, and all separators/newlines.  This is a sealed post-generation
validity rule before library mutation; there is no truncation, dropped skill,
retry, or alternate candidate.

Near the fixed model-input cap, checkpointed teacher-forced Qwen3.5 scoring
stores saved autograd tensors on CPU once the exact prefix-plus-action span
reaches 32,768 tokens. CUDA execution additionally requires the pinned
`flash-linear-attention==0.5.2` gated-delta kernel instead of Transformers'
high-memory PyTorch fallback. The recorded action IDs, logits, gradients,
optimizer step, horizon, and input limits remain unchanged; only the
kernel/storage implementation changes to keep the production shape within the
fixed H800 reserve limit.

Before Gate 4a `@7`, the Retain template and `AuthoringSamplingConfig` are
developed only against a fixed public synthetic development suite covering
short/medium/long instructions, few/many requirements, one/multiple task
families, with/without tools, repetitive/low-redundancy material, and a
near-minimal legal source.  Once frozen, the disjoint public synthetic holdout
suite is executed exactly once per case with the same model, tokenizer,
template, sampling, hardware class, and fixed per-case seeds; no retry or
candidate selection is allowed, and any rejection blocks protocol freeze.
Every Retain prompt names the exact tokenizer identity, source canonical token
count, strict source-minus-one output bound, exact applicability, and exact
requirement IDs.  The validator independently recounts the complete canonical
draft.  The new full-block validator invalidates the old gate for protocol
freeze purposes, so Gate 4a `@7` uses one new source package and one
fixed-protocol-seed execution.  The consumed `@2` failure and superseded `@3`
through `@6` successes are never rerun.

## 11. Closed execution-state representation

A successful `EvolutionLoop.run(plan)` has no phase-state field. A phase
detected in any phase-search slot is resolved in that same call, followed by
the frozen closure tail; any failure invalidates the attempt rather than
publishing a resumable open phase. Private `PhaseOpenForensicSnapshot` values
may describe where a failed attempt stopped, but are not accepted by
application construction, training, evolution, or snapshot loaders.

Diagnostics and detector state each use explicit fresh/active tagged unions.
Full and flow-only projection and runtime states have disjoint tagged schemas.
No live code selects an algorithm based on missing history, empty tuples,
`None`, or a file search.

## 12. Attempt and budget boundary

The parent supervisor starts one child worker exactly once. Only canonical
`AttemptSucceeded` or `AttemptFailed` IPC values cross back; mutable scientific
objects do not. The child is the only broad exception boundary for a complete
attempt.

Within that attempt each rollout, tool, and frozen-base authoring call has one strict budget transition:
`reserve` followed by exact `settle`. There is no release, abort, unknown-use
state, reconciliation receipt, retry scope, or idempotent duplicate. Fixed
attempt capacity is validated before work begins.
