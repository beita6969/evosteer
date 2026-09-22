# Reuse and active-method boundary

`idea.tex` is the scientific authority. Reuse from the paused project is
limited to method-neutral engineering behavior; no FIRE-GFN scientific
mechanism or compatibility API is part of SKILLEV-new.

## Reused engineering behavior

- canonical JSON and stable content identities;
- bounded public execution and budget accounting;
- append-only operational logging;
- immutable skill documents;
- answer-isolated terminal evaluators;
- deterministic test environments, packaging, and offline build patterns.

Reuse means reimplementing and testing the needed invariant in the `skillev`
namespace. Old package names, intervention/Mirror-flow semantics, governance
gates, approval receipts, seals, and training-free source checks are retired.

## Protocol-v3 active graph (runtime wire `@6`)

```text
contracts → policy → scoring → rollout → training
                                  ↓
                       diagnostics + calibration
                                  ↓
                              evolution
                                  ↓
                         application builder
```

The sole scientific full-method composition root is `src/skillev/application.py`.
The historical GPU `run_training.py` path is outside this graph and is not a
BayesianImprove formal entrypoint; Protocol 10 formal execution remains blocked
until the validated GPU infrastructure is adapted to this core. Production H0
in the full-method graph uses `FullRetrievedSkillContext`: metadata plus the
complete immutable skill instructions. Metadata-only placeholders,
progressive-disclosure fallbacks,
serving log probabilities, duplicate trajectory DTOs, and rollout-resume
state are not active scientific paths.

Each live layer has one typed handoff:

| Boundary | Sole handoff |
|---|---|
| policy → rollout | `generate_policy(PolicyGenerationRequest)` |
| policy → scoring | `score(...)` and query-only `z_value(query_ids)` |
| rollout → training | exact `RolloutArtifact` members of one fixed `TrainingBatchPlan` |
| training → projections | immutable `TrainingStepSource` previewed before one source append |
| diagnostics → calibration | one `ProjectionTransition` built from the same source |
| projections → evolution | typed window-flow, posterior, and trajectory evidence views |
| evolution → library | one complete immutable `EvolutionMutation` |
| library → rollout | applicability-matched full skill contexts |
| attempt child → supervisor | one tagged immutable IPC outcome |

## Scientific sources and offline audit

Exactly four event payloads are scientific sources:

1. `LIBRARY_INITIALIZED`
2. `TRAINING_STEP_COMMITTED`
3. `EVOLUTION_PHASE_OPENED`
4. `EVOLUTION_CYCLE_COMMITTED`

Operational rollout/budget events may still be logged, but are never audit
inputs. `src/skillev/audit/` is a one-way offline consumer of source events.
No active package imports it, and no live component exposes `from_events`,
replay verification, or state hydration from logs.

Diagnostics and calibration publish online state through preview/commit:
training constructs the exact source and projection transition, appends
`TRAINING_STEP_COMMITTED`, then publishes the already-computed projection
references. A source append failure ends the attempt and does not publish the
projection. Offline audit independently recomputes the same floats from the
committed source.

## Checkpoint boundary

`RuntimeSnapshot@6` is an execution snapshot only. It uses fixed policy and
optimizer paths and stores one exact tagged `RuntimeExecutionState@6`: task
cursor, exact run cursor, library state, projection state, detector state, and
the immutable builder/config/method/protocol identity. Construction from a
checkpoint requires a caller-selected completed snapshot directory. There is no:

- latest-checkpoint search;
- filename fallback;
- v1/v2 reader;
- fingerprint, signature, receipt, or attestation;
- event-history comparison;
- prepare/apply/recover transaction;
- executable open-phase state;
- automatic repair, retry, or rollback.
- automatic cleanup or pruning in the live writer.

If a live attempt fails, that attempt is dead. A caller may explicitly start a
new attempt from a previously completed exact snapshot; private forensic phase
evidence is never accepted as a resume input.

## Full method versus arms

The full core has raw importance, uncapped flow weighting, confidence bounds,
the residual-and-entropy AND trigger, no gradient clipping, and zero AdamW
weight decay. Six alternate implementations are isolated under
`src/skillev/experiments/arms/`; the full core contains no boolean/optional
switch selecting them. The no-Bayesian application uses arm-owned source and
cycle records rather than empty posterior values shaped like full-method
records.

The full-method experiment protocol and freeze are Protocol v3. The historical
seven-arm benchmark wire belongs to Protocol 9 evidence only. New formal work
uses the three initial Protocol 10 method identities and its population-level
benchmark specification; the current execution gate is documented in
`EVALUATION_SPEC.md`.

Publication, run-plan, budgets, and snapshots are method-neutral runtime
infrastructure. Full and no-Bayesian source contracts are deliberately not
interchangeable. Offline audit kernels consume only published bundles and are
never imported by the live graph. The private fixed benchmark worker and its
catalog remain outside the public wheel; public completion-smoke builders never
enter formal benchmark aggregation.

## Retired scientific components

The following remain outside the active graph and must not be imported:

- exact reverse-flow compilers, Mirror flow, and intervention cuts;
- factorial K-O-E decision logic and causal effect estimators;
- the duplicate serving-logprob/backward-scorer trajectory path;
- legacy Bayesian feature builders and configurable bucket semantics;
- staged admission/defer/survivor experiment runners;
- live full-log replay and recovery proof systems;
- approval, receipt, seal, attestation, and hash-pin governance.

Private benchmark questions, answers, verifier state, and per-item results stay
in local/cluster private storage or `skillev-private-evaluation`; only
answer-free public projections may reach the evaluated model.
