# Protocol-v3 method handoff contracts

`idea.tex` is the scientific-method authority. This table records the sole
live handoff at each boundary; it is not another implementation.

| Upstream → downstream | Sole production handoff | Forbidden bypass |
|---|---|---|
| Policy → rollout | `generate_policy(PolicyGenerationRequest)` | model internals, authoring sampler controls, serving scores |
| Policy → scoring | `score(prefix_ids, action_ids, role)` and query-only `z_value(query_ids)` | action re-encoding, direction strings, H0-conditioned Z |
| Rollout → training | exact `RolloutArtifact` positions in one `TrainingBatchPlan` | task replacement, reward-zero filtering, rollout resume |
| Training → online projections | immutable `TrainingStepSource` previewed before `TRAINING_STEP_COMMITTED` | tensors, gradients, derived events, optional observers |
| Diagnostics → calibration | one atomic `ProjectionTransition` from the same source | another flow config, caps in full core, live replay |
| Projections → evolution | `WindowFlowView`, `PosteriorEvidenceView`, `TrajectoryEvidenceView` | full-log replay, mutable-query fallback, posterior-chain proof |
| Evolution → runtime library | one complete `EvolutionMutation` or explicit verified no-op | partial mutation, truncation, failed authoring disguised as no-op |
| Evolution → training | deterministic `reset_partition(seed)` after complete mutation construction | arbitrary optimizer edits, random reset, component-only recovery |
| Runtime library → rollout | applicability-matched `FullRetrievedSkillContext` tuples | metadata placeholders, ID filler, top-N truncation |
| Application → runtime | one explicit `SKILLEVApplication` object graph | alternate unjournaled assembly, constructor introspection, duplicate authorities |
| Scientific sources → audit | typed source events, offline analysis only | active import of audit; live recovery uses the step transaction and posterior provenance |
| Experiment protocol → arms | six explicit arm builders plus full builder | result-conditioned selection or a core arm-name switch |

## Literal full-method identity

- raw-softmax rollout and teacher-forced scoring;
- no importance clipping;
- no flow-weight cap;
- no gradient clipping;
- AdamW weight decay `0.0`;
- fixed B-task population, attempted once;
- flow-weighted confidence-bound calibration;
- residual-and-entropy phase trigger;
- complete nonempty Φ, verified no-op consuming a durable evidence cutoff until a fully
  fresh comparison window exists (semantics v4), or failed attempt;
- one preregistered primary seed.

Authoring uses a separate frozen-base request. OOD evaluation freezes policy,
skill library, and posterior.
