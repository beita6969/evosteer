# SkillFlow upstream parity contract

Protocol 10 uses two distinct SkillFlow artifacts and never aliases either one to
BayesianImprove no-calibration.

## Frozen upstream identity

- upstream project: Mingda Zhang SkillFlow;
- upstream revision: `74be52bb6bd9f0e9e68dacb72636b75649197983`;
- parity contract: `skillflow-upstream-parity@1`;
- local semantic projection: `configs/baseline/skillflow_upstream_semantics.yaml`.

## Artifact boundary

**Upstream reproduction** uses the official-derived `training.GFlowNetTrainer`, its
native data identity, and the upstream experiment shape. It is evidence that the
upstream method can be reproduced; it is not a Protocol 10 comparison arm.

**Protocol 10 exact-method baseline** preserves the upstream algorithm while adapting
only the frozen Protocol 10 task order, 16-by-288 experiment shape, trusted terminal
evaluators, external SGLang transport, current distributed GPU transport, and durable
checkpoint/adapter transaction boundary.

The Protocol 10 adapter must not change theta/phi parameterization, TTB formula or
normalization, KL regularization, partition head, gradient clipping, plateau detector,
task-type accuracy tracking, CGF D/R/U classification, SkillCreator prompts,
SkillWorkspace metadata, mutation, or pruning rules.

## Parity oracle

The official-derived `training/` path is the executable oracle. Tiny deterministic
fixtures compare the following boundaries before the Protocol 10 builder is admitted:

| Boundary | Oracle | Protocol 10 implementation | Required equality |
|---|---|---|---|
| prompt/action schema | `training.task_prompts`, `training.react_prompts` | exact task/session bridge | exact text and schema |
| theta forward log-probability | `GFlowNetTrainer` | exact baseline training kernel | floating tolerance |
| phi backward log-probability | `GFlowNetTrainer` | exact baseline training kernel | floating tolerance |
| KL and reference policy | `GFlowNetTrainer._run_micro_batches` | distributed exact gradient | floating tolerance |
| partition value and TTB residual | `GFlowNetTrainer` | exact baseline training kernel | floating tolerance |
| batch scaling and clipping | `GFlowNetTrainer` | distributed exact gradient | floating tolerance |
| plateau transition | upstream detector state | exact evolution controller | exact decision |
| CGF D/R/U and mutation | upstream evolution/workspace | exact evolution controller | exact decision and state |

Formal source events record both the upstream revision and this parity-contract version.
An exact baseline application may not inherit from `FlowOnlyApplication`, use
`MethodProjectionPipeline`, or create Bayesian posterior cells.
