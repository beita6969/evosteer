# Executable method specification

This document maps the immutable scientific specification in `idea.tex` to code. If this document
and `idea.tex` disagree, `idea.tex` wins.

## State and trajectory

An orchestration state is the full history `H_t`; an edge appends reasoning, one action, and the
execution observation. `H_0` contains only the query, retrieved skills, and permitted task context.
Evaluation answers are never permitted in it. Terminal reward is supplied by a verifier through
the `TerminalReward` contract and is shifted to positive support as
`R_tilde = R + epsilon_min`.

Implementation: `skillev.runtime.trajectory`, `skillev.runtime.bounded_agent`,
`skillev.runtime.execution`, and `skillev.benchmarks`.

## Tempered trajectory balance

For each action edge, forward and hindsight-conditioned backward log probabilities are normalized
by the number of action tokens. The residual and objective are

```text
Delta(tau) = log Z_theta(q)
             + sum_t mean_token_log_PF(t)
             - beta * log(R_tilde(tau))
             - sum_t mean_token_log_PB(t)
L_TTB(tau) = (Delta(tau) / T)^2
```

This defines a variational trajectory distribution, not a claim of deterministic path-independent
flow. Rollout generation may be served by SGLang, but differentiable forward/backward scoring uses
the frozen Qwen3.5-9B backbone and the relevant trainable LoRA adapter locally. The action-token
span, not reasoning or observations, is scored.

Implementation: `skillev.policy`, `skillev.training.step_math`, `skillev.training.loop`.

## Distributional diagnostics

Local importance is `log I(t) = log PF(t) - log PB(t)`. State flow is an expectation over observed
trajectory prefixes, and skill marginal flow sums state flow over explicit invocations of a skill.
The implementation's finite-sample estimator and uncertainty metadata must remain visible rather
than being presented as an exact conserved field.

Implementation: `skillev.diagnostics` and `skillev.evolution.evidence`.

## Bayesian calibration

The context key is the tuple of task context, failure mode, token bucket, and horizon bucket. Each
skill/context cell carries a Beta-Bernoulli posterior:

```text
alpha = alpha0 + sum_e w_e * 1[y_e = 1]
beta  = beta0  + sum_e w_e * 1[y_e = 0]
mu    = alpha / (alpha + beta)
var   = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
LCB   = mu - k*sqrt(var)
```

Only verified training evidence can update this posterior. Flow-derived weights are normalized
within the invoking batch; validation/test evidence is physically separate and read-only.

Implementation: `skillev.calibration`, `skillev.evidence`.

## Phase boundary and evolution

A phase boundary requires both residual improvement stagnation over the specified window and a
decrease in skill entropy. One condition alone cannot trigger evolution. At a boundary, the
dual-axis operator uses marginal flow, context-conditioned LCB/UCB, and local asymmetry to choose:

- retain/compress: high flow and high LCB;
- refine/patch: high flow with localized low LCB;
- split: high flow with materially distinct context posterior modes;
- prune: low flow and low UCB;
- generate: no skill coverage under high residual asymmetry.

Every mutation produces a new immutable skill version and atomically changes the active alias.
After a library change, `log Z_theta` is reinitialized as specified by `idea.tex`.

Implementation: `skillev.evolution`, `skillev.runtime.skill_library`, and application snapshots.

## Closed interpretation points

`MethodSemanticsConfig` records the implementation choices needed to make abstract terms
executable: task-family context namespacing, terminal success as the calibration outcome,
per-batch invoking-edge mean weight normalization, explicit skill invocation credit, phase-search
scope, a non-empty seed library, and the observed-prefix state-flow estimator. These choices may be
changed only by a versioned method configuration, never during a run.

“Multi-modal posterior” is operationalized over separated context cells; it does not introduce a
Gaussian-process acquisition layer or expected improvement. Those are not in `idea.tex`.
