# Method comparison and claim boundary

## SkillFlow versus BayesianImprove

| Axis | SkillFlow | SkillFlow-BayesianImprove |
|---|---|---|
| Trajectory learning | Tempered trajectory balance with forward/backward policies | Same TTB core |
| Diagnostic signal | Flow/credit used to guide recursive evolution | Distributional step/state/skill flow connected to persistent evidence |
| Reliability model | No feature-conditioned conjugate posterior in the fixed code revision | Flow-weighted Beta-Bernoulli posterior per skill/context cell |
| Decision rule | LLM-authored evolution from accumulated traces | Conservative LCB/UCB decisions with explicit retain/refine/split/prune/generate cases |
| Phase change | Evolution cadence/heuristics | Residual-stagnation **and** skill-entropy-decrease boundary |
| Skill identity | Mutable textual artifacts | Immutable version plus atomically updated active alias |
| Failure/restart | Partial runtime state | Transactional step and full method-state continuation |

The implementation intentionally preserves TTB rather than replacing it with policy-gradient or
generic Bayesian optimization. Bayesian evidence calibrates structural skill decisions; it does
not change the definition of the TTB residual.

## Relationship to Bayesian-Agent

[Bayesian-Agent](https://arxiv.org/abs/2602.13112) also treats skills as uncertain hypotheses and
uses verified trajectories to update reliability/failure beliefs. Its public repository documents
a categorical default likelihood and a legacy Beta-Bernoulli ablation. That is relevant prior art
and must be cited in any paper claim.

The narrower combined claim pursued here is the closed loop specified by `idea.tex`: token-normalized
TTB produces distributional local/state/skill flow; that flow weights feature-conditioned
Beta-Bernoulli evidence; conservative bounds and a joint phase-transition rule drive a dual-axis
skill-version operator; training then restarts its partition head. The code and experiments must
demonstrate the contribution of each link. “Bayesian self-improvement” by itself is not claimed as
novel.

## Permitted result language

- A single seed is evidence for that registered seed, not a robustness claim.
- A corrected baseline is not an upstream-exact reproduction.
- Unit, smoke, or bounded-pilot success is not full-training success.
- Evaluation results are claimable only under the frozen protocol and `TerminalReward` interface.
- Unsupported benchmark or related-work comparisons must be reported as unavailable, not inferred.
