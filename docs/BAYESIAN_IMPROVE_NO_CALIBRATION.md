# BayesianImprove no-calibration formal method

`bayesian-improve-no-calibration` is a formal Protocol 10 method, not the exact
SkillFlow baseline and not a legacy `skillflow-disabled` ablation identity.

It preserves the BayesianImprove TTB objective, exact-token external SGLang rollout,
distributional flow diagnostics, and the residual-stagnation **and** skill-entropy
phase trigger from `idea.tex`. It does not construct alpha/beta posterior state,
calibration cells, LCB/UCB values, or placeholder posterior objects. Evolution uses
only answer-free flow and importance evidence and is limited to the declared
flow-only retain/generate policy.

Formal full and no-calibration runs share the same `DistributedTTBGradientCoordinator`,
`RolloutWorkflowResources`, external base-model skill author, versioned SGLang adapter
publisher, checkpoint store, and step transaction journal. Their differences begin
only at projection, posterior/calibration, evolution decision, and source-event type.

The formal checkpoint identity is method-specific. Resume rejects a full-method or
exact-SkillFlow checkpoint, restores the exact task cursor and flow-only state,
re-publishes the forward adapter, and reconciles only events already sealed in the
step transaction. No-calibration checkpoints intentionally contain no posterior.
