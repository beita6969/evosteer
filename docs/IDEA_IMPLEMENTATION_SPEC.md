# idea.tex implementation mapping

`idea.tex` is the immutable scientific authority. This document describes the
current BayesianImprove method, not the root SkillFlow baseline launcher.

## Current population and execution identity

The seven IID domains are HotpotQA, TriviaQA, AIME2026, HealthBench, ALFWorld,
MBPP+, HumanEval. The current training condition is
`seven-domain-rotating-eighth-8x4-seed0@1`: seven domain occurrences plus one
rotating eighth occurrence, each with four independent rollouts, B=32. An
occurrence is not necessarily a distinct source question. One Qwen3.5-9B task
agent owns each episode; there are no consulting agents or answer mergers.

The debug entry uses production components but is not a formal publication
entry. A long run requires an explicit search/closure/cycle plan. Historical
Protocol 10 numbers (nine domains, B16, 288 steps) and its retired admission
system are not the current run authority.

## Scientific quantities and implementation owners

| Quantity / boundary | Implementation |
|---|---|
| Exact action-token F/B means and TTB residual `(log Z + sum F - beta log(R+epsilon) - sum B)` | policy scoring, training TTB assembly |
| Loss `(delta / T)^2`; precomputed gradients finalized only after full trajectory/batch | training provisional and distributed TTB paths |
| Prefix sample `exp(sum log I)`; skill flow averaged over invoking trajectories | diagnostics core/assembly |
| Full-batch invoking-edge mean-one weights; independent terminal Bernoulli label | calibration core; method projection pipeline |
| Explicit H0-available skill declaration credit | contracts/skill_invocation; training/invocation_evidence |
| Retrospective `z`: family, execution status, initial-H0 token bucket, final horizon | calibration extractors; never an online routing feature |
| One committed posterior query owner; new IDs from prior, unchanged IDs retain history | training projections/posterior state |
| Joint residual stagnation plus falling invocation entropy | evolution detector; production W50 unchanged |
| Retain / Refine / Split / Prune / Generate | evolution decision, execution, authoring and validation |
| Library mutation / Z reset / next diagnostic segment / durable optimizer commit | evolution loop and existing step transaction |

Flow is a single observed-full-history prefix sample. No conditional aggregation
under a coarser state or distributional-robustness guarantee is claimed. Weighted
terminal success is an association under the declared evidence rule, not a causal
skill contribution; repeated calls and repeated source questions are correlated.

## Current repair semantics

`MethodSemanticsConfig` version 5 and evolution configuration version 7 record the
new public-action Generate population (including COMPLETE), bounded public author
material, and raw-response-before-validation recovery. Exposure is distinguished
from explicit invocation. Static skill invocation is a strategy declaration, not
a separately solved subprogram. Retain on an irreducible source aborts the complete
attempt before any author call; an error is not silently converted into no-op.

The complete decision stays private and persistent. Bounded author material has
its own version and records omitted evidence. Unknown network outcomes are not
retried automatically. A successfully received malformed response is a rejected
result, not permission to sample another author draft. New library IDs do not by
themselves prove new capability.

A verified phase no-op does not reset Z or consume a mutation cycle; detection
waits for fresh comparison evidence. Zero coverage is not a special bootstrap
exception to the paper's joint criterion. F/B state must survive a real mutation;
Z parameters, gradient and optimizer state alone are reset as required.

## Evidence status

See [repair status](BAYESIAN_EVOLUTION_REPAIR_STATUS.md) for implemented changes,
remaining work and actual validation. The historical four-step / 128-trajectory
run proved ordinary commits and a 2+2 restore on source `ed04e85`; it had no phase
or mutation and is not validation of the repaired code or natural closure.

Full acceptance still requires actual authoring and new-library continuation,
full mutation H0 executability, real-Qwen parallel equivalence through Phi,
natural W50 observations, and independent calibration/effect evaluation. Neither
all unit tests passing nor a short training run proves that the method improves.
