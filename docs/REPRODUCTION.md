# SkillFlow reproduction contract

## Fixed source

- Paper: [SkillFlow, arXiv:2605.14089](https://arxiv.org/abs/2605.14089)
- Code: [beita6969/SkillFlow](https://github.com/beita6969/SkillFlow)
- Compared revision: `74be52bb6bd9f0e9e68dacb72636b75649197983`
- Local reference: remote `skillflow-upstream`, tag `upstream-skillflow-74be52b`

The upstream tree is a read-only reference. This repository remains the implementation and result
authority. `idea.tex` is immutable and is the final authority for the scientific method.

## Reproduction profiles

`configs/baseline/paper_v1_250step.yaml` freezes the paper-profile run (250 optimizer steps).
`configs/baseline/upstream_head.yaml` freezes the upstream-head profile (300 steps). Both use seed
0 only. Any repair required to make upstream runnable is recorded as a corrected baseline rather
than silently attributed to upstream.

## Verified upstream gaps

Inspection of the fixed revision found these material differences between prose, configuration,
and executable paths:

1. the published configuration runs 300 steps while the paper profile calls for 250;
2. the data loader can silently substitute fallback examples;
3. the documented data preparation import does not match the checked-in preparation module;
4. executor, supervisor, and skill-creator model roles are split across hard-coded services and
   devices rather than one role-aware gateway;
5. device roles are hard-coded rather than supplied by the run topology;
6. context budgets are not represented by one shared contract;
7. skill evolution lacks a complete, durable evidence-to-version transition record;
8. a restart does not restore every RNG, sampler, optimizer, adapter, posterior, and skill-library
   state needed for exact continuation.

This project addresses those gaps through strict data validation, one SGLang gateway, explicit GPU
roles, fail-closed whole-step OOM handling, immutable skill versions, answer-free evidence, and complete
application snapshots. Passing unit tests is not a paper reproduction claim. A reproduction claim
requires a completed run, frozen inputs, and evaluation under `docs/benchmark-evaluation-guide.md`.

## Classification

- **Upstream-exact**: fixed upstream revision with no semantic repair.
- **Corrected baseline**: only documented execution repairs; Bayesian components disabled.
- **BayesianImprove**: the corrected execution layer plus every active component specified by
  `idea.tex`.

Results from these classes must never be merged under one run identity.
