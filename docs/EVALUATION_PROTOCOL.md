# Historical evaluation protocol note

This file described the pre-Protocol-10 binding. The active benchmark science is now frozen in
`evaluation.tex`; `docs/benchmark-evaluation-guide.md` is the evaluation engineering authority and
`docs/EVALUATION_SPEC.md` is its implementation map. This file does not authorize a formal run.

## Frozen inputs

- one registered seed: `0`;
- one immutable model/adapter/skill-library snapshot per evaluated condition;
- preselected task populations, order, budgets, metrics, and terminal evaluators;
- no post-result changes under the same protocol identity.

The benchmark suite and selection rules are those in Protocol 10. Its typed reader is separate from
the historical `@9` reader, whose artifacts continue to list a different suite.

## Isolation

Training, validation, and evaluation stores are physically separate. Licensed task statements,
choices, reference answers, and per-item outcomes remain private and never enter Git. Reference
answers are provided only to the terminal evaluator after generation; they never appear in prompts,
retrieved skills, model messages, evidence available to the policy, or supervisor context.

Every evaluator implements the guide's `TerminalReward` contract. Evaluation snapshots are opened
read-only and cannot update Bayesian posteriors or mutate the skill library.

## Comparisons

Report corrected SkillFlow and BayesianImprove from the same frozen population and budgets. Include
task metrics, resource use, calibration metrics, TTB residual/flow diagnostics, skill transitions,
failures, and run completion status. A single-seed result is described only as registered
primary-seed evidence. Missing or incomplete arms remain visible.
