# BayesianImprove evidence/evolution repair — implementation status

Updated 2026-09-08. **Partial implementation; not full closure acceptance.**
`idea.tex` is immutable. Current IID: HotpotQA, TriviaQA, AIME2026,
HealthBench, ALFWorld, MBPP+, HumanEval. Exactly one Qwen3.5-9B task agent
per episode; frozen-base authoring is only an evolution role, not a second solver.

## Implemented in this repair batch

- Every newly recorded posterior event is associated with an admitted, explicit
  skill declaration, its public execution status, following step coordinates and
  the independent `TerminalReward.success` label. Rejected declarations and
  declarations with no following action remain visible. This is **strategy
  selection / terminal-outcome association**, not independently executed skill
  success or causal credit. Historical missing links remain unknown.
- Author evidence copies only public action and execution-observation wire fields.
  No evaluator/native payload, gold answer, hidden tests or rubric is copied.
  Complete public evidence is retained in private projection/transaction storage.
- Author prompting uses an explicitly versioned bounded projection. Source
  identity arrays and summaries are compacted as well as examples. Public
  context/status groups are sampled deterministically, independent of author
  results. Mandatory applicability and output/immutable constraints are never
  silently truncated. Exact tokenizer capacity is checked before submission.
- Native SGLang author requests persist `prepared`, `submitted`, and the complete
  `response-received` transport result **before token parsing or validation**.
  The outer journal records validated/rejected outcomes. A received response can
  be consumed after interruption without another network generation. A submitted
  request with no saved response remains ambiguous and fails closed; a seed is
  not an idempotency key. Legacy received drafts remain readable.
- Generate includes valid static COMPLETE edges. Its frozen coverage definition
  is **no explicit invocation**, not absence of applicable skills. Exposed-but-not-
  called, no-exposed-skill and unknown-exposure populations are distinguished.
  Authors receive related existing skills' public summaries/applicability. New ID
  uniqueness is not claimed as semantic novelty.
- Calibration reporting adds success/failure weight, weight-concentration ESS,
  first-seen-source prequential errors and reliability bins. Predictions use the
  pre-batch posterior. This is not a fixed source-held-out study, a causal estimate,
  or a proof of LCB coverage. ESS is not an independent-sample count.
- Online retrieval accepts the existing pre-action public feature type, not the
  retrospective calibration type. Retain's existing pre-authoring abort on an
  irreducible source is now explicitly frozen as the operator policy.
- Method semantics is version 5; evolution configuration is version 7. Old
  serialized conditions must not be silently resumed as the new method.

## Preserved scientific/transaction boundaries

Per-token F/B means, TTB residual/loss, complete-batch mean-one invocation-edge
weights, Beta accumulation, posterior ownership, W50 conjunction, and Z reset
only after actual mutation are unchanged. This patch does not normalize per
worker, credit all exposed skills, force calls/phases, reduce rollout budgets,
introduce an extra solver, or relabel a debug entry as a formal entry.

## Acceptance still outstanding

| Work package | Remaining work / evidence |
|---|---|
| Formal seven-IID entry | Implemented subsequently by the training session: shared assembly plus dedicated `bayesian_improve_training` build/full-resume entry, explicit B28/250 configuration and incompatible-condition resume rejection. See [formal condition](bayesian_formal_250.md); this does not complete the other acceptance rows. |
| Full author capacity | Large Generate groups still have a mandatory per-edge output requirement contract. Bounded optional evidence alone cannot guarantee arbitrary-sized mandatory contracts fit; this requires a versioned finite author contract, not silent omission. Ensure selected Refine weak cells and required normal constraints are represented under the budget. |
| Split | Existing routing partitions task families. Same-family public-context partitions, their discovery statistics, applicability, author validation and restore are not implemented by this batch. |
| Mutation executability | Existing per-draft retrieved-block bounds are not a complete next-batch H0 check. Validate affected task/tool/retrieval assemblies before committing the entire mutation. |
| Parallel scientific equivalence | Extend real-Qwen sealed/provisional and single/dual-owner comparison through posterior, diagnostics and Phi proposals. Existing CPU formula tests are reusable, not substitutes for this test. |
| Real author closure | Actual SGLang author → valid mutation → Z/optimizer cleanup → fresh process → new H0 → legitimate skill call → first posterior event. Not run in this batch. |
| Natural W50 | Keep production thresholds; record opportunities/blockers and post-mutation learning. Four steps cannot demonstrate two W50 residual windows. Two recursive segments require roughly 200 accepted batches plus wait/tail, with no guarantee of natural triggers. |
| Statistical effect | Fixed source-disjoint calibration, actual LCB decision behavior and new-skill benefit/regression evaluation remain unmeasured. |

The owner subsequently explicitly requested a250-step formal experiment instead
of another four-step run. That authorization supersedes the earlier launch hold,
not the outstanding scientific/authoring acceptance items above. Do not report
complete `idea.tex` closure from this partial repair. Existing four-step results
used source `ed04e85`, not these edits.

## Verification log

- First related CPU batch on Linux: **40 passed**, including durable transport,
  author journal, external-author contract, bounded material, base-author budget,
  Generate semantics and evidence reporting.
- Local Ruff passed for modified files. Local pytest dependency startup stalled;
  those own processes were terminated and the tests moved to Linux CPU.
- Final private Linux CPU snapshot: `CUDA_VISIBLE_DEVICES="" make check` passed:
  **3,633 passed, 1 real-CUDA opt-in skip**, 142 warnings, pytest 280.66 seconds;
  full Ruff/format, mypy (555 sources), both wheels and model-wheel check passed.
- Earlier full validation stopped at a concurrently incomplete execution-stage
  Protocol; the refreshed snapshot then exposed 12 old Generate assumptions
  (3,621 passes). Two assertions and the shared synthetic no-op witness were
  updated for valid COMPLETE eligibility. The witness uses below-floor asymmetry,
  not altered production thresholds. Affected tests passed **26/26** before the
  final full check. Each rerun followed a relevant source change.
- An assertion fixture was also corrected to construct valid applicability before
  testing rejection of retrospective features, avoiding a vacuous TypeError pass.
- No redundant full builds were run between small edits. No success is claimed
  for terminated local tests, for the skipped CUDA test, or for later concurrent
  source edits outside the verified private snapshot.
- No GPU job, independent review agent, or manual hash check was started by this
  repair batch. Existing serving/training jobs owned by other threads were untouched.
