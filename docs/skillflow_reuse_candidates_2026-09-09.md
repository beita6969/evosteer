# SkillFlow-inspired repair candidates

These are **independent implementations**, not vendored SkillFlow code. The
owner-supplied supplement cites SkillFlow `74be52bb6bd9f0e9e68dacb72636b75649197983`
and arXiv:2605.14089v1. Its reported license status was unresolved; no permission
to copy upstream implementation is assumed. Source inspiration: explicit tool
interfaces, catalog/read separation, reference-policy stability, local contrast
and CGF diagnostics. `idea.tex` remains unchanged and authoritative.

## Separate conditions, never a bundled hot switch

- Legacy rollout formats @3–@6 retain their old wire and exposure defaults.
- `skillev-policy-rollout@7`: `phase_context`, `action_wire`, `skill_exposure`.
  Formal YAML exposes these explicitly through `skillev-bayesian-formal-training@4`;
  older formal versions retain their original serialized defaults.
  The sole construction entry is `rollout.context_assembler(maximum_h0_tokens=...)`.
  Compare phase-only, `native-single-tool-call@1`, and `catalog-then-read@1`
  separately. Reasoning requests and per-domain thinking remain unchanged.
- Native Qwen3.5 uses XML `<function=...><parameter=...>` (confirmed against
  the deployed template), **not** the Qwen2 JSON tool carrier. Exactly one
  explicit carrier is accepted. No repairs, prose search, reasoning extraction,
  alternate answer, grammar mask, or first-of-many selection. Full original
  action tokens are scored; native stopping does not use the JSON-root cutter.
- Persisted H0 includes the phase/contract specification; sampling, sealed and
  provisional scoring and artifact restoration share the same tokenizer path.
  F receives current reasoning; B receives current public observation, never
  current reasoning. Execution objects remain distinct from sampled wire text.
- Catalog mode keeps retrieval order/applicability unchanged. H0 has IDs,
  versions, summaries and applicability; exact bodies arrive only on a valid
  deterministic read. Exposure alone creates no posterior event.
- `ZInitializationSpec`: default seeded Kaiming or independent
  `output-bias-log-epsilon@1`. Initial seed and mutation reset behavior are
  persisted in model configuration. The latter initializes only the final
  output affine layer to log epsilon, retaining q-conditioned capacity.
- Optimizer @4 adds independent B learning rate and an optional
  `reference-categorical-kl-group-clip@1` condition. Default is bare TTB.
  KL is categorical KL(current F || frozen accepted initial F), token/edge/
  trajectory averaged once. F/B/Z clipping happens after complete batch merge.
  A nonzero KL condition without its exact reference scorer fails closed.
  `QwenFrozenForwardReference` saves/loads the frozen F separately and computes
  logits through functional substitution; it cannot generate task answers.
  Application builders expose an explicit `reference_scorer_factory` for this
  private artifact. Missing reference files must not be recaptured on resume.
- Candidate reports keep original TTB loss and diagnostics unchanged, adding
  regularization loss, total loss and pre-clip group norms/scales separately.

## Authoring and diagnostics

Same-source contrasts require complete benchmark/population/source/occurrence,
policy, library and explicit environment-reset identities. Missing identities
produce no claimed match. Pair selection is deterministic, not best-answer
selection. Only bounded public action/observation excerpts and references reach
existing author evidence; private reward payloads/tests/rubrics do not. These
are observational comparisons, not causal counterfactuals. Current production
sources may need explicit reset identity metadata before this evidence exists.

Existing SkillDocument authoring, requirement preservation, Split/Generate
admission, one-shot author journal, Beta/LCB, W50 phase detector, atomic library
mutation and F/B warm start remain authoritative. CGF/Jensen is a read-only
view with explicit missing values, no clipping or no-call ability sentinel;
it cannot trigger or replace a posterior/phase decision.

## Validation scope

Synthetic CPU tests cover native syntax and difficult string payloads, exact
sampling/scoring/restoration paths, catalog read credit, reference KL zero-point
and gradient sensitivity, shared-support masking, Z reset, complete-batch clip,
independent TTB derivatives and source identity/privacy. Private deployed
Qwen tokenizer integration is opt-in via `SKILLEV_PRIVATE_TOKENIZER_DIR`.
These tests are not native-wire model generation, GPU gradient equivalence,
20–25-step quality retention, or natural W50 author→new-library→new-posterior
acceptance. No such experiment is implied by implementation or unit-test success.


## Public semantics repair (explicit new candidate)

The original native candidate omitted the surface's public instruction text.
The repaired candidate is **formal @5 / rollout @8** with
`public_action_semantics: true`, still `native-single-tool-call@1` and the
unchanged reasoning, sampling, full-inline/catalog choice, and task budgets.
Its condition and assembler record `public-action-semantics@1`. Older formats
retain the old native prompt rather than silently enabling this repair on resume.

ActionSurface @3 stores a typed semantic/JSON-wire instruction partition. Its
legacy JSON projection preserves the original ordered public text. The seven-domain
factory supplies public domain answer semantics and the existing Hotpot guidance;
no target, rubric, test, or private answer is an input to that projection. Native
reasoning receives the semantic text in H0, and native action schemas receive the
same semantic text in tool descriptions. JSON-envelope instructions are omitted
by selecting the explicit semantic partition, not by searching or deleting words.
ALFWorld admissible-command and environment-termination meaning is projected
from the existing public action surface fields.

Old artifacts continue using their stored H0 and tools snapshot. Wire parsing and
raw sampled token spans are unchanged. This is a changed prompt condition needing
its own collect-only T0; neither numerical replay of the old condition nor passing
CPU/tokenizer tests demonstrates that first-step task quality has recovered.

## Shared public task semantics (explicit subsequent candidate)

The `@5` repair restored the training adapter's existing semantic instructions;
those were weaker than native evaluation's Hotpot answer-span and MBPP public-
example instructions. `public-task-semantics@1` now supplies one public catalog
for all seven domains. It accepts only the domain, the declared public input
profile and the Hotpot deliberation switch—not task text, references or scores.
Training's declared source is `training-public-source-bridge@1`; Trivia in that
profile is closed-book. Evaluation uses its `PublicTaskView.input_profile`, so
only `released-iid-source@1` receives the reading-context instruction.

Select formal `skillev-bayesian-formal-training@6` with
`task_semantic_guidance: public-task-semantics@1`, retaining the desired explicit
`phase_context`, `action_wire` and `public_action_semantics` controls. This maps to
rollout `@9` and its shared context-assembler factory, including quality probes.
The existing formal `hotpot_deliberation` boolean is forwarded unchanged;
budgets, input material, thinking modes, sampling and TTB are not changed.

Evaluation selects the same catalog through `InferenceArm.task_semantic_guidance`
and its explicit `hotpot_deliberation` boolean (serialized as integrity arm `@6`).
This public guidance switch is separate from native thinking. Training retains
its `answer` parameter and executable action wire; evaluation retains its textual
`Final answer:` convention and environment tool wire. ALF public guidance enters
the evaluation system message without rewriting the authoritative reset task.

Legacy formal/rollout/evaluation conditions omit the new selector and retain
prior behavior. New initial contexts persist the selected catalog/input profile;
old saved contexts continue to restore their original text and tools. This is a
new prompt condition for an independent diagnostic run, not evidence of quality
recovery and not permission to start formal training.

## Committed initial-state evidence for source contrasts

New training source rewards can carry `training-reset-binding@1`. ALFWorld
captures the single real reset result only after the existing instruction,
observation, admissible-command, game, seed and horizon checks pass. Static
tasks explicitly record `static-no-environment-reset`. Both bind the complete
public task projection, excluding only its occurrence-specific task ID, captured
before actions. This is private reward metadata, not a new model-visible prompt.

Same-source contrasts compare the full canonical initial-state value, not a
seed-only label or a newly generated reset ID. `same-source-observational-contrast@2`
exports only the evidence kind and original trajectory references, without
repeating the initial projection in authoring prompts. Missing historical
bindings remain missing; existing explicit legacy reset evidence remains
readable. Reward/success values, F/B inputs and posterior arithmetic are unchanged.
