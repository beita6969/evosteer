# Clean frozen-policy and skill-library evaluation

The clean executor supports an **explicitly selected forward LoRA** and an
**explicitly selected, read-only skill library**, in addition to adapter-free/no-skill
controls. These are separately declared experimental axes. This repairs the former upper-layer adapter-free routing
and missing actor inference state. It is not a candidate selector and does not
change `idea.tex`, training mathematics, benchmark inputs or native scorers.

## Scope and comparison axes

- A single trained arm uses its own owner final from that arm's forward policy; there is no vote, Step-0 fallback, best-of-N,
  token-cost-based problem selection or alternative-derivation rerun.
- A paired run changes **one** axis: permitted single-owner reasoning topology,
  skill access, or forward weights. Consultant agents are not executable arms.
  A weights-only comparison holds topology, skill mode, tools, prompts, thinking
  policy, inputs, native metrics, service and budgets fixed. A trained skill-only
  comparison holds the same trained policy and all other controls fixed.
- Actual model and tool costs remain reported, not equalized after seeing
  outcomes. A non-improvement remains a non-improvement; skill access is not
  assumed to guarantee a higher score. Each episode retains one Qwen3.5-9B owner.
- Weights and libraries are frozen during evaluation. A forward-only/no-skill arm
  remains a weight ablation; a library arm actually uses the chosen active
  `SkillLibraryState`, not a generic replacement. No training application or
  optimizer is restored. Backward policy, log Z and evolution operators are not
  inference dependencies of this executor.
- The existing production `TaskConditionedSkillRetriever` uses document
  applicability, not a trained ranker or posterior/LCB score. Restoring its active
  documents therefore restores all of its current inference state. Calibration,
  detector and training evidence are not sent to the actor. An unsupported
  retrieval rule fails explicitly rather than silently becoming generic advice.

## Explicit private configuration

Keep the existing private runtime configuration and its complete eight-IID
thinking, input, scorer and budget declarations. Add a `trained_policies` mapping:

```text
trained_policies:
  <public-policy-label>:
    checkpoint_directory: <absolute, explicitly chosen immutable checkpoint>
    adapter_name: <preloaded-forward-route-name>
    # Optional, when replicas use different absolute paths:
    served_adapter_paths: [<replica-0-forward-path>, <replica-1-forward-path>]
```

This is a schematic; the runtime file is JSON. The directory must be either a
completed training checkpoint (containing `policy/`, `runtime_state.json` and
`COMPLETE`) or an explicitly exported policy directory containing
`policy_state.json` and `forward_adapter/`. Metadata must state the selected
nonzero optimizer step. No directory scan or implicit `latest`/`best` selection
is performed. The adapter config's base-model path must match the actual service
base-model path. Use published, immutable checkpoint directories throughout the run.
Metadata and paths are recorded privately; model tensors are not loaded or hashed
by the evaluation coordinator. Remote path mappings are explicit operator bindings,
not content verification of independently copied remote weights.

Construct a trained arm from the chosen matched arm with `dataclasses.replace`:

```python
trained = replace(
    matched_arm,
    arm_id="trained-no-skill",
    optimizer_steps=chosen_step,
    policy_id=chosen_public_policy_label,
)
config["arms"] = [matched_arm.to_value(), trained.to_value()]
```

The new `skillev-policy-integrity@4` arm format carries `policy_id` and the existing
tool-call mode. Old untrained arm formats retain their serialization. A trained
label or nonzero step alone cannot imply a loaded adapter. Paths must never be
used as public policy labels or committed in runtime configuration examples.

## Explicit frozen skill source

Use `skill_mode="off"` for no skill: no library is opened or sent to the actor,
no retriever is constructed, and the ordinary task/tool interfaces remain.
For both fixed-initial and evolved libraries, use a `library` arm with explicit
`skill_library_id` and `skill_retrieval_rule="active-applicability@1"`. Serialization
is `skillev-policy-integrity@5`; `policy_id` may be null for an adapter-free skill
ablation. Model checkpoint, skill identity and retrieval rule are independent.

Add to the **private JSON** runtime configuration (schematic):

```text
skill_libraries:
  initial-library:
    kind: initial
    source: planned-advisory-seeds@1
    retrieval_rule: active-applicability@1
  evolved-library:
    kind: evolved
    checkpoint_directory: <explicit completed training checkpoint root>
    retrieval_rule: active-applicability@1
```

An initial source may instead declare `snapshot_file: <absolute file>` containing
an existing `SkillLibraryState.to_value()` export. An evolved source reads exactly
`runtime_state.json -> execution_state -> library` from the chosen completed
checkpoint. It does **not** require or load `optimizer.pt`, restore a trainer, scan
for a best checkpoint, or run skill evolution. Choose the same training root as
its forward policy when testing that checkpoint's complete learned state; a
cross-checkpoint combination is a separately declared ablation, not that result.
The snapshot is loaded once before generation and retained immutably in memory.
An evolved checkpoint can legitimately contain unchanged initial skills if no
mutation occurred; its source step alone is not evidence of a changed library.

Only active canonical skill documents and public source labels reach the actor.
Private checkpoint paths, inactive lineage, training diagnostics, targets and
optimizer state do not. Expanded controls retain the actual active content and
source step privately, so equal labels alone do not establish equal controls.
A weight-only comparison requires the **same** frozen library; changing both
weights and skill source is not presented as a one-axis paired comparison.

Retrieval uses the actual public task family, context and available tool IDs.
Optional `skill_task_features[benchmark]` declares `task_family` and `context_id`
explicitly; these are also frozen controls. Defaults are `public-task` and
`<benchmark>:evaluation`. Training-only applicability does not magically match
evaluation, and a no-match is reported rather than relabelled. Tools absent from
the clean lane are never advertised just to satisfy a skill's applicability.

At episode start the production retriever offers full applicable documents,
subject to the declared skill-context budget. The model may ignore these or use:

- `list_skills(cursor=0, limit=8)`: discover the complete active space, paged,
  including nonmatching documents. No lexical-overlap gate.
- `retrieve_skills()`: apply the declared production rule to the current public
  task. It returns matches and budget diagnostics, not an answer selector.
- `read_skill(skill_id)`: view the complete document, prioritizing the model's
  choice within the skill context budget. Never truncate its instructions.
- `invoke_skill(skill_id)`: explicitly adopt a text skill using the production
  invocation acknowledgement. It does not execute a hidden solver, mandate a
  plan, call another model or submit an answer.

Both plain control objects (`kind="skill", operation=<tool name>`) and native
Qwen tool calls are supported, along with production `kind="skill", name="invoke"`
actions. A tool request spends its ordinary model call; no framework extra call
is used to retrieve, load or invoke the document. The older unnamed generic
modes now use the canonical fixed initial training seeds, not the lexical advice
registry; use explicit library arms for new experiments.

### Observed exposure, not just permission

Candidate `intervention_counts` distinguish `skill_access_episodes`,
`skill_retrieval_requests`, `skill_retrieved` (production-rule matches),
`skill_discovery_calls`, `skill_read_calls`, `skill_blocks_injected`,
`skill_body_tokens` and `skill_calls` (explicit invocation acknowledgements).
Body counts/tokens are measured **after context packing** in actual model
requests, not when a body is merely selected. They count exposures across calls,
not unique skills. Reading a nonmatching skill can expose it without a retrieval
rule match; invocation does not prove the model followed its advice.

`skill_no_match`, `skill_budget_skips` and `skill_context_omissions` are distinct.
Private `skill-access`, `skill-retrieval`, `skill-tool`, `skill-context` and
`rendered-request` events show IDs and reasons. A permission flag, empty match,
or skipped body must never be reported as an actually supplied/invoked skill.

## Serving and execution

Provision the approved SGLang service separately with LoRA enabled, a stable served
base-model name, and the selected **forward** adapter preloaded under `adapter_name`.
Set LoRA rank/target-module options to support the checkpoint. The existing Qwen3.5
training exporter uses the SGLang-compatible projection subset; do not substitute
an unrelated all-linear adapter whose projections the service cannot apply.
The coordinator only reads `/model_info`, `/server_info` and `/v1/models`; it never
loads/unloads an adapter or changes the service. It verifies the registered route's
root and parent on every configured replica, and rechecks after generation.
Native requests send the route through `lora_path`, as documented by
[SGLang's LoRA interface](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/lora.mdx).
Adapter-free requests omit that field. Separate per-policy client pools prevent
concurrent arms from switching each other's route. A failed/missing route is an
infrastructure failure, not a reason to retry against the base model.
Generate the matched Step-0 arm freshly on this same LoRA-enabled service; an old
result from a differently configured deployment is not the paired baseline.

The existing `scripts/run_step0_integrity_paired.py` command accepts these arms;
its historical filename does not imply Step-0-only execution. Trained plans and
summaries use `policy-integrity-*` schema names. Same-run resume retains the exact
checkpoint controls and validates original model-output policy/adapter provenance;
answers from a different run cannot be imported.

No benchmark run is authorized by this document. Before a real round, the owner
must identify the checkpoint and round/population. Keep one seed, AIME thinking-on,
the approved GPU selection rules, and a declared progress threshold with rate/ETA.
The IID list remains HotpotQA, TriviaQA, AIME2026, HealthBench, WebShop, ALFWorld,
**MBPP+ hard**, HumanEval. The existing `mbpp-plus` machine lane is official Base
AND Plus scoring; do not relabel it as a verified hard subset without its definition.
