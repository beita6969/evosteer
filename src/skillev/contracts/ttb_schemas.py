"""Deterministic schema identities for the phase-one TTB contracts.

The existing :mod:`schema_registry` intentionally stores identities rather
than implementing instance validation.  The records themselves remain the
source of validation behavior; this module gives each independently
serializable wire shape a stable descriptor and collects those descriptors in
one sorted registry.
"""

from __future__ import annotations

from typing import Final, TypeAlias

from .canonical import CANONICALIZATION_VERSION, stable_hash
from .schema_registry import SchemaDescriptor, SchemaRegistry, SchemaVersion

SchemaField: TypeAlias = tuple[str, str]
SchemaSpec: TypeAlias = tuple[str, str, tuple[SchemaField, ...]]

TTB_SCHEMA_VERSION: Final = SchemaVersion(4, 1)

# The seven method-level contracts named in docs/phase1-goal.md.
MAIN_TTB_SCHEMA_IDS: Final = (
    "skillev.edge-logprob-record",
    "skillev.evolution-action-record",
    "skillev.phase-transition-event",
    "skillev.posterior-cell-state",
    "skillev.terminal-reward",
    "skillev.trajectory-record",
    "skillev.ttb-batch-stats",
)

# Every entry corresponds to a public ``to_value``/``from_value`` record.  Enum
# values are embedded in their containing field type because enums do not have
# an independent record envelope.
_TTB_SCHEMA_SPECS: Final[tuple[SchemaSpec, ...]] = (
    (
        "skillev.context-feature",
        "ContextFeature",
        (
            ("context", "string"),
            (
                "failure_mode",
                "enum[success,tool_error,schema_invalid,timeout,parse_error,other]",
            ),
            ("horizon_bucket", "enum[le_3,h4_to_8,gt_8]"),
            ("token_bucket", "enum[le_1k,k1_to_4k,gt_4k]"),
        ),
    ),
    (
        "skillev.source-run-cursor",
        "RunCursorValue",
        (
            ("committed_actions", "integer>=0"),
            ("committed_cycles", "integer>=0"),
            ("completed_training_steps", "integer>=0"),
            ("format", "literal[skillev-source-run-cursor@1]"),
            ("run_plan_hash", "sha256"),
        ),
    ),
    (
        "skillev.edge-logprob-record",
        "EdgeLogprobRecord",
        (
            ("context", "object|null"),
            ("backward_adapter_version", "string"),
            ("backward_logprob_per_token", "finite-number"),
            ("forward_adapter_version", "string"),
            ("forward_logprob_per_token", "finite-number"),
            ("scoring_stack_id", "literal[training-stack]"),
            ("step_importance", "finite-number"),
            ("step_index", "integer>=1"),
            ("trajectory_id", "string"),
        ),
    ),
    (
        "skillev.entropy-observation",
        "EntropyObservation",
        (
            ("distinct_skills", "integer>=0"),
            ("entropy", "finite-number>=0"),
            ("total_invocations", "integer>=0"),
            ("window_end_step", "integer>=0"),
        ),
    ),
    (
        "skillev.evolution-action-record",
        "EvolutionActionRecord",
        (
            (
                "variant",
                "union[retain-compress,refine,split,prune,generate]",
            ),
        ),
    ),
    (
        "skillev.evolution-cycle-committed",
        "EvolutionCycleCommitted",
        (
            ("authoring_reservation_ids", "array[string;unique]"),
            (
                "authoring_usage",
                "object[input_tokens:int>=0,output_tokens:int>=0,model_calls:int>=0]",
            ),
            ("decision_content_hash", "sha256"),
            ("format", "literal[skillev-evolution-cycle-commit@7]"),
            ("library_version_after", "string"),
            ("library_version_before", "string"),
            ("mutation", "schema:skillev.evolution-mutation-value@4.1"),
            ("optimizer_step", "integer>=1"),
            ("phase_event_id", "string"),
            ("proposal_content_hashes", "array[sha256;nonempty]"),
            ("run_cursor_after", "schema:skillev.source-run-cursor@4.1"),
            ("z_reset_seed", "uint64"),
            ("z_version_after_reset", "string"),
        ),
    ),
    (
        "skillev.evolution-mutation-value",
        "EvolutionMutationValue",
        (
            ("actions", "array[schema:skillev.evolution-action-record@4.1]"),
            ("active_skill_ids_after", "array[string]"),
            ("library_version_after", "string"),
            ("library_version_before", "string"),
            ("new_documents", "array[json-object]"),
        ),
    ),
    (
        "skillev.evolution-phase-opened",
        "EvolutionPhaseOpened",
        (
            ("format", "literal[skillev-evolution-phase-open@5]"),
            ("phase_event", "schema:skillev.phase-transition-event@4.1"),
            ("run_cursor_at_phase", "schema:skillev.source-run-cursor@4.1"),
        ),
    ),
    (
        "skillev.phase-checkpoint-artifact",
        "PhaseCheckpointArtifact",
        (
            ("artifact_sha256", "sha256"),
            ("format", "literal[skillev-phase-checkpoint-artifact@1]"),
            ("library_version", "string"),
            ("optimizer_step", "integer>=1"),
            ("phase_event_id", "string"),
            ("policy_snapshot_id", "string"),
            ("run_cursor_after", "schema:skillev.source-run-cursor@4.1"),
            ("runtime_state_sha256", "sha256"),
        ),
    ),
    (
        "skillev.phase-checkpoint-published",
        "PhaseCheckpointPublished",
        (
            ("artifact", "schema:skillev.phase-checkpoint-artifact@4.1"),
            ("format", "literal[skillev-phase-checkpoint-published@1]"),
            ("phase_event_id", "string"),
        ),
    ),
    (
        "skillev.generate-action-record",
        "GenerateActionRecord",
        (
            ("action_id", "string"),
            ("action_type", "literal[generate]"),
            ("evidence", "schema:skillev.generate-evidence@4.1"),
            ("format", "literal[skillev-evolution@5]"),
            ("library_version_after", "string"),
            ("library_version_before", "string"),
            ("lineage_ref", "string"),
            ("phase_event_id", "string"),
            ("proposal_content_hash", "sha256"),
            ("produced_skill_id", "string"),
            ("rationale_text", "string"),
        ),
    ),
    (
        "skillev.generate-evidence",
        "GenerateEvidence",
        (
            ("evidence_type", "literal[generate]"),
            ("importance_edge_ids", "array[string]"),
            ("importance_quantile", "finite-number[0,1]"),
            ("importance_semantics", "literal[absolute-log-density-ratio@1]"),
            ("minimum_absolute_log_importance", "finite-number>0"),
        ),
    ),
    (
        "skillev.initial-context",
        "InitialContext",
        (
            ("active_skill_ids", "array[string;sorted;unique]"),
            ("assembled_hash", "sha256"),
            ("assembled_token_count", "integer>=1"),
            ("assembler_version", "string"),
            ("meta", "json-object"),
            ("query", "string"),
            ("retrieved_skill_ids", "array[string]"),
        ),
    ),
    (
        "skillev.library-initialized",
        "LibraryInitialized",
        (
            ("active_skill_ids", "array[string]"),
            ("documents", "array[json-object]"),
            ("format", "literal[skillev-library-initialized@4]"),
            ("initial_optimizer_step", "integer>=0"),
            ("library_version", "string"),
            ("method_identity_hash", "sha256"),
            ("run_cursor", "schema:skillev.source-run-cursor@4.1"),
        ),
    ),
    (
        "skillev.phase-transition-event",
        "PhaseTransitionEvent",
        (
            ("current_window", "schema:skillev.window-stats@4.1"),
            ("entropy_condition_met", "boolean"),
            ("entropy_series", "array[schema:skillev.entropy-observation@4.1]"),
            ("event_id", "string"),
            ("format", "literal[skillev-phase-transition@2]"),
            ("library_version", "string"),
            ("previous_window", "schema:skillev.window-stats@4.1"),
            ("relative_improvement", "finite-number"),
            ("required_consecutive_drops", "integer>=1"),
            ("residual_condition_met", "boolean"),
            ("rho", "finite-number>0"),
            ("trigger_rule", "enum[residual-and-entropy,residual-only]"),
            ("triggered", "literal[true]"),
            ("triggered_at_step", "integer>=0"),
        ),
    ),
    (
        "skillev.posterior-batch-update",
        "PosteriorBatchUpdate",
        (
            ("batch_id", "string"),
            ("updates", "array[schema:skillev.posterior-update-event@4.1]"),
        ),
    ),
    (
        "skillev.posterior-cell-state",
        "PosteriorCellState",
        (
            ("alpha", "finite-number>0"),
            ("alpha_0", "finite-number>0"),
            ("beta_0", "finite-number>0"),
            ("beta_count", "finite-number>0"),
            ("last_event_id", "string|null"),
            ("skill_id", "string"),
            ("update_count", "integer>=0"),
            ("z", "schema:skillev.context-feature@4.1"),
        ),
    ),
    (
        "skillev.posterior-update-event",
        "PosteriorUpdateEvent",
        (
            ("alpha_before", "finite-number>0"),
            ("beta_count_before", "finite-number>0"),
            ("alpha_after", "finite-number>0"),
            ("beta_count_after", "finite-number>0"),
            ("event_id", "string"),
            ("flow_weight", "finite-number>=0"),
            ("outcome", "boolean"),
            ("skill_id", "string"),
            ("step_index", "integer>=1"),
            ("trajectory_id", "string"),
            ("z", "schema:skillev.context-feature@4.1"),
        ),
    ),
    (
        "skillev.posterior-task-family-mode",
        "PosteriorTaskFamilyMode",
        (
            ("cell_keys", "array[string]"),
            ("task_family", "string"),
            ("evidence_mass", "finite-number>=0"),
            ("lcb", "finite-number"),
            ("mean", "finite-number[0,1]"),
            ("posterior_event_ids", "array[string]"),
            ("sigma", "finite-number>=0"),
            ("ucb", "finite-number"),
            ("within_cell_mean_span", "finite-number>=0"),
        ),
    ),
    (
        "skillev.prune-action-record",
        "PruneActionRecord",
        (
            ("action_id", "string"),
            ("action_type", "literal[prune]"),
            ("evidence", "schema:skillev.prune-evidence@4.1"),
            ("format", "literal[skillev-evolution@5]"),
            ("library_version_after", "string"),
            ("library_version_before", "string"),
            ("lineage_ref", "string"),
            ("phase_event_id", "string"),
            ("proposal_content_hash", "sha256"),
            ("rationale_text", "string"),
            ("target_skill_id", "string"),
        ),
    ),
    (
        "skillev.prune-evidence",
        "PruneEvidence",
        (
            ("evidence_type", "literal[prune]"),
            ("flow_evidence_kind", "enum[observed-low,zero-invocation]"),
            ("flow_quantile", "finite-number[0,1]|null"),
            ("k", "finite-number>=0"),
            ("log_skill_marginal_flow", "finite-number|null"),
            ("posterior_event_ids", "array[string]"),
            ("ucb", "finite-number"),
            ("zero_invocation_window_id", "string|null"),
        ),
    ),
    (
        "skillev.refine-action-record",
        "RefineActionRecord",
        (
            ("action_id", "string"),
            ("action_type", "literal[refine]"),
            ("evidence", "schema:skillev.refine-evidence@4.1"),
            ("format", "literal[skillev-evolution@5]"),
            ("library_version_after", "string"),
            ("library_version_before", "string"),
            ("lineage_ref", "string"),
            ("phase_event_id", "string"),
            ("proposal_content_hash", "sha256"),
            ("produced_skill_id", "string"),
            ("rationale_text", "string"),
            ("target_skill_id", "string"),
        ),
    ),
    (
        "skillev.refine-evidence",
        "RefineEvidence",
        (
            ("evidence_type", "literal[refine]"),
            ("flow_quantile", "finite-number[0,1]"),
            ("k", "finite-number>=0"),
            ("lcb", "finite-number"),
            ("log_skill_marginal_flow", "finite-number"),
            ("posterior_event_ids", "array[string]"),
            ("target_context_keys", "array[string]"),
            ("target_contexts", "array[schema:skillev.context-feature@4.1]"),
        ),
    ),
    (
        "skillev.retain-compress-action-record",
        "RetainCompressActionRecord",
        (
            ("action_id", "string"),
            ("action_type", "literal[retain-compress]"),
            (
                "evidence",
                "schema:skillev.retain-evidence@4.1",
            ),
            ("format", "literal[skillev-evolution@5]"),
            ("library_version_after", "string"),
            ("library_version_before", "string"),
            ("lineage_ref", "string"),
            ("phase_event_id", "string"),
            ("proposal_content_hash", "sha256"),
            ("produced_skill_id", "string"),
            ("rationale_text", "string"),
            ("target_skill_id", "string"),
        ),
    ),
    (
        "skillev.retain-evidence",
        "RetainEvidence",
        (
            ("evidence_type", "literal[retain]"),
            ("flow_quantile", "finite-number[0,1]"),
            ("k", "finite-number>=0"),
            ("lcb", "finite-number"),
            ("log_skill_marginal_flow", "finite-number"),
            ("posterior_event_ids", "array[string]"),
        ),
    ),
    (
        "skillev.split-action-record",
        "SplitActionRecord",
        (
            ("action_id", "string"),
            ("action_type", "literal[split]"),
            ("evidence", "schema:skillev.split-evidence@4.1"),
            ("format", "literal[skillev-evolution@5]"),
            ("library_version_after", "string"),
            ("library_version_before", "string"),
            ("lineage_ref", "string"),
            ("phase_event_id", "string"),
            ("proposal_content_hash", "sha256"),
            ("produced_skill_ids", "array[string;length=2]"),
            ("rationale_text", "string"),
            ("target_skill_id", "string"),
        ),
    ),
    (
        "skillev.split-evidence",
        "SplitEvidence",
        (
            ("evidence_type", "literal[split]"),
            ("flow_quantile", "finite-number[0,1]"),
            ("log_skill_marginal_flow", "finite-number"),
            ("modality", "schema:skillev.split-modality-evidence@4.1"),
        ),
    ),
    (
        "skillev.split-modality-evidence",
        "SplitModalityEvidence",
        (
            ("assignments", "array[split-task-family-assignment]"),
            ("between_mean_gap", "finite-number>=0"),
            ("high_mode", "schema:skillev.posterior-task-family-mode@4.1"),
            ("intervals_disjoint", "literal[true]"),
            ("low_mode", "schema:skillev.posterior-task-family-mode@4.1"),
            ("separation_cutpoint", "finite-number"),
            ("source_task_families", "array[string;sorted;unique]"),
        ),
    ),
    (
        "skillev.terminal-reward",
        "TerminalReward",
        (
            ("environment_id", "string"),
            ("native_metric_name", "string"),
            ("native_payload", "json-object"),
            ("success", "boolean"),
            (
                "success_rule",
                "enum[r-equals-one,r-at-threshold,trusted-native-projection]",
            ),
            ("success_threshold", "finite-number(0,1]|null"),
            ("value", "finite-number[0,1]"),
            ("verifier_version", "string"),
        ),
    ),
    (
        "skillev.training-step-commit",
        "TrainingStepCommit",
        (
            ("batch_id", "string"),
            ("edge_records", "array[schema:skillev.edge-logprob-record@4.1]"),
            ("format", "literal[skillev-training-step-commit@4]"),
            ("library_version", "string"),
            ("optimizer_step", "integer>=1"),
            ("policy_snapshot_after", "string"),
            ("policy_snapshot_before", "string"),
            ("posterior_batch", "schema:skillev.posterior-batch-update@4.1"),
            ("records", "array[schema:skillev.trajectory-record@4.1]"),
            ("report", "schema:skillev.training-step-report-value@4.1"),
            ("run_cursor_after", "schema:skillev.source-run-cursor@4.1"),
            ("stats", "schema:skillev.ttb-batch-stats@4.1"),
        ),
    ),
    (
        "skillev.training-step-report-value",
        "TrainingStepReportValue",
        (
            ("audited_batch_loss", "finite-number>=0"),
            ("backward_adapter_version", "string"),
            ("batch_id", "string"),
            ("completed_at", "iso-8601-string"),
            (
                "format",
                "literal[skillev-training-step-report@3,skillev-training-step-report@4,skillev-training-step-report@5]",
            ),
            ("forward_adapter_version", "string"),
            ("grad_norm_backward", "finite-number>=0"),
            ("grad_norm_forward", "finite-number>=0"),
            ("grad_norm_z", "finite-number>=0"),
            ("mean_reward", "finite-number[0,1]"),
            ("optimization_diagnostics", "json-object|null;present-only-in-report@4/@5"),
            ("optimizer_transition", "json-object|null;present-only-in-report@5"),
            ("optimizer_step", "integer>=1"),
            ("started_at", "iso-8601-string"),
            ("torch_batch_loss", "finite-number>=0"),
            ("z_version", "string"),
        ),
    ),
    (
        "skillev.trajectory-record",
        "TrajectoryRecord",
        (
            ("created_at", "iso-8601-string"),
            ("decoding_snapshot_id", "string"),
            ("environment_id", "string"),
            ("epsilon_min", "finite-number>0"),
            ("horizon", "integer>=1"),
            ("initial_context", "schema:skillev.initial-context@4.1"),
            ("reward", "schema:skillev.terminal-reward@4.1"),
            ("shifted_reward", "finite-number"),
            ("steps", "array[schema:skillev.trajectory-step@4.1]"),
            ("task_family", "string"),
            ("tokenizer_id", "string"),
            ("trajectory_id", "string"),
        ),
    ),
    (
        "skillev.trajectory-residual",
        "TrajectoryResidual",
        (
            ("delta", "finite-number"),
            ("horizon", "integer>=1"),
            ("log_shifted_reward", "finite-number"),
            ("log_z", "finite-number"),
            ("raw_reward", "finite-number[0,1]"),
            ("sum_backward", "finite-number"),
            ("sum_forward", "finite-number"),
            ("temperature_beta", "finite-number>0"),
            ("trajectory_id", "string"),
        ),
    ),
    (
        "skillev.trajectory-step",
        "TrajectoryStep",
        (
            ("action_text", "string"),
            ("action_token_count", "integer>=1"),
            ("action_token_ids", "array[integer>=0]"),
            ("forward_prefix_hash", "sha256"),
            ("hindsight_prefix_hash", "sha256"),
            ("index", "integer>=1"),
            ("invoked_skill_ids", "array[string]"),
            (
                "observation_status",
                "enum[success,tool_error,schema_invalid,timeout,parse_error,other]",
            ),
            ("observation_text", "string"),
            ("reasoning_text", "string"),
        ),
    ),
    (
        "skillev.ttb-batch-stats",
        "TTBBatchStats",
        (
            ("batch_id", "string"),
            ("batch_loss", "finite-number>=0"),
            ("created_at", "iso-8601-string"),
            ("library_version", "string"),
            ("mean_reward", "finite-number[0,1]"),
            ("optimizer_step", "integer>=0"),
            ("residuals", "array[schema:skillev.trajectory-residual@4.1]"),
        ),
    ),
    (
        "skillev.window-stats",
        "WindowStats",
        (
            ("batch_count", "integer>=1"),
            ("end_optimizer_step", "integer>=0"),
            ("mean_squared_residual", "finite-number>=0"),
            ("member_batch_ids", "array[string]"),
            ("start_optimizer_step", "integer>=0"),
        ),
    ),
)


def _descriptor(spec: SchemaSpec) -> SchemaDescriptor:
    schema_id, record_name, fields = spec
    normalized_fields = tuple(sorted(fields))
    digest = stable_hash(
        {
            "canonicalization_version": CANONICALIZATION_VERSION,
            "closed_object": True,
            "fields": [
                {"name": field_name, "wire_type": wire_type}
                for field_name, wire_type in normalized_fields
            ],
            "record_name": record_name,
            "schema_id": schema_id,
            "version": str(TTB_SCHEMA_VERSION),
        }
    )
    return SchemaDescriptor(
        schema_id=schema_id,
        version=TTB_SCHEMA_VERSION,
        schema_digest=digest,
    )


def build_ttb_schema_registry() -> SchemaRegistry:
    """Build a fresh, uniquely keyed registry in deterministic sort order."""

    descriptors = tuple(
        sorted(
            (_descriptor(spec) for spec in _TTB_SCHEMA_SPECS),
            key=lambda descriptor: (descriptor.schema_id, descriptor.version),
        )
    )
    return SchemaRegistry(descriptors)


TTB_SCHEMA_REGISTRY: Final = build_ttb_schema_registry()
ALL_TTB_SCHEMA_IDS: Final = tuple(
    descriptor.schema_id for descriptor in TTB_SCHEMA_REGISTRY.schemas
)
