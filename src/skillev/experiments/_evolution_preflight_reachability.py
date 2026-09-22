"""Constructive production-detector and production-policy reachability witness."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from skillev.calibration import CalibrationConfig
from skillev.contracts import (
    ContextFeature,
    FailureMode,
    HorizonBucket,
    JsonValue,
    PosteriorCellState,
    TokenBucket,
)
from skillev.diagnostics import (
    BatchDiagnostics,
    ComparableResidualWindows,
    DiagnosticsConfig,
    EdgeFlowDiagnostic,
    InsufficientResidualWindow,
    SkillFlowStat,
    TrajectoryFlowDiagnostic,
)
from skillev.evolution import (
    AuthoringActionKind,
    AuthoringEdgeEvidence,
    EvolutionConfig,
    EvolutionDecision,
    FullEvolutionPolicy,
    PhaseTransitionDetected,
    PhaseTransitionDetector,
    PosteriorEvidenceView,
    TrajectoryEvidenceView,
    VerifiedNoOpEvolutionDecision,
    WindowFlowView,
    assemble_posterior_evidence_view,
)
from skillev.runtime import SkillDocument, SkillLibraryState


class ReachabilityInputs(Protocol):
    @property
    def seed_documents(self) -> tuple[SkillDocument, ...]: ...

    @property
    def diagnostics_config(self) -> DiagnosticsConfig: ...

    @property
    def calibration_config(self) -> CalibrationConfig: ...

    @property
    def evolution_config(self) -> EvolutionConfig: ...

    @property
    def initial_library(self) -> SkillLibraryState: ...

    @property
    def minimum_batches_per_phase(self) -> int: ...


@dataclass(frozen=True, slots=True)
class EvolutionReachabilityProof:
    phase_transition_detected: bool
    trigger_optimizer_step: int
    triggering_window_batch_count: int
    entropy_values: tuple[float, ...]
    decision_nonempty: bool
    decision_proposal_types: tuple[str, ...]
    verified_no_op_reachable: bool

    def __post_init__(self) -> None:
        if self.phase_transition_detected is not True or self.decision_nonempty is not True:
            raise ValueError("planned reachability proof must trigger and produce a decision")
        if type(self.trigger_optimizer_step) is not int or self.trigger_optimizer_step < 1:
            raise ValueError("trigger_optimizer_step must be positive")
        if (
            type(self.triggering_window_batch_count) is not int
            or self.triggering_window_batch_count < 1
        ):
            raise ValueError("triggering_window_batch_count must be positive")
        if not self.entropy_values or any(not math.isfinite(item) for item in self.entropy_values):
            raise ValueError("entropy witness must contain finite observations")
        if any(
            current >= previous
            for previous, current in zip(
                self.entropy_values,
                self.entropy_values[1:],
                strict=False,
            )
        ):
            raise ValueError("entropy witness must be strictly decreasing")
        if not self.decision_proposal_types:
            raise ValueError("decision witness must contain proposals")
        if self.verified_no_op_reachable is not True:
            raise ValueError("dead-band evidence must produce an explicit verified no-op")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "verified_no_op_reachable": self.verified_no_op_reachable,
            "decision_nonempty": self.decision_nonempty,
            "decision_proposal_types": list(self.decision_proposal_types),
            "entropy_values": list(self.entropy_values),
            "phase_transition_detected": self.phase_transition_detected,
            "trigger_optimizer_step": self.trigger_optimizer_step,
            "triggering_window_batch_count": self.triggering_window_batch_count,
        }


def _diagnostics_witness(
    inputs: ReachabilityInputs,
    *,
    non_skill_action_kind: AuthoringActionKind,
    zero_importance: bool = False,
) -> tuple[tuple[BatchDiagnostics, ...], dict[str, AuthoringEdgeEvidence]]:
    minimum_batches = inputs.minimum_batches_per_phase
    entropy_window = inputs.evolution_config.entropy_window
    drops = inputs.evolution_config.required_consecutive_drops
    trigger_step = minimum_batches
    first_entropy_end = trigger_step - drops
    first_entropy_start = first_entropy_end - entropy_window + 1
    witness_skills = tuple(
        document.manifest.skill_id for document in reversed(inputs.seed_documents[: drops + 1])
    )
    skill_by_step = {
        first_entropy_start + offset: skill_id for offset, skill_id in enumerate(witness_skills)
    }
    diagnostics: list[BatchDiagnostics] = []
    authoring: dict[str, AuthoringEdgeEvidence] = {}
    library_version = inputs.initial_library.current_version
    for step in range(1, trigger_step + 1):
        trajectory_id = f"f2-f3-public-{step:04d}"
        edge_id = f"{trajectory_id}:1"
        skill_id = skill_by_step.get(step)
        invoked = () if skill_id is None else (skill_id,)
        log_importance = (
            0.0
            if zero_importance or skill_id is not None
            else (2.0 if step == trigger_step else 1.0)
        )
        edge = EdgeFlowDiagnostic(
            trajectory_id=trajectory_id,
            step_index=1,
            log_importance=log_importance,
            sample_log_state_weight=log_importance,
            invoked_skill_ids=invoked,
        )
        trajectory = TrajectoryFlowDiagnostic(
            trajectory_id=trajectory_id,
            horizon=1,
            edges=(edge,),
            terminal_sample_log_state_weight=log_importance,
        )
        skill_flows = (
            ()
            if skill_id is None
            else (
                SkillFlowStat(
                    skill_id=skill_id,
                    invoking_trajectory_count=1,
                    invoking_edge_count=1,
                    log_skill_flow=log_importance,
                ),
            )
        )
        residual: InsufficientResidualWindow | ComparableResidualWindows
        if step < trigger_step:
            residual = InsufficientResidualWindow(
                observed_batch_count=step,
                required_batch_count=2 * inputs.diagnostics_config.window_size,
            )
        else:
            residual = ComparableResidualWindows(
                library_version=library_version,
                window_size=inputs.diagnostics_config.window_size,
                previous_window_mean_delta_squared=1.0,
                current_window_mean_delta_squared=1.0,
                relative_improvement=0.0,
                rho=inputs.diagnostics_config.stagnation_rho,
                stagnant=True,
            )
        diagnostics.append(
            BatchDiagnostics(
                batch_id=f"f2-f3-batch-{step:04d}",
                optimizer_step=step,
                library_version=library_version,
                trajectories=(trajectory,),
                skill_flows=skill_flows,
                residual_window=residual,
                config=inputs.diagnostics_config,
            )
        )
        action_kind = AuthoringActionKind.SKILL if skill_id is not None else non_skill_action_kind
        authoring[edge_id] = AuthoringEdgeEvidence(
            edge_id=edge_id,
            task_family="public-preflight",
            context_id="public-preflight",
            action_kind=action_kind,
            tool_or_skill_name=skill_id
            or (
                "preflight.tool"
                if non_skill_action_kind is AuthoringActionKind.TOOL
                else "complete"
            ),
            argument_schema_id="public-preflight-action@1",
            observation_status=FailureMode.SUCCESS,
            token_bucket=TokenBucket.LE_1K,
            horizon_bucket=HorizonBucket.LE_3,
            absolute_log_importance=abs(log_importance),
            log_importance_quantile=0.0,
            invoked_skill_ids=invoked,
            available_tools=("preflight.tool",),
        )
    return tuple(diagnostics), authoring


def _posterior_view(inputs: ReachabilityInputs) -> PosteriorEvidenceView:
    z = ContextFeature(
        context="public-preflight",
        failure_mode=FailureMode.SUCCESS,
        token_bucket=TokenBucket.LE_1K,
        horizon_bucket=HorizonBucket.LE_3,
    )
    cells = tuple(
        PosteriorCellState(
            skill_id=document.manifest.skill_id,
            z=z,
            alpha=inputs.calibration_config.alpha_0 + 1.0,
            beta_count=inputs.calibration_config.beta_0,
            alpha_0=inputs.calibration_config.alpha_0,
            beta_0=inputs.calibration_config.beta_0,
            update_count=1,
            last_event_id=f"public-update-{document.manifest.skill_id}",
        )
        for document in inputs.seed_documents
    )
    event_ids = {
        cell.z.cell_key(cell.skill_id): (f"public-update-{cell.skill_id}",) for cell in cells
    }
    return assemble_posterior_evidence_view(
        active_skill_ids=inputs.initial_library.active_skill_ids,
        cells=cells,
        event_ids_by_cell=event_ids,
    )


def build_evolution_reachability(inputs: ReachabilityInputs) -> EvolutionReachabilityProof:
    diagnostics, authoring = _diagnostics_witness(
        inputs,
        non_skill_action_kind=AuthoringActionKind.TOOL,
    )
    detector = PhaseTransitionDetector.fresh(
        evolution_config=inputs.evolution_config,
        diagnostics_config=inputs.diagnostics_config,
        library_version=inputs.initial_library.current_version,
    )
    observation = None
    for diagnostic in diagnostics:
        observation = detector.observe(diagnostic)
    if not isinstance(observation, PhaseTransitionDetected):
        raise RuntimeError("constructive F2 witness did not trigger the production detector")
    posterior = _posterior_view(inputs)
    policy = FullEvolutionPolicy()
    decision = policy.decide_from_views(
        window_flow=WindowFlowView(
            phase_event=observation.event,
            diagnostics=observation.triggering_window,
            active_skill_ids=inputs.initial_library.active_skill_ids,
            applicability_by_skill={
                skill_id: inputs.initial_library.documents[skill_id].applicability
                for skill_id in inputs.initial_library.active_skill_ids
            },
            task_family_universe=("public-preflight",),
        ),
        posterior=posterior,
        trajectories=TrajectoryEvidenceView(authoring_by_edge_id=authoring),
        config=inputs.evolution_config,
    )

    # Synthetic no-op witness: public completions are now Generate-eligible.
    # Use genuinely below-floor asymmetry, not the retired exclusion of COMPLETE.
    # This does not change the production threshold or claim a natural phase.
    dead_diagnostics, terminal_authoring = _diagnostics_witness(
        inputs,
        non_skill_action_kind=AuthoringActionKind.COMPLETE,
        zero_importance=True,
    )
    dead_band_decision = policy.decide_from_views(
        window_flow=WindowFlowView(
            phase_event=observation.event,
            diagnostics=dead_diagnostics[-2 * inputs.diagnostics_config.window_size :],
            active_skill_ids=inputs.initial_library.active_skill_ids,
            applicability_by_skill={
                skill_id: inputs.initial_library.documents[skill_id].applicability
                for skill_id in inputs.initial_library.active_skill_ids
            },
            task_family_universe=("public-preflight",),
        ),
        posterior=posterior,
        trajectories=TrajectoryEvidenceView(authoring_by_edge_id=terminal_authoring),
        config=inputs.evolution_config,
    )
    if not isinstance(decision, EvolutionDecision):
        raise ValueError("constructive action witness unexpectedly produced a no-op")

    entropy_values = tuple(item.entropy for item in observation.event.entropy_series)
    return EvolutionReachabilityProof(
        phase_transition_detected=True,
        trigger_optimizer_step=observation.event.triggered_at_step,
        triggering_window_batch_count=len(observation.triggering_window),
        entropy_values=entropy_values,
        decision_nonempty=bool(decision.proposals),
        decision_proposal_types=tuple(
            type(proposal).__name__.removesuffix("Proposal").lower()
            for proposal in decision.proposals
        ),
        verified_no_op_reachable=isinstance(
            dead_band_decision,
            VerifiedNoOpEvolutionDecision,
        ),
    )


__all__ = ["EvolutionReachabilityProof", "build_evolution_reachability"]
