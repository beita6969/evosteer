"""Strict no-Bayesian experiment arm with no full-method evolution values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from skillev.contracts import EdgeLogprobRecord, TrajectoryRecord, TTBBatchStats
from skillev.diagnostics import (
    BatchDiagnostics,
    CommittedDiagnostic,
    DiagnosticsConfig,
    DiagnosticsState,
    NoCommittedDiagnostic,
    OnlineFlowDiagnostics,
)
from skillev.evolution.config import EvolutionConfig
from skillev.evolution.evidence import (
    AuthoringEdgeEvidence,
    TrajectoryEvidenceView,
    WindowFlowView,
    authoring_evidence_for_diagnostic,
    generate_candidate_contribution,
    group_generate_exemplars,
    select_uncovered_generate_edges,
)
from skillev.training.projections import (
    FlowOnlyProjectionRuntimeState,
    OnlineDiagnosticsProjection,
    ProjectionBatchRuntimeState,
)

from .no_bayesian_contracts import (
    FlowOnlyDecision,
    FlowOnlyGenerateEvidence,
    FlowOnlyGenerateProposal,
    FlowOnlyRetainEvidence,
    FlowOnlyRetainProposal,
)


@dataclass(frozen=True, slots=True)
class UninvokedFlowOnlySkill:
    skill_id: str


@dataclass(frozen=True, slots=True)
class ObservedFlowOnlySkill:
    skill_id: str
    log_skill_marginal_flow: float
    flow_quantile: float
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...]


FlowOnlySkillEvidence: TypeAlias = UninvokedFlowOnlySkill | ObservedFlowOnlySkill


@dataclass(frozen=True, slots=True)
class FlowOnlyTrainingSource:
    batch_id: str
    optimizer_step: int
    records: tuple[TrajectoryRecord, ...]
    stats: TTBBatchStats
    edge_records: tuple[EdgeLogprobRecord, ...]

    def __post_init__(self) -> None:
        if not self.records or any(not isinstance(item, TrajectoryRecord) for item in self.records):
            raise ValueError("flow-only training source requires trajectory records")


@dataclass(frozen=True, slots=True)
class FlowOnlyProjectionTransition:
    source: FlowOnlyTrainingSource
    next_state: FlowOnlyProjectionRuntimeState

    @property
    def diagnostic(self) -> BatchDiagnostics:
        latest = self.next_state.latest_diagnostic
        if not isinstance(latest, CommittedDiagnostic):
            raise RuntimeError("flow-only transition lacks a committed diagnostic")
        return latest.diagnostic


@dataclass(frozen=True, slots=True)
class FlowOnlyEvidencePack:
    phase_event_id: str
    skills: tuple[FlowOnlySkillEvidence, ...]
    uncovered_edge_ids: tuple[str, ...]
    uncovered_exemplars: tuple[AuthoringEdgeEvidence, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(self.skills, key=lambda item: item.skill_id)) != self.skills:
            raise ValueError("flow-only skill evidence must be sorted")
        if len({item.skill_id for item in self.skills}) != len(self.skills):
            raise ValueError("flow-only skill evidence repeats a skill")
        if tuple(item.edge_id for item in self.uncovered_exemplars) != self.uncovered_edge_ids:
            raise ValueError("flow-only uncovered IDs differ from exemplars")


class FlowOnlyProjectionPipeline:
    """Produce only diagnostics for the no-Bayesian application graph."""

    def __init__(
        self,
        diagnostics: OnlineDiagnosticsProjection,
        state: FlowOnlyProjectionRuntimeState,
    ) -> None:
        self._diagnostics = diagnostics
        self._state = state

    @classmethod
    def fresh(
        cls,
        config: DiagnosticsConfig,
        *,
        library_version: str,
    ) -> FlowOnlyProjectionPipeline:
        diagnostics = OnlineFlowDiagnostics.fresh(config, library_version=library_version)
        return cls(
            diagnostics,
            FlowOnlyProjectionRuntimeState(
                diagnostics_state=diagnostics.state,
                latest_diagnostic=diagnostics.latest_state,
                current_segment_batches=(),
                retained_batch_count=max(1, 2 * config.window_size),
            ),
        )

    def preview(self, source: FlowOnlyTrainingSource) -> FlowOnlyProjectionTransition:
        transition = self._diagnostics.preview_from_state(
            source,
            self._state.diagnostics_state,
        )
        batch = ProjectionBatchRuntimeState(
            diagnostic=transition.diagnostic,
            authoring_evidence=_flow_only_answer_free_evidence(
                source,
                transition.diagnostic,
            ),
        )
        batches = (*self._state.current_segment_batches, batch)[-self._state.retained_batch_count :]
        return FlowOnlyProjectionTransition(
            source=source,
            next_state=FlowOnlyProjectionRuntimeState(
                diagnostics_state=transition.next_state,
                latest_diagnostic=CommittedDiagnostic(transition.diagnostic),
                current_segment_batches=batches,
                retained_batch_count=self._state.retained_batch_count,
            ),
        )

    def commit(self, transition: FlowOnlyProjectionTransition) -> None:
        self._state = transition.next_state

    @property
    def latest_diagnostic(self) -> BatchDiagnostics:
        latest = self._state.latest_diagnostic
        if not isinstance(latest, CommittedDiagnostic):
            raise RuntimeError("no flow-only diagnostic has been committed")
        return latest.diagnostic

    @property
    def diagnostics_state(self) -> DiagnosticsState:
        return self._state.diagnostics_state

    def reset_library_segment(
        self,
        old_library_version: str,
        new_library_version: str,
    ) -> None:
        self.commit_reset(
            self.preview_reset(
                old_version=old_library_version,
                new_version=new_library_version,
            )
        )

    def preview_reset(
        self,
        *,
        old_version: str,
        new_version: str,
    ) -> FlowOnlyProjectionRuntimeState:
        from skillev.diagnostics import reset_diagnostics_segment

        return FlowOnlyProjectionRuntimeState(
            diagnostics_state=reset_diagnostics_segment(
                self._state.diagnostics_state,
                old_library_version=old_version,
                new_library_version=new_version,
            ),
            latest_diagnostic=NoCommittedDiagnostic(),
            current_segment_batches=(),
            retained_batch_count=self._state.retained_batch_count,
        )

    def commit_reset(self, state: FlowOnlyProjectionRuntimeState) -> None:
        self._state = state

    def runtime_state(self) -> FlowOnlyProjectionRuntimeState:
        return self._state

    @classmethod
    def from_runtime_state(
        cls,
        config: DiagnosticsConfig,
        state: FlowOnlyProjectionRuntimeState,
    ) -> FlowOnlyProjectionPipeline:
        return cls(
            OnlineFlowDiagnostics.from_runtime_state(
                config,
                state.diagnostics_state,
                state.latest_diagnostic,
            ),
            state,
        )

    def diagnostics_for_batch_ids(
        self,
        batch_ids: tuple[str, ...],
    ) -> tuple[BatchDiagnostics, ...]:
        by_batch = {
            item.diagnostic.batch_id: item.diagnostic
            for item in self._state.current_segment_batches
        }
        if any(batch_id not in by_batch for batch_id in batch_ids):
            raise KeyError("flow-only phase references an absent diagnostic")
        return tuple(by_batch[batch_id] for batch_id in batch_ids)

    def authoring_evidence_for_diagnostics(
        self,
        diagnostics: tuple[BatchDiagnostics, ...],
    ) -> Mapping[str, AuthoringEdgeEvidence]:
        requested = tuple(item.batch_id for item in diagnostics)
        by_batch = {item.diagnostic.batch_id: item for item in self._state.current_segment_batches}
        if any(batch_id not in by_batch for batch_id in requested):
            raise KeyError("flow-only phase references an absent training source")
        return {
            evidence.edge_id: evidence
            for batch_id in requested
            for evidence in by_batch[batch_id].authoring_evidence
        }

    def authoring_evidence_for_batch_ids(
        self,
        batch_ids: tuple[str, ...],
    ) -> Mapping[str, AuthoringEdgeEvidence]:
        return self.authoring_evidence_for_diagnostics(self.diagnostics_for_batch_ids(batch_ids))


def _flow_only_answer_free_evidence(
    source: FlowOnlyTrainingSource,
    diagnostic: BatchDiagnostics,
) -> tuple[AuthoringEdgeEvidence, ...]:
    return authoring_evidence_for_diagnostic(
        {record.trajectory_id: record for record in source.records},
        diagnostic,
    )


class FlowOnlyEvolutionPolicy:
    def decide(
        self,
        pack: FlowOnlyEvidencePack,
        config: EvolutionConfig,
    ) -> FlowOnlyDecision:
        proposals: list[FlowOnlyRetainProposal | FlowOnlyGenerateProposal] = []
        for skill in pack.skills:
            match skill:
                case UninvokedFlowOnlySkill():
                    pass
                case ObservedFlowOnlySkill() if skill.flow_quantile >= config.high_flow_quantile:
                    proposals.append(
                        FlowOnlyRetainProposal(
                            target_skill_id=skill.skill_id,
                            evidence=FlowOnlyRetainEvidence(
                                log_skill_marginal_flow=skill.log_skill_marginal_flow,
                                flow_quantile=skill.flow_quantile,
                            ),
                            rationale_text="high flow in no-Bayesian arm",
                            edge_exemplars=skill.edge_exemplars,
                        )
                    )
                case ObservedFlowOnlySkill():
                    pass
                case _:
                    raise TypeError(type(skill).__name__)
        for group in group_generate_exemplars(
            importance_edge_ids=pack.uncovered_edge_ids,
            edge_exemplars=pack.uncovered_exemplars,
        ):
            proposals.append(
                FlowOnlyGenerateProposal(
                    evidence=FlowOnlyGenerateEvidence(
                        importance_edge_ids=group.importance_edge_ids,
                        minimum_absolute_log_importance=(
                            config.generate_min_absolute_log_importance
                        ),
                        importance_quantile=config.importance_quantile,
                        importance_semantics=config.generate_importance_semantics,
                    ),
                    rationale_text="uncovered high-|log I| edge in no-Bayesian arm",
                    edge_exemplars=group.edge_exemplars,
                )
            )
        if not proposals:
            raise ValueError(f"phase {pack.phase_event_id} has no flow-only arm action")
        return FlowOnlyDecision(
            phase_event_id=pack.phase_event_id,
            proposals=tuple(proposals),
        )

    def decide_from_views(
        self,
        *,
        window_flow: WindowFlowView,
        trajectories: TrajectoryEvidenceView,
        config: EvolutionConfig,
    ) -> FlowOnlyDecision:
        return self.decide(
            build_flow_only_evidence_pack(
                window_flow=window_flow,
                trajectories=trajectories,
                config=config,
            ),
            config,
        )


def build_flow_only_evidence_pack(
    *,
    window_flow: WindowFlowView,
    trajectories: TrajectoryEvidenceView,
    config: EvolutionConfig,
) -> FlowOnlyEvidencePack:
    phase = window_flow.phase_event
    expected_batches = (
        *phase.previous_window.member_batch_ids,
        *phase.current_window.member_batch_ids,
    )
    if tuple(item.batch_id for item in window_flow.diagnostics) != expected_batches:
        raise ValueError("flow-only diagnostics differ from the phase event")

    calls: dict[str, list[tuple[str, int, float, AuthoringEdgeEvidence]]] = {
        skill_id: [] for skill_id in window_flow.active_skill_ids
    }
    active = set(window_flow.active_skill_ids)
    uncovered: list[tuple[str, float, AuthoringEdgeEvidence]] = []
    all_absolute: list[float] = []
    for diagnostic in window_flow.diagnostics:
        for trajectory in diagnostic.trajectories:
            for edge in trajectory.edges:
                edge_id = f"{trajectory.trajectory_id}:{edge.step_index}"
                try:
                    stored = trajectories.authoring_by_edge_id[edge_id]
                except KeyError as error:
                    raise ValueError(f"missing authoring evidence {edge_id!r}") from error
                absolute = abs(edge.log_importance)
                if stored.absolute_log_importance != absolute:
                    raise ValueError("stored authoring importance differs from diagnostics")
                exemplar = stored
                contribution = generate_candidate_contribution(
                    exemplar=exemplar,
                    absolute_importance=absolute,
                    invoked_skill_ids=edge.invoked_skill_ids,
                )
                if contribution is not None:
                    all_absolute.append(contribution.absolute_importance)
                if edge.invoked_skill_ids:
                    for skill_id in edge.invoked_skill_ids:
                        if skill_id not in active:
                            raise ValueError("flow-only window invokes an inactive skill")
                        calls[skill_id].append(
                            (
                                trajectory.trajectory_id,
                                edge.step_index,
                                edge.sample_log_state_weight,
                                exemplar,
                            )
                        )
                elif contribution is not None and contribution.uncovered is not None:
                    uncovered.append(contribution.uncovered)

    observed: dict[str, tuple[float, tuple[AuthoringEdgeEvidence, ...]]] = {}
    for skill_id in window_flow.active_skill_ids:
        skill_calls = calls[skill_id]
        if not skill_calls:
            continue
        trajectory_ids = {item[0] for item in skill_calls}
        log_flow = _logsumexp(tuple(item[2] for item in skill_calls)) - math.log(
            len(trajectory_ids)
        )
        observed[skill_id] = (log_flow, tuple(item[3] for item in skill_calls))
    population = tuple(item[0] for item in observed.values())
    skills: list[FlowOnlySkillEvidence] = []
    for skill_id in window_flow.active_skill_ids:
        if skill_id not in observed:
            skills.append(UninvokedFlowOnlySkill(skill_id))
            continue
        log_flow, exemplars = observed[skill_id]
        skills.append(
            ObservedFlowOnlySkill(
                skill_id=skill_id,
                log_skill_marginal_flow=log_flow,
                flow_quantile=sum(value <= log_flow for value in population) / len(population),
                edge_exemplars=exemplars,
            )
        )

    selected = select_uncovered_generate_edges(
        uncovered,
        tuple(all_absolute),
        importance_quantile=config.importance_quantile,
        minimum_absolute_log_importance=config.generate_min_absolute_log_importance,
    )
    return FlowOnlyEvidencePack(
        phase_event_id=phase.event_id,
        skills=tuple(skills),
        uncovered_edge_ids=tuple(item[0] for item in selected),
        uncovered_exemplars=tuple(item[2] for item in selected),
    )


def _logsumexp(values: tuple[float, ...]) -> float:
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


__all__ = [
    "FlowOnlyEvidencePack",
    "FlowOnlyEvolutionPolicy",
    "FlowOnlyProjectionPipeline",
    "FlowOnlyProjectionTransition",
    "FlowOnlySkillEvidence",
    "FlowOnlyTrainingSource",
    "ObservedFlowOnlySkill",
    "UninvokedFlowOnlySkill",
    "build_flow_only_evidence_pack",
]
