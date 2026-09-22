"""Total, typed evidence assembly for the full-method Phi policy."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from enum import StrEnum
from itertools import combinations
from types import MappingProxyType
from typing import TypeAlias, cast

from skillev.calibration import CellQuery
from skillev.calibration.core import query_cell
from skillev.contracts import (
    FailureMode,
    HorizonBucket,
    JsonValue,
    PhaseTransitionEvent,
    PosteriorCellState,
    PosteriorTaskFamilyMode,
    SplitBranch,
    SplitModalityEvidence,
    SplitTaskFamilyAssignment,
    TokenBucket,
    TrajectoryRecord,
    stable_hash,
)
from skillev.diagnostics import BatchDiagnostics
from skillev.runtime import SkillApplicability

from .config import EvolutionConfig, SplitCriterionConfig
from .public_execution import PublicExecutionSnippet


@dataclass(frozen=True, slots=True)
class WindowFlowView:
    phase_event: PhaseTransitionEvent
    diagnostics: tuple[BatchDiagnostics, ...]
    active_skill_ids: tuple[str, ...]
    applicability_by_skill: Mapping[str, SkillApplicability]
    task_family_universe: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.phase_event, PhaseTransitionEvent):
            raise TypeError("window flow view requires a phase event")
        if not self.diagnostics:
            raise ValueError("window flow view requires diagnostics")
        if self.diagnostics[-1].optimizer_step != self.phase_event.triggered_at_step or any(
            item.library_version != self.phase_event.library_version for item in self.diagnostics
        ):
            raise ValueError("window flow view differs from the phase library or evidence cutoff")
        if tuple(sorted(set(self.active_skill_ids))) != self.active_skill_ids:
            raise ValueError("active_skill_ids must be sorted and unique")
        if set(self.applicability_by_skill) != set(self.active_skill_ids):
            raise ValueError("applicability view must match the active library exactly")
        if tuple(sorted(set(self.task_family_universe))) != self.task_family_universe:
            raise ValueError("task_family_universe must be sorted and unique")
        if not self.task_family_universe or "*" in self.task_family_universe:
            raise ValueError("task_family_universe must be finite and non-empty")
        universe = set(self.task_family_universe)
        copied = dict(self.applicability_by_skill)
        for skill_id in self.active_skill_ids:
            applicability = self.applicability_by_skill[skill_id]
            if not isinstance(applicability, SkillApplicability):
                raise TypeError("applicability view contains an incompatible value")
            if (
                applicability.task_families != ("*",)
                and not set(applicability.task_families) <= universe
            ):
                raise ValueError("active skill targets a family outside the formal universe")
        object.__setattr__(self, "applicability_by_skill", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class TrajectoryEvidenceView:
    authoring_by_edge_id: Mapping[str, AuthoringEdgeEvidence]
    source_by_trajectory: Mapping[str, tuple[str, str, str] | None] = dataclass_field(
        default_factory=dict
    )


class AuthoringActionKind(StrEnum):
    TOOL = "tool"
    SKILL = "skill"
    COMPLETE = "complete"
    INVALID = "invalid"


def is_generate_candidate_action(action_kind: AuthoringActionKind) -> bool:
    """Eligible public actions, including direct static completion (protocol @2).

    Coverage is explicit invocation coverage, NOT applicability or causal use.
    Exposure is recorded separately; malformed actions are not executions.
    """
    if not isinstance(action_kind, AuthoringActionKind):
        raise TypeError("Generate candidate action kind must be AuthoringActionKind")
    return action_kind is not AuthoringActionKind.INVALID


@dataclass(frozen=True, slots=True)
class GenerateCandidateContribution:
    """One eligible edge's contribution to the shared Generate population."""

    absolute_importance: float
    uncovered: tuple[str, float, AuthoringEdgeEvidence] | None


@dataclass(frozen=True, slots=True)
class AuthoringEdgeEvidence:
    edge_id: str
    task_family: str
    context_id: str
    action_kind: AuthoringActionKind
    tool_or_skill_name: str | None
    argument_schema_id: str
    observation_status: FailureMode
    token_bucket: TokenBucket
    horizon_bucket: HorizonBucket
    absolute_log_importance: float
    log_importance_quantile: float
    invoked_skill_ids: tuple[str, ...]
    available_tools: tuple[str, ...]
    public_execution: PublicExecutionSnippet | None = None
    exposed_skill_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for field in ("edge_id", "task_family", "context_id", "argument_schema_id"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"authoring edge evidence {field} cannot be empty")
        if not isinstance(self.action_kind, AuthoringActionKind):
            raise TypeError("authoring edge action_kind must be AuthoringActionKind")
        if self.tool_or_skill_name is not None and (
            type(self.tool_or_skill_name) is not str or not self.tool_or_skill_name.strip()
        ):
            raise ValueError("tool_or_skill_name must be non-empty text or None")
        if not isinstance(self.observation_status, FailureMode):
            raise TypeError("authoring edge status must be FailureMode")
        if not isinstance(self.token_bucket, TokenBucket):
            raise TypeError("authoring edge token bucket must be TokenBucket")
        if not isinstance(self.horizon_bucket, HorizonBucket):
            raise TypeError("authoring edge horizon bucket must be HorizonBucket")
        if not math.isfinite(self.absolute_log_importance) or self.absolute_log_importance < 0.0:
            raise ValueError("absolute_log_importance must be finite and non-negative")
        if not 0.0 <= self.log_importance_quantile <= 1.0:
            raise ValueError("log importance quantile must lie in [0, 1]")
        if tuple(sorted(set(self.invoked_skill_ids))) != self.invoked_skill_ids:
            raise ValueError("invoked_skill_ids must be sorted and unique")
        if tuple(sorted(set(self.available_tools))) != self.available_tools:
            raise ValueError("authoring evidence tools must be sorted and unique")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "public_execution": self.public_execution.to_value() if self.public_execution else None,
            "exposed_skill_ids": list(self.exposed_skill_ids)
            if self.exposed_skill_ids is not None
            else None,
            "available_tools": list(self.available_tools),
            "context_id": self.context_id,
            "edge_id": self.edge_id,
            "action_kind": self.action_kind.value,
            "argument_schema_id": self.argument_schema_id,
            "horizon_bucket": self.horizon_bucket.value,
            "invoked_skill_ids": list(self.invoked_skill_ids),
            "absolute_log_importance": self.absolute_log_importance,
            "log_importance_quantile": self.log_importance_quantile,
            "observation_status": self.observation_status.value,
            "task_family": self.task_family,
            "token_bucket": self.token_bucket.value,
            "tool_or_skill_name": self.tool_or_skill_name,
        }

    @classmethod
    def from_value(cls, value: object) -> AuthoringEdgeEvidence:
        expected = {
            "action_kind",
            "argument_schema_id",
            "available_tools",
            "absolute_log_importance",
            "context_id",
            "edge_id",
            "horizon_bucket",
            "invoked_skill_ids",
            "log_importance_quantile",
            "observation_status",
            "task_family",
            "token_bucket",
            "tool_or_skill_name",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) - {"public_execution", "exposed_skill_ids"} != expected
        ):
            raise ValueError("AuthoringEdgeEvidence has incompatible fields")
        raw_skills = value["invoked_skill_ids"]
        raw_tools = value["available_tools"]
        name = value["tool_or_skill_name"]
        quantile = value["log_importance_quantile"]
        absolute = value["absolute_log_importance"]
        if not isinstance(raw_skills, list) or not isinstance(raw_tools, list):
            raise TypeError("authoring evidence skill and tool identities must be arrays")
        if name is not None and not isinstance(name, str):
            raise TypeError("authoring evidence tool_or_skill_name must be text or null")
        if isinstance(quantile, bool) or not isinstance(quantile, int | float):
            raise TypeError("authoring evidence quantile must be numeric")
        if isinstance(absolute, bool) or not isinstance(absolute, int | float):
            raise TypeError("authoring evidence absolute importance must be numeric")
        text_fields = (
            "edge_id",
            "task_family",
            "context_id",
            "argument_schema_id",
        )
        if any(not isinstance(value[field], str) for field in text_fields):
            raise TypeError("authoring evidence identity fields must be text")
        return cls(
            edge_id=value["edge_id"],
            task_family=value["task_family"],
            context_id=value["context_id"],
            action_kind=AuthoringActionKind(value["action_kind"]),
            tool_or_skill_name=name,
            argument_schema_id=value["argument_schema_id"],
            observation_status=FailureMode(value["observation_status"]),
            token_bucket=TokenBucket(value["token_bucket"]),
            horizon_bucket=HorizonBucket(value["horizon_bucket"]),
            absolute_log_importance=float(absolute),
            log_importance_quantile=float(quantile),
            invoked_skill_ids=tuple(raw_skills),
            available_tools=tuple(raw_tools),
            public_execution=PublicExecutionSnippet.from_value(value["public_execution"])
            if value.get("public_execution") is not None
            else None,
            exposed_skill_ids=tuple(value["exposed_skill_ids"])
            if value.get("exposed_skill_ids") is not None
            else None,
        )


def generate_candidate_contribution(
    *,
    exemplar: AuthoringEdgeEvidence,
    absolute_importance: float,
    invoked_skill_ids: tuple[str, ...],
) -> GenerateCandidateContribution | None:
    """Build the population/candidate contribution shared by every arm."""

    if not isinstance(exemplar, AuthoringEdgeEvidence):
        raise TypeError("Generate candidate requires AuthoringEdgeEvidence")
    if not math.isfinite(absolute_importance) or absolute_importance < 0.0:
        raise ValueError("absolute Generate importance must be finite and non-negative")
    if not is_generate_candidate_action(exemplar.action_kind):
        return None
    uncovered = None if invoked_skill_ids else (exemplar.edge_id, absolute_importance, exemplar)
    return GenerateCandidateContribution(
        absolute_importance=absolute_importance,
        uncovered=uncovered,
    )


@dataclass(frozen=True, slots=True)
class GenerateAuthorityGroup:
    """One authorable, homogeneous subset of selected uncovered edges.

    The global high-|log I| selection remains upstream.  Generate authoring
    itself has a deliberately narrower authority: one task family, context,
    and available-tool set per proposed skill.
    """

    task_family: str
    context_id: str
    available_tools: tuple[str, ...]
    importance_edge_ids: tuple[str, ...]
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...]

    def __post_init__(self) -> None:
        _non_empty_text(self.task_family, field="generate authority task_family")
        _non_empty_text(self.context_id, field="generate authority context_id")
        if tuple(sorted(set(self.available_tools))) != self.available_tools:
            raise ValueError("generate authority tools must be sorted and unique")
        if not self.importance_edge_ids:
            raise ValueError("generate authority group requires selected edge IDs")
        if len(set(self.importance_edge_ids)) != len(self.importance_edge_ids):
            raise ValueError("generate authority group repeats an edge ID")
        if tuple(item.edge_id for item in self.edge_exemplars) != self.importance_edge_ids:
            raise ValueError("generate authority group evidence differs from exact edge IDs")
        if any(not isinstance(item, AuthoringEdgeEvidence) for item in self.edge_exemplars):
            raise TypeError("generate authority group requires authoring edge evidence")
        for exemplar in self.edge_exemplars:
            if (
                exemplar.task_family != self.task_family
                or exemplar.context_id != self.context_id
                or exemplar.available_tools != self.available_tools
            ):
                raise ValueError("generate authority group mixes incompatible authoring authority")


def group_generate_exemplars(
    *,
    importance_edge_ids: tuple[str, ...],
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...],
) -> tuple[GenerateAuthorityGroup, ...]:
    """Partition selected zero-coverage edges by the exact authoring authority.

    This function is intentionally shared by full and no-Bayesian decisions:
    the ablation changes posterior use, not which heterogeneous uncovered
    evidence can be written into one skill.  Input order is not trusted for
    proposal identity; both group and per-group edge order are explicit.
    """

    if tuple(item.edge_id for item in edge_exemplars) != importance_edge_ids:
        raise ValueError("selected Generate edge IDs differ from their exemplars")
    if len(set(importance_edge_ids)) != len(importance_edge_ids):
        raise ValueError("selected Generate edge IDs must be unique")
    grouped: dict[tuple[str, str, tuple[str, ...]], list[AuthoringEdgeEvidence]] = {}
    for exemplar in edge_exemplars:
        if not isinstance(exemplar, AuthoringEdgeEvidence):
            raise TypeError("selected Generate exemplars must be authoring evidence")
        key = (exemplar.task_family, exemplar.context_id, exemplar.available_tools)
        grouped.setdefault(key, []).append(exemplar)
    return tuple(
        GenerateAuthorityGroup(
            task_family=task_family,
            context_id=context_id,
            available_tools=available_tools,
            importance_edge_ids=tuple(item.edge_id for item in ordered),
            edge_exemplars=ordered,
        )
        for (task_family, context_id, available_tools), exemplars in sorted(grouped.items())
        for ordered in (tuple(sorted(exemplars, key=lambda item: item.edge_id)),)
    )


@dataclass(frozen=True, slots=True)
class ZeroSkillFlowEvidence:
    """An active skill with zero invoking edges in the triggering window."""

    skill_id: str
    window_id: str
    kind: str = "zero-invocation"

    def __post_init__(self) -> None:
        _non_empty_text(self.skill_id, field="skill_id")
        _non_empty_text(self.window_id, field="window_id")

    @property
    def invoking_trajectory_count(self) -> int:
        return 0

    @property
    def invoking_edge_count(self) -> int:
        return 0

    @property
    def invoking_edge_ids(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class ObservedSkillFlow:
    skill_id: str
    log_skill_marginal_flow: float
    flow_quantile: float
    invoking_trajectory_count: int
    invoking_edge_count: int
    invoking_edge_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_empty_text(self.skill_id, field="skill_id")
        if not math.isfinite(self.log_skill_marginal_flow):
            raise ValueError("log skill flow must be finite")
        if not 0.0 <= self.flow_quantile <= 1.0:
            raise ValueError("flow quantile must lie in [0, 1]")
        if type(self.invoking_trajectory_count) is not int or self.invoking_trajectory_count < 1:
            raise ValueError("observed flow requires invoking trajectories")
        if type(self.invoking_edge_count) is not int or self.invoking_edge_count < 1:
            raise ValueError("observed flow requires invoking edges")
        if len(self.invoking_edge_ids) != self.invoking_edge_count:
            raise ValueError("invoking edge count differs from exact IDs")
        if len(set(self.invoking_edge_ids)) != len(self.invoking_edge_ids):
            raise ValueError("invoking edge IDs must be unique")


SkillFlowEvidence: TypeAlias = ZeroSkillFlowEvidence | ObservedSkillFlow


@dataclass(frozen=True, slots=True)
class PosteriorCellEvidence:
    """One observed posterior cell bundled with its exact event provenance."""

    cell: PosteriorCellState
    event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cell, PosteriorCellState):
            raise TypeError("cell must be PosteriorCellState")
        if not self.event_ids:
            raise ValueError("observed posterior cell requires event provenance")
        if any(type(item) is not str or not item.strip() for item in self.event_ids):
            raise ValueError("posterior event IDs must be non-empty text")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("posterior event IDs must be unique")


@dataclass(frozen=True, slots=True)
class UnobservedSkillPosterior:
    skill_id: str

    def __post_init__(self) -> None:
        _non_empty_text(self.skill_id, field="skill_id")


@dataclass(frozen=True, slots=True)
class ObservedSkillPosterior:
    skill_id: str
    cells: tuple[PosteriorCellEvidence, ...]

    def __post_init__(self) -> None:
        _non_empty_text(self.skill_id, field="skill_id")
        if not self.cells:
            raise ValueError("observed skill posterior requires cells")
        if any(item.cell.skill_id != self.skill_id for item in self.cells):
            raise ValueError("posterior cell belongs to another skill")
        cell_keys = tuple(item.cell.z.cell_key(self.skill_id) for item in self.cells)
        if len(set(cell_keys)) != len(cell_keys):
            raise ValueError("posterior evidence repeats a cell")

    @property
    def posterior_event_ids(self) -> tuple[str, ...]:
        return tuple(event_id for item in self.cells for event_id in item.event_ids)


SkillPosteriorEvidence: TypeAlias = UnobservedSkillPosterior | ObservedSkillPosterior


@dataclass(frozen=True, slots=True)
class PosteriorEvidenceView:
    """A total mapping over the exact active-library population."""

    by_skill: Mapping[str, SkillPosteriorEvidence]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_skill", MappingProxyType(dict(self.by_skill)))

    def require_exact_active_set(self, active_skill_ids: tuple[str, ...]) -> None:
        expected = set(active_skill_ids)
        actual = set(self.by_skill)
        if actual != expected:
            raise ValueError(
                "posterior view differs from active skills: "
                f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}"
            )
        for skill_id in active_skill_ids:
            evidence = self.by_skill[skill_id]
            if evidence.skill_id != skill_id:
                raise ValueError("posterior mapping key differs from evidence identity")


@dataclass(frozen=True, slots=True)
class NoSplitModality:
    pass


@dataclass(frozen=True, slots=True)
class SupportedSplitModality:
    value: SplitModalityEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.value, SplitModalityEvidence):
            raise TypeError("supported split modality requires SplitModalityEvidence")


SplitAssessment: TypeAlias = NoSplitModality | SupportedSplitModality


@dataclass(frozen=True, slots=True)
class SkillEvidence:
    skill_id: str
    flow: SkillFlowEvidence
    posterior: SkillPosteriorEvidence
    split_modality: SplitAssessment
    target_context_keys: tuple[str, ...]
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...]

    def __post_init__(self) -> None:
        if self.flow.skill_id != self.skill_id or self.posterior.skill_id != self.skill_id:
            raise ValueError("skill evidence components have different identities")
        if isinstance(self.flow, ZeroSkillFlowEvidence):
            if self.edge_exemplars:
                raise ValueError("uninvoked skill cannot carry invoking exemplars")
        else:
            if len(self.edge_exemplars) != self.flow.invoking_edge_count:
                raise ValueError("observed flow exemplars differ from invoking edges")
            if tuple(item.edge_id for item in self.edge_exemplars) != self.flow.invoking_edge_ids:
                raise ValueError("observed flow exemplar identities differ")
        if isinstance(self.posterior, UnobservedSkillPosterior):
            if not isinstance(self.split_modality, NoSplitModality):
                raise ValueError("unobserved posterior cannot support Split")
            if self.target_context_keys:
                raise ValueError("unobserved posterior cannot target posterior cells")


@dataclass(frozen=True, slots=True)
class EvidencePack:
    phase_event: PhaseTransitionEvent
    skills: tuple[SkillEvidence, ...]
    uncovered_importance_edges: tuple[str, ...]
    uncovered_exemplars: tuple[AuthoringEdgeEvidence, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(self.skills, key=lambda item: item.skill_id)) != self.skills:
            raise ValueError("skill evidence must be sorted")
        if len({item.skill_id for item in self.skills}) != len(self.skills):
            raise ValueError("skill evidence repeats a skill")
        exemplar_ids = tuple(item.edge_id for item in self.uncovered_exemplars)
        if exemplar_ids != self.uncovered_importance_edges:
            raise ValueError("uncovered edge IDs and exemplars differ")


def aggregate_window_skill_flows(
    diagnostics: tuple[BatchDiagnostics, ...],
    *,
    active_skill_ids: tuple[str, ...],
) -> tuple[SkillFlowEvidence, ...]:
    calls: dict[str, list[tuple[str, int, float]]] = {skill_id: [] for skill_id in active_skill_ids}
    window_id = stable_hash(
        {"batch_ids": [item.batch_id for item in diagnostics], "kind": "flow-window"}
    )
    active = set(active_skill_ids)
    for diagnostic in diagnostics:
        for trajectory in diagnostic.trajectories:
            for edge in trajectory.edges:
                for skill_id in edge.invoked_skill_ids:
                    if skill_id not in active:
                        raise ValueError("window invokes a skill outside the active library")
                    calls[skill_id].append(
                        (
                            trajectory.trajectory_id,
                            edge.step_index,
                            edge.sample_log_state_weight,
                        )
                    )

    raw_observed: list[tuple[str, float, tuple[tuple[str, int, float], ...]]] = []
    for skill_id in active_skill_ids:
        skill_calls = tuple(calls[skill_id])
        if skill_calls:
            trajectories = {trajectory_id for trajectory_id, _, _ in skill_calls}
            log_flow = _logsumexp(tuple(item[2] for item in skill_calls)) - math.log(
                len(trajectories)
            )
            raw_observed.append((skill_id, log_flow, skill_calls))

    observed_population = tuple(item[1] for item in raw_observed)
    observed_by_id = {item[0]: item for item in raw_observed}
    output: list[SkillFlowEvidence] = []
    for skill_id in active_skill_ids:
        if skill_id not in observed_by_id:
            output.append(ZeroSkillFlowEvidence(skill_id=skill_id, window_id=window_id))
            continue
        _, log_flow, skill_calls = observed_by_id[skill_id]
        invoking_trajectories = {item[0] for item in skill_calls}
        output.append(
            ObservedSkillFlow(
                skill_id=skill_id,
                log_skill_marginal_flow=log_flow,
                flow_quantile=_empirical_cdf(log_flow, observed_population),
                invoking_trajectory_count=len(invoking_trajectories),
                invoking_edge_count=len(skill_calls),
                invoking_edge_ids=tuple(
                    f"{trajectory_id}:{step_index}" for trajectory_id, step_index, _ in skill_calls
                ),
            )
        )
    return tuple(output)


def assemble_posterior_evidence_view(
    *,
    active_skill_ids: tuple[str, ...],
    cells: tuple[PosteriorCellState, ...],
    event_ids_by_cell: Mapping[str, tuple[str, ...]],
) -> PosteriorEvidenceView:
    cells_by_skill: dict[str, list[PosteriorCellState]] = {
        skill_id: [] for skill_id in active_skill_ids
    }
    active = set(active_skill_ids)
    for cell in cells:
        if cell.skill_id in active:
            cells_by_skill[cell.skill_id].append(cell)

    by_skill: dict[str, SkillPosteriorEvidence] = {}
    for skill_id in active_skill_ids:
        selected = tuple(sorted(cells_by_skill[skill_id], key=lambda cell: cell.z.content_hash))
        if not selected:
            by_skill[skill_id] = UnobservedSkillPosterior(skill_id=skill_id)
            continue
        bundled: list[PosteriorCellEvidence] = []
        for cell in selected:
            key = cell.z.cell_key(skill_id)
            try:
                event_ids = event_ids_by_cell[key]
            except KeyError as error:
                raise ValueError(
                    f"observed posterior cell {key!r} has no exact provenance"
                ) from error
            bundled.append(PosteriorCellEvidence(cell=cell, event_ids=event_ids))
        by_skill[skill_id] = ObservedSkillPosterior(
            skill_id=skill_id,
            cells=tuple(bundled),
        )
    view = PosteriorEvidenceView(by_skill=by_skill)
    view.require_exact_active_set(active_skill_ids)
    return view


def aggregate_task_family_mode(
    posterior: ObservedSkillPosterior,
    *,
    task_family: str,
    k: float,
) -> PosteriorTaskFamilyMode:
    # Protocol-v3 fixes ContextFeature.context to TrajectoryRecord.task_family.
    # It must never be confused with the independent retrieval context_id axis.
    selected = tuple(item for item in posterior.cells if item.cell.z.context == task_family)
    if not selected:
        raise ValueError("task-family posterior mode cannot be empty")
    first = selected[0].cell
    alpha_0 = float(first.alpha_0)
    beta_0 = float(first.beta_0)
    if any(
        float(item.cell.alpha_0) != alpha_0 or float(item.cell.beta_0) != beta_0
        for item in selected
    ):
        raise ValueError("task-family posterior cells use different priors")
    success_mass = math.fsum(float(item.cell.alpha) - alpha_0 for item in selected)
    failure_mass = math.fsum(float(item.cell.beta_count) - beta_0 for item in selected)
    alpha = alpha_0 + success_mass
    beta = beta_0 + failure_mass
    mean = alpha / (alpha + beta)
    sigma = math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0)))
    means = tuple(item.cell.mean() for item in selected)
    cell_keys = tuple(item.cell.z.cell_key(posterior.skill_id) for item in selected)
    event_ids = tuple(event_id for item in selected for event_id in item.event_ids)
    return PosteriorTaskFamilyMode(
        task_family=task_family,
        cell_keys=cell_keys,
        posterior_event_ids=event_ids,
        evidence_mass=success_mass + failure_mass,
        mean=mean,
        sigma=sigma,
        lcb=mean - k * sigma,
        ucb=mean + k * sigma,
        within_cell_mean_span=max(means) - min(means),
    )


def _expand_source_task_families(
    applicability: SkillApplicability,
    *,
    task_family_universe: tuple[str, ...],
) -> tuple[str, ...]:
    """Expand a wildcard exactly once against the frozen formal universe."""

    if applicability.task_families == ("*",):
        return task_family_universe
    if not set(applicability.task_families) <= set(task_family_universe):
        raise ValueError("source applicability exceeds the formal task-family universe")
    return applicability.task_families


def _qualified_task_family_modes(
    posterior: ObservedSkillPosterior,
    *,
    source_task_families: tuple[str, ...],
    config: SplitCriterionConfig,
) -> tuple[PosteriorTaskFamilyMode, ...]:
    modes = tuple(
        aggregate_task_family_mode(posterior, task_family=task_family, k=config.confidence_k)
        for task_family in source_task_families
        if any(item.cell.z.context == task_family for item in posterior.cells)
    )
    return tuple(
        mode
        for mode in modes
        if mode.evidence_mass >= config.min_context_evidence_mass
        and mode.within_cell_mean_span <= config.max_within_context_mean_span
    )


def _complete_split_candidate(
    *,
    source_task_families: tuple[str, ...],
    modes: tuple[PosteriorTaskFamilyMode, ...],
    low_anchor: PosteriorTaskFamilyMode,
    high_anchor: PosteriorTaskFamilyMode,
    config: SplitCriterionConfig,
) -> SplitModalityEvidence | None:
    by_family = {mode.task_family: mode for mode in modes}
    if set(by_family) != set(source_task_families):
        return None
    gap = high_anchor.mean - low_anchor.mean
    if gap < config.min_between_context_mean_gap or not low_anchor.ucb < high_anchor.lcb:
        return None
    cutpoint = (low_anchor.ucb + high_anchor.lcb) / 2.0
    assignments: list[SplitTaskFamilyAssignment] = []
    for task_family in source_task_families:
        mode = by_family[task_family]
        if mode.ucb < cutpoint:
            branch = SplitBranch.LOW
        elif mode.lcb > cutpoint:
            branch = SplitBranch.HIGH
        else:
            return None
        assignments.append(SplitTaskFamilyAssignment(mode=mode, branch=branch))
    return SplitModalityEvidence(
        low_mode=low_anchor,
        high_mode=high_anchor,
        between_mean_gap=gap,
        intervals_disjoint=True,
        source_task_families=source_task_families,
        assignments=tuple(assignments),
        separation_cutpoint=cutpoint,
    )


def detect_split_modality(
    posterior: ObservedSkillPosterior,
    *,
    source_applicability: SkillApplicability,
    task_family_universe: tuple[str, ...],
    config: SplitCriterionConfig,
) -> SplitAssessment:
    source_families = _expand_source_task_families(
        source_applicability,
        task_family_universe=task_family_universe,
    )
    qualified = _qualified_task_family_modes(
        posterior,
        source_task_families=source_families,
        config=config,
    )
    candidates: list[SplitModalityEvidence] = []
    for first, second in combinations(qualified, 2):
        low, high = (first, second) if first.mean <= second.mean else (second, first)
        candidate = _complete_split_candidate(
            source_task_families=source_families,
            modes=qualified,
            low_anchor=low,
            high_anchor=high,
            config=config,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return NoSplitModality()
    return SupportedSplitModality(
        max(
            candidates,
            key=lambda item: (
                item.between_mean_gap,
                item.low_task_families,
                item.high_task_families,
            ),
        )
    )


def build_evidence_pack(
    *,
    window_flow: WindowFlowView,
    posterior: PosteriorEvidenceView,
    trajectories: TrajectoryEvidenceView,
    config: EvolutionConfig,
) -> EvidencePack:
    phase = window_flow.phase_event
    expected_batches = (
        *phase.previous_window.member_batch_ids,
        *phase.current_window.member_batch_ids,
    )
    if tuple(item.batch_id for item in window_flow.diagnostics) != expected_batches:
        raise ValueError("window diagnostics differ from the phase event")
    posterior.require_exact_active_set(window_flow.active_skill_ids)
    flows = aggregate_window_skill_flows(
        window_flow.diagnostics,
        active_skill_ids=window_flow.active_skill_ids,
    )
    flow_by_skill = {item.skill_id: item for item in flows}
    exemplars_by_skill: dict[str, list[AuthoringEdgeEvidence]] = {
        skill_id: [] for skill_id in window_flow.active_skill_ids
    }
    uncovered: list[tuple[str, float, AuthoringEdgeEvidence]] = []
    all_absolute_importance: list[float] = []
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
                    all_absolute_importance.append(contribution.absolute_importance)
                if edge.invoked_skill_ids:
                    for skill_id in edge.invoked_skill_ids:
                        exemplars_by_skill[skill_id].append(exemplar)
                elif contribution is not None and contribution.uncovered is not None:
                    uncovered.append(contribution.uncovered)

    skills: list[SkillEvidence] = []
    for skill_id in window_flow.active_skill_ids:
        flow = flow_by_skill[skill_id]
        skill_posterior = posterior.by_skill[skill_id]
        target_context_keys: tuple[str, ...]
        if isinstance(skill_posterior, ObservedSkillPosterior):
            minimum_lcb = _min_lcb(skill_posterior, config.k)
            split = detect_split_modality(
                skill_posterior,
                source_applicability=window_flow.applicability_by_skill[skill_id],
                task_family_universe=window_flow.task_family_universe,
                config=config.split,
            )
            target_context_keys = (minimum_lcb.z.cell_key(skill_id),)
        else:
            split = NoSplitModality()
            target_context_keys = ()
        skills.append(
            SkillEvidence(
                skill_id=skill_id,
                flow=flow,
                posterior=skill_posterior,
                split_modality=split,
                target_context_keys=target_context_keys,
                edge_exemplars=tuple(exemplars_by_skill[skill_id]),
            )
        )
    selected = select_uncovered_generate_edges(
        uncovered,
        tuple(all_absolute_importance),
        importance_quantile=config.importance_quantile,
        minimum_absolute_log_importance=config.generate_min_absolute_log_importance,
    )
    return EvidencePack(
        phase_event=phase,
        skills=tuple(skills),
        uncovered_importance_edges=tuple(item[0] for item in selected),
        uncovered_exemplars=tuple(item[2] for item in selected),
    )


def _min_lcb(posterior: ObservedSkillPosterior, k: float) -> CellQuery:
    return min(
        (query_cell(item.cell, k) for item in posterior.cells),
        key=lambda item: (item.lcb, item.z.content_hash),
    )


def authoring_edge_evidence(
    record: TrajectoryRecord,
    step_index: int,
    *,
    absolute_log_importance: float,
    log_importance_quantile: float,
) -> AuthoringEdgeEvidence:
    step = record.steps[step_index - 1]
    raw_tools = record.initial_context.meta["available_tools"]
    if not isinstance(raw_tools, list) or any(not isinstance(item, str) for item in raw_tools):
        raise ValueError("trajectory context available_tools must be an array of text")
    tools = tuple(cast(list[str], raw_tools))
    if tuple(sorted(set(tools))) != tools:
        raise ValueError("trajectory available_tools must be sorted and unique")
    raw_context_id = record.initial_context.meta["context_id"]
    if type(raw_context_id) is not str or not raw_context_id.strip():
        raise ValueError("trajectory context_id must be non-empty text")
    from skillev.rollout.codec import codec_for_initial_meta

    action = codec_for_initial_meta(record.initial_context.meta).parse(step.action_text).action
    if action is None:
        action_kind = AuthoringActionKind.INVALID
        tool_or_skill_name = None
        argument_schema_id = stable_hash({"kind": "invalid-structured-action"})
    else:
        action_kind = AuthoringActionKind(action.kind.value)
        tool_or_skill_name = action.name
        argument_schema_id = stable_hash(_json_shape(action.arguments))
    return AuthoringEdgeEvidence(
        edge_id=f"{record.trajectory_id}:{step_index}",
        task_family=record.task_family,
        context_id=raw_context_id,
        action_kind=action_kind,
        tool_or_skill_name=tool_or_skill_name,
        argument_schema_id=argument_schema_id,
        observation_status=FailureMode.from_observation_status(step.observation_status),
        token_bucket=TokenBucket.from_count(record.initial_context.assembled_token_count),
        horizon_bucket=HorizonBucket.from_horizon(record.horizon),
        absolute_log_importance=absolute_log_importance,
        log_importance_quantile=log_importance_quantile,
        invoked_skill_ids=tuple(sorted(step.invoked_skill_ids)),
        available_tools=tools,
        public_execution=PublicExecutionSnippet.from_step(step),
        exposed_skill_ids=record.initial_context.retrieved_skill_ids,
    )


def authoring_evidence_for_diagnostic(
    records_by_id: Mapping[str, TrajectoryRecord],
    diagnostic: BatchDiagnostics,
) -> tuple[AuthoringEdgeEvidence, ...]:
    """Materialize answer-free edge evidence from committed contracts only."""

    quantiles = importance_quantiles_for((diagnostic,))
    output: list[AuthoringEdgeEvidence] = []
    for trajectory in diagnostic.trajectories:
        try:
            record = records_by_id[trajectory.trajectory_id]
        except KeyError as error:
            raise ValueError("diagnostic trajectory lacks its committed record") from error
        for edge in trajectory.edges:
            edge_id = f"{trajectory.trajectory_id}:{edge.step_index}"
            output.append(
                authoring_edge_evidence(
                    record,
                    edge.step_index,
                    absolute_log_importance=abs(edge.log_importance),
                    log_importance_quantile=quantiles[edge_id],
                )
            )
    return tuple(output)


def _json_shape(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _json_shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        array_items = cast(
            list[JsonValue],
            sorted(
                {
                    canonical_shape
                    for item in value
                    if (canonical_shape := json.dumps(_json_shape(item), sort_keys=True))
                }
            ),
        )
        return {"array_items": array_items}
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    return "string"


def importance_quantiles_for(
    diagnostics: tuple[BatchDiagnostics, ...],
) -> Mapping[str, float]:
    edges = tuple(
        (f"{trajectory.trajectory_id}:{edge.step_index}", abs(edge.log_importance))
        for diagnostic in diagnostics
        for trajectory in diagnostic.trajectories
        for edge in trajectory.edges
    )
    population = tuple(value for _, value in edges)
    return {edge_id: _empirical_cdf(value, population) for edge_id, value in edges}


def select_uncovered_generate_edges(
    edges: list[tuple[str, float, AuthoringEdgeEvidence]],
    population: tuple[float, ...],
    *,
    importance_quantile: float,
    minimum_absolute_log_importance: float,
) -> tuple[tuple[str, float, AuthoringEdgeEvidence], ...]:
    if not edges:
        return ()
    if not 0.0 <= importance_quantile <= 1.0:
        raise ValueError("importance_quantile must lie in [0, 1]")
    if not math.isfinite(minimum_absolute_log_importance) or (
        minimum_absolute_log_importance <= 0.0
    ):
        raise ValueError("minimum absolute importance must be finite and positive")
    selected: list[tuple[str, float, AuthoringEdgeEvidence]] = []
    for edge_id, absolute, exemplar in edges:
        percentile = _strict_upper_tail_percentile(absolute, population)
        if absolute < minimum_absolute_log_importance or percentile < importance_quantile:
            continue
        selected.append(
            (
                edge_id,
                absolute,
                replace(exemplar, log_importance_quantile=percentile),
            )
        )
    return tuple(selected)


def _strict_upper_tail_percentile(value: float, population: tuple[float, ...]) -> float:
    if not population:
        raise ValueError("Generate importance population cannot be empty")
    if any(not math.isfinite(item) or item < 0.0 for item in population):
        raise ValueError("Generate population must contain finite non-negative values")
    if len(population) == 1:
        return 1.0
    return sum(candidate < value for candidate in population) / (len(population) - 1)


def _empirical_cdf(value: float, values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("empirical CDF population cannot be empty")
    return sum(candidate <= value for candidate in values) / len(values)


def _logsumexp(values: tuple[float, ...]) -> float:
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def _non_empty_text(value: object, *, field: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")


__all__ = [
    "AuthoringEdgeEvidence",
    "EvidencePack",
    "GenerateAuthorityGroup",
    "GenerateCandidateContribution",
    "NoSplitModality",
    "ObservedSkillFlow",
    "ObservedSkillPosterior",
    "PosteriorCellEvidence",
    "PosteriorEvidenceView",
    "SkillEvidence",
    "SkillFlowEvidence",
    "SkillPosteriorEvidence",
    "SplitAssessment",
    "SupportedSplitModality",
    "TrajectoryEvidenceView",
    "UnobservedSkillPosterior",
    "WindowFlowView",
    "ZeroSkillFlowEvidence",
    "aggregate_task_family_mode",
    "aggregate_window_skill_flows",
    "assemble_posterior_evidence_view",
    "authoring_evidence_for_diagnostic",
    "build_evidence_pack",
    "detect_split_modality",
    "generate_candidate_contribution",
    "group_generate_exemplars",
    "is_generate_candidate_action",
    "query_cell",
    "select_uncovered_generate_edges",
]
