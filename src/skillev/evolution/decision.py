"""Closed, deterministic confidence-bound Phi decision policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, assert_never

from skillev.calibration import CellQuery
from skillev.contracts import (
    GenerateEvidence,
    JsonValue,
    PhaseTriggerRule,
    PruneEvidence,
    RefineEvidence,
    RetainEvidence,
    SplitEvidence,
    stable_hash,
)
from skillev.runtime import BudgetVector

from .config import EvolutionConfig
from .evidence import (
    AuthoringEdgeEvidence,
    EvidencePack,
    ObservedSkillPosterior,
    PosteriorEvidenceView,
    SkillEvidence,
    SupportedSplitModality,
    TrajectoryEvidenceView,
    UnobservedSkillPosterior,
    WindowFlowView,
    ZeroSkillFlowEvidence,
    build_evidence_pack,
    group_generate_exemplars,
    is_generate_candidate_action,
    query_cell,
)


class NoApplicableEvolutionActionError(RuntimeError):
    """A verified phase produced no member of the closed full-method Phi."""


class InsufficientEvolutionBudgetError(RuntimeError):
    """The complete decision cannot fit the predeclared Phi budget."""


def _proposal_common(
    *,
    target_skill_id: str,
    rationale_text: str,
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...],
) -> None:
    if type(target_skill_id) is not str or not target_skill_id.strip():
        raise ValueError("proposal target_skill_id must be non-empty text")
    if type(rationale_text) is not str or not rationale_text.strip():
        raise ValueError("proposal rationale_text must be non-empty text")
    if any(not isinstance(item, AuthoringEdgeEvidence) for item in edge_exemplars):
        raise TypeError("proposal edge_exemplars must contain AuthoringEdgeEvidence values")


@dataclass(frozen=True, slots=True)
class RetainProposal:
    target_skill_id: str
    evidence: RetainEvidence
    rationale_text: str
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...]

    def __post_init__(self) -> None:
        _proposal_common(
            target_skill_id=self.target_skill_id,
            rationale_text=self.rationale_text,
            edge_exemplars=self.edge_exemplars,
        )
        if not isinstance(self.evidence, RetainEvidence):
            raise TypeError("RetainProposal requires RetainEvidence")


@dataclass(frozen=True, slots=True)
class RefineProposal:
    target_skill_id: str
    evidence: RefineEvidence
    rationale_text: str
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...]

    def __post_init__(self) -> None:
        _proposal_common(
            target_skill_id=self.target_skill_id,
            rationale_text=self.rationale_text,
            edge_exemplars=self.edge_exemplars,
        )
        if not isinstance(self.evidence, RefineEvidence):
            raise TypeError("RefineProposal requires RefineEvidence")
        if (
            self.evidence.target_contexts
            and tuple(z.cell_key(self.target_skill_id) for z in self.evidence.target_contexts)
            != self.evidence.target_context_keys
        ):
            raise ValueError("Refine coordinates do not identify the target posterior cells")


@dataclass(frozen=True, slots=True)
class SplitProposal:
    target_skill_id: str
    evidence: SplitEvidence
    rationale_text: str
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...]

    def __post_init__(self) -> None:
        _proposal_common(
            target_skill_id=self.target_skill_id,
            rationale_text=self.rationale_text,
            edge_exemplars=self.edge_exemplars,
        )
        if not isinstance(self.evidence, SplitEvidence):
            raise TypeError("SplitProposal requires SplitEvidence")


@dataclass(frozen=True, slots=True)
class PruneProposal:
    target_skill_id: str
    evidence: PruneEvidence
    rationale_text: str
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...]

    def __post_init__(self) -> None:
        _proposal_common(
            target_skill_id=self.target_skill_id,
            rationale_text=self.rationale_text,
            edge_exemplars=self.edge_exemplars,
        )
        if not isinstance(self.evidence, PruneEvidence):
            raise TypeError("PruneProposal requires PruneEvidence")


@dataclass(frozen=True, slots=True)
class GenerateProposal:
    evidence: GenerateEvidence
    rationale_text: str
    edge_exemplars: tuple[AuthoringEdgeEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, GenerateEvidence):
            raise TypeError("GenerateProposal requires GenerateEvidence")
        if type(self.rationale_text) is not str or not self.rationale_text.strip():
            raise ValueError("proposal rationale_text must be non-empty text")
        if tuple(item.edge_id for item in self.edge_exemplars) != self.evidence.importance_edge_ids:
            raise ValueError("Generate proposal exemplars differ from evidence")
        for exemplar in self.edge_exemplars:
            if exemplar.absolute_log_importance < self.evidence.minimum_absolute_log_importance:
                raise ValueError("Generate exemplar is below the absolute importance floor")
            if exemplar.log_importance_quantile < self.evidence.importance_quantile:
                raise ValueError("Generate exemplar is outside the selected upper tail")
            if exemplar.invoked_skill_ids:
                raise ValueError("Generate exemplar already has skill coverage")
            if not is_generate_candidate_action(exemplar.action_kind):
                raise ValueError("Generate exemplar action kind is ineligible")


FullActionProposal: TypeAlias = (
    RetainProposal | RefineProposal | SplitProposal | PruneProposal | GenerateProposal
)


@dataclass(frozen=True, slots=True)
class NoActionForSkill:
    skill_id: str

    def __post_init__(self) -> None:
        if type(self.skill_id) is not str or not self.skill_id.strip():
            raise ValueError("skill_id must be non-empty text")


SkillDecision: TypeAlias = (
    NoActionForSkill | RetainProposal | RefineProposal | SplitProposal | PruneProposal
)


@dataclass(frozen=True, slots=True)
class EvolutionDecision:
    phase_event_id: str
    proposals: tuple[FullActionProposal, ...]

    def __post_init__(self) -> None:
        if type(self.phase_event_id) is not str or not self.phase_event_id.strip():
            raise ValueError("decision phase_event_id cannot be empty")
        if not self.proposals:
            raise NoApplicableEvolutionActionError(
                "verified phase has no applicable full-method Phi action"
            )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    @property
    def proposal_content_hashes(self) -> tuple[str, ...]:
        return tuple(proposal_content_hash(item) for item in self.proposals)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "phase_event_id": self.phase_event_id,
            "proposals": [proposal_to_value(proposal) for proposal in self.proposals],
        }


@dataclass(frozen=True, slots=True)
class VerifiedNoOpEvolutionDecision:
    """A phase with complete evidence but no applicable frozen Phi action."""

    phase_event_id: str
    reason: str = "no-frozen-threshold-admits-an-evolution-action"

    def __post_init__(self) -> None:
        if not self.phase_event_id.strip() or not self.reason.strip():
            raise ValueError("verified no-op evolution identity is incomplete")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    @property
    def proposal_content_hashes(self) -> tuple[str, ...]:
        return ()

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "kind": "verified-no-op",
            "phase_event_id": self.phase_event_id,
            "reason": self.reason,
        }


FullEvolutionDecision: TypeAlias = EvolutionDecision | VerifiedNoOpEvolutionDecision


_ACTION_PRIORITY = {
    PruneProposal: 0,
    SplitProposal: 1,
    RefineProposal: 2,
    RetainProposal: 3,
    GenerateProposal: 4,
}


class FullEvolutionPolicy:
    def decide(self, pack: EvidencePack, config: EvolutionConfig) -> FullEvolutionDecision:
        if not isinstance(pack, EvidencePack) or not isinstance(config, EvolutionConfig):
            raise TypeError("FullEvolutionPolicy requires EvidencePack and EvolutionConfig")
        if pack.phase_event.trigger_rule is PhaseTriggerRule.ZERO_COVERAGE_COLD_START:
            raise ValueError("cold start requires complete source-aware window views")
        decisions = tuple(_decide_skill(skill, config) for skill in pack.skills)
        proposals: list[FullActionProposal] = [
            decision for decision in decisions if not isinstance(decision, NoActionForSkill)
        ]
        for group in group_generate_exemplars(
            importance_edge_ids=pack.uncovered_importance_edges,
            edge_exemplars=pack.uncovered_exemplars,
        ):
            proposals.append(
                GenerateProposal(
                    evidence=GenerateEvidence(
                        importance_edge_ids=group.importance_edge_ids,
                        minimum_absolute_log_importance=(
                            config.generate_min_absolute_log_importance
                        ),
                        importance_quantile=config.importance_quantile,
                        importance_semantics=config.generate_importance_semantics,
                    ),
                    rationale_text=(
                        "Generate coverage for zero-skill edges satisfying both the "
                        "absolute |log I| floor and the window-relative upper tail"
                    ),
                    edge_exemplars=group.edge_exemplars,
                )
            )
        proposals.sort(key=_proposal_sort_key)
        if not proposals:
            return VerifiedNoOpEvolutionDecision(
                phase_event_id=pack.phase_event.event_id,
            )
        return EvolutionDecision(
            phase_event_id=pack.phase_event.event_id,
            proposals=tuple(proposals),
        )

    def decide_from_views(
        self,
        *,
        window_flow: WindowFlowView,
        posterior: PosteriorEvidenceView,
        trajectories: TrajectoryEvidenceView,
        config: EvolutionConfig,
    ) -> FullEvolutionDecision:
        if window_flow.phase_event.trigger_rule is PhaseTriggerRule.ZERO_COVERAGE_COLD_START:
            from .cold_start import decide_cold_start

            if config.cold_start is None:
                raise ValueError("cold start is not enabled in the frozen method configuration")
            # Still validate the complete window, active posterior and edge identities.
            build_evidence_pack(
                window_flow=window_flow,
                posterior=posterior,
                trajectories=trajectories,
                config=config,
            )
            return decide_cold_start(window_flow, trajectories, config)
        return self.decide(
            build_evidence_pack(
                window_flow=window_flow,
                posterior=posterior,
                trajectories=trajectories,
                config=config,
            ),
            config,
        )


def _decide_skill(skill: SkillEvidence, config: EvolutionConfig) -> SkillDecision:
    posterior = skill.posterior
    if isinstance(skill.flow, ZeroSkillFlowEvidence):
        if isinstance(posterior, UnobservedSkillPosterior):
            return NoActionForSkill(skill.skill_id)
        maximum_ucb = _max_ucb(posterior, config.k)
        if maximum_ucb.ucb <= config.ucb_low:
            return PruneProposal(
                target_skill_id=skill.skill_id,
                evidence=PruneEvidence.zero_invocation(
                    window_id=skill.flow.window_id,
                    ucb=maximum_ucb.ucb,
                    k=config.k,
                    posterior_event_ids=posterior.posterior_event_ids,
                ),
                rationale_text=(f"zero invocation and optimistic posterior {maximum_ucb.ucb:.4f}"),
                edge_exemplars=(),
            )
        return NoActionForSkill(skill.skill_id)

    flow = skill.flow
    high_flow = flow.flow_quantile >= config.high_flow_quantile
    low_flow = flow.flow_quantile <= config.low_flow_quantile
    if isinstance(posterior, UnobservedSkillPosterior):
        return NoActionForSkill(skill.skill_id)

    minimum_lcb = _min_lcb(posterior, config.k)
    maximum_ucb = _max_ucb(posterior, config.k)

    if low_flow and maximum_ucb.ucb <= config.ucb_low:
        return PruneProposal(
            target_skill_id=skill.skill_id,
            evidence=PruneEvidence.observed_low(
                log_skill_marginal_flow=flow.log_skill_marginal_flow,
                flow_quantile=flow.flow_quantile,
                ucb=maximum_ucb.ucb,
                k=config.k,
                posterior_event_ids=posterior.posterior_event_ids,
            ),
            rationale_text=f"low flow and optimistic posterior {maximum_ucb.ucb:.4f}",
            edge_exemplars=skill.edge_exemplars,
        )

    if high_flow and isinstance(skill.split_modality, SupportedSplitModality):
        return SplitProposal(
            target_skill_id=skill.skill_id,
            evidence=SplitEvidence(
                log_skill_marginal_flow=flow.log_skill_marginal_flow,
                flow_quantile=flow.flow_quantile,
                modality=skill.split_modality.value,
            ),
            rationale_text=("high flow with supported, tight, disjoint context posterior modes"),
            edge_exemplars=skill.edge_exemplars,
        )

    if high_flow and minimum_lcb.lcb <= config.lcb_low:
        target_key = minimum_lcb.z.cell_key(skill.skill_id)
        return RefineProposal(
            target_skill_id=skill.skill_id,
            evidence=RefineEvidence(
                log_skill_marginal_flow=flow.log_skill_marginal_flow,
                flow_quantile=flow.flow_quantile,
                lcb=minimum_lcb.lcb,
                k=config.k,
                posterior_event_ids=posterior.posterior_event_ids,
                target_context_keys=(target_key,),
                target_contexts=(minimum_lcb.z,),
            ),
            rationale_text=f"high flow but conservative posterior {minimum_lcb.lcb:.4f}",
            edge_exemplars=skill.edge_exemplars,
        )

    if high_flow and minimum_lcb.lcb >= config.lcb_high:
        return RetainProposal(
            target_skill_id=skill.skill_id,
            evidence=RetainEvidence(
                log_skill_marginal_flow=flow.log_skill_marginal_flow,
                flow_quantile=flow.flow_quantile,
                lcb=minimum_lcb.lcb,
                k=config.k,
                posterior_event_ids=posterior.posterior_event_ids,
            ),
            rationale_text=f"high flow and conservative posterior {minimum_lcb.lcb:.4f}",
            edge_exemplars=skill.edge_exemplars,
        )
    return NoActionForSkill(skill.skill_id)


def required_phi_budget(
    decision: FullEvolutionDecision,
    config: EvolutionConfig,
) -> BudgetVector:
    if not isinstance(
        decision,
        EvolutionDecision | VerifiedNoOpEvolutionDecision,
    ) or not isinstance(config, EvolutionConfig):
        raise TypeError("required_phi_budget requires decision and config")
    if isinstance(decision, VerifiedNoOpEvolutionDecision):
        return BudgetVector()
    model_calls = 0
    for proposal in decision.proposals:
        match proposal:
            case PruneProposal():
                pass
            case SplitProposal() | RetainProposal() | RefineProposal() | GenerateProposal():
                model_calls += 1
            case _:
                raise TypeError(f"unsupported full proposal: {type(proposal).__name__}")
    return BudgetVector(
        input_tokens=model_calls * config.max_authoring_prompt_tokens,
        model_calls=model_calls,
        output_tokens=model_calls * config.max_authoring_completion_tokens,
    )


def _min_lcb(posterior: ObservedSkillPosterior, k: float) -> CellQuery:
    queries = tuple(query_cell(item.cell, k) for item in posterior.cells)
    return min(queries, key=lambda item: (item.lcb, item.z.content_hash))


def _max_ucb(posterior: ObservedSkillPosterior, k: float) -> CellQuery:
    queries = tuple(query_cell(item.cell, k) for item in posterior.cells)
    return max(queries, key=lambda item: (item.ucb, item.z.content_hash))


def _proposal_sort_key(proposal: FullActionProposal) -> tuple[int, str]:
    skill_id = "" if isinstance(proposal, GenerateProposal) else proposal.target_skill_id
    return _ACTION_PRIORITY[type(proposal)], skill_id


def proposal_to_value(proposal: FullActionProposal) -> dict[str, JsonValue]:
    targets: list[JsonValue]
    match proposal:
        case RetainProposal(target_skill_id=skill_id, evidence=evidence):
            kind = "retain-compress"
            targets = [skill_id]
        case RefineProposal(target_skill_id=skill_id, evidence=evidence):
            kind = "refine"
            targets = [skill_id]
        case SplitProposal(target_skill_id=skill_id, evidence=evidence):
            kind = "split"
            targets = [skill_id]
        case PruneProposal(target_skill_id=skill_id, evidence=evidence):
            kind = "prune"
            targets = [skill_id]
        case GenerateProposal(evidence=evidence):
            kind = "generate"
            targets = []
        case _ as unreachable:
            assert_never(unreachable)
    return {
        "action_type": kind,
        "edge_exemplars": [item.to_value() for item in proposal.edge_exemplars],
        "evidence": evidence.to_value(),
        "rationale_text": proposal.rationale_text,
        "target_skill_ids": targets,
    }


def proposal_content_hash(proposal: FullActionProposal) -> str:
    """Stable source provenance for one exact Phi proposal."""

    return stable_hash(proposal_to_value(proposal))


__all__ = [
    "EvolutionDecision",
    "FullActionProposal",
    "FullEvolutionDecision",
    "FullEvolutionPolicy",
    "GenerateProposal",
    "InsufficientEvolutionBudgetError",
    "NoActionForSkill",
    "NoApplicableEvolutionActionError",
    "PruneProposal",
    "RefineProposal",
    "RetainProposal",
    "SkillDecision",
    "SplitProposal",
    "VerifiedNoOpEvolutionDecision",
    "proposal_content_hash",
    "proposal_to_value",
    "required_phi_budget",
]
