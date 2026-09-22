"""Posterior-mean Phi policy for the declared LCB-to-mean arm."""

from __future__ import annotations

from skillev.contracts import (
    GenerateEvidence,
    PruneEvidence,
    RefineEvidence,
    RetainEvidence,
    SplitEvidence,
)
from skillev.evolution import (
    EvidencePack,
    EvolutionConfig,
    EvolutionDecision,
    FullActionProposal,
    GenerateProposal,
    NoApplicableEvolutionActionError,
    ObservedSkillFlow,
    ObservedSkillPosterior,
    PosteriorEvidenceView,
    PruneProposal,
    RefineProposal,
    RetainProposal,
    SplitProposal,
    SupportedSplitModality,
    TrajectoryEvidenceView,
    WindowFlowView,
    build_evidence_pack,
    group_generate_exemplars,
)


class PosteriorMeanEvolutionPolicy:
    """Use posterior means at the same thresholds and change no other axis."""

    def decide(
        self,
        pack: EvidencePack,
        config: EvolutionConfig,
    ) -> EvolutionDecision:
        proposals: list[FullActionProposal] = []
        for skill in pack.skills:
            if not isinstance(skill.flow, ObservedSkillFlow):
                continue
            if not isinstance(skill.posterior, ObservedSkillPosterior):
                continue
            flow = skill.flow
            posterior = skill.posterior
            minimum = min(
                posterior.cells,
                key=lambda item: (item.cell.mean(), item.cell.z.content_hash),
            )
            maximum = max(
                posterior.cells,
                key=lambda item: (item.cell.mean(), item.cell.z.content_hash),
            )
            pessimistic_mean = minimum.cell.mean()
            optimistic_mean = maximum.cell.mean()
            high_flow = flow.flow_quantile >= config.high_flow_quantile
            low_flow = flow.flow_quantile <= config.low_flow_quantile
            if low_flow and optimistic_mean <= config.ucb_low:
                proposals.append(
                    PruneProposal(
                        target_skill_id=skill.skill_id,
                        evidence=PruneEvidence.observed_low(
                            log_skill_marginal_flow=flow.log_skill_marginal_flow,
                            flow_quantile=flow.flow_quantile,
                            ucb=optimistic_mean,
                            k=0.0,
                            posterior_event_ids=posterior.posterior_event_ids,
                        ),
                        rationale_text="low flow and low optimistic posterior mean",
                        edge_exemplars=skill.edge_exemplars,
                    )
                )
            elif high_flow and isinstance(skill.split_modality, SupportedSplitModality):
                proposals.append(
                    SplitProposal(
                        target_skill_id=skill.skill_id,
                        evidence=SplitEvidence(
                            log_skill_marginal_flow=flow.log_skill_marginal_flow,
                            flow_quantile=flow.flow_quantile,
                            modality=skill.split_modality.value,
                        ),
                        rationale_text="high flow and separated posterior context means",
                        edge_exemplars=skill.edge_exemplars,
                    )
                )
            elif high_flow and pessimistic_mean <= config.lcb_low:
                proposals.append(
                    RefineProposal(
                        target_skill_id=skill.skill_id,
                        evidence=RefineEvidence(
                            log_skill_marginal_flow=flow.log_skill_marginal_flow,
                            flow_quantile=flow.flow_quantile,
                            lcb=pessimistic_mean,
                            k=0.0,
                            posterior_event_ids=posterior.posterior_event_ids,
                            target_context_keys=(minimum.cell.z.cell_key(skill.skill_id),),
                            target_contexts=(minimum.cell.z,),
                        ),
                        rationale_text="high flow and low posterior mean",
                        edge_exemplars=skill.edge_exemplars,
                    )
                )
            elif high_flow and pessimistic_mean >= config.lcb_high:
                proposals.append(
                    RetainProposal(
                        target_skill_id=skill.skill_id,
                        evidence=RetainEvidence(
                            log_skill_marginal_flow=flow.log_skill_marginal_flow,
                            flow_quantile=flow.flow_quantile,
                            lcb=pessimistic_mean,
                            k=0.0,
                            posterior_event_ids=posterior.posterior_event_ids,
                        ),
                        rationale_text="high flow and high posterior mean",
                        edge_exemplars=skill.edge_exemplars,
                    )
                )
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
                    rationale_text="uncovered high-|log I| edge",
                    edge_exemplars=group.edge_exemplars,
                )
            )
        if not proposals:
            raise NoApplicableEvolutionActionError(
                f"phase {pack.phase_event.event_id} has no posterior-mean action"
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
    ) -> EvolutionDecision:
        return self.decide(
            build_evidence_pack(
                window_flow=window_flow,
                posterior=posterior,
                trajectories=trajectories,
                config=config,
            ),
            config,
        )


__all__ = ["PosteriorMeanEvolutionPolicy"]
