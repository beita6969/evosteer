from __future__ import annotations

import json
from dataclasses import replace

import pytest

from skillev.contracts import (
    ContextFeature,
    EvolutionActionType,
    FailureMode,
    GenerateEvidence,
    HorizonBucket,
    PosteriorCellState,
    RetainEvidence,
    TokenBucket,
)
from skillev.evolution import (
    AuthoredSkillDraft,
    AuthoringActionKind,
    AuthoringFailedError,
    AuthoringResult,
    EvidencePack,
    EvolutionConfig,
    EvolutionDecision,
    FullEvolutionPolicy,
    GenerateProposal,
    NoSplitModality,
    ObservedSkillFlow,
    ObservedSkillPosterior,
    PosteriorCellEvidence,
    RetainProposal,
    SkillAuthoringAuthority,
    SkillEvidence,
    VerifiedNoOpEvolutionDecision,
    authoring_seed,
    build_evolution_mutation,
    group_generate_exemplars,
    is_generate_candidate_action,
    partition_reset_seed,
    proposal_content_hash,
    required_phi_budget,
)
from skillev.evolution.authoring import uncovered_edge_requirement_id
from skillev.experiments.arms import (
    FlowOnlyEvidencePack,
    FlowOnlyEvolutionPolicy,
    PosteriorMeanEvolutionPolicy,
)
from skillev.experiments.arms.no_bayesian_execution import flow_only_authoring_seed
from skillev.runtime import (
    BudgetVector,
    SkillApplicability,
    SkillLibrary,
    SkillLibraryState,
    SkillRequirement,
)
from tests.fakes.skill_author import ScriptedSkillAuthor
from tests.v3_helpers import (
    TEST_CONTEXT_ID,
    TEST_TASK_FAMILY,
    CharacterTokenizer,
    make_authoring_edge,
    make_phase_event,
    make_skill_document,
)


def _exemplar(edge_id: str):
    return make_authoring_edge(edge_id)


def test_generate_gap_includes_public_completion_and_tool_edges() -> None:
    assert is_generate_candidate_action(AuthoringActionKind.COMPLETE)
    assert not is_generate_candidate_action(AuthoringActionKind.INVALID)
    assert is_generate_candidate_action(AuthoringActionKind.TOOL)


def test_authoring_and_partition_reset_seeds_use_scientific_cycle_coordinates_only() -> None:
    proposal_hash = proposal_content_hash(
        RetainProposal(
            target_skill_id="skill-one",
            evidence=RetainEvidence(
                log_skill_marginal_flow=1.0,
                flow_quantile=1.0,
                lcb=0.9,
                k=1.0,
            ),
            rationale_text="retain",
            edge_exemplars=(_exemplar("edge-seed-probe"),),
        )
    )
    expected = authoring_seed(
        base_seed=23,
        cycle_ordinal=2,
        proposal_content_hash=proposal_hash,
        proposal_index=0,
    )

    assert expected == flow_only_authoring_seed(
        base_seed=23,
        cycle_ordinal=2,
        proposal_content_hash=proposal_hash,
        proposal_index=0,
    )
    assert partition_reset_seed(base_seed=23, cycle_ordinal=2) == partition_reset_seed(
        base_seed=23,
        cycle_ordinal=2,
    )


def _skill_evidence(skill_id: str) -> SkillEvidence:
    z = ContextFeature(
        context="debug-family",
        failure_mode=FailureMode.SUCCESS,
        token_bucket=TokenBucket.LE_1K,
        horizon_bucket=HorizonBucket.LE_3,
    )
    cell = PosteriorCellState(
        skill_id=skill_id,
        z=z,
        alpha_0=1.0,
        beta_0=1.0,
        alpha=99.0,
        beta_count=1.0,
        update_count=98,
        last_event_id="posterior-final",
    )
    exemplar = _exemplar(f"{skill_id}-trajectory:1")
    return SkillEvidence(
        skill_id=skill_id,
        flow=ObservedSkillFlow(
            skill_id=skill_id,
            log_skill_marginal_flow=1.0,
            flow_quantile=1.0,
            invoking_trajectory_count=1,
            invoking_edge_count=1,
            invoking_edge_ids=(exemplar.edge_id,),
        ),
        posterior=ObservedSkillPosterior(
            skill_id=skill_id,
            cells=(PosteriorCellEvidence(cell, ("posterior-final",)),),
        ),
        split_modality=NoSplitModality(),
        target_context_keys=(z.cell_key(skill_id),),
        edge_exemplars=(exemplar,),
    )


def test_full_policy_returns_complete_untruncated_nonempty_decision() -> None:
    phase = make_phase_event()
    gap = _exemplar("trajectory-gap:1")
    pack = EvidencePack(
        phase_event=phase,
        skills=(_skill_evidence("skill-alpha"), _skill_evidence("skill-beta")),
        uncovered_importance_edges=(gap.edge_id,),
        uncovered_exemplars=(gap,),
    )

    config = EvolutionConfig(generate_min_absolute_log_importance=0.1)
    decision = FullEvolutionPolicy().decide(pack, config)
    repeated = FullEvolutionPolicy().decide(pack, config)

    assert tuple(item.evidence.evidence_type for item in decision.proposals) == (
        "retain",
        "retain",
        "generate",
    )
    assert len(decision.proposals) == 3
    assert decision.to_value() == repeated.to_value()
    assert decision.proposal_content_hashes == tuple(
        proposal_content_hash(proposal) for proposal in decision.proposals
    )
    assert decision.proposal_content_hashes == repeated.proposal_content_hashes
    authoring_config = EvolutionConfig(
        generate_min_absolute_log_importance=0.1,
        max_skill_instruction_tokens_per_draft=7,
        max_authoring_completion_tokens=59,
        max_authoring_prompt_tokens=31,
    )
    assert required_phi_budget(decision, authoring_config) == BudgetVector(
        input_tokens=93,
        output_tokens=177,
        model_calls=3,
    )
    no_op = FullEvolutionPolicy().decide(
        EvidencePack(
            phase_event=phase,
            skills=(),
            uncovered_importance_edges=(),
            uncovered_exemplars=(),
        ),
        EvolutionConfig(generate_min_absolute_log_importance=0.1),
    )
    assert isinstance(no_op, VerifiedNoOpEvolutionDecision)
    assert no_op.proposal_content_hashes == ()
    assert required_phi_budget(no_op, config) == BudgetVector()


def test_generate_groups_mixed_authorities_deterministically_for_all_policies() -> None:
    late = make_authoring_edge(
        "z-edge:1",
        task_family="zeta/task",
        context_id="zeta-context",
        available_tools=("zeta-tool",),
        importance_quantile=0.91,
    )
    early = make_authoring_edge(
        "a-edge:1",
        task_family="alpha/task",
        context_id="alpha-context",
        available_tools=("alpha-tool",),
        importance_quantile=0.97,
    )
    groups = group_generate_exemplars(
        importance_edge_ids=(late.edge_id, early.edge_id),
        edge_exemplars=(late, early),
    )

    assert tuple(group.importance_edge_ids for group in groups) == (
        (early.edge_id,),
        (late.edge_id,),
    )
    assert tuple(group.edge_exemplars[0].log_importance_quantile for group in groups) == (
        0.97,
        0.91,
    )

    phase = make_phase_event()
    full_pack = EvidencePack(
        phase_event=phase,
        skills=(),
        uncovered_importance_edges=(late.edge_id, early.edge_id),
        uncovered_exemplars=(late, early),
    )
    config = EvolutionConfig(generate_min_absolute_log_importance=0.1)
    full = FullEvolutionPolicy().decide(full_pack, config)
    posterior_mean = PosteriorMeanEvolutionPolicy().decide(full_pack, config)
    flow_only = FlowOnlyEvolutionPolicy().decide(
        FlowOnlyEvidencePack(
            phase_event_id=phase.event_id,
            skills=(),
            uncovered_edge_ids=(late.edge_id, early.edge_id),
            uncovered_exemplars=(late, early),
        ),
        EvolutionConfig(generate_min_absolute_log_importance=0.1),
    )

    expected = ((early.edge_id,), (late.edge_id,))
    assert tuple(item.evidence.importance_edge_ids for item in full.proposals) == expected
    assert tuple(item.evidence.importance_edge_ids for item in posterior_mean.proposals) == expected
    assert tuple(item.evidence.importance_edge_ids for item in flow_only.proposals) == expected
    assert all(len(item.edge_exemplars) == 1 for item in full.proposals)
    assert all(len(item.edge_exemplars) == 1 for item in flow_only.proposals)
    assert (
        required_phi_budget(
            full, EvolutionConfig(generate_min_absolute_log_importance=0.1)
        ).model_calls
        == 2
    )


def test_generate_mutation_uses_structured_authoring_and_one_new_skill() -> None:
    seed = make_skill_document("alpha")
    library = SkillLibrary(SkillLibraryState.from_seed_documents((seed,)))
    phase = make_phase_event(library_version=library.current_version)
    exemplar = _exemplar("trajectory-gap:1")
    decision = EvolutionDecision(
        phase.event_id,
        (
            GenerateProposal(
                evidence=GenerateEvidence(
                    importance_edge_ids=(exemplar.edge_id,),
                    minimum_absolute_log_importance=0.1,
                    importance_quantile=0.9,
                    importance_semantics="absolute-log-density-ratio@1",
                ),
                rationale_text="cover high asymmetry without an active skill",
                edge_exemplars=(exemplar,),
            ),
        ),
    )
    requirement = SkillRequirement(
        requirement_id=uncovered_edge_requirement_id(exemplar.edge_id),
        text="Cover this public edge.",
        kind="evolvable-strategy",
    )
    author = ScriptedSkillAuthor(
        CharacterTokenizer(),
        (
            AuthoringResult(
                drafts=(
                    AuthoredSkillDraft(
                        title="Gap skill",
                        summary="Covers the uncovered edge.",
                        instructions="Follow the gap procedure.",
                        applicability=SkillApplicability(
                            task_families=(TEST_TASK_FAMILY,),
                            contexts=(TEST_CONTEXT_ID,),
                            required_tools=(),
                            excluded_contexts=(),
                        ),
                        requirements=(requirement,),
                    ),
                )
            ),
        ),
    )
    authority = SkillAuthoringAuthority(
        input_schema_id="input@3",
        output_schema_id="output@3",
        license_id="unit-test",
        allowed_task_families=(TEST_TASK_FAMILY,),
        allowed_tools=(),
    )

    mutation = build_evolution_mutation(
        decision,
        author=author,
        library=library,
        phase_event=phase,
        config=EvolutionConfig(generate_min_absolute_log_importance=0.1),
        authority=authority,
        base_seed=17,
        cycle_ordinal=1,
    )

    assert len(mutation.new_documents) == 1
    assert mutation.actions[0].action_type is EvolutionActionType.GENERATE
    assert mutation.actions[0].proposal_content_hash == decision.proposal_content_hashes[0]
    assert mutation.actions[0].evidence.to_value() == decision.proposals[0].evidence.to_value()
    assert mutation.actions[0].rationale_text == decision.proposals[0].rationale_text
    assert mutation.new_documents[0].manifest.skill_id in mutation.active_skill_ids_after
    assert library.current_version == phase.library_version


def test_non_retain_complete_h0_block_must_fit_the_fixed_skill_block_cap() -> None:
    seed = make_skill_document("alpha")
    library = SkillLibrary(SkillLibraryState.from_seed_documents((seed,)))
    phase = make_phase_event(library_version=library.current_version)
    exemplar = _exemplar("trajectory-oversized-gap:1")
    decision = EvolutionDecision(
        phase.event_id,
        (
            GenerateProposal(
                evidence=GenerateEvidence(
                    importance_edge_ids=(exemplar.edge_id,),
                    minimum_absolute_log_importance=0.1,
                    importance_quantile=0.9,
                    importance_semantics="absolute-log-density-ratio@1",
                ),
                rationale_text="cover the uncovered public tool edge",
                edge_exemplars=(exemplar,),
            ),
        ),
    )
    requirement = SkillRequirement(
        requirement_id=uncovered_edge_requirement_id(exemplar.edge_id),
        text="Preserve the public tool-edge procedure.",
        kind="evolvable-strategy",
    )
    author = ScriptedSkillAuthor(
        CharacterTokenizer(),
        (
            AuthoringResult(
                drafts=(
                    AuthoredSkillDraft(
                        title="Oversized generated skill",
                        summary="x" * 3_500,
                        instructions=(
                            "Use the public tool and inspect its observation before continuing."
                        ),
                        applicability=SkillApplicability(
                            task_families=(TEST_TASK_FAMILY,),
                            contexts=(TEST_CONTEXT_ID,),
                            required_tools=(),
                            excluded_contexts=(),
                        ),
                        requirements=(requirement,),
                    ),
                )
            ),
        ),
    )
    before = library.state

    with pytest.raises(AuthoringFailedError):
        build_evolution_mutation(
            decision,
            author=author,
            library=library,
            phase_event=phase,
            config=EvolutionConfig(generate_min_absolute_log_importance=0.1),
            authority=SkillAuthoringAuthority(
                input_schema_id="input@3",
                output_schema_id="output@3",
                license_id="unit-test",
                allowed_task_families=(TEST_TASK_FAMILY,),
                allowed_tools=(),
            ),
            base_seed=17,
            cycle_ordinal=1,
        )

    assert library.state is before
    assert len(author.requests) == 1


def test_authoring_failure_has_no_library_side_effect() -> None:
    seed = make_skill_document("alpha")
    library = SkillLibrary(SkillLibraryState.from_seed_documents((seed,)))
    phase = make_phase_event(library_version=library.current_version)
    exemplar = _exemplar("alpha-trajectory:1")
    decision = EvolutionDecision(
        phase.event_id,
        (
            RetainProposal(
                target_skill_id=seed.manifest.skill_id,
                evidence=RetainEvidence(
                    log_skill_marginal_flow=1.0,
                    flow_quantile=1.0,
                    lcb=0.8,
                    k=1.0,
                    posterior_event_ids=("posterior-alpha",),
                ),
                rationale_text="high flow and reliable posterior",
                edge_exemplars=(exemplar,),
            ),
        ),
    )
    author = ScriptedSkillAuthor(
        CharacterTokenizer(),
        (AuthoringFailedError("one sealed authoring call failed"),),
    )
    before = library.state

    with pytest.raises(AuthoringFailedError):
        build_evolution_mutation(
            decision,
            author=author,
            library=library,
            phase_event=phase,
            config=EvolutionConfig(generate_min_absolute_log_importance=0.1),
            authority=SkillAuthoringAuthority(
                input_schema_id="input@3",
                output_schema_id="output@3",
                license_id="unit-test",
                allowed_task_families=(TEST_TASK_FAMILY,),
                allowed_tools=(),
            ),
            base_seed=17,
            cycle_ordinal=1,
        )

    assert library.state is before
    assert len(author.requests) == 1


def test_refine_keeps_the_actual_weak_context_coordinates_in_authoring():
    from skillev.evolution import RefineProposal, render_authoring_prompt
    from skillev.evolution.execution import authoring_request_for_proposal

    skill = _skill_evidence("skill-alpha")
    original = skill.posterior.cells[0]
    weak = replace(original.cell, alpha=1.0, beta_count=99.0)
    skill = replace(
        skill,
        posterior=ObservedSkillPosterior(
            skill_id=skill.skill_id,
            cells=(replace(original, cell=weak),),
        ),
    )
    phase = make_phase_event()
    decision = FullEvolutionPolicy().decide(
        EvidencePack(phase, (skill,), (), ()),
        EvolutionConfig(generate_min_absolute_log_importance=0.1),
    )
    proposal = decision.proposals[0]
    assert isinstance(proposal, RefineProposal)
    assert proposal.evidence.target_contexts == (weak.z,)
    library = SkillLibrary(SkillLibraryState.from_seed_documents((make_skill_document("alpha"),)))
    request = authoring_request_for_proposal(
        proposal,
        proposal_index=0,
        phase_event=phase,
        library=library,
        authority=SkillAuthoringAuthority("input", "output", "unit-test", (TEST_TASK_FAMILY,), ()),
        base_seed=0,
        cycle_ordinal=1,
    )
    payload = json.loads(
        render_authoring_prompt(request, tokenizer=CharacterTokenizer()).splitlines()[1]
    )
    evidence = json.loads(payload["material"]["evidence_summary"])["evidence"]
    assert evidence["target_contexts"] == [weak.z.to_value()]
    assert evidence["target_context_keys"] == [weak.z.cell_key(skill.skill_id)]
