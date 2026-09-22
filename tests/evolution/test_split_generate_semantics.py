from __future__ import annotations

from dataclasses import replace

import pytest

from skillev.contracts import (
    ContextFeature,
    FailureMode,
    GenerateEvidence,
    HorizonBucket,
    PosteriorCellState,
    TokenBucket,
)
from skillev.evolution import (
    AuthoringActionKind,
    GenerateProposal,
    NoSplitModality,
    ObservedSkillPosterior,
    PosteriorCellEvidence,
    SplitCriterionConfig,
    SupportedSplitModality,
    detect_split_modality,
    phase_detection_enabled,
    select_uncovered_generate_edges,
)
from skillev.runtime import RunSlotKind, SkillApplicability, require_seed_library
from tests.v3_helpers import make_authoring_edge, make_skill_document


def _posterior(*, missing: str | None = None, crossing: bool = False) -> ObservedSkillPosterior:
    cells: list[PosteriorCellEvidence] = []
    for family, successes, failures in (
        ("domain/a", 0.0, 20.0),
        ("domain/b", 4.0, 4.0) if crossing else ("domain/b", 1.0, 19.0),
        ("domain/c", 20.0, 0.0),
    ):
        if family == missing:
            continue
        z = ContextFeature(
            context=family,
            failure_mode=FailureMode.SUCCESS,
            token_bucket=TokenBucket.LE_1K,
            horizon_bucket=HorizonBucket.LE_3,
        )
        cells.append(
            PosteriorCellEvidence(
                PosteriorCellState(
                    skill_id="skill-split",
                    z=z,
                    alpha_0=1.0,
                    beta_0=1.0,
                    alpha=1.0 + successes,
                    beta_count=1.0 + failures,
                    update_count=int(successes + failures),
                    last_event_id=f"event-{family}",
                ),
                (f"event-{family}",),
            )
        )
    return ObservedSkillPosterior("skill-split", tuple(cells))


def _split_config() -> SplitCriterionConfig:
    return SplitCriterionConfig(
        min_context_evidence_mass=4.0,
        max_within_context_mean_span=0.2,
        min_between_context_mean_gap=0.4,
        confidence_k=1.0,
    )


def test_split_partitions_every_finite_source_family_once() -> None:
    assessment = detect_split_modality(
        _posterior(),
        source_applicability=SkillApplicability(
            task_families=("domain/a", "domain/b", "domain/c"),
            contexts=("*",),
            required_tools=(),
            excluded_contexts=(),
        ),
        task_family_universe=("domain/a", "domain/b", "domain/c"),
        config=_split_config(),
    )

    assert isinstance(assessment, SupportedSplitModality)
    modality = assessment.value
    assert modality.low_task_families == ("domain/a", "domain/b")
    assert modality.high_task_families == ("domain/c",)
    assert modality.source_task_families == ("domain/a", "domain/b", "domain/c")
    assert len(modality.posterior_event_ids) == 3


def test_split_expands_wildcard_and_requires_supported_unambiguous_modes() -> None:
    applicability = SkillApplicability(
        task_families=("*",),
        contexts=("*",),
        required_tools=(),
        excluded_contexts=(),
    )
    universe = ("domain/a", "domain/b", "domain/c")

    supported = detect_split_modality(
        _posterior(),
        source_applicability=applicability,
        task_family_universe=universe,
        config=_split_config(),
    )
    missing = detect_split_modality(
        _posterior(missing="domain/b"),
        source_applicability=applicability,
        task_family_universe=universe,
        config=_split_config(),
    )
    crossing = detect_split_modality(
        _posterior(crossing=True),
        source_applicability=applicability,
        task_family_universe=universe,
        config=replace(_split_config(), min_between_context_mean_gap=0.6),
    )

    assert isinstance(supported, SupportedSplitModality)
    assert supported.value.source_task_families == universe
    assert isinstance(missing, NoSplitModality)
    assert isinstance(crossing, NoSplitModality)


def _candidate(edge_id: str, value: float):
    exemplar = replace(
        make_authoring_edge(edge_id),
        absolute_log_importance=value,
        log_importance_quantile=0.0,
    )
    return edge_id, value, exemplar


def test_generate_requires_absolute_floor_and_strict_relative_tail() -> None:
    zeros = [_candidate("zero-a:1", 0.0), _candidate("zero-b:1", 0.0)]
    uniform = [_candidate("same-a:1", 2.0), _candidate("same-b:1", 2.0)]
    below_floor = [_candidate("low:1", 0.2), _candidate("lower:1", 0.1)]

    assert not select_uncovered_generate_edges(
        zeros,
        (0.0, 0.0),
        importance_quantile=0.9,
        minimum_absolute_log_importance=0.1,
    )
    assert not select_uncovered_generate_edges(
        uniform,
        (2.0, 2.0),
        importance_quantile=0.9,
        minimum_absolute_log_importance=0.1,
    )
    assert not select_uncovered_generate_edges(
        below_floor,
        (0.2, 0.1),
        importance_quantile=0.9,
        minimum_absolute_log_importance=0.3,
    )


def test_generate_singleton_is_selected_only_above_absolute_floor() -> None:
    selected = select_uncovered_generate_edges(
        [_candidate("single:1", 0.5)],
        (0.5,),
        importance_quantile=1.0,
        minimum_absolute_log_importance=0.5,
    )
    rejected = select_uncovered_generate_edges(
        [_candidate("single:1", 0.5)],
        (0.5,),
        importance_quantile=1.0,
        minimum_absolute_log_importance=0.6,
    )

    assert tuple(item[0] for item in selected) == ("single:1",)
    assert selected[0][2].log_importance_quantile == 1.0
    assert rejected == ()


def test_generate_proposal_revalidates_sealed_selection_rule() -> None:
    invalid_exemplars = (
        replace(
            make_authoring_edge("below-floor:1"),
            absolute_log_importance=0.4,
            log_importance_quantile=1.0,
        ),
        replace(
            make_authoring_edge("below-tail:1"),
            absolute_log_importance=1.0,
            log_importance_quantile=0.8,
        ),
        replace(
            make_authoring_edge("covered:1"),
            absolute_log_importance=1.0,
            log_importance_quantile=1.0,
            invoked_skill_ids=("skill-existing",),
        ),
        replace(
            make_authoring_edge("invalid:1"),
            action_kind=AuthoringActionKind.INVALID,
            absolute_log_importance=1.0,
            log_importance_quantile=1.0,
        ),
    )
    for exemplar in invalid_exemplars:
        with pytest.raises(ValueError):
            GenerateProposal(
                evidence=GenerateEvidence(
                    importance_edge_ids=(exemplar.edge_id,),
                    minimum_absolute_log_importance=0.5,
                    importance_quantile=0.9,
                    importance_semantics="absolute-log-density-ratio@1",
                ),
                rationale_text="cover a qualifying tool-use pattern",
                edge_exemplars=(exemplar,),
            )


def test_phase_scope_and_seeded_library_semantics_are_closed() -> None:
    assert phase_detection_enabled(RunSlotKind.PHASE_SEARCH)
    assert not phase_detection_enabled(RunSlotKind.CLOSURE)
    seed = make_skill_document("seed")
    require_seed_library((seed,))

    with pytest.raises(ValueError):
        require_seed_library(())


def test_generate_admits_static_completion_without_claiming_absent_applicable_skills():
    from skillev.evolution.evidence import generate_candidate_contribution

    edge = replace(
        make_authoring_edge("static:1"),
        action_kind=AuthoringActionKind.COMPLETE,
        exposed_skill_ids=("available-but-not-invoked",),
    )
    contribution = generate_candidate_contribution(
        exemplar=edge, absolute_importance=2, invoked_skill_ids=()
    )
    assert contribution.uncovered is not None
    assert contribution.uncovered[2].exposed_skill_ids == ("available-but-not-invoked",)
