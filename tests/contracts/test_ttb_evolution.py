from __future__ import annotations

from dataclasses import replace

import pytest

from skillev.contracts import (
    EvolutionActionRecord,
    EvolutionActionType,
    EvolutionEvidence,
    GenerateActionRecord,
    GenerateEvidence,
    PosteriorTaskFamilyMode,
    PruneActionRecord,
    PruneEvidence,
    RefineActionRecord,
    RefineEvidence,
    RetainCompressActionRecord,
    RetainEvidence,
    SplitActionRecord,
    SplitBranch,
    SplitEvidence,
    SplitModalityEvidence,
    SplitTaskFamilyAssignment,
    canonical_json,
    evolution_action_from_value,
    evolution_evidence_from_value,
    parse_canonical_json,
    stable_hash,
)


def _mode(
    *,
    context: str,
    event_id: str,
    mean: float,
    lcb: float,
    ucb: float,
) -> PosteriorTaskFamilyMode:
    return PosteriorTaskFamilyMode(
        task_family=context,
        cell_keys=(f"cell-{context}",),
        posterior_event_ids=(event_id,),
        evidence_mass=8.0,
        mean=mean,
        sigma=0.05,
        lcb=lcb,
        ucb=ucb,
        within_cell_mean_span=0.02,
    )


def _modality() -> SplitModalityEvidence:
    low = _mode(context="low", event_id="posterior-low", mean=0.2, lcb=0.15, ucb=0.25)
    high = _mode(context="high", event_id="posterior-high", mean=0.8, lcb=0.75, ucb=0.85)
    return SplitModalityEvidence(
        low_mode=low,
        high_mode=high,
        between_mean_gap=0.6,
        intervals_disjoint=True,
        source_task_families=("high", "low"),
        assignments=(
            SplitTaskFamilyAssignment(high, SplitBranch.HIGH),
            SplitTaskFamilyAssignment(low, SplitBranch.LOW),
        ),
        separation_cutpoint=0.5,
    )


def _evidences() -> tuple[EvolutionEvidence, ...]:
    return (
        RetainEvidence(
            log_skill_marginal_flow=1.2,
            flow_quantile=0.9,
            lcb=0.7,
            k=1.0,
            posterior_event_ids=("posterior-retain",),
        ),
        RefineEvidence(
            log_skill_marginal_flow=0.8,
            flow_quantile=0.8,
            lcb=0.2,
            k=1.0,
            posterior_event_ids=("posterior-refine",),
            target_context_keys=("context-a",),
        ),
        SplitEvidence(
            log_skill_marginal_flow=0.7,
            flow_quantile=0.75,
            modality=_modality(),
        ),
        PruneEvidence.observed_low(
            log_skill_marginal_flow=-1.0,
            flow_quantile=0.1,
            ucb=0.3,
            k=1.0,
            posterior_event_ids=("posterior-prune",),
        ),
        GenerateEvidence(
            importance_edge_ids=("edge-1",),
            minimum_absolute_log_importance=0.1,
            importance_quantile=0.9,
            importance_semantics="absolute-log-density-ratio@1",
        ),
    )


def _actions() -> tuple[EvolutionActionRecord, ...]:
    common = {
        "phase_event_id": "phase-1",
        "library_version_before": "library-v1",
        "library_version_after": "library-v2",
        "lineage_ref": "lineage-1",
        "proposal_content_hash": stable_hash({"fixture": "typed-action"}),
        "rationale_text": "The typed evidence selects this Phi action.",
    }
    return (
        RetainCompressActionRecord(
            action_id="action-retain",
            target_skill_id="skill-old",
            produced_skill_id="skill-retained",
            evidence=_evidences()[0],
            **common,
        ),
        RefineActionRecord(
            action_id="action-refine",
            target_skill_id="skill-old",
            produced_skill_id="skill-refined",
            evidence=_evidences()[1],
            **common,
        ),
        SplitActionRecord(
            action_id="action-split",
            target_skill_id="skill-old",
            produced_skill_ids=("skill-a", "skill-b"),
            evidence=_evidences()[2],
            **common,
        ),
        PruneActionRecord(
            action_id="action-prune",
            target_skill_id="skill-old",
            evidence=_evidences()[3],
            **common,
        ),
        GenerateActionRecord(
            action_id="action-generate",
            produced_skill_id="skill-new",
            evidence=_evidences()[4],
            **common,
        ),
    )


@pytest.mark.parametrize("evidence", _evidences())
def test_each_sealed_evidence_round_trips(evidence: EvolutionEvidence) -> None:
    payload = parse_canonical_json(canonical_json(evidence.to_value()))
    rebuilt = evolution_evidence_from_value(payload)

    assert rebuilt == evidence
    assert rebuilt.content_hash == evidence.content_hash


@pytest.mark.parametrize("record", _actions())
def test_each_sealed_action_round_trips(record: EvolutionActionRecord) -> None:
    payload = parse_canonical_json(canonical_json(record.to_value()))
    rebuilt = evolution_action_from_value(payload)

    assert rebuilt == record
    assert rebuilt.content_hash == record.content_hash


def test_action_types_are_the_closed_five_action_set() -> None:
    assert tuple(EvolutionActionType) == (
        EvolutionActionType.RETAIN_COMPRESS,
        EvolutionActionType.REFINE,
        EvolutionActionType.SPLIT,
        EvolutionActionType.PRUNE,
        EvolutionActionType.GENERATE,
    )


@pytest.mark.parametrize(
    ("record", "wrong_evidence"),
    [
        (
            _actions()[0],
            GenerateEvidence(("edge-1",), 0.1, 0.9, "absolute-log-density-ratio@1"),
        ),
        (_actions()[1], RetainEvidence(1.0, 0.9, 0.7, 1.0, ("posterior",))),
        (
            _actions()[2],
            PruneEvidence.observed_low(
                log_skill_marginal_flow=-1.0,
                flow_quantile=0.1,
                ucb=0.3,
                k=1.0,
                posterior_event_ids=("posterior",),
            ),
        ),
        (
            _actions()[3],
            GenerateEvidence(("edge-1",), 0.1, 0.9, "absolute-log-density-ratio@1"),
        ),
        (_actions()[4], RetainEvidence(1.0, 0.9, 0.7, 1.0, ("posterior",))),
    ],
)
def test_action_constructors_reject_other_evidence_variants(
    record: EvolutionActionRecord,
    wrong_evidence: EvolutionEvidence,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(record, evidence=wrong_evidence)  # type: ignore[arg-type]


def test_action_specific_shapes_are_not_optional_runtime_fields() -> None:
    prune = _actions()[3]
    generate = _actions()[4]
    split = _actions()[2]

    assert prune.target_skill_ids == ("skill-old",)
    assert prune.produced_skill_ids == ()
    assert generate.target_skill_ids == ()
    assert generate.produced_skill_ids == ("skill-new",)
    assert split.produced_skill_ids == ("skill-a", "skill-b")


def test_split_rejects_shared_posterior_evidence_instead_of_deduplicating() -> None:
    shared = "posterior-shared"
    with pytest.raises(ValueError):
        SplitModalityEvidence(
            low_mode=_mode(
                context="low",
                event_id=shared,
                mean=0.2,
                lcb=0.15,
                ucb=0.25,
            ),
            high_mode=_mode(
                context="high",
                event_id=shared,
                mean=0.8,
                lcb=0.75,
                ucb=0.85,
            ),
            between_mean_gap=0.6,
            intervals_disjoint=True,
            source_task_families=("high", "low"),
            assignments=(),
            separation_cutpoint=0.5,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"evidence_type": "unknown"},
        {"action_type": "unknown"},
        {
            **_actions()[0].to_value(),
            "action_type": EvolutionActionType.GENERATE.value,
        },
    ],
)
def test_sealed_wire_parsers_reject_incomplete_or_mismatched_variants(
    payload: object,
) -> None:
    parser = (
        evolution_evidence_from_value
        if isinstance(payload, dict) and "evidence_type" in payload
        else evolution_action_from_value
    )
    with pytest.raises(ValueError):
        parser(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"log_skill_marginal_flow": float("nan")},
        {"flow_quantile": 1.1},
        {"k": -1.0},
    ],
)
def test_numeric_evidence_rejects_nonfinite_or_out_of_domain_values(
    kwargs: dict[str, float],
) -> None:
    values = {
        "log_skill_marginal_flow": 1.0,
        "flow_quantile": 0.9,
        "lcb": 0.7,
        "k": 1.0,
        "posterior_event_ids": ("posterior",),
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        RetainEvidence(**values)  # type: ignore[arg-type]
