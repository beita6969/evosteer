from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest

from skillev.contracts import (
    ContextFeature,
    EdgeLogprobRecord,
    EntropyObservation,
    EvolutionCycleCommitted,
    EvolutionMutationValue,
    EvolutionPhaseOpened,
    GenerateActionRecord,
    GenerateEvidence,
    InitialContext,
    LibraryInitialized,
    PhaseCheckpointArtifact,
    PhaseCheckpointPublished,
    PhaseTransitionEvent,
    PosteriorBatchUpdate,
    PosteriorCellState,
    PosteriorTaskFamilyMode,
    PosteriorUpdateEvent,
    PruneActionRecord,
    PruneEvidence,
    RefineActionRecord,
    RefineEvidence,
    RetainCompressActionRecord,
    RetainEvidence,
    SplitActionRecord,
    SplitEvidence,
    SplitModalityEvidence,
    TerminalReward,
    TrainingStepCommit,
    TrainingStepReportValue,
    TrajectoryRecord,
    TrajectoryResidual,
    TrajectoryStep,
    TTBBatchStats,
    WindowStats,
)
from skillev.contracts.schema_registry import SchemaDescriptor
from skillev.contracts.ttb_schemas import (
    _TTB_SCHEMA_SPECS,
    ALL_TTB_SCHEMA_IDS,
    MAIN_TTB_SCHEMA_IDS,
    TTB_SCHEMA_REGISTRY,
    TTB_SCHEMA_VERSION,
    build_ttb_schema_registry,
)

_SERIALIZABLE_RECORDS = (
    (ContextFeature, "skillev.context-feature"),
    (EdgeLogprobRecord, "skillev.edge-logprob-record"),
    (EntropyObservation, "skillev.entropy-observation"),
    (EvolutionCycleCommitted, "skillev.evolution-cycle-committed"),
    (EvolutionMutationValue, "skillev.evolution-mutation-value"),
    (EvolutionPhaseOpened, "skillev.evolution-phase-opened"),
    (GenerateActionRecord, "skillev.generate-action-record"),
    (GenerateEvidence, "skillev.generate-evidence"),
    (InitialContext, "skillev.initial-context"),
    (LibraryInitialized, "skillev.library-initialized"),
    (PhaseCheckpointArtifact, "skillev.phase-checkpoint-artifact"),
    (PhaseCheckpointPublished, "skillev.phase-checkpoint-published"),
    (PhaseTransitionEvent, "skillev.phase-transition-event"),
    (PosteriorBatchUpdate, "skillev.posterior-batch-update"),
    (PosteriorCellState, "skillev.posterior-cell-state"),
    (PosteriorTaskFamilyMode, "skillev.posterior-task-family-mode"),
    (PosteriorUpdateEvent, "skillev.posterior-update-event"),
    (PruneActionRecord, "skillev.prune-action-record"),
    (PruneEvidence, "skillev.prune-evidence"),
    (RefineActionRecord, "skillev.refine-action-record"),
    (RefineEvidence, "skillev.refine-evidence"),
    (RetainCompressActionRecord, "skillev.retain-compress-action-record"),
    (RetainEvidence, "skillev.retain-evidence"),
    (SplitActionRecord, "skillev.split-action-record"),
    (SplitEvidence, "skillev.split-evidence"),
    (SplitModalityEvidence, "skillev.split-modality-evidence"),
    (TerminalReward, "skillev.terminal-reward"),
    (TrainingStepCommit, "skillev.training-step-commit"),
    (TrainingStepReportValue, "skillev.training-step-report-value"),
    (TrajectoryRecord, "skillev.trajectory-record"),
    (TrajectoryResidual, "skillev.trajectory-residual"),
    (TrajectoryStep, "skillev.trajectory-step"),
    (TTBBatchStats, "skillev.ttb-batch-stats"),
    (WindowStats, "skillev.window-stats"),
)

_DERIVED_WIRE_FIELDS = {
    GenerateActionRecord: {"action_type"},
    GenerateEvidence: {"evidence_type"},
    PruneActionRecord: {"action_type"},
    PruneEvidence: {"evidence_type"},
    RefineActionRecord: {"action_type"},
    RefineEvidence: {"evidence_type"},
    RetainCompressActionRecord: {"action_type"},
    RetainEvidence: {"evidence_type"},
    SplitActionRecord: {"action_type"},
    SplitEvidence: {"evidence_type"},
}


def test_ttb_registry_is_unique_sorted_and_reproducible() -> None:
    first = build_ttb_schema_registry()
    second = build_ttb_schema_registry()
    keys = tuple((descriptor.schema_id, descriptor.version) for descriptor in first.schemas)

    assert keys == tuple(sorted(keys))
    assert len(keys) == len(set(keys))
    assert first == second == TTB_SCHEMA_REGISTRY
    assert first.registry_hash == second.registry_hash


@pytest.mark.parametrize("schema_id", MAIN_TTB_SCHEMA_IDS)
def test_each_main_contract_has_an_exact_current_schema(schema_id: str) -> None:
    descriptor = TTB_SCHEMA_REGISTRY.get(schema_id, TTB_SCHEMA_VERSION)

    assert descriptor is not None
    assert descriptor.schema_id in MAIN_TTB_SCHEMA_IDS


@pytest.mark.parametrize(("record_type", "schema_id"), _SERIALIZABLE_RECORDS)
def test_every_independently_serializable_record_has_a_schema(
    record_type: type[object],
    schema_id: str,
) -> None:
    descriptor = TTB_SCHEMA_REGISTRY.get(schema_id, TTB_SCHEMA_VERSION)

    assert callable(record_type.to_value)
    assert callable(record_type.from_value)
    assert isinstance(record_type.content_hash, property)
    assert descriptor is not None
    assert SchemaDescriptor.from_value(descriptor.to_value()) == descriptor


@pytest.mark.parametrize(("record_type", "_schema_id"), _SERIALIZABLE_RECORDS)
def test_every_serializable_record_is_a_frozen_slotted_dataclass(
    record_type: type[object],
    _schema_id: str,
) -> None:
    assert is_dataclass(record_type)
    assert record_type.__dataclass_params__.frozen  # type: ignore[attr-defined]
    assert hasattr(record_type, "__slots__")
    assert "__dict__" not in record_type.__slots__  # type: ignore[attr-defined]


def test_each_schema_lists_the_exact_closed_wire_field_set() -> None:
    schema_fields = {
        schema_id: {field_name for field_name, _wire_type in wire_fields}
        for schema_id, _record_name, wire_fields in _TTB_SCHEMA_SPECS
    }

    for record_type, schema_id in _SERIALIZABLE_RECORDS:
        expected = {item.name for item in fields(record_type)}
        expected.update(_DERIVED_WIRE_FIELDS.get(record_type, set()))
        assert schema_fields[schema_id] == expected


def test_registry_id_projection_matches_registered_descriptors() -> None:
    projected_ids = tuple(descriptor.schema_id for descriptor in TTB_SCHEMA_REGISTRY.schemas)

    assert ALL_TTB_SCHEMA_IDS == projected_ids
    assert ALL_TTB_SCHEMA_IDS == tuple(sorted(ALL_TTB_SCHEMA_IDS))
    assert len(ALL_TTB_SCHEMA_IDS) == len(_SERIALIZABLE_RECORDS) + 2


def test_main_contract_projection_contains_exactly_seven_unique_ids() -> None:
    assert len(MAIN_TTB_SCHEMA_IDS) == 7
    assert len(MAIN_TTB_SCHEMA_IDS) == len(set(MAIN_TTB_SCHEMA_IDS))
    assert set(MAIN_TTB_SCHEMA_IDS).issubset(ALL_TTB_SCHEMA_IDS)
