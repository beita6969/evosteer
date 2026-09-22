from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from skillev.calibration import CalibrationConfig, CalibrationEngine
from skillev.contracts import stable_hash
from skillev.diagnostics import DiagnosticsConfig
from skillev.evolution import AwaitingDetectorSegment
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptRunCursorState,
    ExactAttemptRunPlan,
    FullRuntimeExecutionState,
    OrderedTaskCursorState,
    RuntimeSnapshot,
    RuntimeSnapshotIdentity,
    RuntimeSnapshotStore,
    SkillLibraryState,
)
from skillev.training import MethodProjectionPipeline
from tests.v3_helpers import make_skill_document


def _snapshot() -> RuntimeSnapshot:
    library = SkillLibraryState.from_seed_documents((make_skill_document("seed"),))
    plan = ExactAttemptRunPlan(
        phase_search_steps=2,
        closure_steps=1,
        maximum_cycles=1,
    )
    cursor = AttemptRunCursorState.fresh(plan)
    for _ in range(plan.total_training_steps):
        cursor = cursor.after_training_step(plan)
    identity = RuntimeSnapshotIdentity(
        builder_kind=AttemptBuilderKind.FULL,
        public_identity_content_hash=stable_hash({"public": "snapshot-test"}),
        application_config_hash=stable_hash({"config": "snapshot-test"}),
        method_identity_hash=stable_hash({"method": "snapshot-test"}),
        protocol_hash=stable_hash({"protocol": "snapshot-test"}),
        protocol_freeze_id=stable_hash({"freeze": "snapshot-test"}),
        run_plan_hash=plan.content_hash,
        initial_library_version=library.current_version,
        initial_skill_library_state_hash=library.state_hash,
        initial_trainable_state_hash=stable_hash({"initial": "snapshot-test"}),
        ordered_task_sequence_hash=stable_hash({"tasks": ["a", "b"]}),
        sampling_schedule_algorithm="skillev-scientific-sampling@1",
        sampling_schedule_hash=stable_hash({"sampling": "snapshot-test"}),
    )
    projections = MethodProjectionPipeline.fresh(
        diagnostics_config=DiagnosticsConfig(),
        calibration=CalibrationEngine(CalibrationConfig()),
        library_version=library.current_version,
    )
    return RuntimeSnapshot(
        optimizer_step=3,
        experiment_id="experiment-v4",
        identity=identity,
        policy_directory="policy",
        optimizer_file="optimizer.pt",
        execution_state=FullRuntimeExecutionState(
            task_cursor=OrderedTaskCursorState(curriculum_id="curriculum", cursor=6),
            run_cursor=cursor,
            library=library,
            projections=projections.runtime_state(),
            detector=AwaitingDetectorSegment(library.current_version),
        ),
    )


class ExactArtifactStore:
    def __init__(self, snapshot: RuntimeSnapshot) -> None:
        self.snapshot = snapshot
        self.saved: list[tuple[object, str]] = []
        self.loaded: list[Path] = []

    def save_as(self, snapshot: object, *, name: str) -> Path:
        self.saved.append((snapshot, name))
        return Path("/exact") / name

    def load_metadata(self, directory: Path) -> RuntimeSnapshot:
        self.loaded.append(directory)
        if directory != Path("/exact/phase-step-00000003"):
            raise FileNotFoundError(directory)
        return self.snapshot


def test_runtime_snapshot_round_trip_has_exact_v6_schema() -> None:
    snapshot = _snapshot()

    assert RuntimeSnapshot.from_value(snapshot.to_value()) == snapshot
    with pytest.raises(ValueError):
        RuntimeSnapshot.from_value({**snapshot.to_value(), "snapshot_id": "legacy"})
    with pytest.raises(ValueError):
        RuntimeSnapshot.from_value({**snapshot.to_value(), "format": "skillev-runtime-snapshot@5"})


def test_runtime_snapshot_store_loads_only_the_caller_selected_path() -> None:
    snapshot = _snapshot()
    artifacts = ExactArtifactStore(snapshot)
    store = RuntimeSnapshotStore(artifacts)
    exact = Path("/exact/phase-step-00000003")

    assert store.load_exact(exact) == snapshot
    assert artifacts.loaded == [exact]
    with pytest.raises(FileNotFoundError):
        store.load_exact(Path("/exact/latest"))
    assert artifacts.loaded == [exact, Path("/exact/latest")]


def test_runtime_snapshot_rejects_cross_kind_and_run_plan_identity_mismatch() -> None:
    snapshot = _snapshot()

    with pytest.raises(ValueError):
        replace(
            snapshot,
            identity=replace(snapshot.identity, builder_kind=AttemptBuilderKind.NO_BAYESIAN),
        )
    with pytest.raises(ValueError):
        replace(snapshot.identity, initial_library_version="not-a-library-hash")
    with pytest.raises(ValueError):
        replace(
            snapshot,
            identity=replace(
                snapshot.identity,
                run_plan_hash=stable_hash({"plan": "another"}),
            ),
        )


def test_runtime_state_and_projection_restore_reject_cross_segment_controls() -> None:
    snapshot = _snapshot()
    state = snapshot.execution_state
    other_library = SkillLibraryState.from_seed_documents((make_skill_document("other"),))

    with pytest.raises(ValueError):
        replace(state, library=other_library)
    with pytest.raises(ValueError):
        MethodProjectionPipeline.from_runtime_state(
            diagnostics_config=DiagnosticsConfig(),
            calibration_config=CalibrationConfig(),
            state=replace(
                state.projections,
                retained_batch_count=state.projections.retained_batch_count + 1,
            ),
        )
