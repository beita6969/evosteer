from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from skillev.contracts import (
    GenerateEvidence,
    JsonValue,
    PhaseCheckpointPublished,
    PhaseTransitionEvent,
    PosteriorCellState,
    TrainingStepReportValue,
)
from skillev.diagnostics import (
    BatchDiagnostics,
    DiagnosticsConfig,
    DiagnosticsState,
    FreshDiagnosticsSegment,
    InsufficientResidualWindow,
)
from skillev.evolution import (
    ActiveDetectorSegment,
    AuthoredSkillDraft,
    AuthoringFailedError,
    AuthoringResult,
    AwaitingDetectorSegment,
    DetectorBatchRecord,
    DetectorRuntimeState,
    EntropyWindow,
    EvolutionConfig,
    EvolutionDecision,
    EvolutionLoop,
    FullEvolutionDecision,
    GenerateProposal,
    NoPhaseTransition,
    NoPhaseTransitionReason,
    PhaseTransitionDetected,
    PhiBudgetAuthority,
    SkillAuthoringAuthority,
    VerifiedNoOpEvolutionDecision,
)
from skillev.evolution.authoring import uncovered_edge_requirement_id
from skillev.runtime import (
    AdapterGeneration,
    AttemptRunProgress,
    BudgetLedger,
    BudgetReservation,
    BudgetSettlement,
    BudgetVector,
    EventAppendFailedError,
    EventType,
    ExactAttemptRunPlan,
    RemainingAttemptCapacity,
    RuntimeSnapshotStore,
    SkillApplicability,
    SkillLibrary,
    SkillLibraryState,
    SkillRequirement,
)
from skillev.training import TrainingStepExecutionContext
from tests.fakes.skill_author import ScriptedSkillAuthor
from tests.v3_helpers import (
    TEST_CONTEXT_ID,
    TEST_TASK_FAMILY,
    CharacterTokenizer,
    make_authoring_edge,
    make_phase_event,
    make_skill_document,
)


def _report(step: int) -> TrainingStepReportValue:
    return TrainingStepReportValue(
        optimizer_step=step,
        batch_id=f"batch-{step}",
        torch_batch_loss=1.0,
        audited_batch_loss=1.0,
        mean_reward=0.5,
        grad_norm_forward=1.0,
        grad_norm_backward=1.0,
        grad_norm_z=1.0,
        forward_adapter_version=f"forward@{step}",
        backward_adapter_version=f"backward@{step}",
        z_version=f"z@{step}",
        started_at="2026-07-25T00:00:00Z",
        completed_at="2026-07-25T00:00:01Z",
    )


def _diagnostic(step: int, library_version: str) -> BatchDiagnostics:
    return BatchDiagnostics(
        batch_id=f"batch-{step}",
        optimizer_step=step,
        library_version=library_version,
        trajectories=(),
        skill_flows=(),
        residual_window=InsufficientResidualWindow(
            observed_batch_count=step,
            required_batch_count=max(step + 1, 2),
        ),
        config=DiagnosticsConfig(window_size=1),
    )


@dataclass(slots=True)
class _Projection:
    library_version: str
    latest: BatchDiagnostics = field(init=False)
    reset_versions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.latest = _diagnostic(1, self.library_version)

    @property
    def latest_diagnostic(self) -> BatchDiagnostics:
        return self.latest

    @property
    def diagnostics_state(self) -> DiagnosticsState:
        return FreshDiagnosticsSegment(self.library_version)

    @property
    def calibration_cells(self) -> tuple[PosteriorCellState, ...]:
        return ()

    @property
    def event_ids_by_skill(self) -> dict[str, tuple[str, ...]]:
        return {}

    @property
    def event_ids_by_cell(self) -> dict[str, tuple[str, ...]]:
        return {}

    def freeze_for_phase(self, phase):
        from skillev.evolution.projection_snapshot import PhaseProjectionSnapshot

        window = self.diagnostics_for_batch_ids(
            (*phase.previous_window.member_batch_ids, *phase.current_window.member_batch_ids)
        )
        return PhaseProjectionSnapshot(
            window, (), {}, self.authoring_evidence_for_diagnostics(window)
        )

    def diagnostics_for_batch_ids(
        self,
        batch_ids: tuple[str, ...],
    ) -> tuple[BatchDiagnostics, ...]:
        return tuple(
            _diagnostic(index, self.library_version)
            for index, _batch_id in enumerate(batch_ids, start=1)
        )

    def authoring_evidence_for_diagnostics(
        self,
        diagnostics: tuple[BatchDiagnostics, ...],
    ) -> dict[str, object]:
        del diagnostics
        exemplar = make_authoring_edge("gap:1")
        return {exemplar.edge_id: exemplar}

    def preview_reset(self, *, old_version: str, new_version: str) -> str:
        if old_version != self.library_version:
            raise ValueError("projection reset old version differs")
        return new_version

    def commit_reset(self, state: str) -> None:
        self.library_version = state
        self.reset_versions.append(state)


@dataclass(slots=True)
class _Training:
    projection: _Projection
    optimizer_step: int = 0
    runtime_state: str = "idle"
    reset_seeds: list[int] = field(default_factory=list)
    checkpoint_steps: set[int] = field(default_factory=set)
    ledger: BudgetLedger = field(
        default_factory=lambda: BudgetLedger(
            run_id="loop-test",
            attempt_id="attempt-1",
            cap=BudgetVector(input_tokens=100_000, output_tokens=100_000, model_calls=100),
        )
    )

    @property
    def checkpoint_due(self) -> bool:
        return self.optimizer_step in self.checkpoint_steps

    @property
    def policy_snapshot_id(self) -> str:
        return f"policy@{self.optimizer_step}"

    @property
    def scientific_base_seed(self) -> int:
        return 17

    def validate_attempt_budget(
        self,
        *,
        capacity: RemainingAttemptCapacity,
        phi_per_cycle_maximum: BudgetVector,
    ) -> None:
        if not isinstance(capacity, RemainingAttemptCapacity) or not isinstance(
            phi_per_cycle_maximum, BudgetVector
        ):
            raise ValueError("invalid attempt budget request")

    async def collect_batch(self) -> object:
        if self.runtime_state != "idle":
            raise RuntimeError("collect_batch requires idle training state")
        self.runtime_state = "batch-ready"
        return object()

    def apply_step(
        self,
        context: TrainingStepExecutionContext,
    ) -> TrainingStepReportValue:
        if self.runtime_state != "batch-ready":
            raise RuntimeError("apply_step requires a ready batch")
        if context.run_cursor_after.completed_training_steps != self.optimizer_step + 1:
            raise ValueError("training context cursor differs from next test step")
        self.optimizer_step += 1
        self.projection.latest = _diagnostic(
            self.optimizer_step,
            self.projection.library_version,
        )
        self.runtime_state = "applied"
        return _report(self.optimizer_step)

    def set_execution_stage(self, stage: str) -> None:
        del stage  # This fixture records scientific transactions, not timing.

    def install_applied_projection(self) -> None:
        if self.runtime_state != "applied":
            raise RuntimeError("projection install requires an applied step")
        self.runtime_state = "installed"

    def finalize_applied_step(self) -> TrainingStepReportValue:
        if self.runtime_state != "installed":
            raise RuntimeError("finalize requires an installed step")
        self.runtime_state = "idle"
        return _report(self.optimizer_step)

    def reset_partition(self, seed: int) -> str:
        self.reset_seeds.append(seed)
        return f"z-reset-{seed}"


@dataclass(slots=True)
class _Detector:
    phase: PhaseTransitionEvent
    emitted: bool = False
    reset_versions: list[str] = field(default_factory=list)
    observed_steps: list[int] = field(default_factory=list)
    state: DetectorRuntimeState = field(init=False)

    def __post_init__(self) -> None:
        self.state = AwaitingDetectorSegment(self.phase.library_version)

    @property
    def runtime_state(self) -> DetectorRuntimeState:
        return self.state

    def preview_observation(self, diagnostic: BatchDiagnostics) -> object:
        record = DetectorBatchRecord.from_diagnostic(diagnostic)
        prior = () if isinstance(self.state, AwaitingDetectorSegment) else self.state.batch_records
        active = ActiveDetectorSegment(
            library_version=self.phase.library_version,
            batch_records=(*prior, record),
            entropy_evidence=EntropyWindow(self.phase.entropy_series),
            phase_already_triggered=not self.emitted,
            cursor=diagnostic.optimizer_step,
        )
        if self.emitted or diagnostic.optimizer_step != self.phase.triggered_at_step:
            return NoPhaseTransition(
                next_state=active,
                reason=NoPhaseTransitionReason.INSUFFICIENT_RESIDUAL_WINDOW,
            )
        window = (
            _diagnostic(1, self.phase.library_version),
            _diagnostic(2, self.phase.library_version),
        )
        return PhaseTransitionDetected(
            next_state=active,
            event=self.phase,
            triggering_window=window,
        )

    def close_no_op(self, phase: PhaseTransitionEvent, reason: str) -> None:
        from skillev.evolution.detector import close_no_op_segment

        self.state = close_no_op_segment(self.state, phase, reason)

    def commit_observation(self, observation: object) -> None:
        if not isinstance(observation, NoPhaseTransition | PhaseTransitionDetected):
            raise TypeError("unexpected detector observation")
        self.state = observation.next_state
        self.observed_steps.append(observation.next_state.cursor)
        if isinstance(observation, PhaseTransitionDetected):
            self.emitted = True

    def preview_reset(self, *, old_version: str, new_version: str) -> AwaitingDetectorSegment:
        if old_version != self.phase.library_version:
            raise ValueError("detector reset old version differs")
        return AwaitingDetectorSegment(new_version)

    def commit_reset(self, state: AwaitingDetectorSegment) -> None:
        self.state = state
        self.reset_versions.append(state.expected_library_version)


@dataclass(frozen=True, slots=True)
class _Policy:
    decision: FullEvolutionDecision

    def decide_from_views(self, **_: object) -> FullEvolutionDecision:
        return self.decision


@dataclass(slots=True)
class _SnapshotArtifacts:
    root: Path
    names: list[str] = field(default_factory=list)
    fail_on_name: str | None = None
    retention_calls: list[int] = field(default_factory=list)

    def save_as(self, snapshot: object, *, name: str) -> Path:
        del snapshot
        if name == self.fail_on_name:
            raise OSError("injected checkpoint failure")
        self.names.append(name)
        path = self.root / name
        path.mkdir()
        (path / "runtime_state.json").write_text("{}\n", encoding="utf-8")
        return path

    def load_metadata(self, directory: Path) -> object:
        raise NotImplementedError(directory)

    def retain_recent(self, *, keep_recent: int) -> tuple[Path, ...]:
        self.retention_calls.append(keep_recent)
        return ()


@dataclass(frozen=True, slots=True)
class _SnapshotFactory:
    def snapshot(self) -> object:
        return {"kind": "checkpoint"}


class _LedgerScriptedAuthor:
    def __init__(self, delegate: ScriptedSkillAuthor, ledger: BudgetLedger) -> None:
        self._delegate = delegate
        self._ledger = ledger

    @property
    def tokenizer(self):
        return self._delegate.tokenizer

    @property
    def requests(self):
        return self._delegate.requests

    def author(self, request):
        reservation_id = f"authoring-{len(self.requests) + 1}"
        maximum = BudgetVector(input_tokens=1, output_tokens=1, model_calls=1)
        self._ledger.reserve(
            BudgetReservation(
                reservation_id=reservation_id,
                run_id=self._ledger.run_id,
                attempt_id=self._ledger.attempt_id,
                invocation_id=reservation_id,
                maximum=maximum,
            )
        )
        result = self._delegate.author(request)
        self._ledger.settle(BudgetSettlement(reservation_id, maximum))
        return result


@dataclass(slots=True)
class _Emitter:
    fail_cycle: bool = False
    events: list[tuple[EventType, JsonValue]] = field(default_factory=list)

    def emit(self, event_type: EventType, payload: object) -> None:
        if self.fail_cycle and event_type is EventType.EVOLUTION_CYCLE_COMMITTED:
            raise OSError("injected append failure")
        self.events.append((event_type, payload))  # type: ignore[arg-type]


@dataclass(slots=True)
class _StepPublisher:
    prepared: list[object] = field(default_factory=list)
    committed: list[object] = field(default_factory=list)
    rolled_back: list[object] = field(default_factory=list)

    def restore(self, *, optimizer_step: int, policy_snapshot_id: str) -> AdapterGeneration:
        token = (optimizer_step, policy_snapshot_id)
        self.prepared.append(token)
        self.committed.append(token)
        return AdapterGeneration(
            generation=optimizer_step,
            adapter_name="formal-test-adapter",
            adapter_revision=policy_snapshot_id,
        )

    def prepare(self, *, optimizer_step: int, policy_snapshot_id: str) -> object:
        token = (optimizer_step, policy_snapshot_id)
        self.prepared.append(token)
        return token

    def commit(self, prepared: object) -> object:
        self.committed.append(prepared)
        return prepared

    def rollback(self, prepared: object) -> None:
        self.rolled_back.append(prepared)


def _components(
    tmp_path: Path,
    *,
    author_script: tuple[AuthoringResult | Exception, ...],
    fail_cycle_append: bool = False,
    step_publisher: _StepPublisher | None = None,
    decision_override: FullEvolutionDecision | None = None,
) -> tuple[
    EvolutionLoop,
    SkillLibrary,
    _Training,
    _SnapshotArtifacts,
    _Emitter,
    _LedgerScriptedAuthor,
]:
    seed = make_skill_document("alpha")
    library = SkillLibrary(SkillLibraryState.from_seed_documents((seed,)))
    phase = make_phase_event(library_version=library.current_version)
    exemplar = make_authoring_edge("gap:1")
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
                rationale_text="cover uncovered edge",
                edge_exemplars=(exemplar,),
            ),
        ),
    )
    projection = _Projection(library.current_version)
    training = _Training(projection)
    detector = _Detector(phase)
    artifacts = _SnapshotArtifacts(tmp_path / "snapshots")
    artifacts.root.mkdir()
    emitter = _Emitter(fail_cycle=fail_cycle_append)
    author = _LedgerScriptedAuthor(
        ScriptedSkillAuthor(CharacterTokenizer(), author_script),
        training.ledger,
    )
    loop = EvolutionLoop(
        training_loop=training,
        projections=projection,
        detector=detector,  # type: ignore[arg-type]
        library=library,
        author=author,
        policy=_Policy(decision if decision_override is None else decision_override),
        authority=SkillAuthoringAuthority(
            input_schema_id="input@3",
            output_schema_id="output@3",
            license_id="unit-test",
            allowed_task_families=(TEST_TASK_FAMILY,),
            allowed_tools=(),
        ),
        phi_budget=PhiBudgetAuthority(
            cap=BudgetVector(model_calls=10, input_tokens=100_000, output_tokens=100_000),
        ),
        config=EvolutionConfig(generate_min_absolute_log_importance=0.1),
        emitter=emitter,  # type: ignore[arg-type]
        snapshot_store=RuntimeSnapshotStore(artifacts),
        snapshot_factory=_SnapshotFactory(),  # type: ignore[arg-type]
        run_progress=AttemptRunProgress.fresh(_run_plan()),
        step_adapter_publisher=step_publisher,  # type: ignore[arg-type]
        phase_checkpoint_cycle_ordinals=(1,),
    )
    return loop, library, training, artifacts, emitter, author


def _valid_result() -> AuthoringResult:
    return AuthoringResult(
        drafts=(
            AuthoredSkillDraft(
                title="Gap skill",
                summary="Covers a public gap.",
                instructions="Apply the gap procedure.",
                applicability=SkillApplicability(
                    task_families=(TEST_TASK_FAMILY,),
                    contexts=(TEST_CONTEXT_ID,),
                    required_tools=(),
                    excluded_contexts=(),
                ),
                requirements=(
                    SkillRequirement(
                        requirement_id=uncovered_edge_requirement_id("gap:1"),
                        text="Cover the public gap.",
                        kind="evolvable-strategy",
                    ),
                ),
            ),
        )
    )


def _run_plan() -> ExactAttemptRunPlan:
    """Two search slots expose the fixed phase; one closure slot proves H0 use."""

    return ExactAttemptRunPlan(
        phase_search_steps=2,
        closure_steps=1,
        maximum_cycles=1,
    )


def test_phase_on_last_search_step_completes_before_fixed_closure_training(
    tmp_path: Path,
) -> None:
    loop, library, training, artifacts, emitter, author = _components(
        tmp_path,
        author_script=(_valid_result(),),
    )
    before = library.current_version

    summary = asyncio.run(loop.run(_run_plan()))

    assert summary.cycles_committed_this_attempt == 1
    assert summary.actions_committed_this_attempt == 1
    assert library.current_version != before
    assert training.optimizer_step == 3
    assert summary.completed_training_steps_this_attempt == 3
    assert summary.planned_training_steps_this_attempt == 3
    assert len(training.reset_seeds) == 1
    # A committed phase has its own immutable checkpoint, distinct from the
    # final snapshot taken after the fixed closure step.
    assert artifacts.names == [
        "step-00000001",
        "phase-00000001-step-00000002",
        "step-00000002",
        "step-00000003",
        "final-step-00000003",
    ]
    assert artifacts.retention_calls == [3, 3, 3]
    assert len(author.requests) == 1
    assert tuple(
        item[0] for item in emitter.events if item[0] is not EventType.PHASE_DETECTION_RECORDED
    ) == (
        EventType.EVOLUTION_PHASE_OPENED,
        EventType.EVOLUTION_CYCLE_COMMITTED,
        EventType.PHASE_CHECKPOINT_PUBLISHED,
    )
    checkpoint = PhaseCheckpointPublished.from_value(
        next(
            payload
            for event, payload in emitter.events
            if event is EventType.PHASE_CHECKPOINT_PUBLISHED
        )
    )
    assert emitter.events[-1][0] is EventType.PHASE_DETECTION_RECORDED
    assert emitter.events[-1][1]["reason"] == "slot-does-not-allow-phase-detection"
    assert checkpoint.artifact.library_version == library.current_version
    assert checkpoint.artifact.run_cursor_after.committed_cycles == 1
    assert loop._detector.observed_steps == [1, 2]  # type: ignore[attr-defined]
    assert not hasattr(summary, "pending_phase_event_id")


def test_loop_can_commit_one_non_formal_smoke_step(tmp_path: Path) -> None:
    loop, _, training, artifacts, _, _ = _components(tmp_path, author_script=())

    summary = asyncio.run(loop.run(_run_plan(), maximum_steps_this_attempt=1))

    assert training.optimizer_step == 1
    assert summary.completed_training_steps_this_attempt == 1
    assert summary.planned_training_steps_this_attempt == 1
    assert artifacts.names == ["step-00000001", "final-step-00000001"]


def test_due_checkpoint_is_preserved_as_a_cadence_anchor(tmp_path: Path) -> None:
    loop, _, training, artifacts, _, _ = _components(tmp_path, author_script=())
    training.checkpoint_steps.add(1)

    summary = asyncio.run(loop.run(_run_plan(), maximum_steps_this_attempt=1))

    assert summary.final_optimizer_step == 1
    assert artifacts.names == [
        "step-00000001",
        "cadence-step-00000001",
        "final-step-00000001",
    ]


def test_verified_no_op_closes_phase_without_mutation_or_z_reset(tmp_path: Path) -> None:
    seed = make_skill_document("alpha")
    initial = SkillLibraryState.from_seed_documents((seed,))
    phase = make_phase_event(library_version=initial.current_version)
    no_op = VerifiedNoOpEvolutionDecision(phase.event_id)
    loop, library, training, artifacts, emitter, author = _components(
        tmp_path,
        author_script=(),
        decision_override=no_op,
    )
    before = library.state

    summary = asyncio.run(loop.run(_run_plan()))

    assert summary.cycles_committed_this_attempt == 0
    assert summary.actions_committed_this_attempt == 0
    assert library.state is before
    assert training.reset_seeds == []
    assert author.requests == ()
    assert artifacts.names == [
        "step-00000001",
        "step-00000002",
        "step-00000003",
        "final-step-00000003",
    ]
    assert tuple(
        item[0] for item in emitter.events if item[0] is not EventType.PHASE_DETECTION_RECORDED
    ) == (
        EventType.EVOLUTION_PHASE_OPENED,
        EventType.EVOLUTION_NO_OP_COMMITTED,
    )
    detector_state = loop._detector.runtime_state  # type: ignore[attr-defined]
    assert isinstance(detector_state, ActiveDetectorSegment)
    assert detector_state.phase_already_triggered is True
    from skillev.evolution.detector import PhaseStatus

    assert detector_state.phase_status is PhaseStatus.NO_OP_WAITING
    assert detector_state.no_op_closure.reason == no_op.reason


def test_authoring_failure_raises_in_the_same_call_before_reset_or_library_apply(
    tmp_path: Path,
) -> None:
    loop, library, training, _, _, author = _components(
        tmp_path,
        author_script=(AuthoringFailedError("sealed call failed"),),
    )
    before = library.state

    with pytest.raises(AuthoringFailedError):
        asyncio.run(loop.run(_run_plan()))

    assert library.state is before
    assert training.reset_seeds == []
    assert len(author.requests) == 1
    assert loop._run_progress.state.completed_training_steps == 2  # type: ignore[attr-defined]
    assert loop._run_progress.state.committed_cycles == 0  # type: ignore[attr-defined]


def test_cycle_append_failure_escapes_after_durable_cycle_state(tmp_path: Path) -> None:
    loop, _, _, _, _, author = _components(
        tmp_path,
        author_script=(_valid_result(),),
        fail_cycle_append=True,
    )

    with pytest.raises(EventAppendFailedError):
        asyncio.run(loop.run(_run_plan()))
    assert len(author.requests) == 1
    assert loop._run_progress.state.completed_training_steps == 2  # type: ignore[attr-defined]
    # Source publication follows the durable checkpoint.  A formal journal
    # replays the missing prepared event rather than repeating this mutation.
    assert loop._run_progress.state.committed_cycles == 1  # type: ignore[attr-defined]


def test_checkpoint_failure_keeps_step_unpublished_and_requires_restart(
    tmp_path: Path,
) -> None:
    publisher = _StepPublisher()
    loop, _, training, artifacts, emitter, _ = _components(
        tmp_path,
        author_script=(_valid_result(),),
        step_publisher=publisher,
    )
    artifacts.fail_on_name = "step-00000001"

    with pytest.raises(OSError, match="checkpoint failure"):
        asyncio.run(loop.run(_run_plan()))

    assert training.optimizer_step == 1
    assert emitter.events == []
    assert publisher.prepared == [(0, "policy@0"), (1, "policy@1")]
    assert publisher.committed == [(0, "policy@0")]
    assert publisher.rolled_back == [(1, "policy@1")]
    assert loop._run_progress.state.completed_training_steps == 1  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        asyncio.run(loop.run(_run_plan()))
