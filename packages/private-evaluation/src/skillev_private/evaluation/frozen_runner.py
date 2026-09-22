"""Read-only frozen inference over the fixed official benchmark graph."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from skillev.contracts import ScientificSamplingCoordinate
from skillev.experiments import AttemptPurpose, PublishedAttemptIdentity, SchedulePurpose
from skillev.policy import PrivateInitialCheckpointBinding, QwenDeploymentConfig
from skillev.rollout import DecodingSnapshot, RolloutTask
from skillev.rollout.prompt_profiles import InitialContextProfile
from skillev.training import TrainerConfig
from skillev_private.benchmarks.catalog import PrivateBenchmarkCatalog
from skillev_private.benchmarks.curriculum import PrivateFrozenTaskSequence
from skillev_private.frozen_state import FrozenInferenceState
from skillev_private.initial_baseline import InitialBaselineInferenceState
from skillev_private.phase_anchor import PhaseAnchorInferenceState

from .result_contracts import (
    EvaluationEpisodeTelemetry,
    PrivateEvaluationEpisodeResult,
    public_native_metric_values,
)

if TYPE_CHECKING:
    from skillev.evolution import TaskConditionedSkillRetriever
    from skillev.runtime import (
        FullRetrievedSkillContext,
        RuntimeEventEmitter,
        SkillLibrary,
        SkillLibraryState,
    )


FrozenEvaluationState: TypeAlias = (
    FrozenInferenceState | InitialBaselineInferenceState | PhaseAnchorInferenceState
)


@dataclass(frozen=True, slots=True)
class FrozenEpisodeRequest:
    """One immutable evaluation episode bound to its state and sequence slot."""

    task: RolloutTask
    state: FrozenEvaluationState
    task_sequence: PrivateFrozenTaskSequence
    sequence_position: int

    def __post_init__(self) -> None:
        if not isinstance(self.task, RolloutTask):
            raise TypeError("frozen episode requires a rollout task")
        if not isinstance(
            self.state,
            FrozenInferenceState | InitialBaselineInferenceState | PhaseAnchorInferenceState,
        ):
            raise TypeError("frozen episode requires an admitted frozen state")
        if not isinstance(self.task_sequence, PrivateFrozenTaskSequence):
            raise TypeError("frozen episode requires PrivateFrozenTaskSequence")
        if self.sequence_position < 0:
            raise ValueError("frozen episode position must be non-negative")


@dataclass(frozen=True, slots=True)
class FrozenEvaluationExecutionConfig:
    """Private deployment controls for one actual no-update evaluation pass.

    These values are reconstructed from the verified formal training input by
    the private worker.  There is intentionally no callable execution seam:
    every episode uses the real Qwen policy, the frozen skill retriever, and
    the official catalog session/evaluator route.
    """

    backbone: QwenDeploymentConfig
    trainer: TrainerConfig
    maximum_h0_tokens: int
    initial_checkpoint: PrivateInitialCheckpointBinding
    source_identity: PublishedAttemptIdentity
    output_directory: Path
    run_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.trainer, TrainerConfig):
            raise TypeError("frozen evaluation requires the source trainer config")
        if type(self.maximum_h0_tokens) is not int or self.maximum_h0_tokens < 1:
            raise ValueError("frozen evaluation maximum_h0_tokens must be positive")
        if not isinstance(self.initial_checkpoint, PrivateInitialCheckpointBinding):
            raise TypeError("frozen evaluation requires the formal initial checkpoint")
        if not isinstance(self.source_identity, PublishedAttemptIdentity):
            raise TypeError("frozen evaluation requires the formal source identity")
        if self.source_identity.purpose is not AttemptPurpose.FORMAL_BENCHMARK_TRAINING:
            raise ValueError("frozen evaluation requires a formal training source")
        if not isinstance(self.output_directory, Path) or not self.output_directory.is_absolute():
            raise ValueError("frozen evaluation output directory must be absolute")
        if type(self.run_id) is not str or not self.run_id.strip():
            raise ValueError("frozen evaluation run_id must be non-empty")


class _OfficialFrozenEpisodeRuntime:
    """The one concrete Qwen + retrieval + official-evaluator execution graph."""

    def __init__(
        self,
        *,
        state: FrozenEvaluationState,
        catalog: PrivateBenchmarkCatalog,
        execution: FrozenEvaluationExecutionConfig,
        task_count: int,
        sequence: PrivateFrozenTaskSequence,
    ) -> None:
        # Keep heavy model imports inside the private execution path; importing
        # result contracts or frozen-input schemas must not load Torch.
        from skillev.evolution import TaskConditionedSkillRetriever
        from skillev.policy.hf_backbone import build_qwen_policy_backbone
        from skillev.rollout import (
            CanonicalInitialContextAssembler,
            LocalPolicyGenerator,
            RolloutEngine,
            RolloutRequest,
            StructuredJsonActionCodec,
        )
        from skillev.runtime import (
            BoundedAgent,
            BoundedAgentPolicy,
            BudgetLedger,
            BudgetVector,
            FlowOnlyRuntimeExecutionState,
            FullRuntimeExecutionState,
            LiveAttemptEventLog,
            RuntimeEventEmitter,
            RuntimeSnapshot,
            SkillLibrary,
        )
        from skillev.runtime.attempt_publication import sha256_file, sha256_tree

        _validate_frozen_source(state, execution)
        if not isinstance(catalog, PrivateBenchmarkCatalog):
            raise TypeError("frozen evaluation requires the official private catalog")
        if type(task_count) is not int or task_count < 1:
            raise ValueError("frozen evaluation requires a non-empty task sequence")

        if sha256_tree(state.policy_snapshot_directory) != state.policy_snapshot_hash:
            raise ValueError("frozen policy snapshot bytes differ from the admitted artifact")
        metadata: RuntimeSnapshot | None = None
        if isinstance(state, FrozenInferenceState | PhaseAnchorInferenceState):
            runtime_state_path = state.policy_snapshot_directory / "runtime_state.json"
            if isinstance(state, FrozenInferenceState):
                expected_runtime_state_hash = state.final_training_artifact.runtime_state_sha256
                expected_optimizer_step = state.final_training_artifact.optimizer_step
            else:
                expected_runtime_state_hash = state.phase_checkpoint_artifact.runtime_state_sha256
                expected_optimizer_step = state.phase_checkpoint_artifact.optimizer_step
            if sha256_file(runtime_state_path) != expected_runtime_state_hash:
                raise ValueError("frozen runtime state differs from the admitted artifact")
            metadata = RuntimeSnapshot.from_value(json.loads(runtime_state_path.read_text("utf-8")))
            if (
                metadata.identity.method_identity_hash != state.method_identity_hash
                or metadata.identity.initial_trainable_state_hash
                != execution.source_identity.initial_trainable_state.content_hash
                or metadata.optimizer_step != expected_optimizer_step
                or metadata.execution_state.library != state.library
            ):
                raise ValueError("frozen runtime metadata differs from its admitted source state")
            if isinstance(metadata.execution_state, FullRuntimeExecutionState):
                if (
                    metadata.execution_state.projections.calibration_cells
                    != state.calibration_cells
                ):
                    raise ValueError(
                        "frozen posterior differs from the final full runtime snapshot"
                    )
            elif isinstance(metadata.execution_state, FlowOnlyRuntimeExecutionState):
                if state.calibration_cells:
                    raise ValueError("flow-only frozen state cannot contain posterior cells")
            else:
                raise TypeError("frozen runtime has an unsupported execution state")
        elif state.calibration_cells:
            raise ValueError("initial baseline cannot contain posterior cells")

        backbone = build_qwen_policy_backbone(execution.backbone)
        backbone.load_checkpoint(str(execution.initial_checkpoint.directory))
        backbone.bind_initial_trainable_state(execution.source_identity.initial_trainable_state)
        if metadata is not None:
            backbone.load_checkpoint(
                str(state.policy_snapshot_directory / metadata.policy_directory)
            )
        generator = LocalPolicyGenerator(backbone)
        snapshot = generator.snapshot()
        if snapshot.snapshot_id != state.policy_snapshot_id:
            raise ValueError("loaded frozen policy differs from the admitted snapshot")

        rollout_config = execution.trainer.rollout
        calls_per_rollout = 2 * rollout_config.max_turns
        input_tokens_per_call = rollout_config.per_rollout_maximum.input_tokens // calls_per_rollout
        self._reasoning_call_maximum = BudgetVector(
            input_tokens=input_tokens_per_call,
            output_tokens=rollout_config.max_reasoning_tokens,
            model_calls=1,
        )
        self._action_call_maximum = BudgetVector(
            input_tokens=input_tokens_per_call,
            output_tokens=rollout_config.max_action_tokens,
            model_calls=1,
            agent_turns=1,
        )
        self._tool_call_maximum = BudgetVector(
            tool_calls=1,
            wall_time_milliseconds=(
                rollout_config.per_rollout_maximum.wall_time_milliseconds
                // rollout_config.max_turns
            ),
        )
        self._generator = generator
        self._library = SkillLibrary(state.library)
        self._retriever = TaskConditionedSkillRetriever(library=self._library)
        self._catalog = catalog
        self._max_turns = rollout_config.max_turns
        self._max_reasoning_tokens = rollout_config.max_reasoning_tokens
        self._max_action_tokens = rollout_config.max_action_tokens
        self._base_seed = rollout_config.base_seed
        self._epsilon_min = execution.trainer.method.epsilon_min
        self._state = state
        self._execution_source_identity = execution.source_identity
        self._ledger = BudgetLedger(
            run_id=execution.run_id,
            attempt_id=_evaluation_attempt_id(state, sequence),
            cap=rollout_config.per_rollout_maximum.scale(task_count),
        )
        event_log = LiveAttemptEventLog(
            execution.output_directory / _evaluation_event_file_name(state, sequence),
            run_id=self._ledger.run_id,
            attempt_id=self._ledger.attempt_id,
        )
        self._emitter: RuntimeEventEmitter = RuntimeEventEmitter(
            event_log,
            producer_id="frozen-official-evaluation",
        )
        self._context_assembler = CanonicalInitialContextAssembler(
            maximum_h0_tokens=execution.maximum_h0_tokens
        )
        self._action_codec = StructuredJsonActionCodec()
        self._rollout_engine_type = RolloutEngine
        self._rollout_request_type = RolloutRequest
        self._bounded_agent_type = BoundedAgent
        self._bounded_agent_policy_type = BoundedAgentPolicy

    async def run_episode(self, request: FrozenEpisodeRequest) -> PrivateEvaluationEpisodeResult:
        expected_snapshot_id = self._state.policy_snapshot_id
        if self._generator.snapshot().snapshot_id != expected_snapshot_id:
            raise RuntimeError("frozen policy changed before an evaluation episode")
        base_session = self._catalog.route((request.task,)).create(request.task)
        profile = request.task.budget_profile
        episode_turns = self._max_turns if profile is None else profile.max_turns
        reasoning_tokens = (
            self._max_reasoning_tokens if profile is None else profile.max_reasoning_tokens
        )
        action_tokens = self._max_action_tokens if profile is None else profile.max_action_tokens
        if (
            episode_turns > self._max_turns
            or reasoning_tokens > self._max_reasoning_tokens
            or action_tokens > self._max_action_tokens
        ):
            raise ValueError("task rollout budget profile exceeds the global evaluation cap")
        episode_decoding = DecodingSnapshot.create(
            max_reasoning_tokens=reasoning_tokens,
            max_action_tokens=action_tokens,
            base_seed=self._base_seed,
        )
        engine = self._rollout_engine_type(
            generator=self._generator,
            context_assembler=self._context_assembler,
            action_codec=self._action_codec,
            bounded_agent=self._bounded_agent_type(
                environment=base_session.environment,
                policy=self._bounded_agent_policy_type(max_turns=episode_turns),
                ledger=self._ledger,
                tool_call_maximum=self._tool_call_maximum,
                emitter=self._emitter,
                action_surface=request.task.action_surface,
            ),
            terminal_evaluator=base_session.evaluator,
            ledger=self._ledger,
            reasoning_call_maximum=self._reasoning_call_maximum,
            action_call_maximum=self._action_call_maximum,
            emitter=self._emitter,
            clock=lambda: "1970-01-01T00:00:00Z",
        )
        usage_before = self._ledger.settled
        library_state_before, retrieved_skills = _capture_frozen_skill_binding(
            library=self._library,
            retriever=self._retriever,
            task=request.task,
        )
        started_monotonic_ns = time.monotonic_ns()
        artifact = await engine.run(
            self._rollout_request_type(
                trajectory_id=_evaluation_trajectory_id(
                    request.state,
                    request.task_sequence,
                    request.sequence_position,
                ),
                task=request.task,
                retrieved_skills=retrieved_skills,
                active_skill_ids=library_state_before.active_skill_ids,
                library_version=library_state_before.current_version,
                sampling_coordinate=_evaluation_sampling_coordinate(
                    sampling_schedule_hash=(self._execution_source_identity.sampling_schedule_hash),
                    sequence=request.task_sequence,
                    sequence_position=request.sequence_position,
                    task_id=request.task.task_id,
                ),
                decoding=episode_decoding,
                epsilon_min=self._epsilon_min,
                condition_id="trained-skillev",
                initial_context_profile=InitialContextProfile.TRAINED_SKILLEV,
            )
        )
        elapsed_wall_time_milliseconds = (time.monotonic_ns() - started_monotonic_ns) // 1_000_000
        episode_usage = self._ledger.settled.subtract(usage_before)
        if artifact.manifest.policy_snapshot.snapshot_id != expected_snapshot_id:
            raise RuntimeError("evaluation artifact used another policy snapshot")
        if artifact.manifest.library_version != self._state.library.current_version:
            raise RuntimeError("evaluation artifact used another skill library")
        if self._generator.snapshot().snapshot_id != expected_snapshot_id:
            raise RuntimeError("frozen policy changed during an evaluation episode")
        _require_frozen_skill_binding_unchanged(
            library=self._library,
            expected_state=library_state_before,
        )
        context = request.task.public_context
        if not isinstance(context, dict) or type(context.get("benchmark_id")) is not str:
            raise ValueError("evaluation task lacks its public benchmark identity")
        benchmark_id = context["benchmark_id"]
        if type(benchmark_id) is not str:
            raise TypeError("evaluation benchmark identity must be text")
        from skillev.experiments import Benchmark

        return PrivateEvaluationEpisodeResult(
            task_id=request.task.task_id,
            benchmark=Benchmark(benchmark_id),
            reward=artifact.record.reward,
            native_metrics=public_native_metric_values(artifact.record.reward),
            telemetry=EvaluationEpisodeTelemetry.from_rollout(
                artifact,
                resource_usage=episode_usage,
                elapsed_wall_time_milliseconds=elapsed_wall_time_milliseconds,
            ),
            frozen_state_hash=request.state.content_hash,
            task_sequence_hash=request.task_sequence.identity.ordered_task_ids_hash,
            sequence_position=request.sequence_position,
            policy_snapshot_id=artifact.manifest.policy_snapshot.snapshot_id,
            library_version=artifact.manifest.library_version,
        )

    def assert_fully_settled(self) -> None:
        self._ledger.assert_fully_settled()


@dataclass(frozen=True, slots=True)
class FrozenEvaluationRunner:
    """Execute the one concrete frozen graph for every ordered private task."""

    state: FrozenEvaluationState
    catalog: PrivateBenchmarkCatalog
    execution: FrozenEvaluationExecutionConfig

    def __post_init__(self) -> None:
        if not isinstance(
            self.state,
            FrozenInferenceState | InitialBaselineInferenceState | PhaseAnchorInferenceState,
        ):
            raise TypeError("frozen runner requires an admitted frozen state")
        if not isinstance(self.catalog, PrivateBenchmarkCatalog):
            raise TypeError("frozen runner requires PrivateBenchmarkCatalog")
        if not isinstance(self.execution, FrozenEvaluationExecutionConfig):
            raise TypeError("frozen runner requires FrozenEvaluationExecutionConfig")
        _validate_frozen_source(self.state, self.execution)

    async def _run_exact(
        self,
        tasks: tuple[RolloutTask, ...],
        sequence: PrivateFrozenTaskSequence,
    ) -> tuple[PrivateEvaluationEpisodeResult, ...]:
        state_hash = self.state.content_hash
        runtime = _OfficialFrozenEpisodeRuntime(
            state=self.state,
            catalog=self.catalog,
            execution=self.execution,
            task_count=len(tasks),
            sequence=sequence,
        )
        results: list[PrivateEvaluationEpisodeResult] = []
        for position, task in enumerate(tasks):
            result = await runtime.run_episode(
                FrozenEpisodeRequest(
                    task=task,
                    state=self.state,
                    task_sequence=sequence,
                    sequence_position=position,
                )
            )
            if not isinstance(result, PrivateEvaluationEpisodeResult):
                raise TypeError("official frozen graph returned an invalid result")
            if result.task_id != task.task_id:
                raise ValueError("frozen episode result belongs to another task")
            context = task.public_context
            if (
                not isinstance(context, dict)
                or context.get("benchmark_id") != result.benchmark.value
            ):
                raise ValueError("frozen episode benchmark differs from task public identity")
            if (
                result.frozen_state_hash != state_hash
                or result.task_sequence_hash != sequence.identity.ordered_task_ids_hash
                or result.sequence_position != position
                or result.policy_snapshot_id != self.state.policy_snapshot_id
                or result.library_version != self.state.library.current_version
            ):
                raise ValueError("frozen episode result differs from its sealed state or position")
            results.append(result)
        runtime.assert_fully_settled()
        if self.state.content_hash != state_hash:
            raise RuntimeError("frozen evaluation mutated policy, library, or posterior state")
        return tuple(results)

    async def run_sequence(
        self,
        sequence: PrivateFrozenTaskSequence,
    ) -> tuple[PrivateEvaluationEpisodeResult, ...]:
        if not isinstance(sequence, PrivateFrozenTaskSequence):
            raise TypeError("frozen runner requires PrivateFrozenTaskSequence")
        if sequence.identity.purpose not in {
            SchedulePurpose.IID_PROGRESS,
            SchedulePurpose.IID_EVALUATION,
            SchedulePurpose.OOD_EVALUATION,
        }:
            raise ValueError("frozen runner cannot consume an optimizer training sequence")
        if isinstance(self.state, InitialBaselineInferenceState) and (
            sequence.identity.purpose is SchedulePurpose.IID_PROGRESS
        ):
            raise ValueError("initial baseline cannot consume a progress-anchor sequence")
        return await self._run_exact(sequence.resolve(self.catalog), sequence)


def _validate_frozen_source(
    state: FrozenEvaluationState,
    execution: FrozenEvaluationExecutionConfig,
) -> None:
    identity = execution.source_identity
    if isinstance(state, FrozenInferenceState | PhaseAnchorInferenceState):
        source_identity_hash = state.source_training_identity_hash
    else:
        source_identity_hash = state.source_identity_hash
    if (
        identity.content_hash != source_identity_hash
        or identity.exact_input_sha256 != state.source_exact_input_sha256
        or identity.method_identity_hash != state.method_identity_hash
        or identity.initial_trainable_state != execution.initial_checkpoint.trainable_state
    ):
        raise ValueError("frozen evaluation deployment differs from its formal source")


def _capture_frozen_skill_binding(
    *,
    library: SkillLibrary,
    retriever: TaskConditionedSkillRetriever,
    task: RolloutTask,
) -> tuple[SkillLibraryState, tuple[FullRetrievedSkillContext, ...]]:
    from skillev.evolution import TaskConditionedSkillRetriever
    from skillev.rollout import RolloutTask
    from skillev.runtime import SkillLibrary

    if not isinstance(library, SkillLibrary):
        raise TypeError("frozen skill binding requires SkillLibrary")
    if not isinstance(retriever, TaskConditionedSkillRetriever):
        raise TypeError("frozen skill binding requires TaskConditionedSkillRetriever")
    if not isinstance(task, RolloutTask):
        raise TypeError("frozen skill binding requires RolloutTask")
    if retriever.library is not library:
        raise ValueError("frozen retriever and library must be the same live instance")
    state = library.state
    retrieved = retriever.retrieve(task)
    if not {item.metadata.skill_id for item in retrieved} <= set(state.active_skill_ids):
        raise RuntimeError("frozen retriever returned an inactive skill")
    return state, retrieved


def _require_frozen_skill_binding_unchanged(
    *,
    library: SkillLibrary,
    expected_state: SkillLibraryState,
) -> None:
    from skillev.runtime import SkillLibrary, SkillLibraryState

    if not isinstance(library, SkillLibrary) or not isinstance(expected_state, SkillLibraryState):
        raise TypeError("frozen skill binding comparison requires library state")
    if library.state != expected_state:
        raise RuntimeError("frozen skill library changed during an evaluation episode")


def _evaluation_attempt_id(
    state: FrozenEvaluationState,
    sequence: PrivateFrozenTaskSequence,
) -> str:
    return (
        "frozen-evaluation-"
        f"{state.content_hash.removeprefix('sha256:')[:16]}-"
        f"{sequence.identity.ordered_task_ids_hash.removeprefix('sha256:')[:16]}"
    )


def _evaluation_sampling_coordinate(
    *,
    sampling_schedule_hash: str,
    sequence: PrivateFrozenTaskSequence,
    sequence_position: int,
    task_id: str,
) -> ScientificSamplingCoordinate:
    return ScientificSamplingCoordinate(
        sampling_schedule_hash=sampling_schedule_hash,
        schedule_purpose=sequence.identity.purpose.value,
        ordered_sequence_hash=sequence.identity.ordered_task_ids_hash,
        sequence_position=sequence_position,
        task_id=task_id,
        optimizer_step_or_anchor_ordinal=0,
    )


def _evaluation_event_file_name(
    state: FrozenEvaluationState,
    sequence: PrivateFrozenTaskSequence,
) -> str:
    return f"{_evaluation_attempt_id(state, sequence)}.jsonl"


def _evaluation_trajectory_id(
    state: FrozenEvaluationState,
    sequence: PrivateFrozenTaskSequence,
    position: int,
) -> str:
    return f"{_evaluation_attempt_id(state, sequence)}-t{position + 1:06d}"


__all__ = [
    "FrozenEpisodeRequest",
    "FrozenEvaluationExecutionConfig",
    "FrozenEvaluationRunner",
    "FrozenEvaluationState",
]
