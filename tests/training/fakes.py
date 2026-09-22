"""Deterministic public-only collaborators for the real Protocol-v3 loop."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from skillev.calibration import CalibrationConfig, CalibrationEngine
from skillev.contracts import JsonValue, SuccessRule, TerminalReward, canonical_json, stable_hash
from skillev.diagnostics import DiagnosticsConfig, OnlineFlowDiagnostics
from skillev.policy import AdapterRole, PolicyBackbone
from skillev.rollout import (
    CanonicalInitialContextAssembler,
    EnvironmentObservation,
    GenerationPhase,
    NoTerminalSubmission,
    PolicySnapshot,
    PolicySnapshotMismatchError,
    RolloutGenerationRequest,
    RolloutGenerationResult,
    RolloutSessionBundle,
    RolloutTask,
    RolloutTokenizerProtocol,
    TerminalEvaluationRequest,
)
from skillev.runtime import (
    BudgetLedger,
    BudgetVector,
    LiveAttemptEventLog,
    OrderedTaskCursorState,
    RuntimeEventEmitter,
    SkillLibrary,
    SkillLibraryState,
    StructuredAction,
)
from skillev.training import (
    CheckpointConfig,
    FilesystemTrainingCheckpointStore,
    MethodProjectionPipeline,
    OptimizerConfig,
    PolicyRolloutConfig,
    PrivateCheckpointStorageBinding,
    TrainerConfig,
    TrainingExecutionConfig,
    TrainingLoop,
    TTBMethodConfig,
    conservative_rollout_maximum,
)

SCRIPTED_REASONING_TEXT = "reason about the public task"
SCRIPTED_ANSWER_TEXT = "public answer"
SCRIPTED_ACTION_TEXT = canonical_json(
    {
        "arguments": {"value": {"answer": SCRIPTED_ANSWER_TEXT}},
        "kind": "complete",
        "name": "complete",
        "resource_id": None,
        "skill_id": None,
    }
)


def tiny_tokenizer_corpus() -> tuple[str, ...]:
    return (
        SCRIPTED_REASONING_TEXT,
        SCRIPTED_ACTION_TEXT,
        "### Query\npublic task\n### Retrieved Skills\n(none)\n",
        "### Step 1\nReasoning:\nAction:\nObservation:\n",
    )


@dataclass(slots=True)
class FakeISOClock:
    current: datetime = field(default_factory=lambda: datetime(2026, 7, 25, 12, 0, tzinfo=UTC))
    tick: timedelta = field(default_factory=lambda: timedelta(microseconds=1))

    def __call__(self) -> str:
        value = self.current.isoformat(timespec="microseconds").replace("+00:00", "Z")
        self.current += self.tick
        return value


@dataclass(slots=True)
class OrderedTaskProvider:
    tasks: tuple[RolloutTask, ...]
    cursor: int = 0

    def next_task(self) -> RolloutTask:
        task = self.tasks[self.cursor]
        self.cursor += 1
        return task

    @property
    def runtime_state(self) -> OrderedTaskCursorState:
        return OrderedTaskCursorState(
            curriculum_id="test-public-tasks@3",
            cursor=self.cursor,
        )


def make_public_tasks(count: int) -> tuple[RolloutTask, ...]:
    return tuple(
        RolloutTask(
            task_id=f"task-{index:04d}",
            environment_id=f"environment-{index:04d}",
            task_family="debug-family",
            context_id="debug-family",
            query=f"Answer public task {index}.",
            available_tools=(),
            public_context={"index": index},
        )
        for index in range(1, count + 1)
    )


@dataclass(slots=True)
class DynamicScriptedRolloutGenerator:
    backbone: PolicyBackbone
    fail_on_call: int | None = None
    action_text: str = SCRIPTED_ACTION_TEXT
    calls: list[RolloutGenerationRequest] = field(default_factory=list)

    @property
    def tokenizer(self) -> RolloutTokenizerProtocol:
        return cast(RolloutTokenizerProtocol, self.backbone.tokenizer)

    def snapshot(self) -> PolicySnapshot:
        return PolicySnapshot.create(
            backbone_id=self.backbone.backbone_id,
            forward_adapter_version=self.backbone.adapter_version(AdapterRole.FORWARD_POLICY),
            tokenizer_id=self.tokenizer.tokenizer_id,
            backend_id="scripted-v3",
            initial_trainable_state_hash=self.backbone.initial_trainable_state_hash,
        )

    def begin_episode(self, episode_id: str, expected_policy_snapshot_id: str) -> None:
        del episode_id, expected_policy_snapshot_id

    def end_episode(self, episode_id: str) -> None:
        del episode_id

    async def generate(
        self,
        request: RolloutGenerationRequest,
    ) -> RolloutGenerationResult:
        self.calls.append(request)
        if self.fail_on_call == len(self.calls):
            raise PolicySnapshotMismatchError("scripted infrastructure failure")
        snapshot = self.snapshot()
        if request.expected_policy_snapshot_id != snapshot.snapshot_id:
            raise PolicySnapshotMismatchError("stale scripted snapshot")
        text = (
            SCRIPTED_REASONING_TEXT
            if request.phase is GenerationPhase.REASONING
            else self.action_text
        )
        token_ids = tuple(self.tokenizer.encode(text))
        return RolloutGenerationResult(
            content_token_ids=token_ids,
            stop_token_ids=(),
            finish_reason="length",
            policy_snapshot_id=snapshot.snapshot_id,
            backend_id=snapshot.backend_id,
            usage=BudgetVector(
                input_tokens=len(request.input_ids),
                output_tokens=len(token_ids),
                model_calls=1,
            ),
        )


@dataclass(slots=True)
class PublicEnvironment:
    environment_id: str
    task_family: str
    calls: int = 0

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        del action, step_index
        self.calls += 1
        return EnvironmentObservation(
            public_value={"status": "ok"},
            observation_status="success",
            budget_usage=BudgetVector(tool_calls=1),
        )

    def validate_completion(self, submission: JsonValue) -> bool:
        return (
            isinstance(submission, dict)
            and set(submission) == {"answer"}
            and isinstance(submission["answer"], str)
        )


@dataclass(slots=True)
class PrivateEvaluator:
    task_id: str
    environment_id: str
    reward: float

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.task_id:
            raise RuntimeError("wrong evaluator")
        reward = 0.0 if isinstance(request.evaluation_input, NoTerminalSubmission) else self.reward
        return TerminalReward(
            value=reward,
            success=reward >= 0.5,
            success_rule=SuccessRule.R_AT_THRESHOLD,
            success_threshold=0.5,
            native_metric_name="debug-score",
            native_payload={"task": self.task_id},
            environment_id=self.environment_id,
            verifier_version="debug-verifier@3",
        )


@dataclass(slots=True)
class OrderedSessionFactory:
    rewards: tuple[float, ...]
    cursor: int = 0
    cleanup_count: int = 0

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        reward = self.rewards[self.cursor]
        self.cursor += 1

        async def cleanup() -> None:
            self.cleanup_count += 1

        return RolloutSessionBundle(
            environment=PublicEnvironment(task.environment_id, task.task_family),
            evaluator=PrivateEvaluator(task.task_id, task.environment_id, reward),
            retrieved_skills=(),
            cleanup=cleanup,
        )


@dataclass(slots=True)
class TrainingHarness:
    loop: TrainingLoop
    backbone: PolicyBackbone
    generator: DynamicScriptedRolloutGenerator
    task_provider: OrderedTaskProvider
    session_factory: OrderedSessionFactory
    config: TrainerConfig
    library: SkillLibrary
    event_log: LiveAttemptEventLog
    projections: MethodProjectionPipeline
    checkpoint_store: FilesystemTrainingCheckpointStore


def build_training_harness(
    root: Path,
    *,
    backbone: PolicyBackbone,
    raw_rewards: Sequence[float] | None = None,
    tasks: Iterable[RolloutTask] | None = None,
    config: TrainerConfig | None = None,
    fail_on_generation_call: int | None = None,
    action_text: str = SCRIPTED_ACTION_TEXT,
) -> TrainingHarness:
    root.mkdir(parents=True, exist_ok=True)
    actual_config = config or TrainerConfig(
        method=TTBMethodConfig(epsilon_min=0.01, temperature_beta=1.0),
        rollout=PolicyRolloutConfig(
            base_seed=20260725,
            max_turns=1,
            max_reasoning_tokens=64,
            max_action_tokens=64,
            per_rollout_maximum=conservative_rollout_maximum(
                max_turns=1,
                max_reasoning_tokens=64,
                max_action_tokens=64,
                max_model_input_tokens=2048,
                max_tool_wall_time_milliseconds=1000,
            ),
        ),
        optimizer=OptimizerConfig(
            adapter_learning_rate=1e-3,
            z_learning_rate=2e-3,
            weight_decay=0.0,
        ),
        execution=TrainingExecutionConfig(
            experiment_id="v3-training-test",
            batch_size=2,
        ),
        checkpoint=CheckpointConfig(
            every_n_steps=100,
        ),
    )
    capacity = actual_config.execution.batch_size * 4
    task_provider = OrderedTaskProvider(
        tuple(tasks) if tasks is not None else make_public_tasks(capacity)
    )
    rewards = (
        tuple(float(item) for item in raw_rewards)
        if raw_rewards is not None
        else tuple(0.0 if index % 2 == 0 else 1.0 for index in range(capacity))
    )
    session_factory = OrderedSessionFactory(rewards)
    generator = DynamicScriptedRolloutGenerator(
        backbone,
        fail_on_call=fail_on_generation_call,
        action_text=action_text,
    )
    library = SkillLibrary(SkillLibraryState.from_seed_documents(()))
    diagnostics = OnlineFlowDiagnostics.fresh(
        DiagnosticsConfig(window_size=2),
        library_version=library.current_version,
    )
    calibration = CalibrationEngine(CalibrationConfig())
    projections = MethodProjectionPipeline.from_fresh_components(
        diagnostics=diagnostics,
        calibration=calibration,
    )
    clock = FakeISOClock()
    event_log = LiveAttemptEventLog(
        root / "events.jsonl",
        run_id=actual_config.execution.experiment_id,
        attempt_id="attempt-1",
    )
    emitter = RuntimeEventEmitter(
        log=event_log,
        producer_id="training-test",
        clock=clock,
    )
    ledger = BudgetLedger(
        run_id=actual_config.execution.experiment_id,
        attempt_id="attempt-1",
        cap=actual_config.rollout.per_rollout_maximum.scale(capacity),
    )
    checkpoint_store = FilesystemTrainingCheckpointStore(
        root=PrivateCheckpointStorageBinding(directory=str(root / "checkpoints")).directory,
    )
    loop = TrainingLoop(
        backbone=backbone,
        generator=generator,
        task_provider=task_provider,
        session_factory=session_factory,
        context_assembler=CanonicalInitialContextAssembler(
            maximum_h0_tokens=4096, input_window=actual_config.rollout.input_window
        ),
        library=library,
        config=actual_config,
        ledger=ledger,
        emitter=emitter,
        clock=clock,
        projections=projections,
        checkpoint_store=checkpoint_store,
        sampling_schedule_hash=stable_hash({"sampling": actual_config.rollout.base_seed}),
        ordered_task_sequence_hash=stable_hash({"tasks": "training-harness"}),
    )
    return TrainingHarness(
        loop=loop,
        backbone=backbone,
        generator=generator,
        task_provider=task_provider,
        session_factory=session_factory,
        config=actual_config,
        library=library,
        event_log=event_log,
        projections=projections,
        checkpoint_store=checkpoint_store,
    )
