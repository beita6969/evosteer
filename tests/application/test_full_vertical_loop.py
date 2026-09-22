from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import torch

from skillev.application import (
    ApplicationConfig,
    SKILLEVApplication,
    TerminalComponents,
)
from skillev.calibration import CalibrationConfig, CalibrationEngine
from skillev.contracts import (
    EvolutionPhaseOpened,
    TrainingStepCommit,
    canonical_json,
)
from skillev.diagnostics import DiagnosticsConfig
from skillev.evolution import (
    AuthoringSamplingConfig,
    EvolutionConfig,
    FullEvolutionPolicy,
    PhaseTransitionDetector,
    PhiBudgetAuthority,
    SkillAuthoringAuthority,
)
from skillev.policy import (
    AuthoringGenerationRequest,
    GenerationResult,
    PolicyGenerationRequest,
    QwenBackboneConfig,
    QwenPolicyBackbone,
)
from skillev.rollout import (
    EnvironmentObservation,
    RolloutTask,
    UnskilledRolloutSessionBundle,
)
from skillev.runtime import (
    ActionKind,
    AdapterGeneration,
    AttemptRunProgress,
    BudgetLedger,
    BudgetVector,
    EventType,
    LiveAttemptEventLog,
    SkillLibrary,
    SkillLibraryState,
    StepTransactionJournal,
    StepTransactionState,
    StructuredAction,
)
from skillev.runtime.event_log_reader import read_event_history
from skillev.training import (
    CheckpointConfig,
    MethodProjectionPipeline,
    OptimizerConfig,
    PolicyRolloutConfig,
    PrivateCheckpointStorageBinding,
    TrainerConfig,
    TrainingExecutionConfig,
    TTBMethodConfig,
    conservative_rollout_maximum,
)
from tests.training.fakes import (
    SCRIPTED_ACTION_TEXT,
    SCRIPTED_REASONING_TEXT,
    FakeISOClock,
    OrderedTaskProvider,
    PrivateEvaluator,
    PublicEnvironment,
)
from tests.v3_helpers import (
    make_public_identity,
    make_run_plan,
    make_skill_document,
)

_EXPECTED_SKILL_MARKER = "EXPECTED_SKILL_ID="


def _skill_action_text(skill_id: str) -> str:
    return canonical_json(
        StructuredAction(
            kind=ActionKind.SKILL,
            name="debug-skill",
            arguments={},
            resource_id="debug.tool",
            skill_id=skill_id,
        ).to_value()
    )


class _ScriptedQwenBackbone(QwenPolicyBackbone):
    fixture_library: SkillLibrary

    def __init__(self, config: QwenBackboneConfig) -> None:
        super().__init__(config)
        self.policy_prompts: list[str] = []
        self.authoring_prompts: list[str] = []

    def generate_policy(self, request: PolicyGenerationRequest) -> GenerationResult:
        prompt = self.tokenizer.decode(request.input_ids)
        self.policy_prompts.append(prompt)
        if prompt.endswith("Reasoning:\n"):
            text = SCRIPTED_REASONING_TEXT
        elif _EXPECTED_SKILL_MARKER in prompt:
            suffix = prompt.rsplit(_EXPECTED_SKILL_MARKER, maxsplit=1)[1]
            skill_id = suffix.split(maxsplit=1)[0].rstrip(".\n")
            text = _skill_action_text(skill_id)
        elif "CURRENT_ACTIVE_SKILL_INDEX=" in prompt:
            index = int(prompt.rsplit("CURRENT_ACTIVE_SKILL_INDEX=", 1)[1].split(".", 1)[0])
            text = _skill_action_text(self.fixture_library.active_skill_ids[index])
        else:
            text = SCRIPTED_ACTION_TEXT
        token_ids = tuple(self.tokenizer.encode(text))
        if len(token_ids) > request.max_new_tokens:
            raise RuntimeError("test policy script exceeds its declared generation budget")
        return GenerationResult(
            content_token_ids=token_ids,
            stop_token_ids=(),
            finish_reason="length",
        )

    def generate_base(self, request: AuthoringGenerationRequest) -> GenerationResult:
        prompt = self.tokenizer.decode(request.input_ids)
        self.authoring_prompts.append(prompt)
        marker = "Use only this public model-visible authoring material:\n"
        start = prompt.index(marker) + len(marker)
        json_start = prompt.index("{", start)
        payload, _ = json.JSONDecoder().raw_decode(prompt[json_start:])
        constraints = payload["output_contract"]["action_constraints"]
        applicability = constraints["applicability_by_draft"][0]
        requirement_ids = constraints["required_requirement_ids_by_draft"][0]
        preserved = {
            item["requirement_id"]: item
            for item in constraints.get("immutable_source_requirements", ())
        }
        preserved.update(
            {
                item["requirement_id"]: item
                for item in constraints.get("evolvable_source_strategies", ())
            }
        )
        requirement_ids = list(dict.fromkeys([*preserved, *requirement_ids]))
        text = canonical_json(
            {
                "drafts": [
                    {
                        "applicability": applicability,
                        "instructions": (
                            f"Apply public evidence procedure {len(self.authoring_prompts)}."
                        ),
                        "requirements": [
                            preserved.get(
                                requirement_id,
                                {
                                    "requirement_id": requirement_id,
                                    "text": "Cover one public uncovered trajectory edge.",
                                    "kind": "evolvable-strategy",
                                },
                            )
                            for requirement_id in requirement_ids
                        ],
                        "summary": "A generated public procedure.",
                        "title": "Generated procedure",
                    }
                ]
            }
        )
        token_ids = tuple(self.tokenizer.encode(text))
        if len(token_ids) > request.max_new_tokens:
            raise RuntimeError("test authoring script exceeds its declared generation budget")
        return GenerationResult(
            content_token_ids=token_ids,
            stop_token_ids=(),
            finish_reason="length",
        )


@dataclass(slots=True)
class _BaseSessionFactory:
    rewards: tuple[float, ...]
    cursor: int = 0

    def create(self, task: RolloutTask) -> UnskilledRolloutSessionBundle:
        reward = self.rewards[self.cursor]
        self.cursor += 1
        return UnskilledRolloutSessionBundle(
            environment=_SkillCreditEnvironment(task.environment_id, task.task_family),
            evaluator=PrivateEvaluator(task.task_id, task.environment_id, reward),
        )


@dataclass(slots=True)
class _StepPublisher:
    prepared: list[object]
    committed: list[object]
    rolled_back: list[object]

    def restore(self, *, optimizer_step: int, policy_snapshot_id: str) -> AdapterGeneration:
        token = (optimizer_step, policy_snapshot_id)
        self.prepared.append(token)
        self.committed.append(token)
        return AdapterGeneration(
            generation=optimizer_step,
            adapter_name="formal-test-adapter",
            adapter_revision=f"formal-test-step-{optimizer_step}",
        )

    def prepare(self, *, optimizer_step: int, policy_snapshot_id: str) -> object:
        token = (optimizer_step, policy_snapshot_id)
        self.prepared.append(token)
        return token

    def commit(self, prepared: object) -> object:
        self.committed.append(prepared)
        step, _ = prepared  # type: ignore[misc]
        return AdapterGeneration(
            generation=int(step),
            adapter_name="formal-test-adapter",
            adapter_revision=f"formal-test-step-{step}",
        )

    def rollback(self, prepared: object) -> None:
        self.rolled_back.append(prepared)


@dataclass(slots=True)
class _SkillCreditEnvironment(PublicEnvironment):
    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        del step_index
        self.calls += 1
        invoked = (action.skill_id,) if action.kind is ActionKind.SKILL else ()
        return EnvironmentObservation(
            public_value={"status": "ok"},
            observation_status="success",
            invoked_skill_ids=invoked,
            budget_usage=BudgetVector(tool_calls=1),
        )


def _phase_tasks(*, first_skill_id: str, second_skill_id: str) -> tuple[RolloutTask, ...]:
    selected = (first_skill_id, second_skill_id, first_skill_id, first_skill_id, None, None)
    return tuple(
        RolloutTask(
            task_id=f"real-detector-task-{index:02d}",
            environment_id=f"real-detector-environment-{index:02d}",
            task_family="debug-family",
            context_id="debug-family",
            query=(
                f"Invoke the retrieved skill. {_EXPECTED_SKILL_MARKER}{skill_id}."
                if skill_id is not None
                else "Complete the public closure task."
            ),
            available_tools=(),
            public_context={"position": index},
        )
        for index, skill_id in enumerate(selected, start=1)
    )


def _optimizer_steps(
    optimizer: torch.optim.Optimizer,
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[int, ...]:
    return tuple(int(optimizer.state[parameter]["step"].item()) for parameter in parameters)


def build_application_fixture(
    tmp_path: Path,
    training_backbone_config: QwenBackboneConfig,
    *,
    backbone=None,
    batch_size: int = 2,
    cycles: int = 1,
) -> SimpleNamespace:
    if batch_size < 2 or batch_size % 2:
        raise ValueError("vertical fixture requires an even batch population")
    if backbone is None:
        backbone = _ScriptedQwenBackbone(training_backbone_config)
    initial_policy_directory = tmp_path / "initial-policy"
    backbone.save_checkpoint(str(initial_policy_directory))
    initial_trainable_state = backbone.trainable_state_identity
    first_seed = make_skill_document(
        "first-seed",
        task_family="debug-family",
        context_id="debug-family",
        instructions="Use the first complete seed procedure on public debug tasks.",
    )
    second_seed = make_skill_document(
        "second-seed",
        task_family="debug-family",
        context_id="debug-family",
        instructions="Use the second complete seed procedure on public debug tasks.",
    )
    initial_skill_ids = (first_seed.manifest.skill_id, second_seed.manifest.skill_id)
    tasks = _phase_tasks(
        first_skill_id=first_seed.manifest.skill_id,
        second_skill_id=second_seed.manifest.skill_id,
    )
    if cycles > 1:
        tasks = (
            tuple(
                replace(
                    tasks[index],
                    task_id=f"cycle-{cycle}-task-{index}",
                    query=f"Use the current skill. CURRENT_ACTIVE_SKILL_INDEX={skill_index}.",
                )
                for cycle in range(cycles)
                for index, skill_index in enumerate((0, 1, 0, 0))
            )
            + tasks[-2:]
        )
    if batch_size != 2:
        tasks = tuple(
            replace(task, task_id=f"{task.task_id}-rollout-{repeat}")
            for task in tasks
            for repeat in range(batch_size // 2)
        )
    rewards = tuple(
        reward
        for reward in (1.0, 1.0, 0.0, 0.0) * cycles + (1.0, 1.0)
        for _ in range(batch_size // 2)
    )
    library = SkillLibrary(SkillLibraryState.from_seed_documents((first_seed, second_seed)))
    if cycles > 1:
        backbone.fixture_library = library
    diagnostics_config = DiagnosticsConfig(window_size=1, stagnation_rho=0.999999)
    projections = MethodProjectionPipeline.fresh(
        diagnostics_config=diagnostics_config,
        calibration=CalibrationEngine(CalibrationConfig()),
        library_version=library.current_version,
    )
    evolution_config = EvolutionConfig(
        generate_min_absolute_log_importance=0.1,
        entropy_window=1,
        required_consecutive_drops=1,
        high_flow_quantile=1.0,
        low_flow_quantile=0.0,
        lcb_high=1.0,
        lcb_low=1.0,
        ucb_low=0.0,
        max_skill_instruction_tokens_per_draft=1024,
        max_authoring_completion_tokens=4096,
    )
    detector = PhaseTransitionDetector.fresh(
        evolution_config=evolution_config,
        diagnostics_config=diagnostics_config,
        library_version=library.current_version,
    )
    rollout_maximum = conservative_rollout_maximum(
        max_turns=1,
        max_reasoning_tokens=128,
        max_action_tokens=128,
        max_model_input_tokens=4096,
        max_tool_wall_time_milliseconds=1000,
    )
    trainer_config = TrainerConfig(
        method=TTBMethodConfig(epsilon_min=0.01, temperature_beta=1.0),
        rollout=PolicyRolloutConfig(
            base_seed=20260725,
            max_turns=1,
            max_reasoning_tokens=128,
            max_action_tokens=128,
            per_rollout_maximum=rollout_maximum,
        ),
        optimizer=OptimizerConfig(
            adapter_learning_rate=1e-3,
            z_learning_rate=2e-3,
            weight_decay=0.0,
        ),
        execution=TrainingExecutionConfig(
            experiment_id="v3-application",
            batch_size=batch_size,
        ),
        checkpoint=CheckpointConfig(
            every_n_steps=100,
        ),
    )
    application_config = ApplicationConfig(
        trainer=trainer_config,
        diagnostics=diagnostics_config,
        calibration=CalibrationConfig(),
        evolution=evolution_config,
        authoring_sampling=AuthoringSamplingConfig(temperature=1.0, top_p=1.0),
        maximum_h0_tokens=8192,
    )
    event_log = LiveAttemptEventLog(
        tmp_path / "events.jsonl",
        run_id="v3-application",
        attempt_id="attempt-1",
    )
    phi_maximum = BudgetVector(
        input_tokens=2 * evolution_config.max_authoring_prompt_tokens,
        output_tokens=2 * evolution_config.max_authoring_completion_tokens,
        model_calls=2,
    )
    run_plan = make_run_plan(phase_search_steps=2 * cycles, closure_steps=1, maximum_cycles=cycles)
    attempt_budget = rollout_maximum.scale(
        run_plan.total_training_steps * trainer_config.execution.batch_size
    ).add(phi_maximum.scale(cycles))
    authority = SkillAuthoringAuthority(
        input_schema_id="debug-input@3",
        output_schema_id="debug-output@3",
        license_id="unit-test",
        allowed_task_families=("debug-family",),
        allowed_tools=(),
    )
    public_identity = make_public_identity(
        config=application_config,
        run_plan=run_plan,
        seed_documents=(first_seed, second_seed),
        task_ids=tuple(task.task_id for task in tasks),
        attempt_budget=attempt_budget,
        phi_per_cycle_maximum=phi_maximum,
        authoring_authority=authority,
    )
    step_publisher = _StepPublisher([], [], [])
    application = SKILLEVApplication.build_from_components(
        backbone=backbone,
        task_provider=OrderedTaskProvider(tasks),
        base_session_factory=_BaseSessionFactory(rewards),
        terminal_components=TerminalComponents(
            ledger=BudgetLedger(
                run_id="v3-application",
                attempt_id="attempt-1",
                cap=attempt_budget,
            ),
            authoring_authority=authority,
            phi_budget=PhiBudgetAuthority(cap=phi_maximum),
        ),
        checkpoint_storage=PrivateCheckpointStorageBinding(directory=str(tmp_path / "snapshots")),
        public_identity=public_identity,
        event_log=event_log,
        clock=FakeISOClock(),
        library=library,
        projections=projections,
        detector=detector,
        policy=FullEvolutionPolicy(),
        run_progress=AttemptRunProgress.from_state(
            run_plan,
            public_identity.initial_run_cursor,
        ),
        step_adapter_publisher_factory=lambda _backbone: step_publisher,  # type: ignore[arg-type]
    )

    transaction_journal = application.evolution_loop.step_transaction_journal
    assert isinstance(transaction_journal, StepTransactionJournal)
    return SimpleNamespace(
        application=application,
        run_plan=run_plan,
        publisher=step_publisher,
        journal=transaction_journal,
        event_log=event_log,
        initial_skill_ids=initial_skill_ids,
        tasks=tasks,
        rewards=rewards,
        public_identity=public_identity,
        authority=authority,
        attempt_budget=attempt_budget,
        phi_maximum=phi_maximum,
        initial_policy_directory=initial_policy_directory,
        initial_trainable_state=initial_trainable_state,
    )


def test_application_runs_training_phase_complete_phi_and_new_library_training(
    tmp_path: Path,
    training_backbone_config: QwenBackboneConfig,
) -> None:
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    application, run_plan = fixture.application, fixture.run_plan
    backbone, projections = application.training_loop.backbone, application.projections
    step_publisher, transaction_journal = fixture.publisher, fixture.journal
    event_log, initial_skill_ids = fixture.event_log, fixture.initial_skill_ids

    summary = asyncio.run(application.evolution_loop.run(run_plan))
    groups = backbone.parameter_groups()
    adapter_steps_before = _optimizer_steps(
        application.training_loop.optimizer,
        (*groups.forward, *groups.backward),
    )
    assert summary.cycles_committed_this_attempt == 1
    assert summary.actions_committed_this_attempt == 1
    # The frozen closure slot is ordinary post-mutation training, so the
    # forward/backward Adam moments have advanced through all three steps.
    assert set(adapter_steps_before) == {3}
    # Z was reset at the cycle boundary, then advanced only in the one
    # required closure-training step under the new library.
    assert set(_optimizer_steps(application.training_loop.optimizer, groups.z_head)) == {1}

    source_events = tuple(
        event
        for event in read_event_history(event_log.path)
        if event.event_type
        in {
            EventType.LIBRARY_INITIALIZED,
            EventType.TRAINING_STEP_COMMITTED,
            EventType.EVOLUTION_PHASE_OPENED,
            EventType.EVOLUTION_CYCLE_COMMITTED,
        }
    )
    assert tuple(event.event_type for event in source_events) == (
        EventType.LIBRARY_INITIALIZED,
        EventType.TRAINING_STEP_COMMITTED,
        EventType.TRAINING_STEP_COMMITTED,
        EventType.EVOLUTION_PHASE_OPENED,
        EventType.EVOLUTION_CYCLE_COMMITTED,
        EventType.TRAINING_STEP_COMMITTED,
    )
    assert summary.cycles_committed_in_run == 1
    assert summary.completed_training_steps_this_attempt == run_plan.total_training_steps
    assert len(step_publisher.prepared) == run_plan.total_training_steps + 1
    assert step_publisher.prepared[0][0] == 0
    assert step_publisher.committed == step_publisher.prepared
    assert step_publisher.rolled_back == []
    transactions = tuple(
        transaction_journal.load(step) for step in range(1, run_plan.total_training_steps + 1)
    )
    assert all(item.state is StepTransactionState.COMMITTED for item in transactions)
    assert tuple(item.checkpoint_name for item in transactions) == (
        "step-00000001",
        "step-00000002",
        "step-00000003",
    )

    new_skill_ids = set(application.library.active_skill_ids) - set(initial_skill_ids)
    assert len(new_skill_ids) == 1
    new_skill_id = new_skill_ids.pop()
    new_document = application.library.document(new_skill_id)
    training_commits = tuple(
        TrainingStepCommit.from_value(event.payload)
        for event in source_events
        if event.event_type is EventType.TRAINING_STEP_COMMITTED
    )
    closure_records = training_commits[-1].records
    assert len(closure_records) == 2
    assert all(
        new_skill_id in record.initial_context.retrieved_skill_ids for record in closure_records
    )
    assert new_document.instructions in backbone.policy_prompts[-2]
    retired = set(initial_skill_ids) - set(application.library.active_skill_ids)
    for record in closure_records:
        assert record.initial_context.meta["library_version"] == application.library.current_version
        assert set(record.initial_context.active_skill_ids) == set(
            application.library.active_skill_ids
        )
        assert retired.isdisjoint(record.initial_context.retrieved_skill_ids)
    assert all(application.library.document(skill_id) for skill_id in retired)

    opened_event = next(
        event for event in source_events if event.event_type is EventType.EVOLUTION_PHASE_OPENED
    )
    phase = EvolutionPhaseOpened.from_value(opened_event.payload).phase_event
    assert phase.triggered_at_step == 2
    assert phase.residual_condition_met is True
    assert phase.entropy_condition_met is True
    assert tuple(item.distinct_skills for item in phase.entropy_series) == (2, 1)
    assert tuple(item.total_invocations for item in phase.entropy_series) == (2, 2)
    assert phase.relative_improvement < phase.rho

    assert set(
        _optimizer_steps(
            application.training_loop.optimizer,
            (*groups.forward, *groups.backward),
        )
    ) == {3}
    assert projections.latest_diagnostic.library_version == application.library.current_version
    assert all(cell.skill_id != new_skill_id for cell in projections.calibration_cells)
    assert len(backbone.authoring_prompts) == 1
