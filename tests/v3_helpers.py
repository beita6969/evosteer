"""Small deterministic Protocol-v3 records shared by behavior tests."""

from __future__ import annotations

import math
from dataclasses import replace

from skillev.application import ApplicationConfig
from skillev.contracts import (
    EdgeLogprobRecord,
    EntropyObservation,
    FailureMode,
    HorizonBucket,
    InitialContext,
    PhaseTransitionEvent,
    PhaseTriggerRule,
    ScientificSamplingCoordinate,
    SuccessRule,
    TerminalReward,
    TokenBucket,
    TrajectoryRecord,
    TrajectoryResidual,
    TrajectoryStep,
    TTBBatchStats,
    WindowStats,
    build_trajectory_record,
    stable_hash,
)
from skillev.contracts.ttb_training import EdgeScoreContext
from skillev.evolution import (
    AuthoringActionKind,
    AuthoringEdgeEvidence,
    SkillAuthoringAuthority,
)
from skillev.experiments import (
    AttemptPurpose,
    PublishedAttemptIdentity,
    arm_protocol_for_builder_kind,
)
from skillev.policy import (
    PublicTokenizerIdentity,
    PublicTokenizerKind,
    TrainableStateIdentity,
    encode_rollout_prompt,
)
from skillev.rollout import (
    AssembledInitialContext,
    PolicySnapshot,
    RolloutArtifact,
    RolloutManifest,
    RolloutTermination,
)
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptRunCursorState,
    BudgetVector,
    ExactAttemptRunPlan,
    SkillApplicability,
    SkillDocument,
    SkillLibraryState,
    SkillManifest,
    SkillRequirement,
)
from skillev.scoring import (
    assembled_context_hash,
    render_forward_prefix,
    render_hindsight_prefix,
)
from skillev.training import TrainingStepSource

CREATED_AT = "2026-07-25T00:00:00Z"
TEST_TASK_FAMILY = "debug/task-family"
TEST_CONTEXT_ID = "debug:context-id"


class CharacterTokenizer:
    tokenizer_id = "character-tokenizer@v3-tests"
    public_identity = PublicTokenizerIdentity(
        kind=PublicTokenizerKind.QWEN,
        tokenizer_id=tokenizer_id,
        revision="unit-test-character-codepoints",
        content_hash=stable_hash({"tokenizer": tokenizer_id}),
    )

    def __init__(self) -> None:
        self.encode_calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encode_calls.append(text)
        return [ord(character) for character in text]

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(item) for item in token_ids)

    def encode_authoring_prompt(self, text: str) -> list[int]:
        return self.encode(text)


def make_skill_document(
    name: str,
    *,
    task_family: str = TEST_TASK_FAMILY,
    context_id: str = TEST_CONTEXT_ID,
    instructions: str | None = None,
    required_tools: tuple[str, ...] = (),
) -> SkillDocument:
    """Build one content-addressed full skill document."""

    title = f"{name.title()} procedure"
    summary = f"Public summary for {name}."
    body = instructions or f"Apply the complete public {name} procedure."
    applicability = SkillApplicability(
        task_families=(task_family,),
        contexts=(context_id,),
        required_tools=tuple(sorted(required_tools)),
        excluded_contexts=(),
    )
    requirements = (
        SkillRequirement(
            requirement_id=f"requirement-{name}",
            text=f"Complete the {name} procedure.",
        ),
    )
    content = {
        "applicability": applicability.to_value(),
        "instructions": body,
        "requirements": [item.to_value() for item in requirements],
        "summary": summary,
        "title": title,
    }
    content_hash = stable_hash(content)
    return SkillDocument(
        manifest=SkillManifest(
            skill_id=f"skill-{name}",
            version="1",
            content_hash=content_hash,
            input_schema_id="debug-input@3",
            output_schema_id="debug-output@3",
            license_id="unit-test",
            provenance_hash=stable_hash({"source": "unit-test", "name": name}),
        ),
        title=title,
        summary=summary,
        instructions=body,
        applicability=applicability,
        requirements=requirements,
    )


def make_run_plan(
    *,
    phase_search_steps: int = 2,
    closure_steps: int = 1,
    maximum_cycles: int = 1,
) -> ExactAttemptRunPlan:
    return ExactAttemptRunPlan(
        phase_search_steps=phase_search_steps,
        closure_steps=closure_steps,
        maximum_cycles=maximum_cycles,
    )


def make_public_identity(
    *,
    config: ApplicationConfig,
    run_plan: ExactAttemptRunPlan,
    seed_documents: tuple[SkillDocument, ...],
    task_ids: tuple[str, ...],
    attempt_budget: BudgetVector,
    phi_per_cycle_maximum: BudgetVector,
    builder_kind: AttemptBuilderKind = AttemptBuilderKind.FULL,
    authoring_authority: SkillAuthoringAuthority | None = None,
) -> PublishedAttemptIdentity:
    authority = authoring_authority or SkillAuthoringAuthority(
        input_schema_id="debug-input@3",
        output_schema_id="debug-output@3",
        license_id="unit-test",
        allowed_task_families=(TEST_TASK_FAMILY,),
        allowed_tools=(),
    )
    backbone_deployment_hash = stable_hash({"backbone": "unit-test"})
    initial_trainable_state = TrainableStateIdentity.create(
        backbone_deployment_hash=backbone_deployment_hash,
        forward_adapter_hash=stable_hash({"forward": "unit-test"}),
        backward_adapter_hash=stable_hash({"backward": "unit-test"}),
        z_head_hash=stable_hash({"z": "unit-test"}),
    )
    return PublishedAttemptIdentity(
        builder_kind=builder_kind,
        purpose=AttemptPurpose.CORRECTNESS_FIXTURE,
        arm_protocol=arm_protocol_for_builder_kind(builder_kind),
        application_config=config,
        run_plan=run_plan,
        initial_optimizer_step=0,
        initial_run_cursor=AttemptRunCursorState.fresh(run_plan),
        authoring_authority=authority,
        attempt_budget=attempt_budget,
        phi_per_cycle_maximum=phi_per_cycle_maximum,
        protocol_hash=stable_hash({"protocol": "unit-test"}),
        protocol_freeze_id=stable_hash({"freeze": "unit-test"}),
        exact_input_sha256=stable_hash({"input": "unit-test"}),
        backbone_deployment_hash=backbone_deployment_hash,
        initial_trainable_state=initial_trainable_state,
        tokenizer_identity=PublicTokenizerIdentity(
            kind=PublicTokenizerKind.QWEN,
            tokenizer_id="debug-tokenizer@1",
            revision="unit-test",
            content_hash=stable_hash({"tokenizer": "unit-test"}),
        ),
        initial_library_version=SkillLibraryState.from_seed_documents(
            seed_documents
        ).current_version,
        initial_skill_library_state_hash=SkillLibraryState.from_seed_documents(
            seed_documents
        ).state_hash,
        ordered_task_sequence_hash=stable_hash({"task_ids": list(task_ids)}),
    )


def make_phase_event(
    *,
    library_version: str = "library-v1",
    previous_batch_id: str = "batch-1",
    current_batch_id: str = "batch-2",
    step: int = 2,
) -> PhaseTransitionEvent:
    previous = WindowStats(
        start_optimizer_step=step - 1,
        end_optimizer_step=step - 1,
        batch_count=1,
        mean_squared_residual=1.0,
        member_batch_ids=(previous_batch_id,),
    )
    current = WindowStats(
        start_optimizer_step=step,
        end_optimizer_step=step,
        batch_count=1,
        mean_squared_residual=0.99,
        member_batch_ids=(current_batch_id,),
    )
    entropy = (
        EntropyObservation(
            window_end_step=step - 1,
            entropy=0.5,
            total_invocations=2,
            distinct_skills=2,
        ),
        EntropyObservation(
            window_end_step=step,
            entropy=0.0,
            total_invocations=1,
            distinct_skills=1,
        ),
    )
    return PhaseTransitionEvent(
        event_id=stable_hash(
            {
                "library_version": library_version,
                "previous": previous_batch_id,
                "current": current_batch_id,
            }
        ),
        triggered_at_step=step,
        library_version=library_version,
        previous_window=previous,
        current_window=current,
        relative_improvement=0.01,
        rho=0.05,
        residual_condition_met=True,
        entropy_series=entropy,
        trigger_rule=PhaseTriggerRule.RESIDUAL_AND_ENTROPY,
        required_consecutive_drops=1,
        entropy_condition_met=True,
        triggered=True,
    )


def make_authoring_edge(
    edge_id: str = "trajectory-1:1",
    *,
    task_family: str = TEST_TASK_FAMILY,
    context_id: str = TEST_CONTEXT_ID,
    invoked_skill_ids: tuple[str, ...] = (),
    available_tools: tuple[str, ...] = (),
    importance_quantile: float = 1.0,
) -> AuthoringEdgeEvidence:
    return AuthoringEdgeEvidence(
        edge_id=edge_id,
        task_family=task_family,
        context_id=context_id,
        action_kind=AuthoringActionKind.TOOL,
        tool_or_skill_name="debug.tool",
        argument_schema_id="debug-action-schema@1",
        observation_status=FailureMode.SUCCESS,
        token_bucket=TokenBucket.LE_1K,
        horizon_bucket=HorizonBucket.LE_3,
        absolute_log_importance=1.0,
        log_importance_quantile=importance_quantile,
        invoked_skill_ids=tuple(sorted(invoked_skill_ids)),
        available_tools=tuple(sorted(available_tools)),
    )


def make_record(
    trajectory_id: str,
    *,
    skills_by_step: tuple[tuple[str, ...], ...] = (("skill-alpha",),),
    statuses: tuple[str, ...] | None = None,
    reward_success: bool = True,
    reward_value: float | None = None,
    library_version: str = "library-v1",
    task_id: str | None = None,
    task_family: str = TEST_TASK_FAMILY,
    context_id: str = TEST_CONTEXT_ID,
    initial_text: str | None = None,
    tokenizer: CharacterTokenizer | None = None,
) -> tuple[TrajectoryRecord, AssembledInitialContext, CharacterTokenizer]:
    actual_tokenizer = CharacterTokenizer() if tokenizer is None else tokenizer
    actual_task_id = task_id or f"task-{trajectory_id}"
    text = initial_text or f"public query for {trajectory_id}\n"
    actual_statuses = statuses or tuple("success" for _ in skills_by_step)
    if len(actual_statuses) != len(skills_by_step):
        raise ValueError("statuses and skills_by_step must align")

    placeholders: list[TrajectoryStep] = []
    for step_index, (status, skills) in enumerate(
        zip(actual_statuses, skills_by_step, strict=True),
        start=1,
    ):
        if len(skills) > 1:
            raise ValueError("one canonical action may invoke at most one skill")
        if skills:
            action_text = (
                f'{{"arguments":{{}},"kind":"skill","name":"debug-skill",'
                f'"resource_id":"debug.skill","skill_id":"{skills[0]}"}}'
            )
        else:
            action_text = (
                f'{{"arguments":{{"step":{step_index}}},"kind":"tool",'
                '"name":"debug-tool","resource_id":"debug.tool","skill_id":null}'
            )
        token_ids = tuple(actual_tokenizer.encode(action_text))
        placeholders.append(
            TrajectoryStep(
                index=step_index,
                reasoning_text=f"reasoning-{step_index}",
                action_text=action_text,
                action_token_ids=token_ids,
                action_token_count=len(token_ids),
                observation_text=f"observation-{step_index}",
                observation_status=status,
                invoked_skill_ids=skills,
                forward_prefix_hash=stable_hash({"placeholder": "forward", "step": step_index}),
                hindsight_prefix_hash=stable_hash({"placeholder": "hindsight", "step": step_index}),
            )
        )
    placeholder_tuple = tuple(placeholders)
    steps = tuple(
        replace(
            step,
            forward_prefix_hash=render_forward_prefix(
                text,
                placeholder_tuple,
                step.index,
            ).prefix_hash,
            hindsight_prefix_hash=render_hindsight_prefix(
                text,
                placeholder_tuple,
                step.index,
            ).prefix_hash,
        )
        for step in placeholder_tuple
    )
    value = float(reward_success) if reward_value is None else reward_value
    reward = TerminalReward(
        value=value,
        success=reward_success,
        success_rule=SuccessRule.R_EQUALS_ONE,
        success_threshold=None,
        native_metric_name="debug-success",
        native_payload={"success": reward_success},
        environment_id="debug-environment",
        verifier_version="debug-verifier@3",
    )
    initial = InitialContext(
        query=f"query-{trajectory_id}",
        retrieved_skill_ids=tuple(sorted({item for skills in skills_by_step for item in skills})),
        active_skill_ids=tuple(sorted({item for skills in skills_by_step for item in skills})),
        meta={
            "available_tools": [],
            "context_id": context_id,
            "environment_id": "debug-environment",
            "library_version": library_version,
            "task_family": task_family,
            "task_id": actual_task_id,
        },
        assembler_version="test-assembler@3",
        assembled_hash=assembled_context_hash(text),
        assembled_token_count=len(encode_rollout_prompt(actual_tokenizer, text)),
    )
    record = build_trajectory_record(
        tokenizer=actual_tokenizer,
        trajectory_id=trajectory_id,
        environment_id="debug-environment",
        task_family=task_family,
        initial_context=initial,
        steps=steps,
        horizon=len(steps),
        reward=reward,
        shifted_reward=value + 0.01,
        epsilon_min=0.01,
        tokenizer_id=actual_tokenizer.tokenizer_id,
        decoding_snapshot_id="decoding@v3",
        created_at=CREATED_AT,
    )
    return (
        record,
        AssembledInitialContext(text=text, contract=initial),
        actual_tokenizer,
    )


def make_artifact(
    trajectory_id: str,
    *,
    skills_by_step: tuple[tuple[str, ...], ...] = (("skill-alpha",),),
    statuses: tuple[str, ...] | None = None,
    reward_success: bool = True,
    reward_value: float | None = None,
    library_version: str = "library-v1",
) -> RolloutArtifact:
    task_id = f"task-{trajectory_id}"
    record, assembled, tokenizer = make_record(
        trajectory_id,
        skills_by_step=skills_by_step,
        statuses=statuses,
        reward_success=reward_success,
        reward_value=reward_value,
        library_version=library_version,
        task_id=task_id,
    )
    snapshot = PolicySnapshot.create(
        backbone_id="backbone@v3",
        forward_adapter_version="forward@0",
        tokenizer_id=tokenizer.tokenizer_id,
        backend_id="local-v3-test",
        initial_trainable_state_hash="initial-trainable-state-v3-test",
    )
    return RolloutArtifact(
        initial_context=assembled,
        record=record,
        manifest=RolloutManifest(
            trajectory_id=trajectory_id,
            task_id=task_id,
            policy_snapshot=snapshot,
            library_version=library_version,
            sampling_coordinate=ScientificSamplingCoordinate(
                sampling_schedule_hash=stable_hash({"sampling": "v3-helper"}),
                schedule_purpose="unit-test",
                ordered_sequence_hash=stable_hash([task_id]),
                sequence_position=0,
                task_id=task_id,
                optimizer_step_or_anchor_ordinal=0,
            ),
            decoding_snapshot_id=record.decoding_snapshot_id,
            assembler_version=record.initial_context.assembler_version,
            action_format_version="structured-action@1",
            generator_backend_id=snapshot.backend_id,
            termination=RolloutTermination.COMPLETED,
            reasoning_token_counts=tuple(1 for _ in record.steps),
            started_at=CREATED_AT,
            completed_at=CREATED_AT,
        ),
    )


def make_edge(
    trajectory_id: str,
    step_index: int,
    importance: float,
) -> EdgeLogprobRecord:
    backward = -1.0
    forward = backward + importance
    return EdgeLogprobRecord(
        trajectory_id=trajectory_id,
        step_index=step_index,
        forward_logprob_per_token=forward,
        backward_logprob_per_token=backward,
        step_importance=importance,
        forward_adapter_version="forward@0",
        backward_adapter_version="backward@0",
        scoring_stack_id="training-stack",
    )


def make_residual(
    trajectory_id: str,
    *,
    horizon: int,
    delta: float,
    reward_value: float,
    sum_forward: float = 0.0,
    sum_backward: float = 0.0,
) -> TrajectoryResidual:
    log_reward = math.log(reward_value + 0.01)
    return TrajectoryResidual(
        trajectory_id=trajectory_id,
        log_z=delta + log_reward - sum_forward + sum_backward,
        sum_forward=sum_forward,
        sum_backward=sum_backward,
        log_shifted_reward=log_reward,
        raw_reward=reward_value,
        temperature_beta=1.0,
        delta=delta,
        horizon=horizon,
    )


def make_source(
    *,
    batch_id: str = "batch-v3",
    optimizer_step: int = 1,
    library_version: str = "library-v1",
    artifacts: tuple[RolloutArtifact, ...] | None = None,
    importances: tuple[tuple[float, ...], ...] | None = None,
    deltas: tuple[float, ...] | None = None,
) -> TrainingStepSource:
    actual_artifacts = artifacts or (
        make_artifact(f"trajectory-{batch_id}", library_version=library_version),
    )
    actual_importances = importances or tuple(
        tuple(0.2 for _ in artifact.record.steps) for artifact in actual_artifacts
    )
    actual_deltas = deltas or tuple(0.4 for _ in actual_artifacts)
    if not (len(actual_artifacts) == len(actual_importances) == len(actual_deltas)):
        raise ValueError("source fixture arrays must align")
    residuals = tuple(
        make_residual(
            artifact.record.trajectory_id,
            horizon=artifact.record.horizon,
            delta=delta,
            reward_value=artifact.record.reward.value,
            sum_forward=math.fsum(-1.0 + value for value in importance),
            sum_backward=-float(artifact.record.horizon),
        )
        for artifact, delta, importance in zip(
            actual_artifacts, actual_deltas, actual_importances, strict=True
        )
    )
    stats = TTBBatchStats(
        batch_id=batch_id,
        optimizer_step=optimizer_step,
        library_version=library_version,
        residuals=residuals,
        batch_loss=math.fsum((residual.delta / residual.horizon) ** 2 for residual in residuals)
        / len(residuals),
        mean_reward=math.fsum(residual.raw_reward for residual in residuals) / len(residuals),
        created_at=CREATED_AT,
    )
    edges = tuple(
        replace(
            make_edge(artifact.record.trajectory_id, step_index, importance),
            forward_adapter_version=artifact.manifest.policy_snapshot.forward_adapter_version,
            context=EdgeScoreContext(
                batch_id,
                artifact.manifest.policy_snapshot.snapshot_id,
                library_version,
                artifact.record.steps[step_index - 1].action_token_count,
                artifact.record.steps[step_index - 1].action_token_ids,
            ),
        )
        for artifact, trajectory_importances in zip(
            actual_artifacts,
            actual_importances,
            strict=True,
        )
        for step_index, importance in enumerate(trajectory_importances, start=1)
    )
    return TrainingStepSource(
        batch_id=batch_id,
        optimizer_step=optimizer_step,
        artifacts=actual_artifacts,
        stats=stats,
        edge_records=edges,
    )
