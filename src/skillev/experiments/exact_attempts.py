"""Closed, exact-input builders for one isolated method attempt.

The input is deliberately data, not a Python entrypoint.  It carries public
task projections and trusted terminal rewards separately; the latter are held
by the session factory and are never copied into a rollout task or prompt.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias

from skillev.application import (
    ApplicationConfig,
    SKILLEVApplication,
    TerminalComponents,
)
from skillev.benchmarks import (
    BenchmarkPublicItem,
    CompletionBenchmarkEnvironment,
    OrderedBenchmarkTaskProvider,
)
from skillev.contracts import JsonValue, TerminalReward, normalize_json, stable_hash
from skillev.evolution import PhiBudgetAuthority, SkillAuthoringAuthority
from skillev.experiments.arm_events import LiveArmEventLog
from skillev.experiments.arms.builders import (
    ArmApplicationInputs,
    FlowOnlyApplication,
    FlowOnlyArmApplicationInputs,
    build_capped_flow_weight_application,
    build_clipped_importance_application,
    build_no_bayesian_calibration_application,
    build_no_flow_weighting_application,
    build_posterior_mean_decision_application,
    build_residual_only_phase_application,
)
from skillev.experiments.attempt_identity import AttemptPurpose, PublishedAttemptIdentity
from skillev.experiments.checkpoint_namespace import checkpoint_storage_for_attempt
from skillev.experiments.protocol import AblationArm, arm_protocol_for_builder_kind
from skillev.policy import (
    PrivateInitialCheckpointBinding,
    PublicTokenizerIdentity,
    PublicTokenizerKind,
    QwenBackboneConfig,
    QwenDeploymentConfig,
    QwenMultimodalBackboneConfig,
    public_qwen_deployment_hash,
)
from skillev.rollout import RolloutTask, TerminalEvaluationRequest, UnskilledRolloutSessionBundle
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptRunCursorState,
    AttemptRunProgress,
    BudgetLedger,
    BudgetVector,
    ExactAttemptRunPlan,
    LiveAttemptEventLog,
    SkillDocument,
    SkillLibraryState,
)
from skillev.runtime.attempt_publication import sha256_bytes
from skillev.training import (
    FixedAttemptBudgetPlan,
    PrivateCheckpointStorageBinding,
    RolloutWorkflowBinding,
)

EXACT_ATTEMPT_INPUT_FORMAT = "skillev-exact-attempt-input@5"


class ExactPolicyBackend(StrEnum):
    CAUSAL = "qwen-causal"
    MULTIMODAL = "qwen-multimodal"


def _object(value: object, *, fields: set[str], label: str) -> dict[str, Any]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


@dataclass(frozen=True, slots=True)
class ExactCompletionCase:
    public: BenchmarkPublicItem
    reward: TerminalReward

    def __post_init__(self) -> None:
        if self.reward.environment_id != self.public.environment_id:
            raise ValueError("terminal reward belongs to another public environment")

    def to_value(self) -> dict[str, JsonValue]:
        return {"public": self.public.to_value(), "reward": self.reward.to_value()}

    @classmethod
    def from_value(cls, value: object) -> ExactCompletionCase:
        data = _object(value, fields={"public", "reward"}, label="exact completion case")
        return cls(
            public=BenchmarkPublicItem.from_value(data["public"]),
            reward=TerminalReward.from_value(data["reward"]),
        )


@dataclass(frozen=True, slots=True)
class ExactAttemptInput:
    backbone: QwenDeploymentConfig
    backbone_kind: ExactPolicyBackend
    application: ApplicationConfig
    checkpoint_storage: PrivateCheckpointStorageBinding
    initial_checkpoint: PrivateInitialCheckpointBinding
    cases: tuple[ExactCompletionCase, ...]
    seed_documents: tuple[SkillDocument, ...]
    attempt_budget: BudgetVector
    phi_per_cycle_maximum: BudgetVector
    authoring_authority: SkillAuthoringAuthority
    run_plan: ExactAttemptRunPlan
    protocol_hash: str
    protocol_freeze_id: str
    rollout_workflow: RolloutWorkflowBinding = field(default_factory=RolloutWorkflowBinding)
    format: str = EXACT_ATTEMPT_INPUT_FORMAT

    def __post_init__(self) -> None:
        if self.format != EXACT_ATTEMPT_INPUT_FORMAT:
            raise ValueError("unsupported exact attempt input format")
        match self.backbone_kind:
            case ExactPolicyBackend.CAUSAL:
                if type(self.backbone) is not QwenBackboneConfig:
                    raise TypeError("causal exact input requires QwenBackboneConfig")
            case ExactPolicyBackend.MULTIMODAL:
                if type(self.backbone) is not QwenMultimodalBackboneConfig:
                    raise TypeError("multimodal exact input requires QwenMultimodalBackboneConfig")
            case _ as unreachable:
                from typing import assert_never

                assert_never(unreachable)
        if not isinstance(self.run_plan, ExactAttemptRunPlan):
            raise TypeError("run_plan must be ExactAttemptRunPlan")
        if not isinstance(self.checkpoint_storage, PrivateCheckpointStorageBinding):
            raise TypeError("exact attempt requires private checkpoint storage")
        if not isinstance(self.initial_checkpoint, PrivateInitialCheckpointBinding):
            raise TypeError("exact attempt requires one private initial checkpoint")
        if not isinstance(self.attempt_budget, BudgetVector):
            raise TypeError("attempt_budget must be BudgetVector")
        if not isinstance(self.phi_per_cycle_maximum, BudgetVector):
            raise TypeError("phi_per_cycle_maximum must be BudgetVector")
        if not isinstance(self.authoring_authority, SkillAuthoringAuthority):
            raise TypeError("authoring_authority must be SkillAuthoringAuthority")
        if not isinstance(self.rollout_workflow, RolloutWorkflowBinding):
            raise TypeError("exact attempt requires a rollout workflow binding")
        expected_cases = (
            self.application.trainer.execution.batch_size * self.run_plan.total_training_steps
        )
        if len(self.cases) != expected_cases:
            raise ValueError("exact cases must equal batch_size times total run-plan steps")
        task_ids = tuple(case.public.task_id for case in self.cases)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("exact completion task IDs must be unique")
        if not self.seed_documents:
            raise ValueError("exact attempt requires a non-empty seed library")
        if type(self.protocol_hash) is not str or not self.protocol_hash.startswith("sha256:"):
            raise ValueError("protocol_hash must be a content hash")
        if type(self.protocol_freeze_id) is not str or not self.protocol_freeze_id.startswith(
            "sha256:"
        ):
            raise ValueError("protocol_freeze_id must be a content hash")
        self.required_attempt_budget().validate_against(self.attempt_budget)

    def required_attempt_budget(self) -> FixedAttemptBudgetPlan:
        return FixedAttemptBudgetPlan.from_trainer_and_run_plan(
            trainer=self.application.trainer,
            run_plan=self.run_plan,
            phi_per_cycle_maximum=self.phi_per_cycle_maximum,
        )

    @property
    def initial_library_version(self) -> str:
        return SkillLibraryState.from_seed_documents(self.seed_documents).current_version

    @property
    def initial_skill_library_state_hash(self) -> str:
        return SkillLibraryState.from_seed_documents(self.seed_documents).state_hash

    @property
    def ordered_task_sequence_hash(self) -> str:
        return stable_hash(
            {
                "ordered_public_tasks": [case.public.to_value() for case in self.cases],
                "run_plan_hash": self.run_plan.content_hash,
            }
        )

    @property
    def backbone_deployment_hash(self) -> str:
        return public_qwen_deployment_hash(
            self.backbone,
            backend_kind=self.backbone_kind.value,
        )

    def public_identity(
        self,
        *,
        builder_kind: AttemptBuilderKind,
        exact_input_sha256: str,
    ) -> PublishedAttemptIdentity:
        return PublishedAttemptIdentity(
            builder_kind=builder_kind,
            purpose=AttemptPurpose.CORRECTNESS_FIXTURE,
            arm_protocol=arm_protocol_for_builder_kind(builder_kind),
            application_config=self.application,
            run_plan=self.run_plan,
            initial_optimizer_step=0,
            initial_run_cursor=AttemptRunCursorState.fresh(self.run_plan),
            authoring_authority=self.authoring_authority,
            attempt_budget=self.attempt_budget,
            phi_per_cycle_maximum=self.phi_per_cycle_maximum,
            protocol_hash=self.protocol_hash,
            protocol_freeze_id=self.protocol_freeze_id,
            exact_input_sha256=exact_input_sha256,
            backbone_deployment_hash=self.backbone_deployment_hash,
            initial_trainable_state=self.initial_checkpoint.trainable_state,
            tokenizer_identity=PublicTokenizerIdentity(
                kind=PublicTokenizerKind.QWEN,
                tokenizer_id=self.backbone.tokenizer_id,
                revision=self.backbone.revision,
                content_hash=self.backbone.tokenizer_content_hash,
            ),
            initial_library_version=self.initial_library_version,
            initial_skill_library_state_hash=self.initial_skill_library_state_hash,
            ordered_task_sequence_hash=self.ordered_task_sequence_hash,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "application": normalize_json(self.application.to_value()),
            "attempt_budget": normalize_json(self.attempt_budget.to_value()),
            "authoring_authority": normalize_json(self.authoring_authority.to_value()),
            "backbone": normalize_json(self.backbone.to_value()),
            "backbone_kind": self.backbone_kind.value,
            "cases": [case.to_value() for case in self.cases],
            "checkpoint_storage": normalize_json(self.checkpoint_storage.to_value()),
            "format": self.format,
            "initial_checkpoint": normalize_json(self.initial_checkpoint.to_value()),
            "phi_per_cycle_maximum": normalize_json(self.phi_per_cycle_maximum.to_value()),
            "protocol_freeze_id": self.protocol_freeze_id,
            "protocol_hash": self.protocol_hash,
            "run_plan": self.run_plan.to_value(),
            "rollout_workflow": self.rollout_workflow.to_value(),
            "seed_documents": [document.to_value() for document in self.seed_documents],
        }

    @classmethod
    def from_value(cls, value: object) -> ExactAttemptInput:
        data = _object(
            value,
            fields={
                "application",
                "attempt_budget",
                "authoring_authority",
                "backbone",
                "backbone_kind",
                "cases",
                "checkpoint_storage",
                "format",
                "initial_checkpoint",
                "phi_per_cycle_maximum",
                "protocol_freeze_id",
                "protocol_hash",
                "run_plan",
                "rollout_workflow",
                "seed_documents",
            },
            label="exact attempt input",
        )
        if data["format"] != EXACT_ATTEMPT_INPUT_FORMAT:
            raise ValueError("unsupported exact attempt input format")
        raw_cases = data["cases"]
        raw_documents = data["seed_documents"]
        if not isinstance(raw_cases, list) or not isinstance(raw_documents, list):
            raise TypeError("exact cases and seed_documents must be arrays")
        raw_kind = data["backbone_kind"]
        if type(raw_kind) is not str:
            raise TypeError("backbone_kind must be text")
        backbone_kind = ExactPolicyBackend(raw_kind)
        backbone = (
            QwenBackboneConfig.from_value(data["backbone"])
            if backbone_kind is ExactPolicyBackend.CAUSAL
            else QwenMultimodalBackboneConfig.from_value(data["backbone"])
        )
        for field_name in ("protocol_hash", "protocol_freeze_id", "format"):
            if type(data[field_name]) is not str:
                raise TypeError(f"{field_name} must be text")
        return cls(
            backbone=backbone,
            backbone_kind=backbone_kind,
            application=ApplicationConfig.from_value(data["application"]),
            checkpoint_storage=PrivateCheckpointStorageBinding.from_value(
                data["checkpoint_storage"]
            ),
            initial_checkpoint=PrivateInitialCheckpointBinding.from_value(
                data["initial_checkpoint"]
            ),
            cases=tuple(ExactCompletionCase.from_value(item) for item in raw_cases),
            seed_documents=tuple(SkillDocument.from_value(item) for item in raw_documents),
            attempt_budget=BudgetVector.from_value(data["attempt_budget"]),
            phi_per_cycle_maximum=BudgetVector.from_value(data["phi_per_cycle_maximum"]),
            authoring_authority=SkillAuthoringAuthority.from_value(data["authoring_authority"]),
            run_plan=ExactAttemptRunPlan.from_value(data["run_plan"]),
            rollout_workflow=RolloutWorkflowBinding.from_value(data["rollout_workflow"]),
            protocol_hash=data["protocol_hash"],
            protocol_freeze_id=data["protocol_freeze_id"],
            format=data["format"],
        )

    @classmethod
    def read(cls, path: Path) -> ExactAttemptInput:
        return cls.from_value(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def read_verified(
        cls,
        path: Path,
        *,
        expected_sha256: str,
    ) -> ExactAttemptInput:
        data = path.resolve().read_bytes()
        if sha256_bytes(data) != expected_sha256:
            raise ValueError("exact input content changed after request capture")
        return cls.from_value(json.loads(data.decode("utf-8")))


@dataclass(frozen=True, slots=True)
class _ExactTerminalEvaluator:
    task_id: str
    reward: TerminalReward

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.task_id:
            raise ValueError("terminal evaluation reached another exact task")
        return self.reward


@dataclass(frozen=True, slots=True)
class _ExactCompletionSessionFactory:
    cases: tuple[ExactCompletionCase, ...]

    def create(self, task: RolloutTask) -> UnskilledRolloutSessionBundle:
        matches = tuple(case for case in self.cases if case.public.task_id == task.task_id)
        if len(matches) != 1 or matches[0].public.to_rollout_task() != task:
            raise ValueError("rollout task has no exact completion case")
        case = matches[0]
        return UnskilledRolloutSessionBundle(
            environment=CompletionBenchmarkEnvironment(case.public),
            evaluator=_ExactTerminalEvaluator(task.task_id, case.reward),
        )


ExactApplication: TypeAlias = SKILLEVApplication | FlowOnlyApplication


@dataclass(frozen=True, slots=True)
class ExactBuiltApplication:
    application: ExactApplication
    run_plan: ExactAttemptRunPlan
    public_identity: PublishedAttemptIdentity
    arm_event_log: LiveArmEventLog | None = None

    def __post_init__(self) -> None:
        if self.public_identity.run_plan != self.run_plan:
            raise ValueError("built application identity has another run plan")

    def close_source_logs(self) -> None:
        """Close the persistent arm stream before publication hashes it."""

        if self.arm_event_log is not None:
            self.arm_event_log.close()


def _common(
    path: Path,
    *,
    builder_kind: AttemptBuilderKind,
    exact_input_sha256: str,
    run_id: str,
    attempt_id: str,
    event_log_path: Path,
) -> tuple[
    ExactAttemptInput,
    PublishedAttemptIdentity,
    PrivateCheckpointStorageBinding,
    TerminalComponents,
    OrderedBenchmarkTaskProvider,
    _ExactCompletionSessionFactory,
    LiveAttemptEventLog,
]:
    exact = ExactAttemptInput.read_verified(path, expected_sha256=exact_input_sha256)
    identity = exact.public_identity(
        builder_kind=builder_kind,
        exact_input_sha256=exact_input_sha256,
    )
    checkpoint_storage = checkpoint_storage_for_attempt(
        exact.checkpoint_storage,
        run_id=run_id,
        attempt_id=attempt_id,
        builder_kind=builder_kind,
        exact_input_sha256=exact_input_sha256,
    )
    ledger = BudgetLedger(
        run_id=run_id,
        attempt_id=attempt_id,
        cap=exact.attempt_budget,
    )
    terminal = TerminalComponents(
        ledger=ledger,
        authoring_authority=exact.authoring_authority,
        phi_budget=PhiBudgetAuthority(exact.phi_per_cycle_maximum),
    )
    provider = OrderedBenchmarkTaskProvider(tuple(case.public for case in exact.cases))
    sessions = _ExactCompletionSessionFactory(exact.cases)
    event_log = LiveAttemptEventLog(event_log_path, run_id=run_id, attempt_id=attempt_id)
    return exact, identity, checkpoint_storage, terminal, provider, sessions, event_log


def _clock() -> str:
    return "1970-01-01T00:00:00Z"


def build_full_application(
    path: Path,
    *,
    exact_input_sha256: str,
    run_id: str,
    attempt_id: str,
    event_log_path: Path,
) -> ExactBuiltApplication:
    exact, identity, checkpoint_storage, terminal, provider, sessions, event_log = _common(
        path,
        builder_kind=AttemptBuilderKind.FULL,
        exact_input_sha256=exact_input_sha256,
        run_id=run_id,
        attempt_id=attempt_id,
        event_log_path=event_log_path,
    )
    return ExactBuiltApplication(
        application=SKILLEVApplication.build(
            backbone_config=exact.backbone,
            task_provider=provider,
            base_session_factory=sessions,
            seed_documents=exact.seed_documents,
            terminal_components=terminal,
            checkpoint_storage=checkpoint_storage,
            initial_checkpoint=exact.initial_checkpoint,
            public_identity=identity,
            event_log=event_log,
            clock=_clock,
            workflow_binding=exact.rollout_workflow,
        ),
        run_plan=exact.run_plan,
        public_identity=identity,
    )


ArmBuilder: TypeAlias = Callable[[ArmApplicationInputs], ExactApplication]


def _build_arm_application(
    path: Path,
    *,
    builder_kind: AttemptBuilderKind,
    exact_input_sha256: str,
    run_id: str,
    attempt_id: str,
    event_log_path: Path,
    arm_event_log_path: Path,
    arm: AblationArm,
    builder: ArmBuilder | None,
) -> ExactBuiltApplication:
    exact, identity, checkpoint_storage, terminal, provider, sessions, event_log = _common(
        path,
        builder_kind=builder_kind,
        exact_input_sha256=exact_input_sha256,
        run_id=run_id,
        attempt_id=attempt_id,
        event_log_path=event_log_path,
    )
    if identity.arm is not arm:
        raise ValueError("builder kind and arm protocol disagree")
    run_progress = AttemptRunProgress.from_state(
        exact.run_plan,
        identity.initial_run_cursor,
    )
    snapshot_identity = identity.runtime_snapshot_identity()
    arm_log: LiveArmEventLog | None = None
    application: ExactApplication
    if builder_kind is AttemptBuilderKind.NO_BAYESIAN:
        arm_log = LiveArmEventLog(
            arm_event_log_path,
            arm=arm,
            run_id=run_id,
            attempt_id=attempt_id,
        )
        application = build_no_bayesian_calibration_application(
            FlowOnlyArmApplicationInputs(
                backbone_config=exact.backbone,
                task_provider=provider,
                base_session_factory=sessions,
                seed_documents=exact.seed_documents,
                terminal_components=terminal,
                checkpoint_storage=checkpoint_storage,
                initial_checkpoint=exact.initial_checkpoint,
                public_identity=identity,
                run_progress=run_progress,
                snapshot_identity=snapshot_identity,
                event_log=event_log,
                clock=_clock,
                rollout_workflow=exact.rollout_workflow,
                arm_event_log=arm_log,
            )
        )
    else:
        if builder is None:
            raise TypeError("full-shaped ablation requires an application builder")
        application = builder(
            ArmApplicationInputs(
                backbone_config=exact.backbone,
                task_provider=provider,
                base_session_factory=sessions,
                seed_documents=exact.seed_documents,
                terminal_components=terminal,
                checkpoint_storage=checkpoint_storage,
                initial_checkpoint=exact.initial_checkpoint,
                public_identity=identity,
                run_progress=run_progress,
                snapshot_identity=snapshot_identity,
                event_log=event_log,
                clock=_clock,
                rollout_workflow=exact.rollout_workflow,
            )
        )
    return ExactBuiltApplication(
        application=application,
        run_plan=exact.run_plan,
        public_identity=identity,
        arm_event_log=arm_log,
    )


def build_no_bayesian_application(
    path: Path,
    *,
    exact_input_sha256: str,
    run_id: str,
    attempt_id: str,
    event_log_path: Path,
    arm_event_log_path: Path,
) -> ExactBuiltApplication:
    return _build_arm_application(
        path,
        builder_kind=AttemptBuilderKind.NO_BAYESIAN,
        exact_input_sha256=exact_input_sha256,
        run_id=run_id,
        attempt_id=attempt_id,
        event_log_path=event_log_path,
        arm_event_log_path=arm_event_log_path,
        arm=AblationArm.SKILLFLOW_DISABLED,
        builder=None,
    )


def build_unit_flow_application(
    path: Path,
    *,
    exact_input_sha256: str,
    run_id: str,
    attempt_id: str,
    event_log_path: Path,
    arm_event_log_path: Path,
) -> ExactBuiltApplication:
    return _build_arm_application(
        path,
        builder_kind=AttemptBuilderKind.UNIT_FLOW,
        exact_input_sha256=exact_input_sha256,
        run_id=run_id,
        attempt_id=attempt_id,
        event_log_path=event_log_path,
        arm_event_log_path=arm_event_log_path,
        arm=AblationArm.NO_FLOW_WEIGHTING,
        builder=build_no_flow_weighting_application,
    )


def build_capped_flow_application(
    path: Path,
    *,
    exact_input_sha256: str,
    run_id: str,
    attempt_id: str,
    event_log_path: Path,
    arm_event_log_path: Path,
) -> ExactBuiltApplication:
    return _build_arm_application(
        path,
        builder_kind=AttemptBuilderKind.CAPPED_FLOW,
        exact_input_sha256=exact_input_sha256,
        run_id=run_id,
        attempt_id=attempt_id,
        event_log_path=event_log_path,
        arm_event_log_path=arm_event_log_path,
        arm=AblationArm.CAPPED_FLOW_WEIGHT,
        builder=build_capped_flow_weight_application,
    )


def build_clipped_importance_application_exact(
    path: Path,
    *,
    exact_input_sha256: str,
    run_id: str,
    attempt_id: str,
    event_log_path: Path,
    arm_event_log_path: Path,
) -> ExactBuiltApplication:
    return _build_arm_application(
        path,
        builder_kind=AttemptBuilderKind.CLIPPED_IMPORTANCE,
        exact_input_sha256=exact_input_sha256,
        run_id=run_id,
        attempt_id=attempt_id,
        event_log_path=event_log_path,
        arm_event_log_path=arm_event_log_path,
        arm=AblationArm.CLIPPED_IMPORTANCE,
        builder=build_clipped_importance_application,
    )


def build_posterior_mean_application(
    path: Path,
    *,
    exact_input_sha256: str,
    run_id: str,
    attempt_id: str,
    event_log_path: Path,
    arm_event_log_path: Path,
) -> ExactBuiltApplication:
    return _build_arm_application(
        path,
        builder_kind=AttemptBuilderKind.POSTERIOR_MEAN,
        exact_input_sha256=exact_input_sha256,
        run_id=run_id,
        attempt_id=attempt_id,
        event_log_path=event_log_path,
        arm_event_log_path=arm_event_log_path,
        arm=AblationArm.LCB_TO_MEAN,
        builder=build_posterior_mean_decision_application,
    )


def build_residual_only_application(
    path: Path,
    *,
    exact_input_sha256: str,
    run_id: str,
    attempt_id: str,
    event_log_path: Path,
    arm_event_log_path: Path,
) -> ExactBuiltApplication:
    return _build_arm_application(
        path,
        builder_kind=AttemptBuilderKind.RESIDUAL_ONLY_PHASE,
        exact_input_sha256=exact_input_sha256,
        run_id=run_id,
        attempt_id=attempt_id,
        event_log_path=event_log_path,
        arm_event_log_path=arm_event_log_path,
        arm=AblationArm.AND_TO_SINGLE_CONDITION,
        builder=build_residual_only_phase_application,
    )


__all__ = [
    "EXACT_ATTEMPT_INPUT_FORMAT",
    "ExactAttemptInput",
    "ExactBuiltApplication",
    "ExactCompletionCase",
    "ExactPolicyBackend",
    "build_capped_flow_application",
    "build_clipped_importance_application_exact",
    "build_full_application",
    "build_no_bayesian_application",
    "build_posterior_mean_application",
    "build_residual_only_application",
    "build_unit_flow_application",
]
