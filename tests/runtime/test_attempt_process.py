from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from skillev.application import ApplicationConfig
from skillev.audit import AuditResources, audit_published_attempt
from skillev.audit.complete_method import audit_full_shaped_attempt
from skillev.audit.no_bayesian import audit_no_bayesian_attempt
from skillev.audit.published_identity import read_published_attempt_identity
from skillev.audit.tokenizer import TokenizerArtifactResolver
from skillev.benchmarks import BenchmarkPublicItem
from skillev.calibration import CalibrationConfig
from skillev.contracts import SuccessRule, TerminalReward, canonical_json, stable_hash
from skillev.diagnostics import DiagnosticsConfig
from skillev.evolution import (
    AuthoringSamplingConfig,
    EvolutionConfig,
    SkillAuthoringAuthority,
)
from skillev.experiments.exact_attempts import (
    ExactAttemptInput,
    ExactCompletionCase,
    ExactPolicyBackend,
)
from skillev.experiments.results import require_cross_arm_comparable
from skillev.policy import (
    PrivateInitialCheckpointBinding,
    PublicTokenizerIdentity,
    QwenBackboneConfig,
    QwenDeploymentConfig,
    QwenMultimodalBackboneConfig,
    QwenTokenizerAdapter,
    TrainableStateIdentity,
    build_qwen_policy_backbone,
    public_qwen_deployment_hash,
)
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptFailed,
    AttemptProcessFailedError,
    AttemptPublisher,
    AttemptRequest,
    AttemptSucceeded,
    BudgetVector,
    ExactAttemptRunPlan,
    read_attempt_outcome,
)
from skillev.runtime.attempt_publication import sha256_bytes, sha256_tree
from skillev.runtime.attempt_supervisor import AttemptSupervisor
from skillev.training import (
    CheckpointConfig,
    OptimizerConfig,
    PolicyRolloutConfig,
    PrivateCheckpointStorageBinding,
    RolloutWorkflowBinding,
    TrainerConfig,
    TrainingExecutionConfig,
    TTBMethodConfig,
    conservative_rollout_maximum,
)
from tests.v3_helpers import TEST_TASK_FAMILY, make_skill_document


def _run_plan() -> ExactAttemptRunPlan:
    return ExactAttemptRunPlan(
        phase_search_steps=1,
        closure_steps=1,
        maximum_cycles=1,
    )


def _attempt_budget() -> BudgetVector:
    """Deliberately roomy test cap; the exact plan still computes the envelope."""

    return BudgetVector(
        input_tokens=100_000,
        output_tokens=100_000,
        model_calls=1_000,
        agent_turns=1_000,
        tool_calls=1_000,
        wall_time_milliseconds=1_000_000,
    )


def _exact_input(
    tmp_path: Path,
    backbone: QwenDeploymentConfig,
    *,
    backbone_kind: ExactPolicyBackend = ExactPolicyBackend.CAUSAL,
) -> ExactAttemptInput:
    plan = _run_plan()
    rollout_maximum = conservative_rollout_maximum(
        max_turns=1,
        max_reasoning_tokens=1,
        max_action_tokens=1,
        max_model_input_tokens=4096,
        max_tool_wall_time_milliseconds=100,
    )
    cases = tuple(
        ExactCompletionCase(
            public=BenchmarkPublicItem(
                benchmark_id="child-smoke",
                dataset_revision="local@1",
                split="test",
                task_id=f"tiny-task-{index}",
                task_family="child-smoke/debug-family",
                query="Return one public token.",
                public_context={"ordinal": index},
            ),
            reward=TerminalReward(
                value=1.0,
                success=True,
                success_rule=SuccessRule.R_EQUALS_ONE,
                success_threshold=None,
                native_metric_name="tiny-exact",
                native_payload={"success": True},
                environment_id="benchmark:child-smoke@local@1",
                verifier_version="tiny-verifier@1",
            ),
        )
        for index in range(1, plan.total_training_steps + 1)
    )
    config = ApplicationConfig(
        trainer=TrainerConfig(
            method=TTBMethodConfig(epsilon_min=0.01, temperature_beta=1.0),
            rollout=PolicyRolloutConfig(
                base_seed=4,
                max_turns=1,
                max_reasoning_tokens=1,
                max_action_tokens=1,
                per_rollout_maximum=rollout_maximum,
            ),
            optimizer=OptimizerConfig(
                adapter_learning_rate=1e-3,
                z_learning_rate=1e-3,
                weight_decay=0.0,
            ),
            execution=TrainingExecutionConfig(
                experiment_id="child-smoke",
                batch_size=1,
            ),
            checkpoint=CheckpointConfig(
                every_n_steps=100,
            ),
        ),
        diagnostics=DiagnosticsConfig(window_size=2),
        calibration=CalibrationConfig(),
        evolution=EvolutionConfig(generate_min_absolute_log_importance=0.1),
        authoring_sampling=AuthoringSamplingConfig(temperature=1.0, top_p=1.0),
        maximum_h0_tokens=4096,
    )
    return ExactAttemptInput(
        backbone=backbone,
        backbone_kind=backbone_kind,
        application=config,
        checkpoint_storage=PrivateCheckpointStorageBinding(directory=str(tmp_path / "checkpoints")),
        initial_checkpoint=_initial_checkpoint(tmp_path, backbone),
        cases=cases,
        seed_documents=(make_skill_document("seed"),),
        attempt_budget=_attempt_budget(),
        phi_per_cycle_maximum=BudgetVector(
            input_tokens=10_000,
            output_tokens=10_000,
            model_calls=2,
        ),
        authoring_authority=SkillAuthoringAuthority(
            input_schema_id="tiny-input@1",
            output_schema_id="tiny-output@1",
            license_id="unit-test",
            allowed_task_families=(TEST_TASK_FAMILY,),
            allowed_tools=(),
        ),
        run_plan=plan,
        protocol_hash=sha256_bytes(b"test-protocol"),
        protocol_freeze_id=sha256_bytes(b"test-protocol-freeze"),
        rollout_workflow=RolloutWorkflowBinding(
            max_resident_trajectories=2,
            max_inflight_model_requests=2,
            max_inflight_environment_calls=2,
            max_inflight_terminal_evaluations=2,
            max_inflight_process_graders=1,
            transport_worker_threads=2,
        ),
    )


def _initial_checkpoint(
    root: Path,
    backbone: QwenDeploymentConfig,
) -> PrivateInitialCheckpointBinding:
    """Create the one frozen initial partition used by every child of a test input."""

    directory = root / "initial-trainable-state"
    if type(backbone) is QwenMultimodalBackboneConfig:
        # This serialization-only fixture intentionally does not execute a
        # multimodal worker.  Its production builder still requires a real
        # checkpoint at the private execution boundary.
        deployment = public_qwen_deployment_hash(backbone, backend_kind="qwen-multimodal")
        return PrivateInitialCheckpointBinding(
            directory=str(directory),
            trainable_state=TrainableStateIdentity.create(
                backbone_deployment_hash=deployment,
                forward_adapter_hash=stable_hash({"forward": "multimodal-fixture"}),
                backward_adapter_hash=stable_hash({"backward": "multimodal-fixture"}),
                z_head_hash=stable_hash({"z": "multimodal-fixture"}),
            ),
        )
    policy = build_qwen_policy_backbone(backbone)
    policy.save_checkpoint(str(directory))
    return PrivateInitialCheckpointBinding(
        directory=str(directory),
        trainable_state=policy.trainable_state_identity,
    )


def _write_exact_input(path: Path, exact: ExactAttemptInput) -> str:
    encoded = (canonical_json(exact.to_value()) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    return sha256_bytes(encoded)


def _request(
    tmp_path: Path,
    *,
    attempt_id: str,
    kind: AttemptBuilderKind,
    exact_input_path: Path,
    exact_input_sha256: str,
) -> AttemptRequest:
    return AttemptRequest(
        run_id="child-smoke-run",
        attempt_id=attempt_id,
        builder_kind=kind,
        exact_input_path=exact_input_path,
        exact_input_sha256=exact_input_sha256,
        private_bundle_directory=tmp_path / "private" / attempt_id,
    )


def _supervisor(tmp_path: Path) -> AttemptSupervisor:
    return AttemptSupervisor(
        AttemptPublisher(
            published_root=tmp_path / "published",
            quarantine_root=tmp_path / "quarantine",
        )
    )


@dataclass(frozen=True, slots=True)
class _ExactTokenizerResolver(TokenizerArtifactResolver):
    config: QwenBackboneConfig

    def resolve(self, identity: PublicTokenizerIdentity) -> QwenTokenizerAdapter:
        tokenizer = QwenTokenizerAdapter.from_config(self.config)
        if tokenizer.public_identity != identity:
            raise ValueError("test resolver received another tokenizer identity")
        return tokenizer


def test_exact_input_preserves_explicit_policy_backend_identity(
    tmp_path: Path,
    training_backbone_config: QwenBackboneConfig,
) -> None:
    multimodal = QwenMultimodalBackboneConfig.from_value(training_backbone_config.to_value())
    exact_input = _exact_input(
        tmp_path,
        multimodal,
        backbone_kind=ExactPolicyBackend.MULTIMODAL,
    )

    restored = ExactAttemptInput.from_value(exact_input.to_value())

    assert restored == exact_input
    assert type(restored.backbone) is QwenMultimodalBackboneConfig
    with pytest.raises(TypeError):
        replace(restored, backbone_kind=ExactPolicyBackend.CAUSAL)
    with pytest.raises(ValueError):
        ExactAttemptInput.from_value(
            {**exact_input.to_value(), "format": "skillev-exact-attempt-input@2"}
        )


@pytest.mark.parametrize("kind", tuple(AttemptBuilderKind))
def test_supervisor_runs_the_closed_builder_and_generic_typed_audit(
    tmp_path: Path,
    training_backbone_config: QwenBackboneConfig,
    kind: AttemptBuilderKind,
) -> None:
    exact_path = tmp_path / f"{kind.value}.json"
    exact_input = _exact_input(tmp_path / kind.value, training_backbone_config)
    exact_hash = _write_exact_input(exact_path, exact_input)
    attempt_id = f"attempt-{kind.value}"
    request = _request(
        tmp_path,
        attempt_id=attempt_id,
        kind=kind,
        exact_input_path=exact_path,
        exact_input_sha256=exact_hash,
    )

    published = _supervisor(tmp_path).run(request)
    outcome = read_attempt_outcome(published.outcome_path)

    assert isinstance(outcome, AttemptSucceeded)
    assert (
        outcome.summary.completed_training_steps_this_attempt
        == exact_input.run_plan.total_training_steps
    )
    assert (
        outcome.summary.planned_training_steps_this_attempt
        == exact_input.run_plan.total_training_steps
    )
    assert published.event_log_path.is_file()
    assert not (tmp_path / "private" / attempt_id).exists()

    resources = AuditResources(_ExactTokenizerResolver(training_backbone_config))
    audited = audit_published_attempt(published, resources=resources)
    assert audited.attempt_id == attempt_id
    assert audited.training_step_count == exact_input.run_plan.total_training_steps
    assert audited.final_library_version == outcome.summary.final_library_version
    if kind is AttemptBuilderKind.NO_BAYESIAN:
        with pytest.raises(TypeError):
            audit_full_shaped_attempt(published, resources=resources)
    else:
        with pytest.raises(TypeError):
            audit_no_bayesian_attempt(published, resources=resources)
        assert (
            audit_full_shaped_attempt(
                published,
                resources=resources,
            )
            == audited
        )


def test_seven_published_arms_have_one_cross_arm_execution_identity(
    tmp_path: Path,
    training_backbone_config: QwenBackboneConfig,
) -> None:
    """Gate D: each fixed builder publishes and audits one common exact input."""

    exact_path = tmp_path / "shared-exact-input.json"
    exact_input = _exact_input(tmp_path, training_backbone_config)
    exact_hash = _write_exact_input(exact_path, exact_input)
    supervisor = _supervisor(tmp_path)
    resources = AuditResources(_ExactTokenizerResolver(training_backbone_config))
    published = tuple(
        supervisor.run(
            _request(
                tmp_path,
                attempt_id=f"seven-arm-{kind.value}",
                kind=kind,
                exact_input_path=exact_path,
                exact_input_sha256=exact_hash,
            )
        )
        for kind in AttemptBuilderKind
    )

    identities = tuple(read_published_attempt_identity(item) for item in published)
    assert tuple(identity.builder_kind for identity in identities) == tuple(AttemptBuilderKind)
    require_cross_arm_comparable(identities)
    checkpoint_roots = tuple(sorted((tmp_path / "checkpoints" / "attempts").iterdir()))
    assert len(checkpoint_roots) == len(AttemptBuilderKind)
    snapshots = tuple(
        item
        for root in checkpoint_roots
        for item in root.iterdir()
        if item.is_dir() and item.name != "step-transactions"
    )
    assert all((snapshot / "runtime_state.json").is_file() for snapshot in snapshots)
    snapshot_identities = {sha256_tree(snapshot) for snapshot in snapshots}
    assert all(
        item.final_training_artifact.artifact_sha256 in snapshot_identities for item in published
    )
    audited_ids = tuple(
        audit_published_attempt(item, resources=resources).attempt_id for item in published
    )
    assert audited_ids == tuple(item.attempt_id for item in published)


def test_exact_input_capture_tampering_is_quarantined_without_private_text(tmp_path: Path) -> None:
    private_detail = "private/evaluator/answer-key"
    exact_path = tmp_path / "invalid.json"
    original = json.dumps({"private": private_detail}).encode("utf-8")
    exact_path.write_bytes(original)
    request = _request(
        tmp_path,
        attempt_id="invalid-attempt",
        kind=AttemptBuilderKind.FULL,
        exact_input_path=exact_path,
        exact_input_sha256=sha256_bytes(original),
    )

    with pytest.raises(AttemptProcessFailedError) as captured:
        _supervisor(tmp_path).run(request)

    outcome = read_attempt_outcome(captured.value.quarantine.directory / "outcome.json")
    assert isinstance(outcome, AttemptFailed)
    assert private_detail not in outcome.public_message
    assert private_detail not in canonical_json(outcome.to_value())
    assert not (tmp_path / "published" / request.attempt_id).exists()
