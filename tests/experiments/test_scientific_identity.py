"""Regression tests for the formal seven-arm scientific-identity gate."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from skillev.application import ApplicationConfig
from skillev.application_config import MethodSemanticsConfig
from skillev.calibration import CalibrationConfig
from skillev.contracts import stable_hash
from skillev.diagnostics import DiagnosticsConfig
from skillev.evolution import (
    AuthoringSamplingConfig,
    EvolutionConfig,
    SkillAuthoringAuthority,
    SplitCriterionConfig,
)
from skillev.experiments import (
    FIXED_SEED,
    AttemptPurpose,
    Benchmark,
    BenchmarkTaskCount,
    ExecutionHardwareIdentity,
    FormalRunLedger,
    FormalRunManifest,
    FormalRunSlot,
    FormalTrainingBinding,
    FrozenTaskSequenceIdentity,
    PublishedAttemptIdentity,
    SchedulePurpose,
    arm_protocol_for_builder_kind,
)
from skillev.experiments.protocol import FREEZE_FORMAT, ProtocolFreeze
from skillev.experiments.results import (
    require_cross_arm_comparable,
    require_formal_cross_arm_comparable,
)
from skillev.policy import TrainableStateIdentity
from skillev.runtime import (
    AttemptBuilderKind,
    BudgetVector,
    PublishedSuccessfulAttemptBundle,
)
from skillev.training import (
    CheckpointConfig,
    OptimizerConfig,
    PolicyRolloutConfig,
    TrainerConfig,
    TrainingExecutionConfig,
    TTBMethodConfig,
    conservative_rollout_maximum,
)
from tests.experiments.formal_execution_helpers import (
    make_formal_artifacts,
    make_formal_execution,
    publish_formal_success_for_identity,
)
from tests.v3_helpers import make_public_identity, make_run_plan, make_skill_document


def _config() -> ApplicationConfig:
    rollout_maximum = conservative_rollout_maximum(
        max_turns=1,
        max_reasoning_tokens=2,
        max_action_tokens=2,
        max_model_input_tokens=64,
        max_tool_wall_time_milliseconds=10,
    )
    return ApplicationConfig(
        trainer=TrainerConfig(
            method=TTBMethodConfig(epsilon_min=0.01, temperature_beta=1.0),
            rollout=PolicyRolloutConfig(
                base_seed=FIXED_SEED,
                max_turns=1,
                max_reasoning_tokens=2,
                max_action_tokens=2,
                per_rollout_maximum=rollout_maximum,
            ),
            optimizer=OptimizerConfig(
                adapter_learning_rate=1e-3,
                z_learning_rate=2e-3,
                weight_decay=0.0,
            ),
            execution=TrainingExecutionConfig(
                experiment_id="identity-regression-a",
                batch_size=1,
            ),
            checkpoint=CheckpointConfig(every_n_steps=3),
        ),
        diagnostics=DiagnosticsConfig(window_size=2),
        calibration=CalibrationConfig(alpha_0=1.0, beta_0=1.0, default_k=1.0),
        evolution=EvolutionConfig(
            generate_min_absolute_log_importance=0.1,
            entropy_window=2,
            split=SplitCriterionConfig(confidence_k=1.0),
        ),
        authoring_sampling=AuthoringSamplingConfig(temperature=1.0, top_p=1.0),
        maximum_h0_tokens=512,
    )


def test_method_semantics_are_closed_and_required_by_application_wire() -> None:
    semantics = MethodSemanticsConfig()
    assert MethodSemanticsConfig.from_value(semantics.to_value()) == semantics
    with pytest.raises(ValueError):
        MethodSemanticsConfig(context_feature="environment-id@1")

    value = _config().to_value()
    value.pop("semantics")
    with pytest.raises(ValueError):
        ApplicationConfig.from_value(value)


def _identities() -> tuple[PublishedAttemptIdentity, ...]:
    seed = make_skill_document("identity-seed")
    plan = make_run_plan(phase_search_steps=2, closure_steps=1, maximum_cycles=1)
    authority = SkillAuthoringAuthority(
        input_schema_id="debug-input@3",
        output_schema_id="debug-output@3",
        license_id="unit-test",
        allowed_task_families=("debug/task-family",),
        allowed_tools=(),
    )
    maximum = BudgetVector(input_tokens=4, output_tokens=4, model_calls=1)
    base = make_public_identity(
        config=_config(),
        run_plan=plan,
        seed_documents=(seed,),
        task_ids=("identity-task-1", "identity-task-2", "identity-task-3"),
        attempt_budget=BudgetVector(input_tokens=100, output_tokens=100, model_calls=100),
        phi_per_cycle_maximum=maximum,
        authoring_authority=authority,
    )
    return tuple(
        replace(
            base,
            builder_kind=kind,
            arm_protocol=arm_protocol_for_builder_kind(kind),
        )
        for kind in AttemptBuilderKind
    )


def _replace_one(
    identities: tuple[PublishedAttemptIdentity, ...],
    mutate: Callable[[PublishedAttemptIdentity], PublishedAttemptIdentity],
) -> tuple[PublishedAttemptIdentity, ...]:
    return (*identities[:-1], mutate(identities[-1]))


def _formal_binding(identity: PublishedAttemptIdentity) -> FormalTrainingBinding:
    return FormalTrainingBinding(
        training_sequence=FrozenTaskSequenceIdentity(
            purpose=SchedulePurpose.IID_TRAINING,
            ordered_task_ids_hash=identity.ordered_task_sequence_hash,
            task_count=identity.run_plan.total_training_steps,
            benchmark_counts=(
                BenchmarkTaskCount(
                    benchmark=Benchmark.HOTPOT_QA,
                    count=identity.run_plan.total_training_steps,
                ),
            ),
            schedule_algorithm="identity-regression@1",
        ),
        catalog_freeze_hash=stable_hash({"catalog": "identity-regression"}),
    )


def _formal_identities(
    identities: tuple[PublishedAttemptIdentity, ...],
) -> tuple[PublishedAttemptIdentity, ...]:
    artifacts = make_formal_artifacts()
    return tuple(
        replace(
            identity,
            purpose=AttemptPurpose.FORMAL_BENCHMARK_TRAINING,
            formal_training_binding=_formal_binding(identity),
            formal_execution=make_formal_execution(
                application=identity.application_config,
                run_plan=identity.run_plan,
                training_sequence=_formal_binding(identity).training_sequence,
                backbone_deployment_hash=identity.backbone_deployment_hash,
                initial_trainable_state=identity.initial_trainable_state,
                initial_library_version=identity.initial_library_version,
                initial_skill_library_state_hash=identity.initial_skill_library_state_hash,
                authoring_authority=identity.authoring_authority,
                attempt_budget=identity.attempt_budget,
                phi_per_cycle_maximum=identity.phi_per_cycle_maximum,
                catalog_freeze_hash=_formal_binding(identity).catalog_freeze_hash,
                artifacts=artifacts,
            ),
            base_model_artifact=artifacts.base_model,
            tokenizer_artifact=artifacts.tokenizer,
            implementation_build=artifacts.implementation,
            tokenizer_identity=artifacts.public_tokenizer,
        )
        for identity in identities
    )


def _formal_manifest_selection(
    tmp_path: Path,
    identities: tuple[PublishedAttemptIdentity, ...],
    *,
    hardware_by_kind: dict[AttemptBuilderKind, ExecutionHardwareIdentity] | None = None,
) -> tuple[
    tuple[PublishedAttemptIdentity, ...],
    FormalRunLedger,
    tuple[PublishedSuccessfulAttemptBundle, ...],
]:
    """Make the declared seven successful slots for formal-aggregate tests."""

    protocol_hash = identities[0].protocol_hash
    freeze = ProtocolFreeze(
        protocol_hash=protocol_hash,
        freeze_id=stable_hash(
            {
                "format": FREEZE_FORMAT,
                "protocol_hash": protocol_hash,
                "result_ingestion_count": 0,
            }
        ),
    )
    frozen = tuple(
        replace(
            identity,
            protocol_freeze_id=freeze.freeze_id,
            exact_input_sha256=stable_hash({"formal-input": identity.builder_kind.value}),
        )
        for identity in identities
    )
    slots = tuple(
        FormalRunSlot(
            builder_kind=identity.builder_kind,
            attempt_id=f"formal-{identity.builder_kind.value}",
            exact_input_sha256=identity.exact_input_sha256,
            public_identity_content_hash=identity.content_hash,
        )
        for identity in frozen
    )
    manifest = FormalRunManifest.create(protocol_freeze=freeze, slots=slots)
    ledger = FormalRunLedger.create(
        directory=tmp_path / "formal-run-ledger",
        manifest=manifest,
    )
    bundles = tuple(
        publish_formal_success_for_identity(
            tmp_path / "formal-source",
            identity,
            ledger,
            hardware_identity=(hardware_by_kind or {}).get(identity.builder_kind),
        )
        for identity in frozen
    )
    return frozen, ledger, bundles


def _with_changed_catalog_freeze(identity: PublishedAttemptIdentity) -> PublishedAttemptIdentity:
    binding = identity.formal_training_binding
    assert binding is not None
    return replace(
        identity,
        formal_training_binding=replace(
            binding,
            catalog_freeze_hash=stable_hash({"catalog": "other"}),
        ),
    )


def test_cross_arm_scientific_identity_allows_only_the_declared_arm_protocol() -> None:
    identities = _identities()

    shared = require_cross_arm_comparable(identities)

    assert shared.initial_trainable_state_hash == identities[0].initial_trainable_state.content_hash
    assert tuple(identity.arm for identity in identities) == tuple(
        arm_protocol_for_builder_kind(kind).arm for kind in AttemptBuilderKind
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda identity: replace(
            identity,
            application_config=replace(
                identity.application_config,
                calibration=replace(identity.application_config.calibration, alpha_0=2.0),
            ),
        ),
        lambda identity: replace(
            identity,
            application_config=replace(
                identity.application_config,
                diagnostics=replace(identity.application_config.diagnostics, window_size=3),
            ),
        ),
        lambda identity: replace(
            identity,
            application_config=replace(
                identity.application_config,
                calibration=replace(identity.application_config.calibration, default_k=2.0),
            ),
        ),
        lambda identity: replace(
            identity,
            application_config=replace(
                identity.application_config,
                evolution=replace(identity.application_config.evolution, lcb_high=0.7),
            ),
        ),
        lambda identity: replace(
            identity,
            application_config=replace(
                identity.application_config,
                evolution=replace(
                    identity.application_config.evolution,
                    split=replace(
                        identity.application_config.evolution.split,
                        min_between_context_mean_gap=0.5,
                    ),
                ),
            ),
        ),
        lambda identity: replace(
            identity,
            application_config=replace(
                identity.application_config,
                authoring_sampling=replace(
                    identity.application_config.authoring_sampling,
                    temperature=0.8,
                ),
            ),
        ),
        lambda identity: replace(
            identity,
            authoring_authority=replace(identity.authoring_authority, license_id="other-license"),
        ),
        lambda identity: replace(
            identity,
            phi_per_cycle_maximum=BudgetVector(input_tokens=5, output_tokens=4, model_calls=1),
        ),
        lambda identity: replace(
            identity,
            initial_trainable_state=TrainableStateIdentity.create(
                backbone_deployment_hash=identity.backbone_deployment_hash,
                forward_adapter_hash="sha256:" + "1" * 64,
                backward_adapter_hash=identity.initial_trainable_state.backward_adapter_hash,
                z_head_hash=identity.initial_trainable_state.z_head_hash,
            ),
        ),
        lambda identity: replace(
            identity,
            initial_library_version=stable_hash({"active-library": "other"}),
        ),
        lambda identity: replace(
            identity,
            initial_skill_library_state_hash=stable_hash({"library-state": "other"}),
        ),
    ],
)
def test_cross_arm_scientific_identity_rejects_each_changed_control(
    mutate: Callable[[PublishedAttemptIdentity], PublishedAttemptIdentity],
) -> None:
    with pytest.raises(ValueError):
        require_cross_arm_comparable(_replace_one(_identities(), mutate))


def test_cross_arm_identity_excludes_artifact_namespace_but_formal_gate_requires_purpose(
    tmp_path: Path,
) -> None:
    identities = _identities()
    changed_namespace = _replace_one(
        identities,
        lambda identity: replace(
            identity,
            application_config=replace(
                identity.application_config,
                trainer=replace(
                    identity.application_config.trainer,
                    execution=replace(
                        identity.application_config.trainer.execution,
                        experiment_id="identity-regression-other-namespace",
                    ),
                ),
            ),
        ),
    )

    require_cross_arm_comparable(changed_namespace)
    formal, ledger, bundles = _formal_manifest_selection(
        tmp_path,
        _formal_identities(changed_namespace),
    )
    require_formal_cross_arm_comparable(
        formal,
        formal_run_ledger=ledger,
        bundles=bundles,
    )


def test_formal_cross_arm_aggregate_requires_its_manifest_selected_successes(
    tmp_path: Path,
) -> None:
    formal, ledger, bundles = _formal_manifest_selection(
        tmp_path, _formal_identities(_identities())
    )

    with pytest.raises(ValueError):
        require_cross_arm_comparable(formal)

    require_formal_cross_arm_comparable(
        formal,
        formal_run_ledger=ledger,
        bundles=bundles,
    )
    mismatched = replace(
        bundles[0],
        public_identity_content_hash=stable_hash({"identity": "replacement"}),
    )
    with pytest.raises(ValueError):
        require_formal_cross_arm_comparable(
            formal,
            formal_run_ledger=ledger,
            bundles=(mismatched, *bundles[1:]),
        )


def test_formal_cross_arm_aggregate_rejects_mixed_hardware_runtime(
    tmp_path: Path,
) -> None:
    mixed_hardware = ExecutionHardwareIdentity(
        accelerator_name="NVIDIA H100",
        compute_capability="9.0",
        visible_device_count=1,
        nvidia_driver_version="570.12",
        cuda_runtime_version="12.8",
        cudnn_version="91000",
        nccl_version="2.27.5",
        kernel_release="6.12.0-test",
        safetensors_version="0.5.3",
    )
    formal, ledger, bundles = _formal_manifest_selection(
        tmp_path,
        _formal_identities(_identities()),
        hardware_by_kind={AttemptBuilderKind.FULL: mixed_hardware},
    )

    with pytest.raises(ValueError):
        require_formal_cross_arm_comparable(
            formal,
            formal_run_ledger=ledger,
            bundles=bundles,
        )


def test_formal_identity_requires_its_schedule_and_path_free_catalog_binding() -> None:
    fixture = _identities()[0]

    with pytest.raises(ValueError):
        replace(fixture, purpose=AttemptPurpose.FORMAL_BENCHMARK_TRAINING)
    with pytest.raises(ValueError):
        replace(fixture, formal_training_binding=_formal_binding(fixture))

    formal = _formal_identities((fixture,))[0]

    assert PublishedAttemptIdentity.from_value(formal.to_value()) == formal
    assert formal.formal_training_binding is not None
    assert formal.formal_training_binding.training_sequence.purpose is SchedulePurpose.IID_TRAINING
    assert (
        formal.formal_training_binding.training_sequence_content_hash
        == formal.formal_training_binding.training_sequence.content_hash
    )
    assert (
        formal.formal_training_binding.training_sequence.ordered_task_ids_hash
        == formal.ordered_task_sequence_hash
    )

    malformed = formal.formal_training_binding.to_value()
    malformed["training_sequence_content_hash"] = stable_hash({"wrong": "sequence"})
    with pytest.raises(ValueError):
        FormalTrainingBinding.from_value(malformed)
    with pytest.raises(ValueError):
        replace(
            formal,
            initial_skill_library_state_hash=stable_hash({"library-state": "other"}),
        )


def test_formal_identity_requires_a_schedule_that_fills_its_exact_run_plan() -> None:
    fixture = _identities()[0]
    binding = _formal_binding(fixture)
    incomplete_sequence = replace(
        binding.training_sequence,
        task_count=1,
        benchmark_counts=(BenchmarkTaskCount(benchmark=Benchmark.HOTPOT_QA, count=1),),
    )

    with pytest.raises(ValueError):
        replace(
            fixture,
            purpose=AttemptPurpose.FORMAL_BENCHMARK_TRAINING,
            formal_training_binding=replace(binding, training_sequence=incomplete_sequence),
        )


def test_formal_cross_arm_identity_rejects_a_changed_catalog_freeze() -> None:
    identities = _formal_identities(_identities())
    with pytest.raises(ValueError):
        _replace_one(identities, _with_changed_catalog_freeze)


def test_formal_training_binding_rejects_a_non_training_schedule() -> None:
    identity = _identities()[0]
    training = _formal_binding(identity).training_sequence

    with pytest.raises(ValueError):
        FormalTrainingBinding(
            training_sequence=replace(training, purpose=SchedulePurpose.IID_EVALUATION),
            catalog_freeze_hash=stable_hash({"catalog": "identity-regression"}),
        )
