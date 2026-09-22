from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from skillev_private.experiments.protocol_v10_application_input import (
    SKILLFLOW_PARITY_CONTRACT,
    SKILLFLOW_UPSTREAM_REVISION,
    BayesianImproveApplicationContract,
    ExactSkillFlowApplicationContract,
    require_protocol_v10_method_contract,
)
from skillev_private.experiments.protocol_v10_attempt_input import (
    ProtocolV10AdmissionMode,
    ProtocolV10AttemptInput,
)
from skillev_private.experiments.protocol_v10_attempt_supervisor import (
    PROTOCOL_V10_ATTEMPT_WORKER_MODULE,
    ProtocolV10TorchrunLaunch,
    ProtocolV10TorchrunSupervisor,
)
from skillev_private.experiments.protocol_v10_launch_package import (
    ProtocolV10LaunchPackage,
    ProtocolV10LaunchSlot,
)

from skillev.experiments import FormalApplicationV10, FormalMethodV10
from skillev.rollout.external_sglang import ExternalSGLangRolloutConfig
from skillev.runtime.formal_sglang_runtime import FormalSGLangRuntimeBinding
from skillev.runtime.formal_storage import FormalStorageBinding, FormalStorageBudget
from skillev.runtime.gpu_topology import GPUObservation, ThreeGPURolePolicy
from skillev.runtime.sglang_gateway import SGLangGatewayConfig
from skillev.training import RolloutWorkflowBinding


def _input(
    tmp_path: Path,
    *,
    admission_mode: str = "formal",
    smoke_steps: int | None = None,
    attempt_name: str = "attempt",
) -> ProtocolV10AttemptInput:
    attempt_root = (
        tmp_path / "non-formal-smoke" / attempt_name
        if admission_mode == "integration-smoke"
        else tmp_path / attempt_name
    ).resolve()
    output_root = attempt_root.parent
    workflow = RolloutWorkflowBinding(
        max_resident_trajectories=20,
        max_inflight_model_requests=12,
        max_inflight_environment_calls=4,
        max_inflight_terminal_evaluations=4,
        max_inflight_process_graders=2,
        transport_worker_threads=12,
    )
    return ProtocolV10AttemptInput(
        method=FormalMethodV10.BAYESIAN_IMPROVE_FULL,
        protocol_config=(tmp_path / "protocol.yaml").resolve(),
        experiment_config=(tmp_path / "experiment.yaml").resolve(),
        catalog_root=(tmp_path / "catalog").resolve(),
        session_deployments=(tmp_path / "sessions.json").resolve(),
        application_input=(tmp_path / "application.json").resolve(),
        hardware=ThreeGPURolePolicy(
            inference_physical_index=5,
            coordinator_physical_index=4,
            gradient_primary_physical_index=6,
            forbidden_physical_indices=(0, 1, 2, 3, 7),
            minimum_free_memory_mib=70_000,
        ),
        sglang=FormalSGLangRuntimeBinding(
            gateway=SGLangGatewayConfig(
                endpoint_base="http://127.0.0.1:30000",
                base_model="qwen-base",
                supervisor_adapter="formal-initial",
            ),
            rollout=ExternalSGLangRolloutConfig(
                endpoint_base="http://127.0.0.1:30000",
                transport_worker_threads=12,
            ),
            workflow=workflow,
            adapter_export_root=(attempt_root / "adapters").resolve(),
            adapter_namespace="protocol-v10-full-attempt-1",
        ),
        storage=FormalStorageBinding(
            output_root=output_root,
            temporary_root=output_root,
            budget=FormalStorageBudget(
                estimated_attempt_peak_bytes=1,
                rollback_checkpoint_bytes=1,
                atomic_publication_bytes=1,
                minimum_free_inodes=1,
            ),
        ),
        attempt_root=attempt_root,
        admission_mode=ProtocolV10AdmissionMode(admission_mode),
        integration_smoke_steps=smoke_steps,
    )


def _gpu_observations() -> tuple[GPUObservation, ...]:
    return tuple(
        GPUObservation(
            physical_index=index,
            name="H800",
            uuid=f"GPU-physical-{index}",
            memory_total_mib=80_000,
            memory_used_mib=0,
            utilization_percent=0,
            compute_pids=(1005,) if index == 5 else (),
        )
        for index in range(8)
    )


def test_private_protocol_v10_attempt_input_round_trips(tmp_path: Path) -> None:
    expected = _input(tmp_path)

    assert ProtocolV10AttemptInput.from_value(expected.to_value()) == expected


def test_private_protocol_v10_bounded_smoke_round_trips(tmp_path: Path) -> None:
    expected = _input(tmp_path, admission_mode="integration-smoke", smoke_steps=2)

    assert ProtocolV10AttemptInput.from_value(expected.to_value()) == expected


def test_private_protocol_v10_formal_attempt_rejects_a_smoke_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _input(tmp_path, smoke_steps=1)


def test_torchrun_launch_derives_method_and_gpu_mapping_from_exact_input(tmp_path: Path) -> None:
    expected = _input(tmp_path, admission_mode="integration-smoke", smoke_steps=2)
    exact_path = (tmp_path / "exact.json").resolve()
    exact_path.write_text(json.dumps(expected.to_value()), encoding="utf-8")

    launch = ProtocolV10TorchrunLaunch.from_exact_input(exact_path)

    assert launch.method is expected.method
    assert launch.worker_module == PROTOCOL_V10_ATTEMPT_WORKER_MODULE
    assert launch.training_physical_indices == (4, 6)
    assert (
        launch.environment({}, observations=_gpu_observations())["CUDA_VISIBLE_DEVICES"]
        == "GPU-physical-4,GPU-physical-6"
    )
    environment = launch.environment({}, observations=_gpu_observations())
    assert environment["SKILLEV_DISABLE_GRADIENT_STANDBY"] == "1"
    assert environment["SKILLEV_TRAINING_GPU_MAP"] == "4,6"
    assert environment["SKILLEV_TRAINING_CUDA_VISIBLE_DEVICES"] == ("GPU-physical-4,GPU-physical-6")
    assert environment["SKILLEV_EXTERNAL_INFERENCE_GPU_UUID"] == "GPU-physical-5"


def test_protocol_v10_method_application_tags_reject_cross_method_aliases() -> None:
    baseline = ExactSkillFlowApplicationContract(
        upstream_revision=SKILLFLOW_UPSTREAM_REVISION,
        parity_contract=SKILLFLOW_PARITY_CONTRACT,
        official_configuration_projection={"kl_coefficient": 0.01},
    )
    no_calibration = BayesianImproveApplicationContract(
        method=FormalMethodV10.BAYESIAN_IMPROVE_NO_CALIBRATION,
        calibration_enabled=False,
    )

    require_protocol_v10_method_contract(FormalMethodV10.SKILLFLOW_BASELINE, baseline)
    require_protocol_v10_method_contract(
        FormalMethodV10.BAYESIAN_IMPROVE_NO_CALIBRATION,
        no_calibration,
    )
    with pytest.raises(TypeError):
        require_protocol_v10_method_contract(FormalMethodV10.SKILLFLOW_BASELINE, no_calibration)
    with pytest.raises(ValueError):
        require_protocol_v10_method_contract(
            FormalMethodV10.BAYESIAN_IMPROVE_FULL,
            no_calibration,
        )


def test_torchrun_supervisor_does_not_retry_a_failed_attempt(tmp_path: Path) -> None:
    exact = _input(tmp_path, admission_mode="integration-smoke", smoke_steps=2)
    exact.storage.output_root.mkdir(parents=True)
    exact_path = (tmp_path / "exact.json").resolve()
    exact_path.write_text(json.dumps(exact.to_value()), encoding="utf-8")
    launches: list[ProtocolV10TorchrunLaunch] = []

    def run(launch: ProtocolV10TorchrunLaunch) -> int:
        launches.append(launch)
        return 86

    with pytest.raises(RuntimeError):
        ProtocolV10TorchrunSupervisor(
            ProtocolV10TorchrunLaunch.from_exact_input(exact_path),
            process_runner=run,
        ).run()

    assert len(launches) == 1
    assert launches[0].training_physical_indices == (4, 6)


def test_formal_exact_input_cannot_bypass_the_launch_package(tmp_path: Path) -> None:
    exact = _input(tmp_path)
    exact_path = (tmp_path / "formal-exact.json").resolve()
    exact_path.write_text(json.dumps(exact.to_value()), encoding="utf-8")

    with pytest.raises(ValueError):
        ProtocolV10TorchrunLaunch.from_exact_input(exact_path)


def test_formal_launch_is_derived_from_the_unchanged_frozen_package(tmp_path: Path) -> None:
    applications = (
        FormalApplicationV10.EXACT_SKILLFLOW_BASELINE,
        FormalApplicationV10.BAYESIAN_IMPROVE_FULL,
        FormalApplicationV10.BAYESIAN_IMPROVE_NO_CALIBRATION,
    )
    slots: list[ProtocolV10LaunchSlot] = []
    for method, application in zip(FormalMethodV10, applications, strict=True):
        source = _input(tmp_path / method.value, attempt_name=f"formal-{method.value}")
        exact = replace(
            source,
            method=method,
            sglang=replace(source.sglang, adapter_namespace=f"formal-{method.value}"),
        )
        exact_path = (tmp_path / f"{method.value}.json").resolve()
        exact_path.write_text(json.dumps(exact.to_value()), encoding="utf-8")
        digest = hashlib.blake2b(exact_path.read_bytes(), digest_size=32).hexdigest()
        slots.append(
            ProtocolV10LaunchSlot(
                method=method,
                application=application,
                attempt_id=exact.attempt_root.name,
                exact_input_path=exact_path,
                exact_input_digest=f"blake2b:{digest}",
                attempt_root=exact.attempt_root,
                checkpoint_root=exact.attempt_root / "checkpoints",
                adapter_namespace=f"formal-{method.value}",
                method_contract={"method": method.value},
            )
        )
    package = ProtocolV10LaunchPackage(
        run_group_id="formal-seed-0",
        source_commit="a" * 40,
        protocol_identity="protocol-v10",
        formal_experiment_identity="experiment-v1",
        model_revision="qwen-revision",
        tokenizer_identity="tokenizer-revision",
        initial_checkpoint_identity="initial-checkpoint",
        initial_skill_library_identity="initial-skill-library",
        training_catalog_identity="catalog-v10",
        ordered_episode_sequence_identity="sequence-seed-0",
        sglang_endpoint="http://127.0.0.1:30000",
        sglang_model_identity="qwen-base",
        hardware=_input(tmp_path / "hardware").hardware,
        slots=tuple(slots),
    )
    package_path = (tmp_path / "launch-package.json").resolve()
    package.write_once(package_path)

    launch = ProtocolV10TorchrunLaunch.from_launch_package(
        package_path,
        method=FormalMethodV10.SKILLFLOW_BASELINE,
    )

    assert launch.exact_input == slots[0].exact_input_path
    assert launch.method is FormalMethodV10.SKILLFLOW_BASELINE
