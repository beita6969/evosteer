from __future__ import annotations

from pathlib import Path

import pytest

from skillev.policy import AdapterRole
from skillev.rollout.external_sglang import (
    ExternalSGLangRolloutConfig,
    ExternalSGLangRolloutGenerator,
)
from skillev.runtime.formal_sglang_runtime import (
    BoundFormalSGLangRuntime,
    FormalSGLangRuntimeBinding,
)
from skillev.runtime.sglang_gateway import SGLangGatewayConfig
from skillev.runtime.sglang_step_publisher import SGLangStepAdapterPublisher
from skillev.training import RolloutWorkflowBinding
from skillev.training.distributed_ttb import (
    DistributedTTBGradientCoordinator,
    DistributedTTBTopology,
)


class _Tokenizer:
    tokenizer_id = "tokenizer"

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return ""


class _Backbone:
    backbone_id = "backbone"
    tokenizer = _Tokenizer()
    initial_trainable_state_hash = "sha256:" + "1" * 64

    def adapter_version(self, role: AdapterRole) -> str:
        assert role is AdapterRole.FORWARD_POLICY
        return "forward@0"


def _binding(tmp_path: Path) -> FormalSGLangRuntimeBinding:
    workflow = RolloutWorkflowBinding(
        max_resident_trajectories=20,
        max_inflight_model_requests=12,
        max_inflight_environment_calls=4,
        max_inflight_terminal_evaluations=4,
        max_inflight_process_graders=2,
        transport_worker_threads=12,
    )
    return FormalSGLangRuntimeBinding(
        gateway=SGLangGatewayConfig(
            endpoint_base="http://127.0.0.1:30000",
            base_model="qwen-base",
            supervisor_adapter="initial-forward",
        ),
        rollout=ExternalSGLangRolloutConfig(
            endpoint_base="http://127.0.0.1:30000",
            transport_worker_threads=12,
        ),
        workflow=workflow,
        adapter_export_root=(tmp_path / "adapters").resolve(),
        adapter_namespace="run-full-attempt-1",
    )


def test_formal_runtime_constructs_exact_external_dependencies(tmp_path: Path) -> None:
    coordinator = DistributedTTBGradientCoordinator(
        DistributedTTBTopology(rank=0, world_size=2, local_rank=0, backend="nccl")
    )
    runtime = BoundFormalSGLangRuntime.build(
        binding=_binding(tmp_path),
        gradient_preparer=coordinator,
    )
    dependencies = runtime.dependencies()
    backbone = _Backbone()

    generator = dependencies.rollout_generator_factory(  # type: ignore[arg-type]
        backbone,
        dependencies.workflow_resources,
    )
    publisher = dependencies.step_adapter_publisher_factory(backbone)  # type: ignore[arg-type]

    assert isinstance(generator, ExternalSGLangRolloutGenerator)
    assert generator.snapshot().backend_id == "sglang-native-exact-token"
    assert isinstance(publisher, SGLangStepAdapterPublisher)
    assert callable(dependencies.skill_author_factory)
    assert dependencies.gradient_preparer is coordinator
    assert dependencies.workflow_resources.binding.max_inflight_model_requests == 12
    generator.close()


def test_formal_runtime_rejects_endpoint_or_thread_mismatch(tmp_path: Path) -> None:
    binding = _binding(tmp_path)
    with pytest.raises(ValueError):
        FormalSGLangRuntimeBinding(
            gateway=binding.gateway,
            rollout=ExternalSGLangRolloutConfig(
                endpoint_base="http://127.0.0.1:30001",
                transport_worker_threads=12,
            ),
            workflow=binding.workflow,
            adapter_export_root=binding.adapter_export_root,
            adapter_namespace=binding.adapter_namespace,
        )
    with pytest.raises(ValueError):
        FormalSGLangRuntimeBinding(
            gateway=binding.gateway,
            rollout=ExternalSGLangRolloutConfig(
                endpoint_base=binding.gateway.endpoint_base,
                transport_worker_threads=8,
            ),
            workflow=binding.workflow,
            adapter_export_root=binding.adapter_export_root,
            adapter_namespace=binding.adapter_namespace,
        )


def test_formal_runtime_binding_round_trips_private_execution_controls(tmp_path: Path) -> None:
    binding = _binding(tmp_path)

    restored = FormalSGLangRuntimeBinding.from_value(binding.to_value())

    assert restored == binding


def test_explicit_actor_and_author_pools_survive_runtime_binding_restore(tmp_path):
    from dataclasses import replace

    from skillev.runtime.sglang_pool import SGLangActorPool

    base = _binding(tmp_path)
    actor_b = replace(base.gateway, endpoint_base="http://127.0.0.1:30001")
    author = replace(base.gateway, endpoint_base="http://127.0.0.1:30002")
    binding = replace(
        base,
        actor_replicas=(base.gateway, actor_b),
        author_replicas=(author,),
        request_journal_path=(tmp_path / "requests.sqlite3").resolve(),
    )
    restored = FormalSGLangRuntimeBinding.from_value(binding.to_value())
    assert restored == binding
    runtime = BoundFormalSGLangRuntime.build(
        binding=restored,
        gradient_preparer=DistributedTTBGradientCoordinator(
            DistributedTTBTopology(rank=0, world_size=2, local_rank=0, backend="nccl")
        ),
    )
    assert isinstance(runtime.gateway, SGLangActorPool)
    assert len(runtime.gateway.members) == 2
    generator = runtime.dependencies().rollout_generator_factory(_Backbone(), runtime.resources)
    assert generator.request_journal.path == binding.request_journal_path
    generator.close()
