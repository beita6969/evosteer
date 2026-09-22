from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from skillev.application import FormalRuntimeDependencies, SKILLEVApplication
from skillev.evolution import ExternalSGLangSkillAuthor
from skillev.rollout.external_sglang import ExternalSGLangRolloutGenerator
from skillev.runtime.sglang_step_publisher import SGLangStepAdapterPublisher
from skillev.training import RolloutWorkflowBinding, RolloutWorkflowResources
from skillev.training.distributed_ttb import (
    DistributedTTBGradientCoordinator,
    DistributedTTBTopology,
)


def _dependencies() -> FormalRuntimeDependencies:
    coordinator = DistributedTTBGradientCoordinator(
        DistributedTTBTopology(rank=0, world_size=2, local_rank=0, backend="nccl")
    )
    resources = RolloutWorkflowResources(RolloutWorkflowBinding())
    return FormalRuntimeDependencies(
        rollout_generator_factory=lambda backbone, workflow: None,  # type: ignore[return-value]
        gradient_preparer=coordinator,
        workflow_resources=resources,
        step_adapter_publisher_factory=lambda backbone: None,  # type: ignore[return-value]
        skill_author_factory=lambda *args: None,  # type: ignore[return-value]
    )


def test_formal_runtime_dependencies_reject_local_gradient_fallback() -> None:
    resources = RolloutWorkflowResources(RolloutWorkflowBinding())
    with pytest.raises(TypeError):
        FormalRuntimeDependencies(
            rollout_generator_factory=lambda backbone, workflow: None,  # type: ignore[return-value]
            gradient_preparer=object(),  # type: ignore[arg-type]
            workflow_resources=resources,
            step_adapter_publisher_factory=lambda backbone: None,  # type: ignore[return-value]
            skill_author_factory=lambda *args: None,  # type: ignore[return-value]
        )


def test_formal_runtime_binding_requires_external_generation_gradient_and_publisher() -> None:
    dependencies = _dependencies()
    application = SimpleNamespace(
        generator=object.__new__(ExternalSGLangRolloutGenerator),
        training_loop=SimpleNamespace(
            gradient_preparer=dependencies.gradient_preparer,
            workflow_resources=dependencies.workflow_resources,
        ),
        evolution_loop=SimpleNamespace(
            step_adapter_publisher=object.__new__(SGLangStepAdapterPublisher),
            step_transaction_journal=object(),
            author=object.__new__(ExternalSGLangSkillAuthor),
        ),
    )

    dependencies.require_bound(cast(SKILLEVApplication, application))

    application.generator = object()
    with pytest.raises(TypeError):
        dependencies.require_bound(cast(SKILLEVApplication, application))

    application.generator = object.__new__(ExternalSGLangRolloutGenerator)
    application.evolution_loop.step_transaction_journal = None
    with pytest.raises(TypeError):
        dependencies.require_bound(cast(SKILLEVApplication, application))

    application.evolution_loop.step_transaction_journal = object()
    application.evolution_loop.author = object()
    with pytest.raises(TypeError):
        dependencies.require_bound(cast(SKILLEVApplication, application))
