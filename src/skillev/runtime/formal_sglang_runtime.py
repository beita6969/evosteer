"""Fail-closed external SGLang dependencies for a formal distributed run."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from skillev.application import FormalRuntimeDependencies, FormalSharedRuntimeDependencies
from skillev.contracts import JsonValue, normalize_json
from skillev.evolution import (
    AuthoringCallMaximum,
    AuthoringSamplingConfig,
    EvolutionConfig,
    ExternalSGLangSkillAuthor,
)
from skillev.policy import AdapterRole, PolicyBackbone
from skillev.policy.hf_backbone import QwenMultimodalPolicyBackbone, QwenPolicyBackbone
from skillev.rollout import PolicySnapshot
from skillev.rollout.external_sglang import (
    ExternalSGLangRolloutConfig,
    ExternalSGLangRolloutGenerator,
)
from skillev.training import RolloutWorkflowBinding, RolloutWorkflowResources
from skillev.training.distributed_ttb import DistributedTTBGradientCoordinator
from skillev.training.performance_config import TrainingPerformanceConfig

from .budget_ledger import BudgetLedger
from .emitter import RuntimeEventEmitter
from .request_journal import DurableRequestJournal
from .sglang_gateway import SGLangGateway, SGLangGatewayConfig
from .sglang_pool import SGLangActorPool
from .sglang_step_publisher import SGLangStepAdapterPublisher


@dataclass(frozen=True, slots=True)
class FormalSGLangRuntimeBinding:
    """Private deployment values shared by all three Protocol 10 methods."""

    gateway: SGLangGatewayConfig
    rollout: ExternalSGLangRolloutConfig
    workflow: RolloutWorkflowBinding
    adapter_export_root: Path
    adapter_namespace: str
    adapter_keep_recent: int = 3
    performance: TrainingPerformanceConfig | None = None
    actor_replicas: tuple[SGLangGatewayConfig, ...] = ()
    author_replicas: tuple[SGLangGatewayConfig, ...] = ()
    request_journal_path: Path | None = None

    def __post_init__(self) -> None:
        if self.gateway.endpoint_base.rstrip("/") != self.rollout.endpoint_base.rstrip("/"):
            raise ValueError("SGLang control and rollout endpoints differ")
        if self.rollout.transport_worker_threads != self.workflow.transport_worker_threads:
            raise ValueError("SGLang transport workers differ from the workflow binding")
        if self.request_journal_path is not None and not self.request_journal_path.is_absolute():
            raise ValueError("request journal must be an absolute private path")
        if not self.adapter_export_root.is_absolute():
            raise ValueError("formal adapter export root must be absolute")
        if type(self.adapter_keep_recent) is not int or self.adapter_keep_recent < 1:
            raise ValueError("formal adapter retention must be positive")

    def to_value(self) -> dict[str, JsonValue]:
        """Serialize the exact private SGLang execution binding."""

        result: dict[str, JsonValue] = {
            "adapter_export_root": str(self.adapter_export_root),
            "adapter_keep_recent": self.adapter_keep_recent,
            "adapter_namespace": self.adapter_namespace,
            "gateway": self.gateway.to_value(),
            "rollout": self.rollout.to_value(),
            "workflow": self.workflow.to_value(),
        }
        if self.performance is not None:
            result["performance"] = self.performance.to_value()
        for name in ("actor_replicas", "author_replicas"):
            replicas = getattr(self, name)
            if replicas:
                result[name] = [v.to_value() for v in replicas]
        if self.request_journal_path is not None:
            result["request_journal_path"] = str(self.request_journal_path)
        return result

    @classmethod
    def from_value(cls, value: object) -> FormalSGLangRuntimeBinding:
        normalized = normalize_json(value)
        fields = {
            "adapter_export_root",
            "adapter_keep_recent",
            "adapter_namespace",
            "gateway",
            "rollout",
            "workflow",
        }
        if isinstance(normalized, dict):
            fields.update(
                set(normalized)
                & {"performance", "actor_replicas", "author_replicas", "request_journal_path"}
            )
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("formal SGLang runtime binding has incompatible fields")
        export_root = normalized["adapter_export_root"]
        namespace = normalized["adapter_namespace"]
        keep_recent = normalized["adapter_keep_recent"]
        if type(export_root) is not str or type(namespace) is not str:
            raise TypeError("formal SGLang runtime path and namespace must be text")
        if type(keep_recent) is not int:
            raise TypeError("formal SGLang adapter retention must be an integer")
        journal = normalized.get("request_journal_path")
        if journal is not None and (
            not isinstance(journal, str) or not Path(journal).is_absolute()
        ):
            raise ValueError("request journal must be an absolute private path")
        replicas = {}
        for name in ("actor_replicas", "author_replicas"):
            values = normalized.get(name, [])
            if not isinstance(values, list):
                raise TypeError("replica configurations must be a list")
            replicas[name] = tuple(SGLangGatewayConfig.from_value(v) for v in values)
        return cls(
            **replicas,
            request_journal_path=None if journal is None else Path(journal),
            gateway=SGLangGatewayConfig.from_value(normalized["gateway"]),
            rollout=ExternalSGLangRolloutConfig.from_value(normalized["rollout"]),
            workflow=RolloutWorkflowBinding.from_value(normalized["workflow"]),
            adapter_export_root=Path(export_root),
            adapter_namespace=namespace,
            adapter_keep_recent=keep_recent,
            performance=(
                TrainingPerformanceConfig.from_value(normalized["performance"])
                if "performance" in normalized
                else None
            ),
        )


@dataclass(slots=True)
class BoundFormalSGLangRuntime:
    """One coordinator-owned gateway and its mandatory application dependencies."""

    binding: FormalSGLangRuntimeBinding
    gateway: SGLangGateway
    resources: RolloutWorkflowResources
    gradient_preparer: DistributedTTBGradientCoordinator | None

    @classmethod
    def build(
        cls,
        *,
        binding: FormalSGLangRuntimeBinding,
        gradient_preparer: DistributedTTBGradientCoordinator,
    ) -> BoundFormalSGLangRuntime:
        if not isinstance(binding, FormalSGLangRuntimeBinding):
            raise TypeError("formal SGLang runtime requires a private binding")
        if not isinstance(gradient_preparer, DistributedTTBGradientCoordinator):
            raise TypeError("formal SGLang runtime requires the distributed coordinator")
        if binding.performance is not None:
            binding.performance.configure_process()
            profile = binding.performance
            binding = replace(
                binding,
                workflow=profile.workflow(),
                rollout=replace(
                    binding.rollout, transport_worker_threads=profile.transport_threads
                ),
            )
            gradient_preparer.coordinator_participates = profile.coordinator_participates
            gradient_preparer.pipeline_mode = profile.pipeline_mode
            gradient_preparer.stream_directory = (
                Path(tempfile.gettempdir()) / "skillev-gradient-stream"
            )
        return cls(
            binding=binding,
            gateway=(
                SGLangActorPool([SGLangGateway(v) for v in binding.actor_replicas])
                if binding.actor_replicas
                else SGLangGateway(binding.gateway)
            ),
            resources=RolloutWorkflowResources(binding.workflow),
            gradient_preparer=gradient_preparer,
        )

    @classmethod
    def build_local_diagnostic(
        cls, *, binding: FormalSGLangRuntimeBinding
    ) -> BoundFormalSGLangRuntime:
        """Explicit single-owner diagnostic, not a formal/distributed fallback.

        ``None`` selects the existing local prepare_ttb_step/LocalGradientStepStream
        implementation in TrainingUpdater. Formal dependencies remain unavailable.
        """
        if not isinstance(binding, FormalSGLangRuntimeBinding):
            raise TypeError("local diagnostic requires a private SGLang binding")
        if binding.actor_replicas or binding.author_replicas:
            raise ValueError("local diagnostic requires one explicit external service")
        if binding.performance is None or not binding.performance.coordinator_participates:
            raise ValueError("local diagnostic must declare its participating gradient owner")
        binding.performance.configure_process()
        binding = replace(
            binding,
            workflow=binding.performance.workflow(),
            rollout=replace(
                binding.rollout, transport_worker_threads=binding.performance.transport_threads
            ),
        )
        return cls(
            binding,
            SGLangGateway(binding.gateway),
            RolloutWorkflowResources(binding.workflow),
            None,
        )

    def dependencies(self) -> FormalRuntimeDependencies:
        if not isinstance(self.gradient_preparer, DistributedTTBGradientCoordinator):
            raise TypeError("formal dependencies require the distributed TTB coordinator")
        shared = self.shared_dependencies()
        return FormalRuntimeDependencies(
            rollout_generator_factory=shared.rollout_generator_factory,
            gradient_preparer=self.gradient_preparer,
            workflow_resources=shared.workflow_resources,
            step_adapter_publisher_factory=shared.step_adapter_publisher_factory,
            skill_author_factory=shared.skill_author_factory,
        )

    def shared_dependencies(self) -> FormalSharedRuntimeDependencies:
        binding = self.binding
        gateway = self.gateway
        workflow_resources = self.resources

        def generator_factory(
            backbone: PolicyBackbone,
            resources: RolloutWorkflowResources,
        ) -> ExternalSGLangRolloutGenerator:
            if resources is not workflow_resources:
                raise ValueError("formal generator received another workflow resource set")

            if binding.performance is not None:
                if not isinstance(backbone, QwenPolicyBackbone | QwenMultimodalPolicyBackbone):
                    raise TypeError("performance profile requires a Qwen policy backbone")
                backbone.configure_performance(binding.performance)

            def snapshot() -> PolicySnapshot:
                return PolicySnapshot.create(
                    backbone_id=backbone.backbone_id,
                    forward_adapter_version=backbone.adapter_version(AdapterRole.FORWARD_POLICY),
                    tokenizer_id=backbone.tokenizer.tokenizer_id,
                    backend_id="sglang-native-exact-token",
                    initial_trainable_state_hash=backbone.initial_trainable_state_hash,
                )

            return ExternalSGLangRolloutGenerator(
                config=binding.rollout,
                tokenizer=backbone.tokenizer,
                gateway=gateway,
                snapshot_provider=snapshot,
                request_journal=(
                    DurableRequestJournal(binding.request_journal_path)
                    if binding.request_journal_path is not None
                    else None
                ),
            )

        def publisher_factory(backbone: PolicyBackbone) -> SGLangStepAdapterPublisher:
            return SGLangStepAdapterPublisher(
                backbone=backbone,
                gateway=gateway,
                export_root=binding.adapter_export_root,
                adapter_namespace=binding.adapter_namespace,
                keep_recent=binding.adapter_keep_recent,
            )

        def skill_author_factory(
            backbone: PolicyBackbone,
            config: EvolutionConfig,
            sampling: AuthoringSamplingConfig,
            ledger: BudgetLedger,
            emitter: RuntimeEventEmitter,
            maximum: AuthoringCallMaximum,
        ) -> ExternalSGLangSkillAuthor:
            return ExternalSGLangSkillAuthor(
                tokenizer=backbone.tokenizer,
                config=config,
                sampling=sampling,
                ledger=ledger,
                emitter=emitter,
                maximum=maximum,
                rollout_config=binding.rollout,
                gateway=gateway,
                replica_configs=binding.author_replicas,
                request_journal=(
                    DurableRequestJournal(binding.request_journal_path)
                    if binding.request_journal_path is not None
                    else None
                ),
            )

        return FormalSharedRuntimeDependencies(
            rollout_generator_factory=generator_factory,
            workflow_resources=workflow_resources,
            step_adapter_publisher_factory=publisher_factory,
            skill_author_factory=skill_author_factory,
        )


__all__ = ["BoundFormalSGLangRuntime", "FormalSGLangRuntimeBinding"]
