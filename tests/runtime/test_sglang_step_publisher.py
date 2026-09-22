from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from skillev.runtime.sglang_gateway import (
    AdapterGeneration,
    PreparedAdapterSwap,
)
from skillev.runtime.sglang_step_publisher import SGLangStepAdapterPublisher


@dataclass(slots=True)
class _Backbone:
    saves: int = 0

    def save_checkpoint(self, directory: str) -> None:
        self.saves += 1
        adapter = Path(directory) / "forward_adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")


@dataclass(slots=True)
class _Gateway:
    generation: AdapterGeneration = field(
        default_factory=lambda: AdapterGeneration(0, "theta", "initial")
    )
    prepared: PreparedAdapterSwap | None = None

    def prepare_supervisor_adapter(
        self,
        *,
        adapter_path: str,
        adapter_revision: str,
    ) -> PreparedAdapterSwap:
        candidate = AdapterGeneration(
            self.generation.generation + 1,
            f"theta_{adapter_revision}",
            adapter_revision,
        )
        self.prepared = PreparedAdapterSwap(
            previous=self.generation,
            previous_path=None,
            candidate=candidate,
            candidate_path=adapter_path,
        )
        self.generation = candidate
        return self.prepared

    def commit_supervisor_adapter(self, prepared: PreparedAdapterSwap) -> AdapterGeneration:
        assert prepared == self.prepared
        self.prepared = None
        return self.generation

    def rollback_supervisor_adapter(self, prepared: PreparedAdapterSwap) -> None:
        assert prepared == self.prepared
        self.generation = prepared.previous
        self.prepared = None

    def restore_supervisor_adapter(
        self,
        *,
        adapter_path: str,
        adapter_revision: str,
    ) -> AdapterGeneration:
        candidate = AdapterGeneration(
            self.generation.generation + 1,
            f"theta_{adapter_revision}",
            adapter_revision,
        )
        self.generation = candidate
        return candidate


def _publisher(tmp_path: Path, *, keep_recent: int = 3):
    backbone = _Backbone()
    gateway = _Gateway()
    publisher = SGLangStepAdapterPublisher(
        backbone=backbone,  # type: ignore[arg-type]
        gateway=gateway,  # type: ignore[arg-type]
        export_root=tmp_path / "exports",
        adapter_namespace="formal-full-attempt-1",
        keep_recent=keep_recent,
    )
    return publisher, backbone, gateway


def test_prepare_is_reusable_and_rollback_restores_the_prior_generation(tmp_path: Path) -> None:
    publisher, backbone, gateway = _publisher(tmp_path)

    prepared = publisher.prepare(optimizer_step=1, policy_snapshot_id="policy@1")
    assert prepared.export_directory.name == "formal-full-attempt-1-step-00000001"
    assert prepared.gateway_swap.candidate.adapter_revision.startswith(
        "formal-full-attempt-1-step-00000001-"
    )
    assert gateway.generation == prepared.gateway_swap.candidate
    publisher.rollback(prepared)
    assert gateway.generation == prepared.gateway_swap.previous

    retried = publisher.prepare(optimizer_step=1, policy_snapshot_id="policy@1")
    publisher.commit(retried)
    assert backbone.saves == 1


def test_restore_reuses_the_immutable_export_and_binds_the_revision(tmp_path: Path) -> None:
    publisher, backbone, gateway = _publisher(tmp_path)

    first = publisher.restore(optimizer_step=1, policy_snapshot_id="policy@1")
    restored = publisher.restore(optimizer_step=1, policy_snapshot_id="policy@1")

    assert restored.adapter_revision == first.adapter_revision
    assert gateway.generation == restored
    assert backbone.saves == 1


def test_commit_retains_only_recent_immutable_exports(tmp_path: Path) -> None:
    publisher, _, _ = _publisher(tmp_path, keep_recent=2)

    for step in range(1, 4):
        prepared = publisher.prepare(
            optimizer_step=step,
            policy_snapshot_id=f"policy@{step}",
        )
        publisher.commit(prepared)

    assert sorted(path.name for path in publisher.export_root.iterdir()) == [
        "formal-full-attempt-1-step-00000002",
        "formal-full-attempt-1-step-00000003",
    ]


def test_adapter_namespace_is_required_and_safe(tmp_path: Path) -> None:
    publisher, _, _ = _publisher(tmp_path)
    assert publisher.adapter_namespace == "formal-full-attempt-1"
    try:
        SGLangStepAdapterPublisher(
            backbone=_Backbone(),  # type: ignore[arg-type]
            gateway=_Gateway(),  # type: ignore[arg-type]
            export_root=tmp_path / "unsafe",
            adapter_namespace="contains spaces",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe adapter namespace must be rejected")


def test_existing_step_rejects_another_policy_identity(tmp_path: Path) -> None:
    publisher, _, _ = _publisher(tmp_path)
    prepared = publisher.prepare(optimizer_step=1, policy_snapshot_id="policy@1")
    publisher.rollback(prepared)

    try:
        publisher.prepare(optimizer_step=1, policy_snapshot_id="different")
    except FileExistsError:
        pass
    else:
        raise AssertionError("one adapter step must not be replaced by another policy")
