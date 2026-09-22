from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from training.formal_checkpoint import FormalCheckpointStore
from training.gflownet_trainer import GFlowNetTrainer
from training.step_transaction import StepPhase, StepTransaction


def _store(root: Path, *, keep_recent: int = 3) -> FormalCheckpointStore:
    return FormalCheckpointStore(
        root,
        keep_recent=keep_recent,
        initial_checkpoint_estimate_bytes=0,
        minimum_free_headroom_bytes=0,
    )


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict(
            {
                "phi": torch.nn.Linear(2, 2, bias=False),
                "theta": torch.nn.Linear(2, 2, bias=False),
            }
        )


@dataclass
class FakeWorkspace:
    skills_dir: Path
    _skills: dict[str, object] = field(default_factory=dict)

    def save_all(self) -> None:
        self.skills_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class FakeGradientClient:
    current_micro_batch: int = 4
    restored: dict[int, torch.Tensor] | None = None

    def capture_worker_rng(self) -> dict[int, torch.Tensor]:
        return {1: torch.tensor([4, 2], dtype=torch.uint8)}

    def restore_worker_rng(self, value: dict[int, torch.Tensor]) -> None:
        self.restored = value


class FakeTrainer:
    def __init__(self, root: Path) -> None:
        self.shared_model = TinyPolicy()
        self.partition_fn = torch.nn.Linear(2, 1)
        theta = list(self.shared_model.lora_A["theta"].parameters())
        phi = list(self.shared_model.lora_A["phi"].parameters())
        self._supervisor_optimizer = torch.optim.AdamW(theta)
        self._phi_optimizer = torch.optim.AdamW(phi)
        self._partition_optimizer = torch.optim.AdamW(self.partition_fn.parameters())
        self.workspace = FakeWorkspace(root / "skills")
        self.distributed_gradient_client = FakeGradientClient()
        self.device = "cpu"
        self._experience_buffer: list[object] = []
        self._observation_buffer: dict[str, object] = {}
        self._per_type_acc_history: dict[str, object] = {}
        self._per_type_balance_history: dict[str, object] = {}
        self._per_type_evolution_helped: dict[str, object] = {}
        self._per_type_last_evolution: dict[str, object] = {}
        self._phase_trajectories: list[object] = []
        self._plateau_detector: dict[str, object] = {"window": [1.0]}
        self._skill_negative_counter: dict[str, int] = {}
        self._skills_just_updated = False
        self.accuracy_tracker: dict[str, object] = {}
        self.config = {
            "base_model": "Qwen/Qwen3.5-9B",
            "batch_size": 16,
            "max_steps": 288,
            "parity_contract": "skillflow-upstream-parity@1",
            "upstream_revision": "74be52bb6bd9f0e9e68dacb72636b75649197983",
        }


@pytest.fixture
def cuda_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: torch.tensor([1, 2]))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)


def test_checkpoint_is_atomic_complete_and_restorable(tmp_path: Path, cuda_rng: None) -> None:
    trainer = FakeTrainer(tmp_path)
    store = _store(tmp_path)

    checkpoint = store.save(trainer, committed_step=1)
    trainer.distributed_gradient_client.current_micro_batch = 1
    store.restore(trainer, checkpoint)

    assert checkpoint.name == "checkpoint_step_000001"
    assert trainer.distributed_gradient_client.current_micro_batch == 4
    assert (checkpoint / "COMPLETE").is_file()
    assert not tuple(store.root.glob("checkpoint_tmp_*"))
    store.verify(checkpoint)


def test_checkpoint_retention_keeps_recent_steps(tmp_path: Path, cuda_rng: None) -> None:
    trainer = FakeTrainer(tmp_path)
    store = _store(tmp_path, keep_recent=2)

    for step in range(1, 5):
        store.save(trainer, committed_step=step)

    assert [path.name for path in sorted(store.root.glob("checkpoint_step_*"))] == [
        "checkpoint_step_000003",
        "checkpoint_step_000004",
    ]


def test_incomplete_checkpoint_is_rejected(tmp_path: Path) -> None:
    incomplete = tmp_path / "checkpoints" / "checkpoint_step_000001"
    incomplete.mkdir(parents=True)

    with pytest.raises(ValueError):
        _store(tmp_path).verify(incomplete)


def test_checkpoint_rejects_another_method_identity(tmp_path: Path, cuda_rng: None) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    checkpoint = _store(source).save(FakeTrainer(source), committed_step=1)
    other = FakeTrainer(destination)
    other.config["upstream_revision"] = "another-revision"

    with pytest.raises(ValueError):
        _store(destination).verify(checkpoint, trainer=other)


def test_real_trainer_checkpointed_trackers_are_serializable(tmp_path: Path) -> None:
    trainer = GFlowNetTrainer(
        {"base_model": "unused", "output_dir": str(tmp_path), "tracking_mode": "disabled"}
    )

    torch.save(trainer._per_type_evolution_helped, io.BytesIO())
    assert trainer._per_type_evolution_helped["new_task_type"] is True


def test_step_recovery_is_private_singleton_and_cleared_after_commit(tmp_path: Path) -> None:
    transaction = StepTransaction(tmp_path)
    transaction.transition(step=2, phase=StepPhase.PREPARED, token_count=10)
    transaction.seal(
        step=2,
        trajectories=[{"private": "payload"}],
        adapter_version="theta_step_000002",
        micro_batch=4,
    )

    recovered = transaction.load(expected_step=2)

    assert recovered is not None
    assert recovered["micro_batch"] == 4
    assert transaction.recovery.stat().st_mode & 0o777 == 0o600
    assert len(tuple(transaction.directory.glob("current_step*.pt"))) == 1
    transaction.transition(step=2, phase=StepPhase.COMMITTED)
    transaction.clear()
    assert not transaction.recovery.exists()


def test_step_recovery_restores_rng_and_micro_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transaction = StepTransaction(tmp_path)
    client = FakeGradientClient(current_micro_batch=1)
    transaction.seal(
        step=2,
        trajectories=[object()],
        adapter_version="theta_step_000002",
        micro_batch=4,
        worker_cuda_rng={1: torch.tensor([9, 3], dtype=torch.uint8)},
    )
    recovery = transaction.load(expected_step=2)
    assert recovery is not None
    monkeypatch.setattr(torch, "set_rng_state", lambda state: None)

    transaction.restore_rng(recovery, coordinator_device="cpu", gradient_client=client)

    assert client.current_micro_batch == 4
    assert client.restored is not None
    torch.testing.assert_close(client.restored[1], torch.tensor([9, 3], dtype=torch.uint8))


def test_recovery_refuses_a_different_step(tmp_path: Path) -> None:
    transaction = StepTransaction(tmp_path)
    transaction.seal(
        step=3,
        trajectories=[object()],
        adapter_version="theta_step_000003",
        micro_batch=1,
    )

    with pytest.raises(ValueError):
        transaction.load(expected_step=4)
