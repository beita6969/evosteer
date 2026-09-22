"""Private journal and replay state for one uncommitted SkillFlow step."""

from __future__ import annotations

import json
import os
import random
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch


class StepPhase(StrEnum):
    PREPARED = "prepared"
    ROLLOUT_COMPLETE = "rollout_complete"
    GRADIENT_IN_PROGRESS = "gradient_in_progress"
    GRADIENT_COMPLETE = "gradient_complete"
    OPTIMIZER_COMMITTED = "optimizer_committed"
    SGLANG_SYNCED = "sglang_synced"
    COMMITTED = "committed"


class StepTransaction:
    def __init__(self, run_directory: Path) -> None:
        self.directory = run_directory / "recovery"
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.journal = self.directory / "step_journal.jsonl"
        self.recovery = self.directory / "current_step.pt"

    def transition(self, *, step: int, phase: StepPhase, **metadata: object) -> None:
        record = {"phase": phase.value, "step": step, **metadata}
        descriptor = os.open(
            self.journal,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def seal(
        self,
        *,
        step: int,
        trajectories: list[Any],
        adapter_version: str,
        micro_batch: int,
        coordinator_device: str | torch.device = "cpu",
        worker_cuda_rng: dict[int, torch.Tensor] | None = None,
    ) -> None:
        temporary = self.directory / f"current_step_tmp_{uuid.uuid4().hex}.pt"
        torch.save(
            {
                "adapter_version": adapter_version,
                "coordinator_cuda_rng": (
                    torch.cuda.get_rng_state(coordinator_device)
                    if torch.cuda.is_available()
                    else None
                ),
                "micro_batch": micro_batch,
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "step": step,
                "torch_cpu_rng": torch.get_rng_state(),
                "trajectories": trajectories,
                "worker_cuda_rng": worker_cuda_rng or {},
            },
            temporary,
        )
        temporary.chmod(0o600)
        os.replace(temporary, self.recovery)

    def load(self, *, expected_step: int) -> dict[str, Any] | None:
        if not self.recovery.is_file():
            return None
        value = torch.load(self.recovery, map_location="cpu", weights_only=False)
        if value.get("step") != expected_step or not isinstance(value.get("trajectories"), list):
            raise ValueError("private recovery state does not match the next committed step")
        return value

    def restore_rng(
        self,
        value: dict[str, Any],
        *,
        coordinator_device: str | torch.device,
        gradient_client: Any,
    ) -> None:
        random.setstate(value["python_rng"])
        np.random.set_state(value["numpy_rng"])
        torch.set_rng_state(value["torch_cpu_rng"])
        coordinator_cuda = value.get("coordinator_cuda_rng")
        if coordinator_cuda is not None:
            torch.cuda.set_rng_state(coordinator_cuda, coordinator_device)
        worker_cuda = value.get("worker_cuda_rng")
        if worker_cuda:
            gradient_client.restore_worker_rng(worker_cuda)
        gradient_client.current_micro_batch = int(value["micro_batch"])

    def clear(self) -> None:
        self.recovery.unlink(missing_ok=True)
