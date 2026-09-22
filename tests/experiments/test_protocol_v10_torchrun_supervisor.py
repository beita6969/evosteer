from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import skillev_private.experiments.protocol_v10_attempt_supervisor as supervisor_module
from skillev_private.experiments.protocol_v10_attempt_supervisor import (
    ProtocolV10TorchrunLaunch,
)

from skillev.experiments import FormalMethodV10
from skillev.runtime.formal_storage import FormalStorageBinding, FormalStorageBudget
from skillev.runtime.gpu_topology import GPUObservation, ThreeGPURolePolicy


def _hardware() -> ThreeGPURolePolicy:
    return ThreeGPURolePolicy(
        inference_physical_index=5,
        coordinator_physical_index=4,
        gradient_primary_physical_index=6,
        forbidden_physical_indices=(0, 1, 2, 3, 7),
        minimum_free_memory_mib=70_000,
    )


def _launch(tmp_path: Path) -> ProtocolV10TorchrunLaunch:
    return ProtocolV10TorchrunLaunch(
        method=FormalMethodV10.BAYESIAN_IMPROVE_FULL,
        exact_input=Path("/private/protocol-v10/full.json"),
        hardware=_hardware(),
        storage=FormalStorageBinding(
            output_root=tmp_path,
            temporary_root=tmp_path,
            budget=FormalStorageBudget(
                estimated_attempt_peak_bytes=1,
                rollback_checkpoint_bytes=1,
                atomic_publication_bytes=1,
                minimum_free_inodes=1,
            ),
        ),
        worker_module="private_runtime.protocol_v10_worker",
    )


def _observations() -> tuple[GPUObservation, ...]:
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


def test_launch_exposes_only_coordinator_and_gradient_gpus(tmp_path: Path) -> None:
    launch = _launch(tmp_path)

    assert launch.training_physical_indices == (4, 6)
    assert launch.world_size == 2
    environment = launch.environment({"PATH": "/bin"}, observations=_observations())
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-physical-4,GPU-physical-6"
    assert "GPU-physical-5" not in environment["CUDA_VISIBLE_DEVICES"]
    assert "GPU-physical-7" not in environment["CUDA_VISIBLE_DEVICES"]
    assert "--nproc-per-node=2" in launch.command()


def test_formal_launcher_requires_the_frozen_clean_source_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter((SimpleNamespace(stdout="a" * 40 + "\n"), SimpleNamespace(stdout="")))
    monkeypatch.setattr(supervisor_module.subprocess, "run", lambda *args, **kwargs: next(outputs))

    supervisor_module._require_clean_source_commit("a" * 40)

    outputs = iter((SimpleNamespace(stdout="b" * 40 + "\n"), SimpleNamespace(stdout="")))
    monkeypatch.setattr(supervisor_module.subprocess, "run", lambda *args, **kwargs: next(outputs))
    with pytest.raises(RuntimeError):
        supervisor_module._require_clean_source_commit("a" * 40)
