from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from skillev_private.experiments.protocol_v10_attempt_worker import (
    _bind_skillflow_authoring_service,
    _bind_skillflow_distributed_gradient,
    _initialize_formal_topology,
    _skillflow_runtime_values,
)

import training.distributed_gradient as distributed
from skillev.experiments import FormalMethodV10
from skillev.training.distributed_ttb import DistributedTTBTopology


def test_skillflow_runtime_uses_the_pinned_split_tokenizer_path(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class FakeConfig:
        def runtime_values(self, **values: object) -> dict[str, object]:
            calls.append(values)
            return dict(values)

    application_input = SimpleNamespace(
        backbone=SimpleNamespace(
            base_model_path="/models/weights-only",
            tokenizer_path="/models/tokenizer",
        )
    )
    exact = SimpleNamespace(
        attempt_root=tmp_path,
        sglang=SimpleNamespace(
            gateway=SimpleNamespace(
                openai_base="http://inference/v1",
                base_model="frozen-base",
            ),
            rollout=SimpleNamespace(request_timeout_seconds=321.0),
            adapter_namespace="attempt-adapter",
        ),
    )

    values = _skillflow_runtime_values(
        config=FakeConfig(),  # type: ignore[arg-type]
        application_input=application_input,  # type: ignore[arg-type]
        exact=exact,  # type: ignore[arg-type]
    )

    assert calls[0]["base_model_path"] == "/models/weights-only"
    assert values["tokenizer_path"] == "/models/tokenizer"
    assert values["tokenizer_path"] != values["base_model_path"]
    assert values["distributed_micro_batch"] == 1
    assert values["distributed_minimum_micro_batch"] == 1
    assert values["executor_request_timeout_seconds"] == 321.0
    assert values["supervisor_request_timeout_seconds"] == 321.0


def test_skillflow_authoring_uses_the_frozen_base_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKILL_CREATOR_API_BASE", raising=False)
    monkeypatch.delenv("SKILL_CREATOR_MODEL", raising=False)
    exact = SimpleNamespace(
        method=FormalMethodV10.SKILLFLOW_BASELINE,
        sglang=SimpleNamespace(
            gateway=SimpleNamespace(
                api_root="http://inference",
                base_model="frozen-base",
            )
        ),
    )

    _bind_skillflow_authoring_service(exact)  # type: ignore[arg-type]

    assert os.environ["SKILL_CREATOR_API_BASE"] == "http://inference/v1/messages"
    assert os.environ["SKILL_CREATOR_MODEL"] == "frozen-base"


def test_formal_process_group_covers_coordinator_only_evolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, int] = {}
    topology = DistributedTTBTopology(rank=0, world_size=2, local_rank=0, backend="nccl")

    def initialize(*, timeout_minutes: int) -> DistributedTTBTopology:
        observed["timeout_minutes"] = timeout_minutes
        return topology

    monkeypatch.setattr(
        "skillev_private.experiments.protocol_v10_attempt_worker.initialize_distributed_ttb",
        initialize,
    )

    assert _initialize_formal_topology() is topology
    assert observed["timeout_minutes"] > 30


def test_skillflow_coordinator_joins_worker_startup_barrier_before_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    client = object()

    class FakeDistributedGradientClient:
        @classmethod
        def create(
            cls,
            model: object,
            *,
            initial_micro_batch: int,
            minimum_micro_batch: int,
        ) -> object:
            assert model == "authority-model"
            assert initial_micro_batch == 8
            assert minimum_micro_batch == 2
            calls.append("create-client")
            return client

    trainer = SimpleNamespace(
        config={"distributed_micro_batch": 8, "distributed_minimum_micro_batch": 2},
        shared_model="authority-model",
    )

    def resume(path: str) -> None:
        assert trainer.distributed_gradient_client is client
        calls.append(f"resume:{path}")

    trainer.resume = resume
    monkeypatch.setattr(distributed, "DistributedGradientClient", FakeDistributedGradientClient)
    monkeypatch.setattr(
        distributed,
        "coordinator_barrier",
        lambda: calls.append("coordinator-barrier"),
    )

    _bind_skillflow_distributed_gradient(
        trainer,
        resume_snapshot=Path("/private/checkpoint"),
    )

    assert calls == ["create-client", "coordinator-barrier", "resume:/private/checkpoint"]
    assert trainer.distributed_gradient_client is client
