from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
import torch

import training.distributed_gradient as distributed
from training.distributed_gradient import (
    DistributedGradientClient,
    StandbyGradientRequiredError,
)


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict(
            {"phi": torch.nn.Linear(2, 2, bias=False), "theta": torch.nn.Linear(2, 2, bias=False)}
        )

    def set_adapter(self, adapter_name: str) -> None:
        del adapter_name


def _items() -> list[tuple[torch.Tensor, int, float]]:
    return [(torch.tensor([1, 2, 3]), 1, 0.25), (torch.tensor([4, 5]), 1, 0.75)]


def _ok(model: TinyPolicy, rank: int, value: float) -> dict[str, Any]:
    return {
        "gradients": {
            name: torch.full_like(parameter, value)
            for name, parameter in model.named_parameters()
            if ".theta." in name
        },
        "items": [rank],
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
        "rank": rank,
        "status": "ok",
        "token_count": 3,
    }


def test_worker_gradients_are_summed_without_rank_mean_bias(monkeypatch) -> None:
    model = TinyPolicy()
    client = DistributedGradientClient.create(model, initial_micro_batch=4, minimum_micro_batch=1)
    monkeypatch.setattr(
        distributed,
        "_exchange",
        lambda command, local_result: [None, _ok(model, 1, 2.0), _ok(model, 2, 3.0)],
    )

    client.compute(_items(), adapter_name="theta", kl_coeff=0.01, batch_size=2)

    for name, parameter in model.named_parameters():
        if ".theta." in name:
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 5.0))


def test_typed_oom_replays_same_items_at_lower_micro_batch(monkeypatch) -> None:
    model = TinyPolicy()
    client = DistributedGradientClient.create(model, initial_micro_batch=4, minimum_micro_batch=1)
    commands: list[dict[str, Any]] = []

    def exchange(command, local_result):
        del local_result
        commands.append(command)
        if len(commands) == 1:
            return [None, {"rank": 1, "status": "oom"}]
        return [None, _ok(model, 1, 1.0)]

    monkeypatch.setattr(distributed, "_exchange", exchange)

    client.compute(_items(), adapter_name="theta", kl_coeff=0.01, batch_size=2)

    assert [command["micro_batch"] for command in commands] == [4, 2]
    assert [tuple(item[0].tolist() for item in command["items"]) for command in commands] == [
        ([1, 2, 3], [4, 5]),
        ([1, 2, 3], [4, 5]),
    ]


def test_minimum_micro_batch_oom_requires_process_restart(monkeypatch) -> None:
    model = TinyPolicy()
    client = DistributedGradientClient.create(model, initial_micro_batch=1, minimum_micro_batch=1)
    monkeypatch.setattr(
        distributed,
        "_exchange",
        lambda command, local_result: [None, {"rank": 1, "status": "oom"}],
    )

    with pytest.raises(StandbyGradientRequiredError):
        client.compute(_items(), adapter_name="theta", kl_coeff=0.01, batch_size=2)


def test_non_oom_rank_failure_is_not_retried(monkeypatch) -> None:
    model = TinyPolicy()
    client = DistributedGradientClient.create(model, initial_micro_batch=4, minimum_micro_batch=1)
    calls = 0

    def exchange(command, local_result):
        nonlocal calls
        del command, local_result
        calls += 1
        return [None, {"rank": 1, "status": "error"}]

    monkeypatch.setattr(distributed, "_exchange", exchange)
    with pytest.raises(RuntimeError):
        client.compute(_items(), adapter_name="theta", kl_coeff=0.01, batch_size=2)
    assert calls == 1


def test_worker_failure_preserves_diagnostic_message(monkeypatch) -> None:
    from training.gflownet_trainer import GFlowNetTrainer

    model = TinyPolicy()
    monkeypatch.setattr(distributed.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(distributed.torch.cuda, "reset_peak_memory_stats", lambda device: None)

    def fail(*args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError("worker diagnostic")

    monkeypatch.setattr(GFlowNetTrainer, "_run_micro_batches", fail)
    result = distributed._compute_worker_gradient(
        model,
        {
            "adapter_name": "theta",
            "batch_size": 2,
            "items": [(torch.tensor([1, 2, 3]), 1, 0.25, None)],
            "kl_coeff": 0.01,
            "micro_batch": 1,
            "state": {},
        },
        worker_index=0,
        worker_count=1,
        device=torch.device("cpu"),
    )

    assert result["status"] == "error"
    assert result["error_class"] == "RuntimeError"
    assert isinstance(result["error_message"], str)
    assert result["error_message"]


def test_worker_rng_expands_deterministically_for_new_standby_rank(monkeypatch) -> None:
    model = TinyPolicy()
    client = DistributedGradientClient.create(model, initial_micro_batch=1, minimum_micro_batch=1)
    commands: list[dict[str, Any]] = []
    monkeypatch.setattr(distributed.dist, "get_world_size", lambda: 3)

    def exchange(command, local_result):
        del local_result
        commands.append(command)
        return [None, {"status": "rng_restored"}, {"status": "rng_restored"}]

    monkeypatch.setattr(distributed, "_exchange", exchange)
    original = torch.tensor([4, 2], dtype=torch.uint8)

    client.restore_worker_rng({1: original})

    assert commands[0]["states"] == {1: original, 2: original}


def test_controlled_oom_injection_stops_after_process_expansion(monkeypatch) -> None:
    distributed._CONTROLLED_OOM_RANKS.clear()
    monkeypatch.setenv("SKILLEV_INJECT_PRIMARY_OOM_UNTIL_EXPANSION", "1")
    monkeypatch.setattr(distributed.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(distributed.dist, "get_world_size", lambda: 2)

    assert distributed._inject_controlled_oom() is True

    monkeypatch.setattr(distributed.dist, "get_world_size", lambda: 3)
    assert distributed._inject_controlled_oom() is False


def test_controlled_once_oom_is_not_repeated(monkeypatch) -> None:
    distributed._CONTROLLED_OOM_RANKS.clear()
    monkeypatch.setenv("SKILLEV_INJECT_PRIMARY_OOM_ONCE", "1")
    monkeypatch.setattr(distributed.dist, "get_rank", lambda: 1)

    assert distributed._inject_controlled_oom() is True
    assert distributed._inject_controlled_oom() is False


def test_process_group_timeout_covers_long_formal_rollouts(monkeypatch) -> None:
    observed = {}
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(distributed.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(
        distributed.dist,
        "init_process_group",
        lambda **kwargs: observed.update(kwargs),
    )

    assert distributed.initialize_process_group() == (0, 2, 0)
    assert observed["timeout"] == timedelta(minutes=30)
