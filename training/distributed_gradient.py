"""NCCL process workers for weighted theta/phi LoRA gradients."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from skillev.training.gradient_worker import GradientWorkItem, partition_items_by_token_cost

logger = logging.getLogger(__name__)
_CONTROLLED_OOM_RANKS: set[int] = set()


class StandbyGradientRequiredError(RuntimeError):
    """The primary worker still OOMed at the configured minimum micro-batch."""


@dataclass(slots=True)
class DistributedGradientClient:
    model: Any
    initial_micro_batch: int
    minimum_micro_batch: int
    current_micro_batch: int

    @classmethod
    def create(cls, model: Any, *, initial_micro_batch: int, minimum_micro_batch: int):
        return cls(model, initial_micro_batch, minimum_micro_batch, initial_micro_batch)

    def compute(self, items, *, adapter_name: str, kl_coeff: float, batch_size: int) -> None:
        micro_batch = self.current_micro_batch
        initial_worker_rng = self.capture_worker_rng() if dist.is_initialized() else {}
        while True:
            command = {
                "adapter_name": adapter_name,
                "batch_size": batch_size,
                "items": [
                    (
                        item[0].detach().cpu(),
                        int(item[1]),
                        float(item[2]),
                        None if len(item) < 4 else float(item[3]),
                    )
                    for item in items
                ],
                "kind": "gradient",
                "kl_coeff": float(kl_coeff),
                "micro_batch": micro_batch,
                "state": {
                    name: parameter.detach().cpu()
                    for name, parameter in self.model.named_parameters()
                    if "lora_" in name
                },
            }
            results = _exchange(command, local_result=None)
            failures = [result for result in results[1:] if result.get("status") != "ok"]
            if not failures:
                logger.info(
                    "distributed_gradient=%s",
                    json.dumps(
                        {
                            "micro_batch": micro_batch,
                            "workers": [
                                {
                                    "items": len(result["items"]),
                                    "peak_allocated_bytes": result["peak_allocated_bytes"],
                                    "peak_reserved_bytes": result["peak_reserved_bytes"],
                                    "rank": result["rank"],
                                    "token_count": result["token_count"],
                                }
                                for result in results[1:]
                            ],
                        },
                        sort_keys=True,
                    ),
                )
                self._apply_gradients(results[1:], adapter_name)
                self.current_micro_batch = micro_batch
                return
            if initial_worker_rng:
                self.restore_worker_rng(initial_worker_rng)
            logger.warning(
                "distributed_oom_recovery=%s",
                json.dumps(
                    {
                        "failures": failures,
                        "micro_batch": micro_batch,
                        "next_micro_batch": (
                            None
                            if micro_batch <= self.minimum_micro_batch
                            else max(self.minimum_micro_batch, micro_batch // 2)
                        ),
                    },
                    sort_keys=True,
                ),
            )
            non_oom = [result for result in failures if result.get("status") != "oom"]
            if non_oom:
                raise RuntimeError("distributed gradient worker failed")
            if micro_batch <= self.minimum_micro_batch:
                raise StandbyGradientRequiredError("primary worker OOM at minimum micro-batch")
            micro_batch = max(self.minimum_micro_batch, micro_batch // 2)

    def _apply_gradients(self, results: list[dict[str, Any]], adapter_name: str) -> None:
        by_name = dict(self.model.named_parameters())
        expected = {name for name in by_name if f".{adapter_name}." in name and "lora_" in name}
        if not expected:
            raise RuntimeError("authority model exposes no parameters for requested adapter")
        for name in expected:
            combined = None
            for result in results:
                gradient = result["gradients"].get(name)
                if gradient is None:
                    raise RuntimeError("worker gradient parameter set differs")
                combined = gradient if combined is None else combined.add(gradient)
            assert combined is not None
            if not bool(torch.isfinite(combined).all()):
                raise RuntimeError("distributed gradient is non-finite")
            parameter = by_name[name]
            parameter.grad = combined.to(device=parameter.device, dtype=parameter.dtype)

    def close(self) -> None:
        _exchange({"kind": "stop"}, local_result=None)

    def capture_worker_rng(self) -> dict[int, torch.Tensor]:
        results = _exchange({"kind": "capture_rng"}, local_result=None)
        return {int(result["rank"]): result["rng_state"] for result in results[1:]}

    def restore_worker_rng(self, states: dict[int, torch.Tensor]) -> None:
        if not states:
            raise ValueError("checkpoint has no gradient-worker RNG state")
        fallback = states[min(states)]
        expanded = {rank: states.get(rank, fallback) for rank in range(1, dist.get_world_size())}
        results = _exchange({"kind": "restore_rng", "states": expanded}, local_result=None)
        if any(result.get("status") != "rng_restored" for result in results[1:]):
            raise RuntimeError("a gradient worker did not restore its CUDA RNG")


def initialize_process_group() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30),
        device_id=device,
    )
    return rank, world_size, local_rank


def build_worker_model(config: dict[str, object], device: torch.device):
    base = AutoModelForCausalLM.from_pretrained(
        str(config["base_model"]),
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model = get_peft_model(
        base,
        LoraConfig(
            r=int(config["lora_rank"]),
            lora_alpha=int(config["lora_alpha"]),
            target_modules=list(config["lora_target_modules"]),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        adapter_name="theta",
    )
    phi_rank = int(config["backward_lora_rank"])
    model.add_adapter(
        "phi",
        LoraConfig(
            r=phi_rank,
            lora_alpha=phi_rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.gradient_checkpointing_enable()
    model.train()
    return model


def worker_loop(config: dict[str, object]) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    model = build_worker_model(config, device)
    dist.barrier(device_ids=[device.index])
    while True:
        command = _receive_command()
        if command["kind"] == "stop":
            _exchange_result({"rank": rank, "status": "stopped"})
            return
        if command["kind"] == "capture_rng":
            _exchange_result(
                {"rank": rank, "rng_state": torch.cuda.get_rng_state(device), "status": "rng"}
            )
            continue
        if command["kind"] == "restore_rng":
            torch.cuda.set_rng_state(command["states"][rank], device)
            _exchange_result({"rank": rank, "status": "rng_restored"})
            continue
        result = _compute_worker_gradient(
            model,
            command,
            worker_index=rank - 1,
            worker_count=world_size - 1,
            device=device,
        )
        _exchange_result(result)


def coordinator_barrier() -> None:
    dist.barrier(device_ids=[torch.cuda.current_device()])


def _receive_command() -> dict[str, Any]:
    values: list[Any] = [None]
    device = torch.device("cuda", torch.cuda.current_device())
    dist.broadcast_object_list(values, src=0, device=device)
    return values[0]


def _exchange(command: dict[str, Any], local_result: Any) -> list[dict[str, Any]]:
    values: list[Any] = [command]
    device = torch.device("cuda", torch.cuda.current_device())
    dist.broadcast_object_list(values, src=0, device=device)
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_result)
    return gathered


def _exchange_result(result: dict[str, Any]) -> None:
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, result)


def _compute_worker_gradient(
    model,
    command: dict[str, Any],
    *,
    worker_index: int,
    worker_count: int,
    device: torch.device,
) -> dict[str, Any]:
    adapter_name = command["adapter_name"]
    model.set_adapter(adapter_name)
    parameters = dict(model.named_parameters())
    for name, tensor in command["state"].items():
        parameters[name].data.copy_(tensor.to(device=parameters[name].device))
    model.zero_grad(set_to_none=True)
    items = command["items"]
    work = tuple(
        GradientWorkItem(str(index), int(item[0].shape[0]), f"private://{index}")
        for index, item in enumerate(items)
    )
    partitions = partition_items_by_token_cost(work, worker_count)
    selected_indices = [int(item.item_id) for item in partitions[worker_index]]
    selected = []
    for index in selected_indices:
        ids, context_length, scale, reference = items[index]
        selected.append(
            (ids, context_length, scale)
            if reference is None
            else (ids, context_length, scale, reference)
        )
    torch.cuda.reset_peak_memory_stats(device)
    controlled_oom = False
    try:
        controlled_oom = _inject_controlled_oom()
        if controlled_oom:
            raise torch.cuda.OutOfMemoryError("controlled formal OOM injection")
        from training.gflownet_trainer import GFlowNetTrainer

        GFlowNetTrainer._run_micro_batches(
            selected,
            model,
            device,
            int(command["micro_batch"]),
            float(command["kl_coeff"]),
            int(command["batch_size"]),
        )
        gradients = {}
        for name, parameter in model.named_parameters():
            if f".{adapter_name}." in name and "lora_" in name:
                gradient = parameter.grad
                gradients[name] = (
                    torch.zeros_like(parameter, device="cpu")
                    if gradient is None
                    else gradient.detach().float().cpu()
                )
        finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients.values())
        if not finite:
            return {"rank": dist.get_rank(), "status": "nonfinite"}
        return {
            "gradients": gradients,
            "items": selected_indices,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "rank": dist.get_rank(),
            "status": "ok",
            "token_count": sum(int(items[index][0].shape[0]) for index in selected_indices),
        }
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "controlled_injection": controlled_oom,
            "micro_batch": command["micro_batch"],
            "rank": dist.get_rank(),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "stage": "gradient",
            "status": "oom",
            "token_count": sum(int(items[index][0].shape[0]) for index in selected_indices),
        }
    except Exception as error:
        model.zero_grad(set_to_none=True)
        return {
            "error_class": type(error).__name__,
            "error_message": str(error),
            "rank": dist.get_rank(),
            "status": "error",
        }


def _inject_controlled_oom() -> bool:
    """Inject a typed OOM without allocating memory; disabled unless explicitly requested."""

    rank = dist.get_rank()
    if rank != 1:
        return False
    once = os.environ.get("SKILLEV_INJECT_PRIMARY_OOM_ONCE") == "1"
    until_expansion = os.environ.get("SKILLEV_INJECT_PRIMARY_OOM_UNTIL_EXPANSION") == "1"
    if once and rank not in _CONTROLLED_OOM_RANKS:
        _CONTROLLED_OOM_RANKS.add(rank)
        return True
    return until_expansion and dist.get_world_size() == 2
