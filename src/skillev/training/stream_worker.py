"""One CUDA owner serving ready trajectories, never summing separate positions."""

from __future__ import annotations

import time
from multiprocessing.connection import Client
from typing import TYPE_CHECKING, cast

import torch

from skillev.policy import PolicyBackbone
from skillev.policy.fla_execution import fla_execution_metrics
from skillev.policy.versions import TrainableVersions
from skillev.rollout import RolloutArtifact
from skillev.scoring.edge_plan import PreparedEdgePlan

from .gradient_stream_transport import receive_message, send_contribution, send_message
from .provisional_math import PreparedProvisionalStep, ProvisionalGradientPool
from .scoring_telemetry import scoring_telemetry
from .step_math import TTBArtifactMath, compute_ttb_artifact_contribution, named_ttb_parameters

if TYPE_CHECKING:
    from .distributed_ttb import DistributedTTBTopology


def serve_stream_worker(
    command: dict[str, object], *, topology: DistributedTTBTopology, backbone: PolicyBackbone
) -> None:
    from .distributed_ttb import (
        _all_gather_object,
        _synchronize_parameters,
        should_inject_distributed_ttb_oom,
    )

    if topology.backend == "nccl":
        torch.cuda.set_device(topology.local_rank)
    parameters = backbone.parameter_groups()
    named = named_ttb_parameters(parameters)
    _synchronize_parameters(named, source=0)
    backbone.synchronize_trainable_versions(cast(TrainableVersions, command["trainable_versions"]))
    address = cast(list[str], command["addresses"])[topology.rank - 1]
    connection = Client(address, family="AF_UNIX")
    partials = (
        ProvisionalGradientPool(
            backbone,
            parameters,
            device_budget_bytes=cast(int, command.get("provisional_device_bytes", 0)),
        )
        if command.get("provisional_edges")
        else None
    )
    received: set[int] = set()
    items: list[TTBArtifactMath] = []
    failed = False
    compute_seconds = copy_seconds = 0.0
    receive_wait_seconds = 0.0
    edge_tokens = transferred_bytes = 0
    status: dict[str, object] = {"status": "ok", "rank": topology.rank}
    try:
        while True:
            receive_started = time.perf_counter()
            message = receive_message(connection)
            receive_wait_seconds += time.perf_counter() - receive_started
            if message["kind"] in {"seal", "abort"}:
                failed |= message["kind"] == "abort" or bool(partials and partials.partials)
                break
            try:
                position = message["position"]
                count = cast(int, command["global_batch_size"])
                if (
                    failed
                    or type(position) is not int
                    or not 0 <= position < count
                    or position in received
                ):
                    raise ValueError("worker received a repeated or invalid trajectory position")
                if should_inject_distributed_ttb_oom(topology, cast(str, command["batch_id"])):
                    raise torch.cuda.OutOfMemoryError("controlled stream OOM")
                if message["kind"] == "provisional-edge":
                    if (
                        partials is None
                        or message.get("batch_id") != command["batch_id"]
                        or message.get("optimizer_step") != command["optimizer_step"]
                    ):
                        raise ValueError("provisional worker channel was not enabled")
                    edge = PreparedProvisionalStep.from_value(
                        cast(dict[str, object], message["edge"])
                    )
                    if (
                        edge.trajectory_id != cast(list[str], command["trajectory_ids"])[position]
                        or edge.task_id != cast(list[str], command["task_ids"])[position]
                        or edge.policy_snapshot_id != command["policy_snapshot_id"]
                        or edge.library_version != command["library_version"]
                    ):
                        raise ValueError("provisional worker input belongs to another batch")
                    started = time.perf_counter()
                    partials.advance(
                        position,
                        edge,
                        lambda p: send_message(connection, {"kind": "progress", "progress": p}),
                    )
                    compute_seconds += time.perf_counter() - started
                    edge_tokens += edge.token_cost
                    send_message(
                        connection,
                        {"kind": "edge-complete", "position": position, "step": edge.step.index},
                    )
                    continue
                if message["kind"] != "artifact":
                    raise ValueError("unknown gradient worker command")
                received.add(position)
                artifact = RolloutArtifact.from_value(
                    message["artifact"], tokenizer=backbone.tokenizer
                )
                if (
                    artifact.record.trajectory_id
                    != cast(list[str], command["trajectory_ids"])[position]
                    or artifact.manifest.task_id != cast(list[str], command["task_ids"])[position]
                ):
                    raise ValueError("worker artifact differs from the fixed task plan")
                edges = PreparedEdgePlan.from_wire_edges(
                    message["prepared_edges"],
                    record=artifact.record,
                    initial_text=artifact.initial_context.text,
                    tokenizer_id=backbone.tokenizer.tokenizer_id,
                )
                if partials is None or position not in partials.partials:
                    edge_tokens += edges.token_cost

                def progress(value: dict[str, object]) -> None:
                    send_message(connection, {"kind": "progress", "progress": value})

                started = time.perf_counter()
                contribution = compute_ttb_artifact_contribution(
                    backbone=backbone,
                    parameters=parameters,
                    artifact=artifact,
                    position=position,
                    batch_id=cast(str, command["batch_id"]),
                    optimizer_step=cast(int, command["optimizer_step"]),
                    policy_snapshot_id=cast(str, command["policy_snapshot_id"]),
                    library_version=cast(str, command["library_version"]),
                    global_batch_size=count,
                    temperature_beta=cast(float, command["temperature_beta"]),
                    prepared_edges=edges,
                    provisional=partials,
                    progress=progress,
                )
                compute_seconds += time.perf_counter() - started
                items.extend(contribution.artifacts)
                started = time.perf_counter()
                transferred_bytes += send_contribution(connection, contribution)
                copy_seconds += time.perf_counter() - started
                del contribution
            except (ValueError, TypeError, KeyError, RuntimeError) as error:
                failed = True
                status["error_class"] = type(error).__name__
                send_message(connection, {"kind": "error", "error_class": type(error).__name__})
    finally:
        connection.close()
        if partials is not None:
            partials.clear()
        for parameter in named.values():
            parameter.grad = None
    status.update(
        {
            "status": "abort" if failed else "ok",
            "positions": sorted(received),
            "compute_seconds": compute_seconds,
            "receive_wait_seconds": receive_wait_seconds,
            "receive_wait_scope": "dispatch-wait-plus-command-transfer",
            "fla_process_cumulative": fla_execution_metrics(),
            "compute_timing_kind": "host-wall-not-kernel-time",
            "gradient_copy_transport_host_seconds": copy_seconds,
            "gradient_transferred_bytes": transferred_bytes,
            "edge_tokens": edge_tokens,
            "provisional_storage": None if partials is None else partials.storage_metrics(),
            "scoring": scoring_telemetry(items),
            "z_cache": getattr(backbone, "z_feature_cache_metrics", {}),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()
            if topology.backend == "nccl"
            else 0,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if topology.backend == "nccl"
            else 0,
        }
    )
    # Only status rendezvous: rank-zero already owns every individual contribution.
    _all_gather_object(status, topology.world_size)
