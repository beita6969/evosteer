from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
import torch.distributed as dist

from skillev.policy import QwenPolicyBackbone
from skillev.policy.versions import TrainableVersions
from skillev.training.distributed_ttb import (
    DistributedTTBError,
    DistributedTTBGradientCoordinator,
    DistributedTTBTopology,
    serve_distributed_ttb_worker,
)
from skillev.training.planning import PlannedRollout, TrainingBatchPlan
from skillev.training.step_math import (
    compute_ttb_gradient_shard,
    create_ttb_optimizer,
    named_ttb_parameters,
)
from skillev.training.streaming_step import GradientStepStream


def _worker(config, rendezvous, inject, versions_path):
    torch.set_num_threads(1)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    if inject:
        os.environ["SKILLEV_INJECT_TTB_OOM_ONCE_RANK"] = "1"
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=1,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        backbone = QwenPolicyBackbone(config)
        serve_distributed_ttb_worker(
            topology=DistributedTTBTopology(1, 2, 1, "gloo"), backbone=backbone
        )
        Path(versions_path).write_text(
            json.dumps(asdict(TrainableVersions.from_backbone(backbone)))
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("provisional", [False, True])
@pytest.mark.parametrize(
    ("mode", "failure", "participates"),
    [
        ("within-step", None, True),
        ("within-step", None, False),
        ("sealed-batch", None, True),
        ("within-step", "last-artifact", True),
        ("within-step", "worker-oom", True),
        ("within-step", "worker-oom", False),
    ],
)
def test_same_step_stream_matches_sealed_32_and_aborts_without_update(
    make_training_harness,
    make_training_backbone,
    training_backbone_config,
    tmp_path,
    mode,
    failure,
    participates,
    provisional,
):
    # A real two-process CPU transport exercises collective order and rank ownership.
    torch.set_num_threads(1)
    harness = make_training_harness()
    config = replace(harness.config, execution=replace(harness.config.execution, batch_size=32))
    harness = make_training_harness(config=config)
    # Parameters are broadcast, but the worker must receive this new Z lineage
    # too; marking optimizer_step-1 alone used to retain the pre-reset version.
    harness.backbone.reset_z(17)
    batch = asyncio.run(harness.loop.collect_batch())
    backbone = harness.backbone
    parameters = backbone.parameter_groups()
    original = {name: p.detach().clone() for name, p in named_ttb_parameters(parameters).items()}
    reference = compute_ttb_gradient_shard(
        backbone=backbone,
        parameters=parameters,
        batch=batch,
        positions=tuple(range(32)),
        global_batch_size=32,
        temperature_beta=config.method.temperature_beta,
    )
    optimizer, parameters = create_ttb_optimizer(backbone, config.optimizer)
    plan = TrainingBatchPlan(
        batch.batch_id,
        batch.optimizer_step,
        batch.policy_snapshot_id,
        batch.library_version,
        tuple(
            PlannedRollout(
                i + 1,
                harness.task_provider.tasks[i],
                a.record.trajectory_id,
                harness.loop.decoding_snapshot,
            )
            for i, a in enumerate(batch.artifacts)
        ),
    )
    rendezvous = str(tmp_path / "gloo")
    process = multiprocessing.get_context("spawn").Process(
        target=_worker,
        args=(
            training_backbone_config,
            rendezvous,
            failure == "worker-oom",
            str(tmp_path / "worker-versions.json"),
        ),
    )
    process.start()
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=0,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    with TemporaryDirectory(prefix="ttb-test-", dir="/tmp") as sockets:
        coordinator = DistributedTTBGradientCoordinator(
            DistributedTTBTopology(0, 2, 0, "gloo"),
            coordinator_participates=participates,
            pipeline_mode=mode,
            stream_directory=Path(sockets),
        )
        stream = GradientStepStream(
            coordinator=coordinator,
            backbone=backbone,
            parameters=parameters,
            optimizer=optimizer,
            temperature_beta=config.method.temperature_beta,
            clock=lambda: "2026-09-05T00:00:00Z",
            provisional_edges=provisional,
        )

        async def run():
            if mode == "sealed-batch":
                coordinator.prepare(
                    backbone=backbone,
                    parameters=parameters,
                    optimizer=optimizer,
                    batch=batch,
                    snapshot_before=harness.generator.snapshot(),
                    temperature_beta=config.method.temperature_beta,
                    clock=lambda: "2026-09-05T00:00:00Z",
                )
                return
            await stream.begin(plan, harness.generator.snapshot())
            try:
                # Reverse arrival is scheduled dynamically, but reduced in global order.
                for position in reversed(range(32)):
                    if failure == "last-artifact" and position == 0:
                        await stream.discard_uncommitted()
                        return
                    if provisional:
                        from skillev.rollout.provisional import ProvisionalStep

                        artifact = batch.artifacts[position]
                        for index, step in enumerate(artifact.record.steps):
                            await stream.accept_step(
                                position,
                                ProvisionalStep(
                                    artifact.record.trajectory_id,
                                    artifact.manifest.task_id,
                                    artifact.manifest.policy_snapshot,
                                    artifact.manifest.library_version,
                                    artifact.record.initial_context.query,
                                    artifact.initial_context.text,
                                    artifact.record.steps[:index],
                                    step,
                                ),
                            )
                    await stream.accept(position, batch.artifacts[position])
                if failure is None:
                    with pytest.raises(RuntimeError):
                        await stream.accept(0, batch.artifacts[0])
                    prepared = await stream.seal(batch)
                    assert tuple(r.trajectory_id for r in prepared.residuals) == tuple(
                        a.record.trajectory_id for a in batch.artifacts
                    )
                else:
                    with pytest.raises(DistributedTTBError):
                        await stream.seal(batch)
            except RuntimeError:
                # Streaming can observe a worker failure while later actors are
                # still delivering edges, not only at the terminal seal.
                if failure != "worker-oom" or not stream._aborted:
                    raise
                await stream.discard_uncommitted()
                assert any(m.get("error_class") == "OutOfMemoryError" for m in stream.rank_metrics)
            finally:
                if stream.prepared is None:
                    await stream.discard_uncommitted()

        try:
            asyncio.run(run())
            if failure is None:
                for name, parameter in named_ttb_parameters(parameters).items():
                    torch.testing.assert_close(
                        parameter.grad, reference.gradients[name], rtol=0, atol=0
                    )
            else:
                assert all(p.grad is None for p in named_ttb_parameters(parameters).values())
            assert not optimizer.state
            if failure is None:
                reference_parameters = {
                    name: torch.nn.Parameter(value.clone()) for name, value in original.items()
                }
                reference_optimizer = torch.optim.AdamW(
                    [
                        {
                            **{k: v for k, v in group.items() if k != "params"},
                            "params": [
                                reference_parameters[
                                    next(
                                        name
                                        for name, value in named_ttb_parameters(parameters).items()
                                        if value is p
                                    )
                                ]
                                for p in group["params"]
                            ],
                        }
                        for group in optimizer.param_groups
                    ]
                )
                for name, p in reference_parameters.items():
                    p.grad = reference.gradients[name].clone()
                optimizer.step()
                reference_optimizer.step()
                for name, p in named_ttb_parameters(parameters).items():
                    torch.testing.assert_close(p, reference_parameters[name], rtol=0, atol=0)
                    for key, value in optimizer.state[p].items():
                        torch.testing.assert_close(
                            value,
                            reference_optimizer.state[reference_parameters[name]][key],
                            rtol=0,
                            atol=0,
                        )
            if failure is not None:
                for name, parameter in named_ttb_parameters(parameters).items():
                    torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
        finally:
            coordinator.close()
            dist.destroy_process_group()
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            assert process.exitcode == 0
            assert json.loads((tmp_path / "worker-versions.json").read_text()) == asdict(
                TrainableVersions.from_backbone(backbone)
            )
