"""Completed-edge calculation must not learn from an unclosed trajectory."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
import torch

from skillev.policy.interface import ModelInputWindow
from skillev.policy.versions import TrainableVersions
from skillev.rollout.provisional import ProvisionalStep
from skillev.scoring.edge_plan import prepare_edge_plan
from skillev.scoring.objective import ScoringConfig
from skillev.training.provisional_math import (
    PreparedProvisionalStep,
    ProvisionalGradientPool,
    prepare_provisional_step,
)
from skillev.training.step_math import compute_ttb_artifact_contribution, named_ttb_parameters


@pytest.mark.parametrize("enabled", [False, True])
def test_collector_routes_real_public_alfworld_family_to_provisional_channel(
    make_training_harness, enabled
):
    from skillev.evolution.task_features import public_task_features
    from tests.training.fakes import make_public_tasks

    tasks = tuple(
        replace(task, task_family=public_task_features(domain).task_family)
        for task, domain in zip(make_public_tasks(2), ("alfworld", "humaneval"), strict=True)
    )
    harness = make_training_harness(tasks=tasks)
    received, artifacts = [], []

    class RecordingStream:
        provisional_edges = enabled

        async def begin(self, plan, snapshot):
            assert plan.policy_snapshot_id == snapshot.snapshot_id

        async def accept_step(self, position, edge):
            assert position == 0
            assert not artifacts  # The terminal artifact has not yet been produced.
            received.append(edge)

        async def accept(self, position, artifact):
            if position == 0:
                assert bool(received) == enabled
            artifacts.append(artifact)

    batch = asyncio.run(
        harness.loop._collector.collect(optimizer_step=1, gradient_stream=RecordingStream())
    )
    assert len(batch.artifacts) == 2
    assert tuple(edge.step for edge in received) == (
        batch.artifacts[0].record.steps if enabled else ()
    )
    assert harness.loop.optimizer_step == 0


def inputs(backbone, artifact):
    return [
        prepare_provisional_step(
            backbone,
            ProvisionalStep(
                artifact.record.trajectory_id,
                artifact.manifest.task_id,
                artifact.manifest.policy_snapshot,
                artifact.manifest.library_version,
                artifact.record.initial_context.query,
                artifact.initial_context.text,
                artifact.record.steps[:i],
                step,
                ModelInputWindow.from_meta(artifact.record.initial_context.meta),
            ),
            TrainableVersions.from_backbone(backbone),
        )
        for i, step in enumerate(artifact.record.steps)
    ]


@pytest.mark.parametrize("resident_slots", [0, 1, 4])
@pytest.mark.parametrize("windowed", [False, True])
def test_provisional_interleaving_and_terminal_finalization_are_exact(
    make_training_harness, resident_slots, windowed
):
    torch.set_num_threads(1)
    harness = make_training_harness()
    harness = make_training_harness(
        config=replace(harness.config, execution=replace(harness.config.execution, batch_size=4))
    )
    if windowed:
        rollout = harness.config.rollout
        harness = make_training_harness(
            config=replace(
                harness.config,
                rollout=replace(
                    rollout,
                    format="skillev-policy-rollout@5",
                    input_window=ModelInputWindow(64),
                    per_rollout_maximum=replace(
                        rollout.per_rollout_maximum, input_tokens=2 * rollout.max_turns * 64
                    ),
                ),
            )
        )
    batch = asyncio.run(harness.loop.collect_batch())
    backbone, parameters = harness.backbone, harness.backbone.parameter_groups()
    kwargs = {
        "backbone": backbone,
        "parameters": parameters,
        "batch_id": batch.batch_id,
        "optimizer_step": batch.optimizer_step,
        "policy_snapshot_id": batch.policy_snapshot_id,
        "library_version": batch.library_version,
        "global_batch_size": 4,
        "temperature_beta": 1.0,
    }
    references = [
        compute_ttb_artifact_contribution(**kwargs, position=i, artifact=a)
        for i, a in enumerate(batch.artifacts)
    ]
    size = sum(p.numel() * p.element_size() for p in named_ttb_parameters(parameters).values())
    pool = ProvisionalGradientPool(backbone, parameters, device_budget_bytes=size * resident_slots)
    prepared = [inputs(backbone, a) for a in batch.artifacts]
    if windowed:
        assert any(len(e.forward.prefix_ids) == 64 for edges in prepared for e in edges)
        assert all(len(e.backward.prefix_ids) <= 64 for edges in prepared for e in edges)
    for edge_index in range(max(map(len, prepared))):
        for position in (3, 1, 2, 0):
            if edge_index < len(prepared[position]):
                edge = prepared[position][edge_index]
                assert PreparedProvisionalStep.from_value(edge.to_value()) == edge
                pool.advance(position, edge)
                assert all(p.grad is None for p in named_ttb_parameters(parameters).values())
    for position, artifact in enumerate(batch.artifacts):
        result = compute_ttb_artifact_contribution(
            **kwargs, position=position, artifact=artifact, provisional=pool
        )
        assert result.artifacts == references[position].artifacts
        for name, gradient in result.gradients.items():
            torch.testing.assert_close(
                gradient, references[position].gradients[name], atol=0, rtol=0
            )
        # One actual AdamW update must remain equal, including both moment slots.
        if windowed:
            updated = []
            for contribution in (references[position], result):
                named = named_ttb_parameters(parameters)
                copied = {name: torch.nn.Parameter(p.detach().clone()) for name, p in named.items()}
                optimizer = torch.optim.AdamW(copied.values(), lr=1e-4, weight_decay=0)
                for name, parameter in copied.items():
                    parameter.grad = contribution.gradients[name].clone()
                optimizer.step()
                updated.append((copied, optimizer.state_dict()))
            for name in updated[0][0]:
                torch.testing.assert_close(updated[0][0][name], updated[1][0][name], atol=0, rtol=0)
            for index, state in updated[0][1]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(
                        value, updated[1][1]["state"][index][key], atol=0, rtol=0
                    )
    assert not pool.partials
    assert pool.device_bytes == 0
    assert pool.peak_device_bytes <= size * resident_slots


def test_changed_or_repeated_provisional_evidence_is_rejected_and_discarded(make_training_harness):
    torch.set_num_threads(1)
    harness = make_training_harness()
    batch = asyncio.run(harness.loop.collect_batch())
    artifact = batch.artifacts[0]
    pool = ProvisionalGradientPool(harness.backbone, harness.backbone.parameter_groups())
    edges = inputs(harness.backbone, artifact)
    pool.advance(0, edges[0])
    with pytest.raises(ValueError):
        pool.advance(0, edges[0])
    for edge in edges[1:]:
        pool.advance(0, edge)
    plan = prepare_edge_plan(
        harness.backbone.tokenizer, artifact.record, artifact.initial_context.text
    )
    corrupted = replace(plan, edges=(replace(plan.edges[0], prefix_ids=(0,)), *plan.edges[1:]))
    with pytest.raises(ValueError):
        pool.finalize(0, artifact, corrupted, ScoringConfig())
    pool.clear()
    assert not pool.partials
    assert all(p.grad is None for p in pool.named.values())
