"""Attribution observes the actual backward path and one stateful Adam update."""

import asyncio
from dataclasses import replace

import pytest
import torch
from skillev_private.experiments.ttb_update_attribution import ProtocolProbe, attribute_update

from skillev.policy import AdapterRole
from skillev.rollout import GenerationPhase
from skillev.scoring.edge_plan import prepare_edge_plan
from skillev.training.step_math import (
    compute_ttb_gradient_shard,
    create_ttb_optimizer,
    named_ttb_parameters,
)
from tests.training.fakes import SCRIPTED_ACTION_TEXT


@pytest.mark.parametrize("mixed", [False, True])
def test_isolated_attribution_matches_full_adam_and_does_not_touch_source(
    make_training_harness,
    make_training_backbone,
    monkeypatch,
    mixed,
):
    harness = make_training_harness(raw_rewards=(0.0, 1.0))
    if mixed:
        config = replace(
            harness.config,
            rollout=replace(
                harness.config.rollout,
                max_turns=3,
                per_rollout_maximum=harness.config.rollout.per_rollout_maximum.scale(3),
            ),
        )
        harness = make_training_harness(raw_rewards=(0.0, 1.0), config=config)
        original = type(harness.generator).generate
        finish_turns = {}

        async def scripted(generator, request):
            finish_turn = finish_turns.setdefault(request.episode_id, len(finish_turns) + 2)
            generator.action_text = (
                "not-json " * request.turn_index
                if request.phase is GenerationPhase.ACTION and request.turn_index < finish_turn
                else SCRIPTED_ACTION_TEXT
            )
            return await original(generator, request)

        monkeypatch.setattr(type(harness.generator), "generate", scripted)
    batch = asyncio.run(harness.loop.collect_batch())
    source = named_ttb_parameters(harness.backbone.parameter_groups())
    saved = {name: p.detach().clone() for name, p in source.items()}
    replicas = []

    def replica_factory():
        backbone = make_training_backbone()
        named = named_ttb_parameters(backbone.parameter_groups())
        with torch.no_grad():
            for name, p in named.items():
                p.copy_(saved[name])
        optimizer, _ = create_ttb_optimizer(backbone, harness.config.optimizer)
        # Existing Adam history is part of the test, not four fresh optimizers.
        for p in named.values():
            optimizer.state[p] = {
                "step": torch.tensor(3.0),
                "exp_avg": torch.full_like(p, 0.001),
                "exp_avg_sq": torch.full_like(p, 0.01),
            }
        replicas.append((backbone, optimizer))
        return backbone, optimizer

    plan = prepare_edge_plan(
        harness.backbone.tokenizer,
        batch.artifacts[0].record,
        batch.artifacts[0].initial_context.text,
    )
    edge = plan.edge(batch.artifacts[0].record.horizon, AdapterRole.FORWARD_POLICY)
    probe = ProtocolProbe("synthetic-complete", edge.prefix_ids, edge.action_ids)
    observed = []
    progress_rows = []

    def observe(stage, model, adam):
        observed.append((stage, id(model), id(adam)))
        return {"optimizer_steps": [float(s["step"]) for s in adam.state.values()]}

    report = attribute_update(
        replica_factory=replica_factory,
        batch=batch,
        expected_batch_size=2,
        temperature_beta=1.0,
        relative_l2_tolerance=2e-5,
        absolute_tolerance=2e-6,
        protocol_panel=(probe,),
        observe=observe,
        progress=progress_rows.append,
    )
    reference, optimizer = replica_factory()
    params = reference.parameter_groups()
    shard = compute_ttb_gradient_shard(
        backbone=reference,
        parameters=params,
        batch=batch,
        positions=(0, 1),
        global_batch_size=2,
        temperature_beta=1.0,
    )
    for name, p in named_ttb_parameters(params).items():
        p.grad = shard.gradients[name]
    optimizer.step()
    actual, actual_optimizer = replicas[0]
    for name, p in named_ttb_parameters(actual.parameter_groups()).items():
        expected = named_ttb_parameters(params)[name]
        torch.testing.assert_close(p, expected, rtol=2e-5, atol=2e-6)
        for key, value in optimizer.state[expected].items():
            torch.testing.assert_close(actual_optimizer.state[p][key], value)
        assert torch.equal(source[name], saved[name])
        assert source[name].grad is None
    assert report["diagnostic_optimizer_updates"] == 1
    assert report["global_batch_size"] == 2
    assert [row[0] for row in observed] == ["before", "after"]
    assert observed[0][1:] == observed[1][1:]
    assert [row["completed_trajectories"] for row in progress_rows] == [1, 2]
    assert [row["position"] for row in progress_rows] == [0, 1]
    assert progress_rows[-1]["completed_edges"] == len(report["edges"])
    assert set(report["boundary_observations"]["before"]["optimizer_steps"]) == {3}
    assert set(report["boundary_observations"]["after"]["optimizer_steps"]) == {4}
    assert sum(row["edge_count"] for row in report["category_evidence"].values()) == len(
        report["edges"]
    )
    if not mixed:
        empty = report["category_evidence"]["structure-False/terminal-success-True"]
        assert empty["evidence_status"] == "no-evidence"
        assert empty["gradient_norms"] is None
    assert len(report["trajectories"]) == 2
    assert len(report["edges"]) == (5 if mixed else 2)
    assert report["fixed_action_nll_before"]["synthetic-complete"] > 0
    assert report["category_gradient_norms"]["partition"]["z_head"] > 0
    invalid_success = report["category_gradient_norms"]["structure-False/terminal-success-True"]
    assert (invalid_success["forward"] > 0) is mixed

    assert all(v["max_absolute"] < 2e-6 for v in report["canonical_reconstruction_error"].values())


def test_attribution_rejects_partial_population_before_allocating_model(make_training_harness):
    harness = make_training_harness()
    batch = asyncio.run(harness.loop.collect_batch())

    def forbidden_factory():
        pytest.fail("partial population should be rejected before creating an optimizer")

    with pytest.raises(ValueError):
        attribute_update(
            replica_factory=forbidden_factory,
            batch=batch,
            expected_batch_size=28,
            temperature_beta=1.0,
            relative_l2_tolerance=2e-5,
            absolute_tolerance=2e-6,
        )
