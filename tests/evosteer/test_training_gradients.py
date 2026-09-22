"""Exact streaming gradients against dense AnchorTB on a real tiny causal LM."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from skillev.contracts.canonical import canonical_json, stable_hash
from skillev.contracts.evosteer import DecisionRecord, EvoTask, EvoTrajectory
from skillev.scoring.anchor_tb import anchor_tb_scalar
from skillev.training.evosteer import ADAM_BETA2, EvoOptimizerConfig, EvoTrainer, structure_visit
from skillev.training.reference_statistics import ReferenceObservation, ReferenceStatistics
from tests.evosteer.test_application import application


def initialized(mode, *, gradient_clip=100.0):
    app = application(candidate=False)
    with torch.no_grad():
        app.policy.model.actor_bias.copy_(torch.linspace(-0.4, 0.6, 259))
        output = app.heads.residual_head[-1]
        output.weight.copy_(
            torch.linspace(-0.08, 0.12, output.weight.numel()).reshape_as(output.weight)
        )
        output.bias.fill_(0.17)
    trainer = EvoTrainer(
        app.policy,
        app.heads,
        ("synthetic",),
        replace(app.config.optimizer, gradient_mode=mode, gradient_clip=gradient_clip),
    )
    return app.policy, app.heads, trainer


def trajectory(policy, source, reward, *, cached_encoding=False):
    candidate = source == "paired_treatment"
    add = {"kind": "ADD_AGENT", "node_id": "n0", "role_id": "solver"}
    with_skill = {**add, "skill_id": "candidate-one"}
    second = {"kind": "ADD_AGENT", "node_id": "n1", "role_id": "verifier"}
    selected = {"kind": "SET_OUTPUT", "node_id": "n1"}
    menus = (
        (add, with_skill),
        (second, {"kind": "RERUN_AGENT", "node_id": "n0"}),
        (
            selected,
            {"kind": "SET_OUTPUT", "node_id": "n0"},
            {"kind": "RERUN_AGENT", "node_id": "n1"},
        ),
        ({"kind": "STOP"},),
    )
    chosen = (with_skill if candidate else add, second, selected, {"kind": "STOP"})
    nodes = [
        {
            "node_id": "n0",
            "role_id": "solver",
            "skill_ids": ["candidate-one"] if candidate else [],
            "output": "draft",
        },
        {"node_id": "n1", "role_id": "verifier", "skill_ids": [], "output": "checked"},
    ]
    decisions = []
    for index, (choices, action) in enumerate(zip(menus, chosen, strict=True)):
        state = {
            "stopped": False,
            "graph": {
                "nodes": nodes[: min(index, 2)],
                "edges": [],
                "output_node_id": "n1" if index == 3 else None,
            },
            "history": [{"action": item} for item in chosen[:index]],
            "public_step": index,
        }
        menu = policy.menu(choices)
        action_json = canonical_json(action)
        prompt = policy.encode_prompt(canonical_json(state))
        decisions.append(
            DecisionRecord(
                stable_hash(state),
                canonical_json(state),
                action_json,
                prompt,
                menu.paths[menu.actions.index(action_json)],
                menu.paths,
                tuple((index + 1) * (i + 1) / 150 for i in range(30)),
                policy.encode_state(prompt) if cached_encoding else (),
                0.5,
                forced=source.startswith("paired_") and index == 0,
            )
        )
    paired = source.startswith("paired_")
    return EvoTrajectory(
        source,
        "gradient-batch",
        EvoTask("gradient-task", "synthetic", "Check a calculation."),
        source,
        policy.actor_id if source == "current" else policy.reference_id,
        policy.reference_id,
        "executor",
        "menu",
        "value",
        "statistics",
        tuple(decisions),
        canonical_json({"stopped": True}),
        reward,
        "checked",
        canonical_json({}),
        pair_id="pair" if paired else None,
        candidate_id="candidate-one" if paired else None,
    )


def examples(policy, *, cached_encoding=False):
    trajectories = tuple(
        trajectory(policy, source, reward, cached_encoding=cached_encoding)
        for source, reward in (
            ("current", 0.7),
            ("natural_reference", 0.2),
            ("paired_treatment", 1.0),
            ("paired_control", 0.0),
        )
    )
    snapshot = ReferenceStatistics("statistics").snapshot(
        "gradient-batch",
        (
            ReferenceObservation(
                "natural_reference", "gradient-task", "synthetic", 0.2, "statistics"
            ),
        ),
    )
    return trajectories, {"statistics": snapshot}


@pytest.mark.parametrize(("cached_encoding", "gradient_clip"), [(False, 100.0), (True, 0.05)])
def test_streaming_matches_dense_gradients_and_updates_from_nonzero_actor_and_residual(
    cached_encoding,
    gradient_clip,
):
    dense_policy, dense_heads, dense = initialized("dense", gradient_clip=gradient_clip)
    stream_policy, stream_heads, streaming = initialized("streaming", gradient_clip=gradient_clip)
    trajectories, snapshots = examples(dense_policy, cached_encoding=cached_encoding)
    actor_before = dense_policy.model.actor_bias.detach().clone()
    heads_before = {name: value.clone() for name, value in dense_heads.state_dict().items()}
    reference_before = {
        name: value.clone() for name, value in dense_policy.reference_model.state_dict().items()
    }
    assert trajectories[2].decisions[0].forced
    assert trajectories[3].decisions[0].forced
    assert len(trajectories[2].decisions[0].legal_token_paths) == 2
    dense_metrics = dense.update(trajectories, snapshots)
    stream_metrics = streaming.update(trajectories, snapshots)
    for name in dense_metrics:
        assert stream_metrics[name] == pytest.approx(dense_metrics[name], rel=2e-6, abs=2e-7)
    for dense_parameter, stream_parameter in zip(
        (*dense_policy.trainable_parameters(), *dense_heads.parameters()),
        (*stream_policy.trainable_parameters(), *stream_heads.parameters()),
        strict=True,
    ):
        assert dense_parameter.grad is not None
        assert stream_parameter.grad is not None
        torch.testing.assert_close(
            stream_parameter.grad, dense_parameter.grad, rtol=3e-5, atol=1e-7
        )
        torch.testing.assert_close(stream_parameter, dense_parameter, rtol=3e-5, atol=2e-6)
    assert not torch.equal(stream_policy.model.actor_bias, actor_before)
    assert not torch.equal(
        stream_heads.residual_head[-1].weight, heads_before["residual_head.2.weight"]
    )
    assert not torch.equal(
        stream_heads.runtime_head[-1].weight, heads_before["runtime_head.2.weight"]
    )
    for index, head in enumerate(stream_heads.outcome_heads):
        assert not torch.equal(head[2].weight, heads_before[f"outcome_heads.{index}.2.weight"])
    assert dense_metrics["natural_reference_states"] == 4
    assert dense_metrics["paired_reference_states"] == 6
    assert dense_metrics["supervised_reference_states"] == 10
    assert dense_policy.version == stream_policy.version == 1
    for policy in (dense_policy, stream_policy):
        assert all(parameter.grad is None for parameter in policy.reference_model.parameters())
        for name, value in policy.reference_model.state_dict().items():
            assert torch.equal(value, reference_before[name])
    assert all(int(state["step"].item()) == 1 for state in streaming.optimizer.state.values())


def test_streaming_backpropagates_each_action_before_the_next_and_uses_distinct_exact_coefficients(
    monkeypatch,
):
    policy, heads, trainer = initialized("streaming")
    item = trajectory(policy, "paired_treatment", 1.0)
    snapshot = ReferenceStatistics("statistics").snapshot("gradient-batch")
    root = snapshot.task_anchor(item.task.task_id, item.task.family)
    initial_actor = policy.model.actor_bias.detach().clone()
    with torch.no_grad():
        ratios = [float(policy.score(d) - policy.score(d, reference=True)) for d in item.decisions]
        flows = []
        for decision in item.decisions:
            encoding = policy.encode_state(decision.prompt_ids)
            residual = heads.flow_residual(
                torch.tensor([encoding]), torch.tensor([decision.features])
            ).item()
            measured = min(
                trainer.config.beta,
                max(
                    0.0, root + snapshot.structure_correction(structure_visit(decision.state_json))
                ),
            )
            flows.append(residual + measured)
    oracle = anchor_tb_scalar(ratios, [*flows, 0.0], reward=item.reward)
    assert oracle.state_coefficients[0] != 0.0
    assert oracle.state_coefficients[-1] == 0.0
    assert len({round(value, 5) for value in oracle.action_coefficients}) > 1
    events = []
    scores = policy.score
    residuals = heads.flow_residual
    flow_gradients = []

    def score(record, *, reference=False):
        assert torch.equal(policy.model.actor_bias, initial_actor)
        gradient = torch.is_grad_enabled()
        if reference:
            assert not gradient
        events.append(("score", record.state_id, gradient, reference))
        result = scores(record, reference=reference)
        if gradient and not reference:
            result.register_hook(
                lambda value: events.append(("backward", record.state_id, float(value)))
            )
        return result

    def flow(encoding, features):
        result = residuals(encoding, features)
        index = len(flow_gradients)
        flow_gradients.append(None)
        result.register_hook(lambda value: flow_gradients.__setitem__(index, float(value.item())))
        return result

    combined = getattr(policy, "reference_score_and_encoding", None)

    def reference_pass(record):
        # The trainer scores rho and encodes the state in one frozen forward.
        assert torch.equal(policy.model.actor_bias, initial_actor)
        assert not torch.is_grad_enabled()
        events.append(("score", record.state_id, False, True))
        return combined(record)

    monkeypatch.setattr(policy, "score", score)
    if combined is not None:
        monkeypatch.setattr(policy, "reference_score_and_encoding", reference_pass)
    monkeypatch.setattr(heads, "flow_residual", flow)
    trainer.update((item,), {"statistics": snapshot})
    phase_one = events[: 2 * len(item.decisions)]
    assert all(event[0] == "score" and event[2] is False for event in phase_one)
    phase_two = events[2 * len(item.decisions) :]
    assert len(phase_two) == 2 * len(item.decisions)
    for index, decision in enumerate(item.decisions):
        scored, backward = phase_two[2 * index : 2 * index + 2]
        assert scored == ("score", decision.state_id, True, False)
        assert backward[:2] == ("backward", decision.state_id)
        assert backward[2] == pytest.approx(oracle.action_coefficients[index], rel=2e-6, abs=1e-7)
    assert flow_gradients == pytest.approx(oracle.state_coefficients[:-1], rel=2e-6, abs=1e-7)
    assert flow_gradients[0] != 0.0


def test_streaming_is_default_and_unknown_gradient_modes_fail():
    assert EvoOptimizerConfig().gradient_mode == "streaming"
    with pytest.raises(ValueError, match="gradient_mode"):
        EvoOptimizerConfig(gradient_mode="old-single-coefficient")


def test_each_parameter_group_carries_its_own_beta1_at_the_default_beta2():
    """The actor's first moment is set apart from the heads', beta2 is not."""
    default = EvoOptimizerConfig()
    assert (default.actor_beta1, default.head_beta1) == (0.9, 0.9)
    probe = torch.optim.AdamW([torch.zeros(1, requires_grad=True)])
    assert probe.defaults["betas"] == (default.actor_beta1, ADAM_BETA2)
    policy, heads, trainer = initialized("streaming")

    def betas(**settings):
        replaced = EvoTrainer(policy, heads, ("synthetic",), replace(trainer.config, **settings))
        return [group["betas"] for group in replaced.optimizer.param_groups]

    # The actor group is first, the heads second, as the trainer builds them.
    assert [group["betas"] for group in trainer.optimizer.param_groups] == [
        (0.9, ADAM_BETA2),
        (0.9, ADAM_BETA2),
    ]
    assert betas(actor_beta1=0.95) == [(0.95, ADAM_BETA2), (0.9, ADAM_BETA2)]
    assert betas(head_beta1=0.5) == [(0.9, ADAM_BETA2), (0.5, ADAM_BETA2)]
    assert betas(actor_beta1=0.0, head_beta1=0.99) == [(0.0, ADAM_BETA2), (0.99, ADAM_BETA2)]


@pytest.mark.parametrize("bad", [1.0, 1.5, -0.1, float("nan"), float("inf")])
@pytest.mark.parametrize("name", ["actor_beta1", "head_beta1"])
def test_first_moment_decays_must_be_finite_and_in_the_unit_interval(name, bad):
    with pytest.raises(ValueError, match=name):
        EvoOptimizerConfig(**{name: bad})


def test_value_and_outcome_labels_include_paired_suffixes_but_never_forced_roots_or_current(
    monkeypatch,
):
    policy, heads, trainer = initialized("streaming")
    trajectories, snapshots = examples(policy)
    observed_value = []
    observed_outcomes = []
    value_loss = heads.runtime_loss
    outcome_losses = heads.outcome_losses

    def record_value(features, task_types, rewards):
        observed_value.extend(zip(features.detach().clone(), rewards.detach().clone(), strict=True))
        return value_loss(features, task_types, rewards)

    def record_outcomes(encoding, features, rewards):
        observed_outcomes.extend(
            zip(features.detach().clone(), rewards.detach().clone(), strict=True)
        )
        return outcome_losses(encoding, features, rewards)

    monkeypatch.setattr(heads, "runtime_loss", record_value)
    monkeypatch.setattr(heads, "outcome_losses", record_outcomes)
    metrics = trainer.update(trajectories, snapshots)
    expected = [
        (torch.tensor(decision.features), torch.tensor(item.reward))
        for item in trajectories
        for index, decision in enumerate(item.decisions)
        if item.source == "natural_reference" or (item.source.startswith("paired_") and index >= 1)
    ]
    assert len(expected) == 10
    assert metrics["supervised_reference_states"] == 10
    for observed in (observed_value, observed_outcomes):
        assert len(observed) == len(expected)
        for actual, wanted in zip(observed, expected, strict=True):
            torch.testing.assert_close(actual[0], wanted[0])
            torch.testing.assert_close(actual[1], wanted[1])
        assert all(float(reward) != pytest.approx(0.7) for _, reward in observed)


@pytest.mark.parametrize("changed_loss", ["value", "outcome"])
def test_supervised_labels_and_weights_cannot_change_actor_or_flow_even_when_clipping(
    monkeypatch,
    changed_loss,
):
    baseline_policy, baseline_heads, baseline = initialized("streaming", gradient_clip=0.01)
    changed_policy, changed_heads, initial_trainer = initialized("streaming", gradient_clip=0.01)
    config = replace(initial_trainer.config, **{f"{changed_loss}_loss_weight": 1000.0})
    changed = EvoTrainer(changed_policy, changed_heads, ("synthetic",), config)
    trajectories, snapshots = examples(baseline_policy, cached_encoding=True)
    if changed_loss == "outcome":
        original = changed_heads.outcome_losses

        def flipped_outcome(encoding, features, targets):
            return original(encoding, features, 1 - targets)

        monkeypatch.setattr(changed_heads, "outcome_losses", flipped_outcome)
    else:
        original = changed_heads.runtime_loss

        def flipped_value(features, types, targets):
            return original(features, types, 1 - targets)

        monkeypatch.setattr(changed_heads, "runtime_loss", flipped_value)
    baseline_metrics = baseline.update(trajectories, snapshots)
    changed_metrics = changed.update(trajectories, snapshots)
    assert baseline_metrics["gradient_norm"] > baseline.config.gradient_clip
    assert changed_metrics[f"{changed_loss}_gradient_norm"] > changed.config.gradient_clip
    assert baseline_metrics["anchor_tb_loss"] == changed_metrics["anchor_tb_loss"]
    assert baseline_metrics["gradient_norm"] == changed_metrics["gradient_norm"]
    for left, right in zip(
        (*baseline_policy.trainable_parameters(), *baseline_heads.residual_head.parameters()),
        (*changed_policy.trainable_parameters(), *changed_heads.residual_head.parameters()),
        strict=True,
    ):
        assert torch.equal(left.grad, right.grad)
        assert torch.equal(left, right)
    left_head = (
        baseline_heads.runtime_head if changed_loss == "value" else baseline_heads.outcome_heads
    )
    right_head = (
        changed_heads.runtime_head if changed_loss == "value" else changed_heads.outcome_heads
    )
    assert any(
        not torch.equal(left, right)
        for left, right in zip(
            left_head.parameters(),
            right_head.parameters(),
            strict=True,
        )
    )


def test_task_external_prior_reaches_measured_training_anchor(monkeypatch):
    policy, heads, trainer = initialized("streaming")
    item = trajectory(policy, "current", 0.7)
    item = replace(item, task=replace(item.task, prior_rate=0.9, prior_count=17.0))
    snapshot = ReferenceStatistics("statistics").snapshot("gradient-batch")
    seen = []
    original = type(snapshot).task_anchor

    def anchor(self, task_id, task_family, **kwargs):
        seen.append(kwargs)
        return original(self, task_id, task_family, **kwargs)

    monkeypatch.setattr(type(snapshot), "task_anchor", anchor)
    trainer.update((item,), {"statistics": snapshot})
    assert seen == [{"exclude_observation_id": None, "prior_rate": 0.9, "prior_count": 17.0}]
