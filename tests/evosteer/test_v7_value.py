"""Algorithm 1 step 5: the runtime value head is fitted, not stepped once per batch."""

from __future__ import annotations

import asyncio
import io
from dataclasses import replace

import pytest
import torch

from skillev.evosteer_application import EvoSteerApplication
from skillev.training.evosteer import (
    VALUE_REPLAY_KEY,
    EvoOptimizerConfig,
    EvoTrainer,
    family_mean_loss,
)
from tests.evosteer.test_application import application, binding
from tests.evosteer.test_training_gradients import examples, initialized


def fitted(**optimizer):
    """Identical tiny actor/heads per call, trained under the given value-fit settings."""
    policy, heads, base = initialized("streaming")
    trainer = EvoTrainer(policy, heads, ("synthetic",), replace(base.config, **optimizer))
    return policy, heads, trainer


def batch(policy, batch_id):
    trajectories, snapshots = examples(policy)
    return tuple(replace(item, batch_id=batch_id) for item in trajectories), snapshots


def non_value_parameters(policy, heads):
    return (
        *policy.trainable_parameters(),
        *heads.residual_head.parameters(),
        *heads.outcome_heads.parameters(),
    )


def optimizer_steps(trainer, parameters):
    return {int(trainer.optimizer.state[parameter]["step"].item()) for parameter in parameters}


def eligible_states(trajectories):
    """Legal reference states in trainer order: natural states and paired suffixes."""
    return [
        (decision.features, item.reward)
        for item in trajectories
        for index, decision in enumerate(item.decisions)
        if item.source == "natural_reference" or (item.source.startswith("paired_") and index >= 1)
    ]


@pytest.mark.parametrize("bad", [0, -1, 1.5, 2.0, True, "2", None])
def test_value_fit_settings_must_be_positive_integers(bad):
    with pytest.raises(ValueError, match="value_fit_steps"):
        EvoOptimizerConfig(value_fit_steps=bad)
    with pytest.raises(ValueError, match="value_replay_batches"):
        EvoOptimizerConfig(value_fit_steps=2, value_replay_batches=bad)


def test_defaults_keep_the_single_joint_step_and_an_unused_window_is_rejected():
    config = EvoOptimizerConfig()
    assert (config.value_fit_steps, config.value_replay_batches) == (1, 1)
    with pytest.raises(ValueError, match="requires value_fit_steps"):
        EvoOptimizerConfig(value_replay_batches=4)
    policy, _, trainer = fitted()
    metrics = trainer.update(*examples(policy))
    assert metrics["value_fit_steps"] == 1
    assert metrics["value_fit_states"] == metrics["supervised_reference_states"] == 10
    assert all(int(state["step"].item()) == 1 for state in trainer.optimizer.state.values())
    # The default optimizer checkpoint format is unchanged.
    assert set(trainer.optimizer.state_dict()) == {"state", "param_groups"}
    # Labels: natural 4 x 0.2, treatment suffix 3 x 1.0, control suffix 3 x 0.0.
    assert metrics["value_family_mean_loss"] == pytest.approx(
        (4 * 0.18**2 + 3 * 0.62**2 + 3 * 0.38**2) / 10
    )


def test_family_mean_loss_is_the_in_sample_family_base_rate_error():
    assert family_mean_loss([]) == 0.0
    labels = [(0, 1.0, 3), (0, 0.0, 1), (1, 0.5, 4)]
    assert family_mean_loss(labels) == pytest.approx((3 * 0.25**2 + 0.75**2) / 8)


def test_extra_value_passes_fit_only_the_runtime_head_and_keep_pre_fit_value_loss():
    base_policy, base_heads, baseline = fitted()
    fit_policy, fit_heads, fitter = fitted(value_fit_steps=50)
    for heads in (base_heads, fit_heads):
        with torch.no_grad():
            # Start far from every label (v_hat near 0.88) so the loss
            # comparison below does not depend on where the random init lands.
            heads.runtime_head[-1].bias.fill_(2.0)
    trajectories, snapshots = examples(base_policy)
    base_metrics = baseline.update(trajectories, snapshots)
    fit_metrics = fitter.update(trajectories, snapshots)
    # The actor/flow objective, its clipping and the joint-step diagnostics
    # are identical; value_loss is the loss before any value step.
    for name in (
        "anchor_tb_loss",
        "gradient_norm",
        "value_loss",
        "value_gradient_norm",
        "outcome_loss",
        "outcome_gradient_norm",
        "value_family_mean_loss",
        "value_fit_states",
    ):
        assert fit_metrics[name] == base_metrics[name]
    base_rest = non_value_parameters(base_policy, base_heads)
    fit_rest = non_value_parameters(fit_policy, fit_heads)
    for left, right in zip(base_rest, fit_rest, strict=True):
        assert torch.equal(left, right)
        # Joint-step gradients are restored after the value-only passes.
        assert torch.equal(left.grad, right.grad)
        left_state, right_state = baseline.optimizer.state[left], fitter.optimizer.state[right]
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(left_state[key], right_state[key])
    assert optimizer_steps(fitter, fit_rest) == {1}
    assert optimizer_steps(fitter, fit_heads.runtime_head.parameters()) == {50}
    assert fit_metrics["value_fit_steps"] == 50
    assert any(
        not torch.equal(left, right)
        for left, right in zip(
            base_heads.runtime_head.parameters(), fit_heads.runtime_head.parameters(), strict=True
        )
    )
    # The extra passes reduce the post-fit error on the same states.
    assert fit_metrics["value_fit_loss"] < base_metrics["value_fit_loss"]
    assert fit_metrics["value_fit_loss"] < fit_metrics["value_loss"]


def test_zero_value_weight_leaves_the_runtime_head_untouched_even_with_extra_passes():
    policy, heads, trainer = fitted(value_fit_steps=5, value_loss_weight=0.0)
    before = {name: value.clone() for name, value in heads.runtime_head.state_dict().items()}
    metrics = trainer.update(*examples(policy))
    assert metrics["value_fit_steps"] == 0
    for name, value in heads.runtime_head.state_dict().items():
        assert torch.equal(value, before[name])


def test_replay_window_is_bounded_and_each_extra_pass_uses_the_whole_window(monkeypatch):
    policy, heads, trainer = fitted(value_fit_steps=4, value_replay_batches=2)
    sizes = []
    original = heads.runtime_loss

    def record(features, task_types, rewards):
        sizes.append(int(rewards.numel()))
        return original(features, task_types, rewards)

    monkeypatch.setattr(heads, "runtime_loss", record)
    for index in (1, 2, 3):
        sizes.clear()
        trajectories, snapshots = batch(policy, f"batch-{index}")
        metrics = trainer.update(trajectories, snapshots)
        window = 10 if index == 1 else 20
        # Joint step per trajectory (natural 4, paired suffixes 3 and 3), then
        # three full-window value-only passes.
        assert sizes == [4, 3, 3, window, window, window]
        assert metrics["value_fit_states"] == window
        assert metrics["value_fit_steps"] == 4
        stored = trainer.optimizer.state_dict()[VALUE_REPLAY_KEY]
        assert stored["capacity"] == 2
        assert [item["batch_id"] for item in stored["batches"]] == [f"batch-{index}"]
        expected = eligible_states(trajectories)
        (entry,) = stored["batches"]
        torch.testing.assert_close(
            entry["features"], torch.tensor([f for f, _ in expected], dtype=torch.float32)
        )
        torch.testing.assert_close(
            entry["rewards"], torch.tensor([r for _, r in expected], dtype=torch.float32)
        )
        assert torch.equal(entry["task_types"], torch.zeros(len(expected), dtype=torch.long))
    version = policy.version
    with pytest.raises(ValueError, match="already fitted"):
        trainer.update(*batch(policy, "batch-3"))
    assert policy.version == version


def test_value_fit_with_replay_is_deterministic():
    runs = []
    for _ in range(2):
        policy, heads, trainer = fitted(value_fit_steps=8, value_replay_batches=3)
        metrics = [trainer.update(*batch(policy, f"batch-{index}")) for index in range(1, 5)]
        runs.append((heads.runtime_head.state_dict(), metrics))
    (left_head, left_metrics), (right_head, right_metrics) = runs
    assert left_metrics == right_metrics
    for name, value in left_head.items():
        assert torch.equal(value, right_head[name])


def test_value_replay_survives_a_weights_only_checkpoint_and_resumes_exactly():
    settings = {"value_fit_steps": 6, "value_replay_batches": 3}
    policy, heads, trainer = fitted(**settings)
    for index in (1, 2):
        trainer.update(*batch(policy, f"batch-{index}"))
    stream = io.BytesIO()
    # The same payload shape and loader the application checkpoint uses.
    torch.save(
        {
            "actor": policy.adapter_state(),
            "heads": heads.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
        },
        stream,
    )
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    restored_policy, restored_heads, restored = fitted(**settings)
    restored_policy.load_adapter_state(saved["actor"])
    restored_heads.load_state_dict(saved["heads"], strict=True)
    restored.optimizer.load_state_dict(saved["optimizer"])
    # Loading consumes only torch's shallow copy of the caller's dict.
    assert VALUE_REPLAY_KEY in saved["optimizer"]
    stored = restored.optimizer.state_dict()[VALUE_REPLAY_KEY]
    assert [item["batch_id"] for item in stored["batches"]] == ["batch-1", "batch-2"]
    uninterrupted = trainer.update(*batch(policy, "batch-3"))
    resumed = restored.update(*batch(restored_policy, "batch-3"))
    assert resumed == uninterrupted
    assert uninterrupted["value_fit_states"] == 30
    for name, value in heads.state_dict().items():
        assert torch.equal(value, restored_heads.state_dict()[name])


def test_value_replay_state_is_validated_before_it_replaces_the_window():
    settings = {"value_fit_steps": 3, "value_replay_batches": 3}
    policy, _, trainer = fitted(**settings)
    trainer.update(*batch(policy, "batch-1"))
    good = trainer.optimizer.state_dict()
    # A config without a window cannot silently drop a stored one, and vice versa.
    _, _, plain = fitted(value_fit_steps=3)
    with pytest.raises(ValueError, match="replay"):
        plain.optimizer.load_state_dict(good)
    _, _, other = fitted(**settings)
    with pytest.raises(ValueError, match="replay"):
        other.optimizer.load_state_dict({k: v for k, v in good.items() if k != VALUE_REPLAY_KEY})
    entry = good[VALUE_REPLAY_KEY]["batches"][0]
    nan_rewards = entry["rewards"].clone()
    nan_rewards[0] = float("nan")
    malformed_batches = [
        [{**entry, "rewards": nan_rewards}],
        [{**entry, "rewards": entry["rewards"] + 1.5}],
        [{**entry, "features": entry["features"][:, :-1]}],
        [{**entry, "features": entry["features"].double()}],
        [{**entry, "task_types": entry["task_types"] + 1}],
        [{**entry, "batch_id": ""}],
        [{key: value for key, value in entry.items() if key != "rewards"}],
        [entry, entry],
        [entry, {**entry, "batch_id": "b2"}, {**entry, "batch_id": "b3"}],
    ]
    variants = [
        {**good[VALUE_REPLAY_KEY], "batches": batches} for batches in malformed_batches
    ]
    variants.append({**good[VALUE_REPLAY_KEY], "capacity": 4})
    variants.append({**good[VALUE_REPLAY_KEY], "format": "evosteer-value-replay@0"})
    for variant in variants:
        with pytest.raises(ValueError, match="replay"):
            other.optimizer.load_state_dict({**good, VALUE_REPLAY_KEY: variant})
        # Nothing was committed: no AdamW state and an empty window.
        assert not other.optimizer.state
        assert other.optimizer.state_dict()[VALUE_REPLAY_KEY]["batches"] == []
    other.optimizer.load_state_dict(good)
    stored = other.optimizer.state_dict()[VALUE_REPLAY_KEY]
    assert [item["batch_id"] for item in stored["batches"]] == ["batch-1"]


def test_nonfinite_value_fit_raises_and_restores_joint_gradients_without_committing(
    monkeypatch,
):
    policy, heads, trainer = fitted(value_fit_steps=3, value_replay_batches=2)
    original = heads.runtime_loss
    calls = []

    def poisoned(features, task_types, rewards):
        calls.append(int(rewards.numel()))
        loss = original(features, task_types, rewards)
        # The first three calls are the joint step's per-trajectory losses.
        return loss if len(calls) <= 3 else loss * float("nan")

    monkeypatch.setattr(heads, "runtime_loss", poisoned)
    with pytest.raises(FloatingPointError, match="value-head fit"):
        trainer.update(*examples(policy))
    assert trainer.optimizer.state_dict()[VALUE_REPLAY_KEY]["batches"] == []
    assert policy.model.actor_bias.grad is not None


def value_application(**optimizer):
    base = application()
    config = replace(base.config, optimizer=replace(base.config.optimizer, **optimizer))
    return EvoSteerApplication(base.policy, config, admission=base.admission)


def test_application_checkpoint_carries_the_value_replay_window(tmp_path):
    settings = {"value_fit_steps": 5, "value_replay_batches": 3}
    app = value_application(**settings)
    first = asyncio.run(app.train_batch((binding("task-1"),)))
    assert first.metrics["value_fit_steps"] == 5
    assert first.metrics["value_fit_states"] == first.metrics["supervised_reference_states"]
    path = app.save_checkpoint(tmp_path / "step-one")
    restored = value_application(**settings)
    restored.load_checkpoint(path)
    saved = app.trainer.optimizer.state_dict()[VALUE_REPLAY_KEY]["batches"]
    loaded = restored.trainer.optimizer.state_dict()[VALUE_REPLAY_KEY]["batches"]
    assert [item["batch_id"] for item in loaded] == [first.batch_id]
    for left, right in zip(saved, loaded, strict=True):
        assert left["batch_id"] == right["batch_id"]
        for key in ("features", "task_types", "rewards"):
            assert torch.equal(left[key], right[key])
    second = asyncio.run(restored.train_batch((binding("task-2"),)))
    assert second.metrics["value_fit_states"] == (
        first.metrics["supervised_reference_states"]
        + second.metrics["supervised_reference_states"]
    )
