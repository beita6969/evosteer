from dataclasses import replace

import torch

from skillev.training.performance_config import TrainingPerformanceConfig


def test_cache_reuses_only_frozen_features_and_keeps_z_trainable(backbone, monkeypatch):
    backbone.configure_performance(replace(TrainingPerformanceConfig(), z_feature_cache_entries=2))
    original = backbone._model.forward
    calls = []

    def forward(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(backbone._model, "forward", forward)
    first = backbone.z_value((1, 5))
    second = backbone.z_value((1, 5))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert len(calls) == 1
    second.backward()
    groups = backbone.parameter_groups()
    assert all(p.grad is not None for p in groups.z_head)
    assert all(p.grad is None for p in (*groups.forward, *groups.backward))
    with torch.no_grad():
        groups.z_head[-1].add_(1)
    assert not torch.equal(first, backbone.z_value((1, 5)))
    backbone.reset_z(42)
    backbone.z_value((1, 5)).backward()
    assert len(calls) == 1
    backbone.z_value((1, 6))
    backbone.z_value((1, 7))
    backbone.z_value((1, 5))
    assert len(calls) == 4
    assert backbone.z_feature_cache_metrics["entries"] == 2
    backbone.clear_z_feature_cache()
    assert backbone.z_feature_cache_metrics["entries"] == 0
