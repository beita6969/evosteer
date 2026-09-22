from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from skillev.training.optimizer_state import (
    checkpoint_optimizer_state,
    require_optimizer_state_layout,
)


class NamedParameters:
    def __init__(self):
        self.values = {
            name: torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            for name in ("forward.a", "forward.b", "backward.a", "z.weight")
        }

    def named_trainable_parameters(self):
        return self.values


def fixture():
    backbone = NamedParameters()
    optimizer = torch.optim.AdamW(
        [
            {"name": role, "params": [p for n, p in backbone.values.items() if n.startswith(role)]}
            for role in ("forward", "backward", "z")
        ]
    )
    for index, parameter in enumerate(backbone.values.values()):
        parameter.grad = torch.full_like(parameter, index + 1.0)
    optimizer.step()
    return backbone, optimizer


def test_named_optimizer_moments_roundtrip_without_changing_group_identity():
    backbone, optimizer = fixture()
    saved = checkpoint_optimizer_state(backbone, optimizer)
    restored_backbone, restored = fixture()
    require_optimizer_state_layout(saved, restored_backbone, restored)
    restored.load_state_dict(saved)
    assert restored.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]
    torch.testing.assert_close(restored.state_dict()["state"], optimizer.state_dict()["state"])


def test_same_shape_parameter_reordering_is_rejected():
    backbone, optimizer = fixture()
    saved = checkpoint_optimizer_state(backbone, optimizer)
    optimizer.param_groups[0]["params"].reverse()
    with pytest.raises(ValueError):
        require_optimizer_state_layout(saved, backbone, optimizer)


@pytest.mark.parametrize("fault", ["missing-layout", "saved-order", "group", "shape", "nonfinite"])
def test_optimizer_restore_rejects_semantically_incompatible_state(fault):
    backbone, optimizer = fixture()
    saved = deepcopy(checkpoint_optimizer_state(backbone, optimizer))
    if fault == "missing-layout":
        saved.pop("skillev_parameter_layout")
    elif fault == "saved-order":
        saved["param_groups"][0]["params"].reverse()
    elif fault == "group":
        saved["param_groups"][0]["name"] = "backward"
    elif fault == "shape":
        saved["state"][0]["exp_avg"] = torch.zeros(3)
    else:
        saved["state"][0]["exp_avg"][0] = float("nan")
    with pytest.raises(ValueError):
        require_optimizer_state_layout(saved, backbone, optimizer)
