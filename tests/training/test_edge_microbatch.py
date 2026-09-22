import asyncio

import pytest
import torch

from skillev.policy.scoring_execution import TeacherForcingConfig
from skillev.policy.teacher_forcing import TeacherForcingExecutor
from skillev.training.step_math import compute_ttb_gradient_shard


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("bias", [-64.0, 64.0])
def test_microbatch_keeps_ttb_k_t_and_batch_scaling(make_training_harness, size, bias):
    torch.set_num_threads(1)
    harness = make_training_harness()
    batch = asyncio.run(harness.loop.collect_batch())
    backbone = harness.backbone
    with torch.no_grad():
        backbone._z_head[-1].bias.fill_(bias)
    args = {
        "backbone": backbone,
        "parameters": backbone.parameter_groups(),
        "batch": batch,
        "positions": tuple(range(len(batch.artifacts))),
        "global_batch_size": len(batch.artifacts),
        "temperature_beta": harness.config.method.temperature_beta,
    }

    reference = compute_ttb_gradient_shard(**args)
    backbone._teacher_forcing = TeacherForcingExecutor(TeacherForcingConfig(microbatch_size=size))
    actual = compute_ttb_gradient_shard(**args)
    for name in reference.gradients:
        torch.testing.assert_close(
            actual.gradients[name], reference.gradients[name], rtol=1e-5, atol=1e-6
        )
    for item, expected in zip(actual.artifacts, reference.artifacts, strict=True):
        assert item.residual.delta == pytest.approx(expected.residual.delta, rel=1e-5, abs=1e-6)
        assert (item.residual.delta > 0) == (bias > 0)


def test_microbatch_oom_retains_failure_coordinate_and_clears_temporary_gradients(
    make_training_harness, monkeypatch
):
    from skillev.policy.interface import PolicyScoringMemoryError
    from skillev.scoring.objective import TrajectoryScoringMemoryError
    from skillev.training.step_math import named_ttb_parameters

    harness = make_training_harness()
    batch = asyncio.run(harness.loop.collect_batch())
    backbone = harness.backbone
    backbone._teacher_forcing = TeacherForcingExecutor(TeacherForcingConfig(microbatch_size=2))

    def fail(edges, role):
        raise PolicyScoringMemoryError(
            prefix_token_count=len(edges[0][0]), action_token_count=len(edges[0][1]), role=role
        )

    monkeypatch.setattr(backbone, "score_edge_microbatch", fail)
    with pytest.raises(TrajectoryScoringMemoryError) as caught:
        compute_ttb_gradient_shard(
            backbone=backbone,
            parameters=backbone.parameter_groups(),
            batch=batch,
            positions=(0,),
            global_batch_size=len(batch.artifacts),
            temperature_beta=1.0,
        )
    assert caught.value.trajectory_id == batch.artifacts[0].record.trajectory_id
    assert caught.value.microbatch_steps
    assert all(p.grad is None for p in named_ttb_parameters(backbone.parameter_groups()).values())
