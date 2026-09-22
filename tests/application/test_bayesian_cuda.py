"""Opt-in real-Qwen numerical integration, not a benchmark or on-policy experiment.

The existing vertical fixture supplies synthetic calls, terminal labels and
author text. Only scoring/backpropagation, AdamW, Z, projection, Phi and durable
resume are under test. Production thresholds and sampling are not changed.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path

import pytest
import torch

from skillev.application_reporting import resolved_method_state
from skillev.policy import QwenMultimodalBackboneConfig, QwenMultimodalPolicyBackbone
from skillev.training.performance_config import TrainingPerformanceConfig
from skillev.training.step_math import named_ttb_parameters
from tests.application.test_bayesian_recovery import restore_fixture
from tests.application.test_full_vertical_loop import (
    _optimizer_steps,
    _ScriptedQwenBackbone,
    build_application_fixture,
)


class _ScriptedCudaQwen(QwenMultimodalPolicyBackbone):
    generate_policy = _ScriptedQwenBackbone.generate_policy
    generate_base = _ScriptedQwenBackbone.generate_base


@pytest.mark.skipif(
    not os.environ.get("SKILLEV_CUDA_BACKBONE_CONFIG"),
    reason="requires an explicitly selected CUDA card and private real-Qwen configuration",
)
def test_real_qwen_full_batch_posterior_evolution_z_and_resume(tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    config = QwenMultimodalBackboneConfig.from_value(
        json.loads(Path(os.environ["SKILLEV_CUDA_BACKBONE_CONFIG"]).read_text())
    )
    assert config.device == "cuda"
    assert config.torch_dtype == "bfloat16"
    performance = TrainingPerformanceConfig()
    performance.configure_process()
    torch.manual_seed(0)
    started = time.monotonic()
    backbone = _ScriptedCudaQwen(config, performance=performance)
    backbone.policy_prompts = []
    backbone.authoring_prompts = []
    fixture = build_application_fixture(tmp_path, config, backbone=backbone, batch_size=32)
    application = fixture.application
    initial_library = application.library.current_version
    print("Qwen loaded; starting two full 32-trajectory synthetic batches", flush=True)
    asyncio.run(application.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=2))
    state = application.projections.runtime_state()
    assert application.library.current_version != initial_library
    state.posterior_provenance.require_reconstructed_cells(state.calibration_cells)
    for batch in state.posterior_provenance.batches:
        assert len(batch.trajectory_ids) == 32
        assert len(batch.posterior.updates) == 32
        assert math.fsum(update.flow_weight for update in batch.posterior.updates) == pytest.approx(
            32
        )
        assert all(
            math.isfinite(update.flow_weight) and update.flow_weight > 0
            for update in batch.posterior.updates
        )
    groups = backbone.parameter_groups()
    optimizer = application.training_loop.optimizer
    assert set(_optimizer_steps(optimizer, (*groups.forward, *groups.backward))) == {2}
    assert all(
        parameter not in optimizer.state and parameter.grad is None for parameter in groups.z_head
    )
    before_resume = resolved_method_state(application)
    parameters = named_ttb_parameters(groups)
    expected = {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}
    with torch.no_grad():
        for parameter in parameters.values():
            parameter.add_(1.0)
    directory = tmp_path / "snapshots" / "step-00000002"
    recovered, _ = restore_fixture(fixture, directory, config, backbone=backbone)
    assert recovered.projections.runtime_state() == state
    assert resolved_method_state(recovered) == before_resume
    for name, parameter in parameters.items():
        torch.testing.assert_close(parameter.detach().cpu(), expected[name], rtol=0, atol=0)
    print(
        "Post-evolution checkpoint restored exactly; starting new-library closure batch", flush=True
    )
    summary = asyncio.run(recovered.evolution_loop.run(fixture.run_plan))
    assert summary.final_optimizer_step == 3
    assert summary.cycles_committed_in_run == 1
    batches = recovered.projections.posterior_provenance.batches
    assert [batch.optimizer_step for batch in batches] == [1, 2, 3]
    assert batches[-1].library_version == recovered.library.current_version
    assert batches[-1].posterior.updates == ()
    assert set(_optimizer_steps(recovered.training_loop.optimizer, groups.z_head)) == {1}
    assert set(
        _optimizer_steps(recovered.training_loop.optimizer, (*groups.forward, *groups.backward))
    ) == {3}
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "synthetic_trajectories": 96,
                "steps": 3,
                "seconds": elapsed,
                "trajectories_per_second": 96 / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        ),
        flush=True,
    )
