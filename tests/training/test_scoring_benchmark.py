"""A completed baseline is not evidence that an interrupted candidate passed."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import benchmark_teacher_forcing as benchmark
from skillev.training.performance_config import TrainingPerformanceConfig


@pytest.mark.parametrize("interrupt_candidate", [False, True])
def test_comparison_requires_every_requested_variant(tmp_path, monkeypatch, interrupt_candidate):
    output = tmp_path / "comparison.json"
    inputs = tmp_path / "batch.json"
    inputs.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_teacher_forcing",
            "--preparation",
            "unused",
            "--batch",
            str(inputs),
            "--performance",
            "unused",
            "--output",
            str(output),
            "--sizes",
            "1",
            "--checkpoint-min-tokens",
            "1",
            "4096",
            "--temperature-beta",
            "1",
        ],
    )
    monkeypatch.setattr(benchmark, "gpu_state", lambda: "CPU report-lifecycle fixture")
    monkeypatch.setattr(
        benchmark.TrainingPerformanceConfig, "load", lambda _: TrainingPerformanceConfig()
    )
    monkeypatch.setattr(
        benchmark,
        "_read_preparation",
        lambda _: (
            SimpleNamespace(torch_dtype="bfloat16"),
            SimpleNamespace(directory=tmp_path, trainable_state=None),
        ),
    )
    backbone = SimpleNamespace(
        tokenizer=None,
        load_checkpoint=lambda _: None,
        bind_initial_trainable_state=lambda _: None,
        configure_performance=lambda _: None,
        parameter_groups=lambda: None,
    )
    monkeypatch.setattr(benchmark, "build_qwen_policy_backbone", lambda *a, **kw: backbone)
    monkeypatch.setattr(
        benchmark.CollectedTrainingBatch,
        "from_value",
        lambda *a, **kw: SimpleNamespace(artifacts=(None,)),
    )
    for name in (
        "synchronize",
        "reset_peak_memory_stats",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(benchmark, "scoring_telemetry", lambda _: {})
    monkeypatch.setattr(benchmark, "fla_execution_metrics", dict)
    calls = 0

    def compute(**kwargs):
        nonlocal calls
        calls += 1
        if interrupt_candidate and calls == 3:
            raise RuntimeError("candidate scoring interrupted")
        return SimpleNamespace(
            gradients={"forward.weight": torch.tensor([1.0])},
            artifacts=(
                SimpleNamespace(
                    edges=(
                        SimpleNamespace(
                            forward_logprob_per_token=-1.0, backward_logprob_per_token=-2.0
                        ),
                    )
                ),
            ),
        )

    monkeypatch.setattr(benchmark, "compute_ttb_gradient_shard", compute)
    if interrupt_candidate:
        with pytest.raises(RuntimeError):
            benchmark.main()
    else:
        benchmark.main()
    report = json.loads(output.read_text())
    assert report["complete"] is (not interrupt_candidate)
    assert report["all_within_bounds"] is (not interrupt_candidate)
    assert len(report["variants"]) == (1 if interrupt_candidate else report["planned_variants"])
