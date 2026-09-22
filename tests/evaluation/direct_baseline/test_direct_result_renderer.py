from __future__ import annotations

import argparse
import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from skillev.evaluation.direct_baseline.protocol import load_direct_reference_protocol

_ROOT = Path(__file__).parents[3]
_CONFIG = _ROOT / "configs/evaluation/qwen35_skillflow_direct_reference.yaml"


def _renderer() -> Callable[[argparse.Namespace], str]:
    namespace = runpy.run_path(str(_ROOT / "scripts/render_qwen35_direct_results.py"))
    return cast(Callable[[argparse.Namespace], str], namespace["_render"])


def _aggregate(tmp_path: Path) -> Path:
    protocol = load_direct_reference_protocol(_CONFIG)
    result: dict[str, object] = {}
    for spec in protocol.benchmarks:
        if spec.role != "iid":
            continue
        result[spec.benchmark.value] = {
            "benchmark": spec.benchmark.value,
            "comparability": spec.comparability.value,
            "population": spec.population,
            "dataset_revision": spec.dataset_revision,
            "selection_rule": spec.selection_rule,
            "prompt_profile": spec.prompt_profile,
            "decoding_profile": spec.decoding_profile,
            "parser_profile": spec.parser_profile,
            "scorer_profile": spec.scorer_profile,
            "seed_aggregation": spec.seed_aggregation.mode.value,
            "availability": "complete",
            "coverage": {
                "planned_count": spec.sample_count,
                "final_record_count": spec.sample_count,
                "candidate_response_count": spec.sample_count,
                "definitive_verdict_count": spec.sample_count,
                "generation_infrastructure_failures": 0,
                "scorer_infrastructure_failures": 0,
                "environment_infrastructure_failures": 0,
            },
            "metrics": [
                {
                    "metric": metric.metric_id,
                    "reference_percent": str(metric.reference_percent),
                    "diagnostic_observed_percent": str(metric.reference_percent),
                    "formal_observed_percent": str(metric.reference_percent),
                    "absolute_gap_pp": "0",
                    "numeric_status": "PASS",
                }
                for metric in spec.metrics
            ],
            "submission_rate_percent": "100",
            "scorer_reach_rate_percent": "100",
        }
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def _arguments(aggregate: Path, receipt: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=_CONFIG,
        aggregate_json=[aggregate],
        output=aggregate.parent / "results.md",
        role="iid",
        engineering_check="not-run",
        engineering_receipt=receipt,
    )


def test_renderer_projects_complete_machine_results(tmp_path: Path) -> None:
    aggregate = _aggregate(tmp_path)
    receipt = tmp_path / "engineering.json"
    receipt.write_text(
        json.dumps({"status": "passed", "command": 'CUDA_VISIBLE_DEVICES="" make check'}),
        encoding="utf-8",
    )

    rendered = _renderer()(_arguments(aggregate, receipt))

    assert "Observed (full)" in rendered
    assert "Scientific comparability" in rendered
    assert "Infra (gen/scorer/env)" in rendered
    assert "`<7.0 pp`" in rendered
    assert "Published aggregate rows" in rendered
    assert "Engineering validation: **PASSED**" in rendered
    assert "Formal training: **NOT-EVALUATED**" in rendered


def test_renderer_rejects_missing_requested_benchmark(tmp_path: Path) -> None:
    aggregate = _aggregate(tmp_path)
    value = json.loads(aggregate.read_text(encoding="utf-8"))
    del value["swe-bench"]
    aggregate.write_text(json.dumps(value), encoding="utf-8")
    receipt = tmp_path / "engineering.json"
    receipt.write_text(json.dumps({"status": "not-run"}), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly cover"):
        _renderer()(_arguments(aggregate, receipt))


def test_renderer_rejects_forged_pass_status(tmp_path: Path) -> None:
    aggregate = _aggregate(tmp_path)
    value = json.loads(aggregate.read_text(encoding="utf-8"))
    metric = value["hotpotqa"]["metrics"][0]
    metric["formal_observed_percent"] = "0"
    metric["absolute_gap_pp"] = "0"
    metric["numeric_status"] = "PASS"
    aggregate.write_text(json.dumps(value), encoding="utf-8")
    receipt = tmp_path / "engineering.json"
    receipt.write_text(json.dumps({"status": "not-run"}), encoding="utf-8")

    with pytest.raises(ValueError, match="recomputed"):
        _renderer()(_arguments(aggregate, receipt))
