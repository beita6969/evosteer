from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

from skillev.evaluation.direct_baseline import (
    DirectBenchmark,
    NativeInteractiveAttempt,
)
from skillev.evaluation.direct_baseline.protocol import load_direct_reference_protocol

_ROOT = Path(__file__).parents[3]


def test_interactive_aggregate_partitions_infrastructure_failures() -> None:
    namespace = runpy.run_path(str(_ROOT / "scripts/run_qwen35_direct_interactive.py"))
    aggregate = cast(
        Callable[[list[NativeInteractiveAttempt], object], dict[str, object]],
        namespace["_aggregate"],
    )
    protocol = load_direct_reference_protocol(
        _ROOT / "configs/evaluation/qwen35_skillflow_direct_reference.yaml"
    )
    attempts = [
        NativeInteractiveAttempt(
            "webshop:generation",
            DirectBenchmark.WEB_SHOP,
            None,
            None,
            0,
            0,
            0,
            False,
            False,
            "DirectGenerationError",
        ),
        NativeInteractiveAttempt(
            "webshop:environment",
            DirectBenchmark.WEB_SHOP,
            None,
            None,
            0,
            0,
            0,
            False,
            False,
            "OfficialEnvironmentInfrastructureError",
        ),
    ]

    result = cast(dict[str, object], aggregate(attempts, protocol)["webshop"])
    coverage = cast(dict[str, object], result["coverage"])

    assert coverage["generation_infrastructure_failures"] == 1
    assert coverage["environment_infrastructure_failures"] == 1
    assert result["infrastructure_failures"] == 2
