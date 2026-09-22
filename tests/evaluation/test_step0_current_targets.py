from pathlib import Path

import pytest
import yaml

from skillev.evaluation.input_metric_contracts import IID_BENCHMARKS

TARGETS = Path("configs/evaluation/step0_integrity_targets.yaml")


def test_current_targets_describe_the_single_untrained_owner() -> None:
    targets = yaml.safe_load(TARGETS.read_text())

    assert targets["agent_topology"] == "single-controller"
    assert targets["skill_mode"] == "off"
    assert targets["native_thinking"] is False
    assert targets["optimizer_steps"] == 0


def test_current_targets_cover_only_the_supported_iid_domains() -> None:
    targets = yaml.safe_load(TARGETS.read_text())

    for benchmark in IID_BENCHMARKS:
        assert benchmark in targets["benchmarks"]
    for benchmark in targets["benchmarks"]:
        assert benchmark in IID_BENCHMARKS
    assert "webshop" not in targets["benchmarks"]
    assert targets["benchmarks"]["mbpp-plus"]["display_name"] == "MBPP+"
    assert targets["development_panel"] != targets["panel"]


@pytest.mark.parametrize("benchmark", ["hotpotqa", "aime-2026", "alfworld", "mbpp-plus"])
def test_current_eighty_point_targets_do_not_retain_older_lower_floors(benchmark: str) -> None:
    target = yaml.safe_load(TARGETS.read_text())["benchmarks"][benchmark]

    assert target["threshold"] >= 80.0
    assert target["operator"] in {"greater-than", "greater-than-or-equal"}


def test_health_diagnostic_floor_does_not_replace_backbone_improvement() -> None:
    target = yaml.safe_load(TARGETS.read_text())["benchmarks"]["healthbench"]

    assert target["operator"] == "greater-than"
    assert target["threshold"] > target["diagnostic_floor_percent"]
    assert target["reference_comparison"] == "unmatched-historical-snapshot"
    assert target["official_gpt_judge_comparable"] is False
