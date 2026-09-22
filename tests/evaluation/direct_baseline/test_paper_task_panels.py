from pathlib import Path

from skillev.evaluation.direct_baseline.config import DirectBenchmark
from skillev.evaluation.direct_baseline.protocol import load_direct_reference_protocol


def test_paper_config_has_all_fourteen_unique_benchmarks_and_counts() -> None:
    protocol = load_direct_reference_protocol(
        Path("configs/evaluation/qwen35_skillflow_direct_reference.yaml")
    )
    assert len(protocol.benchmarks) == 14
    assert len({item.benchmark for item in protocol.benchmarks}) == 14
    assert {item.role for item in protocol.benchmarks} == {"iid", "ood"}


def test_ood_panels_remain_conservatively_labeled() -> None:
    protocol = load_direct_reference_protocol(
        Path("configs/evaluation/qwen35_skillflow_direct_reference.yaml")
    )
    ood = [item for item in protocol.benchmarks if item.role == "ood"]
    assert {item.comparability.value for item in ood} == {"approximate-only"}


def test_released_interactive_rows_are_not_overclaimed_as_exact_environments() -> None:
    protocol = load_direct_reference_protocol(
        Path("configs/evaluation/qwen35_skillflow_direct_reference.yaml")
    )
    assert protocol.benchmark(DirectBenchmark.WEB_SHOP).comparability.value == "approximate-only"
    assert protocol.benchmark(DirectBenchmark.ALF_WORLD).comparability.value == "approximate-only"
