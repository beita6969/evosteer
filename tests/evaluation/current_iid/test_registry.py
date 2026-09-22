from pathlib import Path

from skillev.evaluation.current_iid.config import load_current_iid_config
from skillev.evaluation.current_iid.targets import TargetStatus, load_target_registry
from skillev.evaluation.direct_baseline.prompts import STATIC_PROMPT_REGISTRY
from skillev.evaluation.direct_baseline.protocol import load_direct_reference_protocol
from skillev.experiments.protocol_v11 import ACTIVE_BENCHMARKS_V11, BenchmarkV11

ROOT = Path(__file__).parents[3]


def test_config_contains_authoritative_current_ten() -> None:
    config = load_current_iid_config(ROOT / "configs/evaluation/qwen35_current_iid.yaml")
    assert tuple(item.benchmark for item in config.benchmarks) == ACTIVE_BENCHMARKS_V11
    counts = {item.benchmark: item.expected_count for item in config.benchmarks}
    assert counts[BenchmarkV11.AIME_2026] == 30
    assert all(
        count == 128
        for benchmark, count in counts.items()
        if benchmark is not BenchmarkV11.AIME_2026
    )
    assert {condition.context_length for condition in config.conditions.values()} == {98_304}
    appworld = config.conditions["appworld-benchmark-native-no-skill@1"]
    assert appworld.tool_surface == ("execute", "submit")
    mbpp = config.conditions["mbpp-plus-hard-deterministic-direct@1"]
    assert mbpp.tool_surface == ()


def test_targets_are_explicitly_undefined_until_condition_matched() -> None:
    targets = load_target_registry(ROOT / "configs/evaluation/qwen35_current_iid_targets.yaml")
    assert set(targets) == set(ACTIVE_BENCHMARKS_V11)
    assert all(item.status is TargetStatus.UNDEFINED for item in targets.values())


def test_executable_direct_condition_profiles_exist_in_runtime_registries() -> None:
    current = load_current_iid_config(ROOT / "configs/evaluation/qwen35_current_iid.yaml")
    direct = load_direct_reference_protocol(
        ROOT / "configs/evaluation/qwen35_skillflow_direct_reference.yaml"
    )
    profile_ids = {profile.profile_id for profile in direct.decoding_profiles}
    for condition_id in (
        "hotpotqa-full-context-direct@1",
        "aime2026-thinking-direct@1",
        "mbpp-plus-hard-deterministic-direct@1",
        "humaneval-deterministic-direct@1",
    ):
        condition = current.conditions[condition_id]
        assert condition.prompt_profile in STATIC_PROMPT_REGISTRY
        assert condition.decoding_profile in profile_ids
