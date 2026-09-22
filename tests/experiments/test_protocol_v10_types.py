from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from skillev.contracts import SuccessRule, TerminalReward
from skillev.experiments.protocol_v10 import (
    ACTIVE_BENCHMARKS_V10,
    TRAINING_EPISODES_PER_BENCHMARK,
    BenchmarkV10,
    FormalMethodV10,
    PopulationRole,
    PopulationSampling,
    PosteriorSuccessRule,
    ProtocolV10Error,
    load_active_protocol_v10,
    parse_active_protocol_v10,
)

ROOT = Path(__file__).parents[2]
SPEC_PATH = ROOT / "configs" / "evaluation" / "protocol_v10.yaml"


def _raw_spec() -> dict[str, object]:
    value = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_active_protocol_loads_population_records_and_fixed_skillflow_mix() -> None:
    protocol = load_active_protocol_v10(SPEC_PATH)

    assert protocol.seed == 0
    assert tuple(item.benchmark for item in protocol.benchmarks) == ACTIVE_BENCHMARKS_V10
    assert protocol.methods == tuple(FormalMethodV10)
    assert protocol.training_mix.episodes_per_benchmark == TRAINING_EPISODES_PER_BENCHMARK
    assert protocol.training_mix.total_episodes == 9 * 512
    assert protocol.training_mix.benchmark_order == ACTIVE_BENCHMARKS_V10
    assert protocol.training_mix.final_mix_order == "deterministic-global-shuffle-seed-0"

    for benchmark in protocol.benchmarks:
        training = benchmark.population(PopulationRole.TRAINING)
        validation = benchmark.population(PopulationRole.VALIDATION)
        final = benchmark.population(PopulationRole.FINAL_EVALUATION)
        assert len(training) == 1
        assert len(validation) == 1
        assert final
        assert training[0].sampling is PopulationSampling.FIXED_TRAINING_MIX
        assert validation[0].sampling is PopulationSampling.HELD_OUT_VALIDATION
        assert all(item.sampling is PopulationSampling.FULL for item in final)
        assert all(
            item.source_version == benchmark.source_version for item in benchmark.populations
        )
        assert all(
            item.evaluator_identity == benchmark.evaluator_identity
            for item in benchmark.populations
        )


def test_protocol_v10_keeps_method_identities_distinct() -> None:
    assert FormalMethodV10.SKILLFLOW_BASELINE is not FormalMethodV10.BAYESIAN_IMPROVE_FULL
    assert FormalMethodV10.SKILLFLOW_BASELINE is not FormalMethodV10.BAYESIAN_IMPROVE_NO_CALIBRATION
    assert (
        FormalMethodV10.BAYESIAN_IMPROVE_FULL is not FormalMethodV10.BAYESIAN_IMPROVE_NO_CALIBRATION
    )


def test_active_reader_rejects_historical_protocol_and_nonuniform_mix() -> None:
    historical = _raw_spec()
    historical["format"] = "skillev-benchmark-protocol@9"
    with pytest.raises(ProtocolV10Error):
        parse_active_protocol_v10(historical)

    wrong_mix = _raw_spec()
    policy = wrong_mix["population_policy"]
    assert isinstance(policy, dict)
    policy["training_episodes_per_benchmark"] = 511
    policy["training_episode_total"] = 9 * 511
    with pytest.raises(ProtocolV10Error):
        parse_active_protocol_v10(wrong_mix)


def test_protocol_v10_types_preserve_special_metric_semantics() -> None:
    protocol = load_active_protocol_v10(SPEC_PATH)

    health = protocol.benchmark(BenchmarkV10.HEALTHBENCH)
    assert health.success_projection.rule is PosteriorSuccessRule.SCORE_AND_NO_NEGATIVE_RUBRIC
    assert health.success_projection.threshold == 0.60
    assert health.success_projection.sources == (
        "official-rubric-score",
        "triggered-negative-rubric-count",
    )

    appworld = protocol.benchmark(BenchmarkV10.APPWORLD)
    assert appworld.reward_projection.source == "task-goal-completion"
    assert appworld.aggregation_constraint == (
        "scenario-goal-completion-requires-complete-scenario-group"
    )

    mbpp = protocol.benchmark(BenchmarkV10.MBPP_PLUS_FIXED_100)
    assert mbpp.forbidden_label == "hard-100"
    assert mbpp.final_selection == "first-100-by-ascending-canonical-evalplus-task-id"


def test_protocol_v10_projects_native_reward_and_binary_success_separately() -> None:
    protocol = load_active_protocol_v10(SPEC_PATH)

    hotpot = protocol.benchmark(BenchmarkV10.HOTPOT_QA)
    native = {"answer-f1": 0.73, "answer-exact-match": 0}
    assert hotpot.reward_projection.project(native) == 0.73
    assert hotpot.success_projection.project(native) is False

    health = protocol.benchmark(BenchmarkV10.HEALTHBENCH)
    safe = {"official-rubric-score": 1.2, "triggered-negative-rubric-count": 0}
    unsafe = {"official-rubric-score": 0.91, "triggered-negative-rubric-count": 1}
    assert health.reward_projection.project(safe) == 1.0
    assert health.success_projection.project(safe) is True
    assert health.reward_projection.project(unsafe) == 0.91
    assert health.success_projection.project(unsafe) is False


def test_protocol_v10_freezes_privacy_isolation_and_open_execution_gate() -> None:
    protocol = load_active_protocol_v10(SPEC_PATH)

    assert protocol.training_mix.require_disjoint_source_ids is True
    assert protocol.training_mix.require_disjoint_normalized_content is True
    assert protocol.training_mix.final_evaluation_is_read_only is True
    assert protocol.terminal_contract.reward_range == (0.0, 1.0)
    assert protocol.terminal_contract.posterior_success_values == (0, 1)
    assert protocol.terminal_contract.private_evaluator_payload_model_visible is False
    protocol.require_execution_ready()


def test_population_ids_are_global_and_final_populations_are_full() -> None:
    duplicate = deepcopy(_raw_spec())
    benchmarks = duplicate["benchmarks"]
    assert isinstance(benchmarks, list)
    first = benchmarks[0]
    second = benchmarks[1]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    first_populations = first["populations"]
    second_populations = second["populations"]
    assert isinstance(first_populations, list)
    assert isinstance(second_populations, list)
    first_training = first_populations[0]
    second_training = second_populations[0]
    assert isinstance(first_training, dict)
    assert isinstance(second_training, dict)
    second_training["id"] = first_training["id"]
    with pytest.raises(ProtocolV10Error):
        parse_active_protocol_v10(duplicate)

    non_full = deepcopy(_raw_spec())
    rows = non_full["benchmarks"]
    assert isinstance(rows, list)
    benchmark = rows[0]
    assert isinstance(benchmark, dict)
    populations = benchmark["populations"]
    assert isinstance(populations, list)
    final = populations[-1]
    assert isinstance(final, dict)
    final["sampling"] = "fixed-500"
    with pytest.raises(ProtocolV10Error):
        parse_active_protocol_v10(non_full)


def test_trusted_native_projection_can_express_healthbench_conjunction() -> None:
    reward = TerminalReward(
        value=0.91,
        success=False,
        success_rule=SuccessRule.TRUSTED_NATIVE_PROJECTION,
        success_threshold=None,
        native_metric_name="official-rubric-score",
        native_payload={
            "official-rubric-score": 0.91,
            "triggered-negative-rubric-count": 1,
        },
        environment_id="healthbench",
        verifier_version="healthbench-protocol-v10",
    )

    assert TerminalReward.from_value(reward.to_value()) == reward
    assert reward.success is False

    with pytest.raises(ValueError):
        TerminalReward(
            value=0.91,
            success=False,
            success_rule=SuccessRule.TRUSTED_NATIVE_PROJECTION,
            success_threshold=0.60,
            native_metric_name="official-rubric-score",
            native_payload={},
            environment_id="healthbench",
            verifier_version="healthbench-protocol-v10",
        )
