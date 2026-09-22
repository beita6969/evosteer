from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parents[2]
SPEC_PATH = ROOT / "configs" / "evaluation" / "protocol_v10.yaml"

EXPECTED_BENCHMARKS = (
    "hotpotqa",
    "triviaqa",
    "aime-2026",
    "healthbench",
    "webshop",
    "alfworld",
    "spreadsheetbench",
    "appworld",
    "mbpp-plus-fixed-100",
)


def _spec() -> dict[str, Any]:
    value = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _benchmarks() -> tuple[dict[str, Any], ...]:
    rows = _spec()["benchmarks"]
    assert isinstance(rows, list)
    assert all(isinstance(row, dict) for row in rows)
    return tuple(rows)


def test_protocol_v10_freezes_nine_benchmarks_and_skillflow_sized_mix() -> None:
    spec = _spec()
    policy = spec["population_policy"]

    assert spec["format"] == "skillev-benchmark-protocol@10"
    assert spec["status"] == "frozen-scientific-specification"
    assert spec["seed"] == 0
    assert tuple(row["id"] for row in _benchmarks()) == EXPECTED_BENCHMARKS
    assert policy["training_episodes_per_benchmark"] == 512
    assert policy["training_episode_total"] == 512 * len(EXPECTED_BENCHMARKS)
    assert tuple(policy["training_block_order"]) == EXPECTED_BENCHMARKS
    assert policy["within_block_order"] == "deterministic-shuffle-seed-0"
    assert policy["training_count_rule_applies_to"] == "training-only"
    assert "distinct-episode-ids" in policy["undersized_population"]


def test_protocol_v10_has_disjoint_population_roles_and_identities() -> None:
    all_population_ids: list[str] = []

    for benchmark in _benchmarks():
        populations = benchmark["populations"]
        roles = [population["role"] for population in populations]
        assert roles.count("training") == 1
        assert roles.count("validation") == 1
        assert roles.count("final-evaluation") >= 1

        population_ids = [population["id"] for population in populations]
        assert len(population_ids) == len(set(population_ids))
        all_population_ids.extend(population_ids)

    assert len(all_population_ids) == len(set(all_population_ids))


def test_every_native_reward_has_an_explicit_binary_success_projection() -> None:
    for benchmark in _benchmarks():
        assert benchmark["native_metrics"]
        reward = benchmark["ttb_reward_projection"]
        success = benchmark["posterior_success_projection"]
        assert reward["source"]
        assert reward["rule"]
        assert success["source"]
        assert success["rule"]

    by_id = {row["id"]: row for row in _benchmarks()}
    health = by_id["healthbench"]["posterior_success_projection"]
    assert health["threshold"] == 0.60
    assert health["rule"] == "score-at-least-and-no-negative-rubric-triggered"
    assert by_id["hotpotqa"]["ttb_reward_projection"]["source"] == "answer-f1"
    assert by_id["hotpotqa"]["posterior_success_projection"]["source"] == ("answer-exact-match")
    assert by_id["webshop"]["ttb_reward_projection"]["source"] == "native-score"
    assert by_id["webshop"]["posterior_success_projection"]["source"] == ("native-success")


def test_special_population_and_aggregation_choices_are_not_aliases() -> None:
    by_id = {row["id"]: row for row in _benchmarks()}

    health = by_id["healthbench"]
    assert health["version"] == "openai-full-2025-05-07"
    assert health["excluded_primary_variants"] == [
        "healthbench-hard",
        "healthbench-consensus",
    ]

    spreadsheet = by_id["spreadsheetbench"]
    assert spreadsheet["version"] == "v1-verified-400"
    assert "spreadsheetbench-2" in spreadsheet["excluded_primary_variants"]

    appworld = by_id["appworld"]
    assert appworld["aggregation_constraint"] == (
        "scenario-goal-completion-requires-complete-scenario-group"
    )
    assert appworld["ttb_reward_projection"]["source"] == "task-goal-completion"

    mbpp = by_id["mbpp-plus-fixed-100"]
    assert mbpp["version"] == "evalplus-v0.2.0"
    assert mbpp["forbidden_label"] == "hard-100"
    assert mbpp["final_selection"] == ("first-100-by-ascending-canonical-evalplus-task-id")


def test_protocol_v10_is_frozen_and_admitted_after_integration_evidence() -> None:
    spec = _spec()
    assert spec["formal_methods"] == [
        "skillflow-baseline",
        "bayesian-improve-full",
        "bayesian-improve-no-calibration",
    ]
    assert spec["execution_gate"]["executable"] is True
    assert spec["terminal_contract"]["private_evaluator_payload_model_visible"] is False
    assert spec["terminal_contract"]["infrastructure_failure"] == ("abort-uncommitted-step")
    assert spec["amendment"] == {
        "id": "protocol-v10-healthbench-qwen-judge@1",
        "reason": "remove-external-api-dependency-and-use-the-frozen-local-base-model-judge",
        "adopted_before_first_formal_run": True,
        "prior_gpt_4_1_profile_used_for_formal_run": False,
    }
    health = next(row for row in spec["benchmarks"] if row["id"] == "healthbench")
    assert health["evaluator"] == "healthbench-qwen3.5-9b-sglang-temperature-0@1"
