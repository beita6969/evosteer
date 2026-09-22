from pathlib import Path

import yaml

from skillev.evaluation.step0_receipts import ArchitectureIdentityReceipt


def test_frozen_step0_condition_names_exact_eight_and_no_trained_components() -> None:
    value = yaml.safe_load(
        Path("configs/evaluation/step0_architecture_conditions.yaml").read_text()
    )
    assert value["catalog"] == [
        "hotpotqa",
        "triviaqa",
        "aime-2026",
        "healthbench",
        "webshop",
        "alfworld",
        "mbpp-plus",
        "humaneval",
    ]
    assert "mbpp-plus-hard" not in value["catalog"]
    assert value["model"]["optimizer_steps"] == 0
    assert value["model"]["adapter_active"] is False
    assert value["condition_id"].startswith("skillev-bayesian-improve-step-zero-exact-eight-128@")
    assert value["population"]["default_count"] == 128
    assert value["execution"]["interactive_history"] == {
        "policy": "bounded-recent-turns-plus-persistent-controller-memory",
        "window_steps": 4,
        "maximum_characters": 24000,
        "reason": "prevent-stale-action-surfaces-and-context-growth",
    }
    assert value["execution"]["internal_action_repair_turns"] == 8
    assert value["execution"]["source_prompt_behavior_authority"] == "none-interface-only"
    assert value["execution"]["task_behavior_authority"] == "retrieved-typed-skill"
    assert value["execution"]["controller_action_selection"] is False
    assert value["execution"]["retrieved_skill_orchestration"] == {
        "activation": "exact-retrieved-skill-id",
        "evidence": "current-public-task-state-actions-and-tools-only",
        "evaluator_target_or_reward_visible": False,
    }
    assert value["execution"]["reasoning_action_transcription"] == (
        "exact-current-surface-action-only"
    )
    assert value["execution"]["controller_memory_refresh"] == (
        "live-after-every-environment-transition"
    )
    assert value["benchmarks"]["webshop"]["max_steps"] == 75
    assert value["benchmarks"]["webshop"]["prompt_profile"] == (
        "webshop-step0-skill-owned-react-memory@1"
    )
    assert value["benchmarks"]["webshop"]["evaluator_target_or_reward_in_catalog"] is False
    assert value["benchmarks"]["webshop"]["public_constraint_plan"] == (
        "answer-independent-public-instruction-parser"
    )
    assert value["benchmarks"]["webshop"]["public_catalog_selector"] == (
        "skillev-step0-webshop-expanded-public-structured-consensus@1"
    )
    assert value["benchmarks"]["webshop"]["official_environment_java"] == "pinned-java11"
    assert value["benchmarks"]["webshop"]["public_catalog_executor"] == (
        "retrieved-skill-owned-native-action-operator"
    )
    assert value["benchmarks"]["alfworld"]["max_steps"] == 75
    assert value["benchmarks"]["alfworld"]["prompt_profile"] == (
        "alfworld-step0-skill-owned-react-memory@1"
    )
    assert value["benchmarks"]["alfworld"]["preserve_numeric_location_instances"] is True
    assert value["benchmarks"]["alfworld"]["public_action_projection"] == (
        "retrieved-skill-owned-public-state-machine"
    )
    assert value["method_id"] == "skillev-bayesian-improve-step-zero@1"
    assert value["architecture"]["family"] == "skillev-bayesian-improve"
    assert value["architecture"]["lineage"] == "skillflow-plus-idea-tex"
    assert value["promotion_target_id"] == (
        "qwen35-current-eight-backbone-strict-improvement-2026-08-31@2"
    )
    assert value["promotion_rule"] == (
        "every-headline-metric-strictly-greater-than-current-backbone"
    )
    assert value["reasoning"]["authority"] == "frozen-benchmark-conditioned-mixed"
    assert value["reasoning"]["benchmark_overrides"]["aime-2026"] == {
        "authority": "frozen-benchmark-matched",
        "source": "protocol14-public-backbone-decoding-contract",
        "sampling_mode": "sampling",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "max_new_tokens": 81920,
        "seed": 42,
        "probability_in_forward_edge": False,
    }
    assert value["root_query"]["stable_across_episode"] is True
    assert value["execution"]["terminal_prompt"] == "benchmark-native-no-json@1"
    assert value["benchmarks"]["aime-2026"]["backbone_external_minimum_goal_percent"] == 80.0
    assert value["benchmarks"]["aime-2026"]["terminal_projection"] == (
        "frozen-greedy-last-explicit-integer-transcription"
    )
    assert value["benchmarks"]["aime-2026"]["terminal_recomputation"] == "forbidden"
    assert value["benchmarks"]["triviaqa"] == {
        "terminal": "short-answer",
        "terminal_projection": "frozen-greedy-reasoning-hypothesis-transcription",
        "retrieval": "detailed-answer-isolated-diverse-complementary-search-v4",
        "scorer": "official-complete-alias-em-f1",
        "scorer_alias_source": "pinned-unfiltered-nocontext-validation-parquet",
        "search_calls": 3,
        "hits_per_search": 8,
        "max_hits_per_document": 2,
        "deduplicate_passages_across_searches": True,
        "snippet_token_window": 96,
        "maximum_snippet_characters": 1200,
        "primary_metric": "f1",
        "owner_minimum_goal_percent": 81.0,
    }
    assert value["benchmarks"]["mbpp-plus"]["reasoning_template"] == ("qwen-native-thinking")
    assert value["benchmarks"]["humaneval"]["reasoning_template"] == ("qwen-native-thinking")


def test_step0_targets_are_strictly_above_the_owner_backbone_snapshot() -> None:
    value = yaml.safe_load(Path("configs/evaluation/step0_architecture_targets.yaml").read_text())

    assert value["comparison"]["operator"] == "greater-than"
    assert value["comparison"]["equality_passes"] is False
    assert value["catalog"] == [
        "hotpotqa",
        "triviaqa",
        "aime-2026",
        "healthbench",
        "webshop",
        "alfworld",
        "mbpp-plus",
        "humaneval",
    ]
    assert value["benchmarks"]["aime-2026"]["metrics"]["accuracy"] == {
        "strictly_greater_than_percent": 86.67
    }
    assert value["benchmarks"]["healthbench"]["metrics"]["native_rubric_mean"] == {
        "strictly_greater_than_percent": 53.17
    }
    assert value["benchmarks"]["mbpp-plus"]["metrics"]["base_plus_pass_at_1"] == {
        "strictly_greater_than_percent": 67.19
    }
    assert value["benchmarks"]["webshop"]["metrics"] == {
        "average_score": {"strictly_greater_than_percent": 80.0},
        "success_rate": {"strictly_greater_than_percent": 80.0},
    }
    assert value["benchmarks"]["alfworld"]["metrics"] == {
        "success_rate": {"strictly_greater_than_percent": 80.0}
    }


def test_identity_records_no_active_training_state() -> None:
    receipt = ArchitectureIdentityReceipt(
        method_id="skillev-bayesian-improve-step-zero@1",
        architecture_family="skillev-bayesian-improve",
        architecture_lineage="skillflow-plus-idea-tex",
        architecture_components=(
            "seeded-skill-controller",
            "trajectory-balance-gflownet",
            "beta-bernoulli-lcb-calibration",
            "operator-driven-skill-evolution",
        ),
        controller_id="two-pass-controller@2",
        optimizer_steps=0,
        forward_adapter_active=False,
        backward_adapter_active=False,
        posterior_active=False,
        calibration_active=False,
        operator_active=False,
        seed_library_id="step0-exact-eight-answer-free-seeds@1",
        seed_skill_ids=("skill-hotpot-evidence-chain",),
        retrieval_policy_id="step0-exact-eight-deterministic-top1@1",
        initial_context_profile="seeded-step-zero@1",
        reasoning_authority="frozen-deterministic",
        reasoning_contract_resolved=True,
        action_policy_authority="adapter-free-evaluation-matched-base",
    )
    assert receipt.optimizer_steps == 0
