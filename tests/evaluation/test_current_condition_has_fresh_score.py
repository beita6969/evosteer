from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONDITION = ROOT / "configs/evaluation/step0_architecture_conditions.yaml"
TARGETS = ROOT / "configs/evaluation/step0_architecture_targets.yaml"
RESULT = ROOT / "docs/SKILLEV_BAYESIAN_IMPROVE_STEP0_IID_128_RESULTS.md"


def test_current_step_zero_condition_has_a_fresh_answer_free_result() -> None:
    condition = yaml.safe_load(CONDITION.read_text(encoding="utf-8"))
    assert condition["condition_id"].startswith(
        "skillev-bayesian-improve-step-zero-exact-eight-128@"
    )
    assert condition["method_id"] == "skillev-bayesian-improve-step-zero@1"
    assert condition["architecture"]["components"] == [
        "seeded-skill-controller",
        "trajectory-balance-gflownet",
        "beta-bernoulli-lcb-calibration",
        "operator-driven-skill-evolution",
    ]
    assert condition["catalog"] == [
        "hotpotqa",
        "triviaqa",
        "aime-2026",
        "healthbench",
        "webshop",
        "alfworld",
        "mbpp-plus",
        "humaneval",
    ]
    default_count = condition["population"]["default_count"]
    aime_count = condition["population"]["overrides"]["aime-2026"]["count"]
    assert 7 * default_count + aime_count == 926
    assert condition["benchmarks"]["aime-2026"]["backbone_external_minimum_goal_percent"] == 80.0

    targets = yaml.safe_load(TARGETS.read_text(encoding="utf-8"))
    assert targets["comparison"]["operator"] == "greater-than"
    assert targets["comparison"]["equality_passes"] is False
    assert targets["benchmarks"]["aime-2026"]["metrics"]["accuracy"] == {
        "strictly_greater_than_percent": 86.67
    }

    report = RESULT.read_text(encoding="utf-8")
    assert condition["condition_id"] in report
    assert condition["method_id"] in report
    assert "926" in report
    assert "80.00%" in report
    assert "86.67%" in report
    assert "<!--" not in report
    assert "pending" not in report.casefold()
    for private_marker in (
        "/ssd",
        "/home/",
        "OPENSSH PRIVATE KEY",
        "IdentityFile",
    ):
        assert private_marker not in report
    assert "@" + "185.212.56.211" not in report
