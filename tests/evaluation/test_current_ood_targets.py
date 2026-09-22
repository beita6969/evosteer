import json
from pathlib import Path

from scripts.report_current_ood_targets import assess


def test_gate_is_strict_unrounded_and_independent_of_panel_size():
    root = Path(__file__).resolve().parents[2]
    policy = json.loads((root / "configs/evaluation/current_ood_targets.json").read_text())
    entry = {
        "count": 32,
        "metric": "answer-f1",
        "value": 0.675,
        "execution_status": "complete",
        "native_metric_contract_status": "pass",
        "native_diagnostics": {"owner": {"status_counts": {"scored": 32}}},
    }
    summary = {"run_id": "fixture", "benchmarks": {"musique": entry}}

    def result():
        return assess(summary, policy)["benchmarks"]["musique"]

    assert result()["status"] == "below-gate"
    entry["value"] = 0.67500001
    assert result()["promote_to_64"]
    assert not result()["accepted_64"]
    entry["count"] = 64
    entry["native_diagnostics"]["owner"]["status_counts"] = {"scored": 63, "candidate-failure": 1}
    assert result()["accepted_64"]
    assert not result()["promote_to_64"]
    entry["native_diagnostics"]["owner"]["status_counts"] = {
        "scored": 63,
        "infrastructure-failure": 1,
    }
    assert result()["status"] == "incomplete"


def test_missing_or_wrong_native_metric_never_passes():
    policy = {
        "targets": {"scienceworld": 56.17},
        "gate_ratio": 0.9,
        "initial_samples": 32,
        "confirmation_samples": 64,
        "condition": "development",
    }
    for entry in ({}, {"metric": "clipped-reward", "value": 1.0, "count": 64}):
        report = assess({"benchmarks": {"scienceworld": entry}}, policy)
        assert report["benchmarks"]["scienceworld"]["status"] == "incomplete"
