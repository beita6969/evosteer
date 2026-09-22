import sqlite3

import pytest

from skillev.training.action_metrics import ActionMetrics
from skillev.training.source_group_metrics import source_group_summary


def test_rollouts_are_averaged_before_equal_source_weighting():
    summary = source_group_summary(
        [(("humaneval", "one"), {"score": v}) for v in (0.0, 0.0, 0.0, 1.0)]
        + [(("humaneval", "two"), {"score": 1.0})],
        allow_standard_error=True,
    )
    assert summary["source_group_count"] == 2
    assert summary["score/source_group_mean"] == 0.625  # not 2/5 rollout-weighted
    assert summary["score/source_group_sample_sd"] == pytest.approx(0.75 / 2**0.5)
    assert summary["score/source_group_standard_error"] == pytest.approx(0.375)


def test_four_rollouts_are_one_group_and_missing_evidence_is_not_dropped():
    rows = [(("humaneval", "same"), {"score": v}) for v in (0.0, 1.0, 0.0, 1.0)]
    summary = source_group_summary(rows, allow_standard_error=True)
    assert summary["source_group_count"] == 1
    assert summary["score/source_group_mean"] == 0.5
    assert summary["score/source_group_sample_sd"] is None
    assert summary["score/source_group_standard_error"] is None
    rows.append((("humaneval", "other"), {"score": None}))
    summary = source_group_summary(rows, allow_standard_error=True)
    assert summary["score/source_group_observed_count"] == 1
    assert summary["score/source_group_mean"] is None
    assert summary["score/source_group_standard_error"] is None


def test_mixed_domain_panel_has_no_unjustified_pooled_standard_error():
    summary = source_group_summary(
        [(("humaneval", "one"), {"score": 0.0}), (("mbpp", "one"), {"score": 1.0})],
        allow_standard_error=False,
    )
    assert summary["source_group_count"] == 2
    assert summary["score/source_group_sample_sd"] == pytest.approx(2**-0.5)
    assert summary["score/source_group_standard_error"] is None


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([(True, False)], 1),
        ([(False, True)], 1),
        ([(False, False)] * 3, 0),
        ([(True, True), (True, True)], 1),
    ],
    ids=[
        "failed-valid-submission",
        "environment-terminal",
        "horizon-exhausted-no-submission",
        "count-once-per-record",
    ],
)
def test_valid_terminal_is_trajectory_or_not_native_success_or_action_count(flags, expected):
    record = {
        "trajectory_id": "synthetic",
        "reward": {"value": 0.0, "success": False},
        "steps": [
            {"action_token_ids": [i], "observation_text": f"observation-{i}"}
            for i in range(len(flags))
        ],
    }
    with sqlite3.connect(":memory:") as connection:
        store = ActionMetrics(connection)
        assert store.for_records("run", [record])["valid_terminal_record_count"] is None
        for i, (accepted, terminal) in enumerate(flags):
            store.observe(
                {
                    "run_id": "run",
                    "payload": {
                        "trajectory_id": "synthetic",
                        "turn": i + 1,
                        **record["steps"][i],
                        "assessment": {
                            "admitted": True,
                            "executed": not accepted,
                            "accepted_submission": accepted,
                            "environment_terminal": terminal,
                            "execution_status": "success",
                        },
                    },
                }
            )
        counts = store.for_records("run", [record])
        assert counts["terminal_evidence_record_count"] == 1
        assert counts["valid_terminal_record_count"] == expected
        # A committed unassessed tail cannot silently shrink the denominator.
        record["steps"].append({"action_token_ids": [999], "observation_text": "missing"})
        counts = store.for_records("run", [record])
        assert counts["terminal_evidence_record_count"] == 0
        assert counts["valid_terminal_record_count"] is None
