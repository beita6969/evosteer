"""Distinct actual skill use is separate from success and total action count."""

import json
import math
from copy import deepcopy

from skillev.contracts import canonical_json
from skillev.contracts.skill_exposure import TWO_SKILL_CATALOG_EXPOSURE
from skillev.training.metrics_contract import TrainingMetricsSnapshot
from skillev.training.metrics_telemetry import skill_target_facts, step_facts
from tests.training.test_metrics_contract import event
from tests.training.test_metrics_telemetry import setup_export


def source():
    result = event()
    for i, record in enumerate(result["payload"]["records"]):
        record["initial_context"] = {
            "meta": {"skill_exposure": TWO_SKILL_CATALOG_EXPOSURE},
            "retrieved_skill_ids": ["PRIVATE_FIRST", "PRIVATE_SECOND"],
        }
        terminal = record["steps"][0]
        terminal["invoked_skill_ids"] = []
        record["steps"] = [
            {
                "action_text": canonical_json(
                    {
                        "kind": "skill",
                        "name": "invoke",
                        "arguments": {},
                        "resource_id": "skill-runtime",
                        "skill_id": skill,
                    }
                ),
                "action_token_count": 7,
                "invoked_skill_ids": [skill],
            }
            for skill in ("PRIVATE_FIRST", "PRIVATE_FIRST" if i < 14 else "PRIVATE_SECOND")
        ] + [terminal]
    for residual in result["payload"]["stats"]["residuals"]:
        residual["horizon"] = 3
    result["payload"]["stats"]["batch_loss"] = 4 / 9
    return result


def test_repeated_reads_are_actions_but_not_two_distinct_skills_and_no_rows_are_filtered(tmp_path):
    committed = source()
    original = deepcopy(committed)
    metrics = TrainingMetricsSnapshot.from_event(committed, condition_id="two-skill-target").metrics
    assert metrics["action_count"] == 84  # Two reads plus a submission for all 28 trajectories.
    assert metrics["horizon_mean"] == 3
    assert metrics["trajectory_count"] == 28
    assert metrics["ttb_loss"] == math.fsum([(2 / 3) ** 2] * 28) / 28
    store, collector = setup_export(tmp_path)
    collector.observe(committed)
    assert collector.flush({}) == 1
    values = store.values()[0]
    facts = values["telemetry"]
    assert facts["skill_invocation_count"] == 56
    assert facts["skill_target_trajectory_count"] == 28
    assert facts["skill_target_met_trajectory_count"] == 14
    assert facts["skill_target_met_fraction"] == 0.5
    assert facts["skill_distinct_per_trajectory_mean"] == 1.5
    assert facts["skill_target_unknown_trajectory_count"] == 0
    assert facts["skill_catalog_below_target_trajectory_count"] == 0
    assert committed == original
    assert "PRIVATE_" not in json.dumps(values)
    store.close()


def test_missing_invocations_and_unavailable_catalog_are_not_invented_success():
    committed = source()
    records = committed["payload"]["records"]
    del records[0]["steps"][0]["invoked_skill_ids"]
    records[1]["initial_context"]["retrieved_skill_ids"] = ["PRIVATE_FIRST"]
    facts = step_facts(committed)
    assert facts["skill_invocation_count"] is None
    assert facts["skill_target_met_trajectory_count"] is None
    assert facts["skill_target_met_fraction"] is None
    assert facts["skill_target_unknown_trajectory_count"] == 1
    assert facts["skill_catalog_below_target_trajectory_count"] == 1
    for record in records:
        for edge in record["steps"]:
            edge["invoked_skill_ids"] = []
    assert skill_target_facts(records)["skill_target_met_fraction"] == 0


def test_legacy_condition_does_not_acquire_a_retroactive_skill_target():
    committed = source()
    records = committed["payload"]["records"]
    for record in records:
        record["initial_context"]["meta"]["skill_exposure"] = "catalog-then-read@1"
    assert skill_target_facts(records) == {}
