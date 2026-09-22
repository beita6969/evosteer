import asyncio
import copy
import json

import pytest

from skillev.training.action_failure_report import committed_action_outcomes
from skillev.training.learning_behavior_report import committed_learning_report
from tests.training.test_metrics_contract import event


def test_original_loss_global_b_terms_and_population_alias_contrasts():
    source = event()
    for index, (record, residual) in enumerate(
        zip(source["payload"]["records"], source["payload"]["stats"]["residuals"], strict=True)
    ):
        record["reward"]["native_payload"]["training_evidence_source"]["population_id"] = (
            f"alias-{index}"
        )
        residual.update(
            log_z=0,
            sum_forward=-1,
            sum_backward=-2,
            log_shifted_reward=-1,
            temperature_beta=1,
            raw_reward=0.5,
            delta=2,
        )
    # One longer trajectory keeps the same residual but contributes one quarter loss.
    source["payload"]["records"][0]["steps"] *= 2
    source["payload"]["stats"]["residuals"][0]["horizon"] = 2
    original = copy.deepcopy(source)
    report = committed_learning_report(source, condition_id="fixed")
    assert source == original
    assert report["summary"]["raw_delta_squared_mean"] == 4
    assert report["summary"]["normalized_ttb_loss_mean"] == pytest.approx(109 / 28)
    assert report["summary"]["reward_term_mean"] == -1
    assert report["summary"]["signed_reward_term_mean"] == 1
    assert report["summary"]["signed_backward_term_mean"] == 2
    assert sum(
        g["global_batch_loss_contribution"] for g in report["groups"]["horizon"]
    ) == pytest.approx(109 / 28)
    assert len(report["source_contrasts"]) == 7
    assert all(g["pair_count"] == 6 for g in report["source_contrasts"])
    assert any(p["terminal_success_disagrees"] for p in report["source_contrasts"][0]["pairs"])
    assert any(p["differing_action_positions"] == 1 for p in report["source_contrasts"][0]["pairs"])
    assert "PRIVATE_" not in json.dumps(report)
    assert report["component_parameter_changes"] is None


def test_missing_terms_actions_and_success_are_not_reconstructed_from_reward():
    source = event()
    del source["payload"]["records"][0]["steps"][0]["action_text"]
    del source["payload"]["records"][0]["reward"]["success"]
    report = committed_learning_report(source, condition_id="fixed")
    assert report["summary"]["reward_term_mean"] is None
    assert report["source_contrasts"][0]["pairs"][0]["raw_action_sequence_equal"] is None
    assert report["source_contrasts"][0]["pairs"][0]["terminal_success_disagrees"] is None
    assert all(v is None for v in report["component_gradient_norms"].values())


def test_action_report_uses_exact_assessment_and_preserves_unknown_requests():
    source = event()
    record = source["payload"]["records"][0]
    record["initial_context"] = {"meta": {}}
    step = record["steps"][0]
    step.update(
        action_token_ids=[1, 2], observation_text="public error", observation_status="tool_error"
    )
    record["reward"]["success"] = False
    assessment = {
        "run_id": source["run_id"],
        "event_type": "agent_step_recorded",
        "payload": {
            "trajectory_id": record["trajectory_id"],
            "turn": 1,
            "action_token_ids": [1, 2],
            "observation_text": "public error",
            "assessment": {
                "admitted": True,
                "executed": True,
                "accepted_submission": False,
                "environment_terminal": False,
            },
        },
    }
    result = committed_action_outcomes(source, source_events=[assessment])
    assert "admitted-execution-unsuccessful" in result["actions"][0]["labels"]
    assert "admitted-action-in-terminal-failed-trajectory" in result["actions"][0]["labels"]
    assert result["actions"][0]["finish_reason"] is None
    assert result["actions"][1]["admitted"] is None
    assert "PRIVATE_" not in json.dumps(result)
    assessment["payload"]["action_token_ids"] = [9]
    assert (
        committed_action_outcomes(source, source_events=[assessment])["actions"][0]["admitted"]
        is None
    )


def test_engine_persists_one_diagnostic_on_original_environment_event(
    make_training_harness, monkeypatch
):
    from skillev.diagnostics.rollout_trace import NullRolloutTraceSink, RolloutTraceStage

    observed = []

    async def record(self, event):
        observed.append(event)

    monkeypatch.setattr(NullRolloutTraceSink, "record", record)
    harness = make_training_harness()
    batch = asyncio.run(harness.loop.collect_batch())
    outcomes = [e for e in observed if e.stage == RolloutTraceStage.ENVIRONMENT_RESULT]
    actions = [e for e in observed if e.stage == RolloutTraceStage.ACTION_RESULT]
    assert len(outcomes) == len(actions) == sum(a.record.horizon for a in batch.artifacts)
    for outcome in outcomes:
        evidence = outcome.public_payload["action_diagnostics"]
        assert evidence["format"] == "observed-action-outcomes@2"
        assert evidence["admitted"] is None
        assert evidence["terminal_success"] is None
    assert harness.loop.optimizer_step == 0


def test_native_outcome_uses_persisted_surface_and_valid_xml_code_is_not_json_failure():
    from tests.rollout.test_native_tool_wire import call, contract

    source = event()
    record = source["payload"]["records"][0]
    record["initial_context"] = {
        "meta": {
            "action_wire": "native-single-tool-call@3",
            "action_contract": contract().to_scoring_metadata(),
        }
    }
    step = record["steps"][0]
    step["action_text"] = call("read_skill", skill_id="absent")
    labels = committed_action_outcomes(source)["actions"][0]["labels"]
    assert "action-not-in-current-surface" in labels
    assert "json-schema-mismatch" not in labels
    step["action_text"] = call("execute", code='print("中文\\n")\n# <literal>')
    step["observation_status"] = "success"
    labels = committed_action_outcomes(source)["actions"][0]["labels"]
    assert "json-string-escaping" not in labels
    assert "native-schema-mismatch" not in labels


def test_stop_only_original_tokens_are_not_replaced_and_trace_labels_empty_stop(
    tmp_path, monkeypatch
):
    from skillev.diagnostics.rollout_trace import NullRolloutTraceSink, RolloutTraceStage
    from tests.rollout.engine_fakes import ByteTokenizer, GenerationScript, make_harness

    observed = []

    async def record(self, value):
        observed.append(value)

    monkeypatch.setattr(NullRolloutTraceSink, "record", record)
    tokenizer = ByteTokenizer(decode_overrides={(201,): "<stop>"})
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=[
            GenerationScript.text(tokenizer, "reason"),
            GenerationScript(content_token_ids=(), stop_token_ids=(201,)),
        ],
        max_turns=1,
    )
    artifact = asyncio.run(harness.engine.run(harness.request))
    assert artifact.record.steps[0].action_token_ids == (201,)
    assert artifact.record.steps[0].action_text == "<stop>"
    outcomes = [e for e in observed if e.stage == RolloutTraceStage.ENVIRONMENT_RESULT]
    assert len(outcomes) == 1
    assert "empty-stop" in outcomes[0].public_payload["action_diagnostics"]["labels"]


def test_residual_signs_and_missing_members_keep_full_batch_denominator():
    source = event()
    residuals = source["payload"]["stats"]["residuals"]
    residuals[0]["delta"] = -2.0
    residuals[1]["delta"] = 0.0
    report = committed_learning_report(source, condition_id="fixed")
    assert report["summary"]["negative_delta_count"] == 1
    assert report["summary"]["zero_delta_count"] == 1
    assert report["summary"]["positive_delta_count"] == 26
    assert report["summary"]["normalized_ttb_loss_mean"] == pytest.approx(108 / 28)
    residuals[0]["delta"] = None
    report = committed_learning_report(source, condition_id="fixed")
    assert report["summary"]["normalized_ttb_loss_mean"] is None
    assert report["summary"]["normalized_ttb_loss_observed_count"] == 27
    assert report["summary"]["positive_delta_count"] is None
    assert report["summary"]["trajectory_count"] == 28
