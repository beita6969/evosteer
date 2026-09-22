"""Synthetic-only LiveMedBench source, single-owner and rubric integration."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from skillev_private.evaluation.ood_export import export_sources
from skillev_private.evaluation.ood_livemedbench import (
    grade_case,
    parse_verdict,
    request_rubric,
    rubric_result,
)
from skillev_private.evaluation.ood_scoring import score_ood
from skillev_private.evaluation.ood_sources import load_ood_panel

from skillev.evaluation.input_metric_contracts import OOD_BENCHMARKS, PublicTaskView
from skillev.evaluation.integrity_results import NativeScore, native_mean
from skillev.evaluation.livemedbench import (
    BENCHMARK,
    METRIC,
    PROFILE,
    VERIFIER,
    generation_instruction,
)
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from tests.evaluation.test_integrity_broker_boundary import runtime


def case(index=1):
    return {
        "case_id": index,
        "post_time": "2026-01-01T00:00:00",
        "narrative": f"Synthetic patient question {index}.",
        "core_request": "What should I do?",
        "doctor_advice": "PRIVATE_DOCTOR",
        "rubric_items": [
            {"criterion": "PRIVATE_POSITIVE", "points": 5},
            {"criterion": "PRIVATE_NEGATIVE", "points": -10},
        ],
    }


def settings(tmp_path):
    return {
        "profile": PROFILE,
        "judge_model": "lab-gpt-5.6-luna",
        "reasoning_effort": "medium",
        "cache_directory": str(tmp_path / "judge"),
    }


def api_response(criterion, met, finish="stop"):
    return SimpleNamespace(
        id="synthetic-response",
        model="lab-gpt-5.6-luna",
        usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 5}),
        choices=[
            SimpleNamespace(
                finish_reason=finish,
                message=SimpleNamespace(
                    content=json.dumps(
                        [{"question": criterion, "met": met, "reasoning": "Evidence"}]
                    ),
                    refusal=None,
                ),
            )
        ],
    )


class Client:
    def __init__(self, responses):
        self.responses, self.calls = iter(responses), []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


def test_livemedbench_replaces_only_the_medical_ood_slot():
    assert BENCHMARK not in OOD_BENCHMARKS
    assert "gpqa-diamond-bioorganic" in OOD_BENCHMARKS
    assert len(OOD_BENCHMARKS) == 6


def test_frozen_source_uses_one_snapshot_and_never_projects_rubrics(tmp_path):
    source = tmp_path / "LiveMedBench_v202601.json"
    rows = [case(index) for index in range(80)]
    source.write_text(json.dumps(rows))
    config = {
        "sample_count": 64,
        "seed": 0,
        "sources": {
            BENCHMARK: {"paths": [str(source)], "snapshot": "v202601"},
        },
    }
    exported = export_sources(config, tmp_path / "frozen")
    with pytest.raises(ValueError):
        load_ood_panel(exported)
    records = [json.loads(line) for line in open(exported["ood_sources"][BENCHMARK])]
    assert len(records) == 64
    assert all("PRIVATE" not in str(record["public"]) for record in records)
    assert all("doctor_advice" not in record["target"] for record in records)
    rows = [
        {
            **row,
            "doctor_advice": "CHANGED",
            "rubric_items": [
                {"criterion": "CHANGED", "points": 1},
            ],
        }
        for row in rows
    ]
    source.write_text(json.dumps(rows))
    second = export_sources(config, tmp_path / "second")
    second_records = [json.loads(line) for line in open(second["ood_sources"][BENCHMARK])]
    assert [record["public"] for record in records] == [
        record["public"] for record in second_records
    ]
    source.write_text(json.dumps([*rows, rows[0]]))
    with pytest.raises(ValueError):
        export_sources(config, tmp_path / "duplicates")


def test_complete_response_prompt_preserves_language_without_banning_explanation():
    assert "中文" in generation_instruction("我感冒了", "怎么办?")
    instruction = generation_instruction("Patient question", "Advice?")
    assert "complete" in instruction
    assert "Explain" in instruction
    assert "without any explanation" not in instruction
    assert "PRIVATE" not in instruction


@pytest.mark.parametrize("native_tools", [False, True])
def test_single_owner_can_submit_complete_medical_reply(tmp_path, native_tools):
    entry = PublicTaskView.from_record("synthetic", BENCHMARK, case())
    instance = runtime(tmp_path, entry, ["This is the complete patient-facing response."])
    arm = InferenceArm(
        "medical",
        tool_call_mode=(ToolCallMode.QWEN_XML if native_tools else ToolCallMode.PLAIN_TEXT),
    )
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
        assert final.text == "This is the complete patient-facing response."
        assert final.intervention_counts["model_calls"] == 1
        outputs = instance.journal.model_outputs(("synthetic", arm.arm_id, entry.task_id))
        assert all(output.participant == "owner" for output in outputs)
    finally:
        instance.journal.close()


def test_official_points_keep_negatives_until_aggregate_clipping():
    result = rubric_result(case()["rubric_items"], [True, True])
    assert result["raw_score"] == -1
    assert result["terminal_reward_projection"] == 0
    assert result["triggered_negative_count"] == 1
    scores = tuple(
        NativeScore(
            str(i),
            BENCHMARK,
            METRIC,
            value,
            secondary_metrics={"triggered-negative-rubric-count": float(value < 0)},
            verifier_version=VERIFIER,
            grader_used=True,
        )
        for i, value in enumerate((-1, 1))
    )
    assert native_mean(scores) == 0  # Not mean(clip(each)), which would be 0.5.
    with pytest.raises(ValueError):
        rubric_result(case()["rubric_items"], [True])


def test_negative_only_released_case_keeps_official_zero_denominator_behavior():
    result = rubric_result([{"criterion": "Error present?", "points": -5}], [True])
    assert result["raw_score"] == 0
    assert result["earned_points"] == -5
    assert result["triggered_negative_count"] == 1


@pytest.mark.parametrize(
    "value", [[], [{"met": True}], [{"question": "C", "met": 1, "reasoning": "x"}]]
)
def test_missing_or_malformed_judge_output_is_not_a_zero_score(value):
    with pytest.raises(ValueError):
        parse_verdict(json.dumps(value), "C")


def test_one_medium_request_preserves_truncation_and_usage():
    client = Client([api_response("C", True, finish="length")])
    result = request_rubric(client, "Synthetic prompt", "C")
    assert result["status"] == "incomplete-or-refused"
    assert result["usage"]["completion_tokens"] == 5
    request = client.calls[0]
    assert request["model"] == "lab-gpt-5.6-luna"
    assert request["reasoning_effort"] == "medium"
    assert request["max_completion_tokens"] == 8000
    assert request["store"] is False


def test_completed_true_and_false_verdicts_are_not_rejudged(tmp_path):
    target = case()
    candidate = SimpleNamespace(
        run_id="run", arm_id="arm", episode_id="case", owner_call_id="owner", text="Reply"
    )
    client = Client(
        [api_response("PRIVATE_POSITIVE", False), api_response("PRIVATE_NEGATIVE", False)]
    )
    first = grade_case(candidate, target, settings(tmp_path), client=client)
    second = grade_case(candidate, target, settings(tmp_path), client=client)
    assert first == second
    assert len(client.calls) == 2
    assert first["raw_score"] == 0
    assert first["cost"]["output_tokens"] == 10
    candidate.text = "Changed answer"
    with pytest.raises(ValueError):
        grade_case(candidate, target, settings(tmp_path), client=client)


def test_failed_request_is_preserved_and_never_silently_retried(tmp_path):
    candidate = SimpleNamespace(
        run_id="run", arm_id="arm", episode_id="case", owner_call_id="owner", text="Reply"
    )
    client = Client([api_response("PRIVATE_POSITIVE", True, finish="length")])
    for _ in range(2):
        with pytest.raises(RuntimeError):
            grade_case(candidate, case(), settings(tmp_path), client=client)
    assert len(client.calls) == 1
    saved = json.loads(next((tmp_path / "judge").glob("*.json")).read_text())
    assert not saved["complete"]
    assert saved["attempts"][0]["usage"]["completion_tokens"] == 5


def test_empty_owner_candidate_stays_in_denominator_without_paid_judge():
    candidate = SimpleNamespace(text="", episode_id="empty")
    score = asyncio.run(
        score_ood(
            SimpleNamespace(get=lambda *args: candidate),
            ("run", "arm", "empty"),
            BENCHMARK,
            case(),
            settings={},
            sandbox=None,
            diagnostics=lambda value: None,
        )
    )
    assert score.value == 0
    assert score.grader_used is False
