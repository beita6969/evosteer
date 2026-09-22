"""Synthetic CPU transport only; these are not medical or real model verdicts."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from skillev_private.benchmarks import healthbench_api as api
from skillev_private.benchmarks.healthbench_ledger import official_criterion_messages
from skillev_private.evaluation.integrity_grader_usage import (
    IncompleteNativeGradingError,
    MeteredCompletions,
)

from skillev.runtime.request_journal import UnknownRequestOutcomeError
from skillev.training import AsyncResourceLimiter
from tests.evaluation.test_healthbench_luna_route import case, fake_official


def grader(tmp_path):
    return api.OpenAIHealthBenchGrader(
        {"task": case()},
        Path("synthetic-source"),
        AsyncResourceLimiter(1),
        criterion_ledger_root=tmp_path,
    )


def test_all_criteria_and_negative_raw_score_persist_then_resume_without_more_calls(
    tmp_path, monkeypatch
):
    requests, _ = fake_official(monkeypatch, iter([{"criteria_met": True}, {"criteria_met": True}]))
    result = asyncio.run(grader(tmp_path).grade("task", "fixed synthetic submission"))
    assert len(requests) == 2
    path = Path(result.criterion_ledger["path"])
    original = path.read_bytes()
    state = json.loads(original)
    assert state["status"] == "completed"
    assert state["binding"]["candidate_answer"] == "fixed synthetic submission"
    assert len(state["requests"]) == len(state["diagnostics"]["criterion_results"]) == 2
    assert [r["criterion_index"] for r in state["diagnostics"]["criterion_results"]] == [0, 1]
    assert state["result"]["native_raw_score"] == -0.5
    assert state["result"]["learning_reward"] == 0
    assert state["result"]["binary_success"] is False
    assert state["result"]["negative_criterion_count"] == 1
    assert all(r["request"]["reasoning_effort"] == "medium" for r in state["requests"])
    assert all(r["response"]["content"] for r in state["requests"])
    assert all(r["elapsed_seconds"] >= 0 for r in state["requests"])
    restored = asyncio.run(grader(tmp_path).grade("task", "fixed synthetic submission"))
    assert restored == result
    assert len(requests) == 2
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        asyncio.run(grader(tmp_path).grade("task", "different submission"))
    assert path.read_bytes() == original


def test_unknown_criterion_is_durable_not_negative_or_resampled(tmp_path, monkeypatch):
    requests, _ = fake_official(monkeypatch, iter([TimeoutError("synthetic timeout")]))
    with pytest.raises(IncompleteNativeGradingError):
        asyncio.run(grader(tmp_path).grade("task", "fixed"))
    path = tmp_path / "task.json"
    original = path.read_bytes()
    state = json.loads(original)
    assert state["status"] == "incomplete-grading"
    assert state["result"] is None
    assert state["diagnostics"]["native_raw_score"] is None
    assert state["requests"][0]["error_type"] == "TimeoutError"
    assert "response" not in state["requests"][0]
    with pytest.raises(UnknownRequestOutcomeError):
        asyncio.run(grader(tmp_path).grade("task", "fixed"))
    assert len(requests) == 1
    assert path.read_bytes() == original


def test_request_is_recorded_before_dispatch_and_full_invalid_response_kept():
    published = []

    class Response:
        choices: ClassVar[list] = [
            SimpleNamespace(
                message=SimpleNamespace(content='{"criteria_met":true}'), finish_reason="length"
            )
        ]
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
        id = "synthetic-response"
        model = "synthetic-judge"
        _request_id = "synthetic-request"

        def model_dump(self, **kwargs):
            return {"id": self.id, "raw": "full synthetic response"}

    def create(**kwargs):
        assert len(published) == 1
        assert published[0][0]["request"]["messages"][0]["content"] == "private synthetic rubric"
        assert "response" not in published[0][0]
        return Response()

    meter = MeteredCompletions(
        api.BoundedAPICompletions(SimpleNamespace(create=create)), record_evidence=published.append
    )
    with pytest.raises(api.IncompleteHealthBenchResponseError):
        meter.create(messages=[{"role": "user", "content": "private synthetic rubric"}], timeout=1)
    row = published[-1][0]
    assert row["response"]["finish_reason"] == "length"
    assert row["response"]["raw_response"]["raw"] == "full synthetic response"
    assert row["response"]["request_id"] == "synthetic-request"
    assert meter.snapshot()["known_output_tokens"] == 20
    assert meter.snapshot()["unknown_usage_calls"] == 0
    assert "private synthetic rubric" not in repr(meter.snapshot())


GRADER_TEMPLATE = "Official synthetic: <<conversation>>\nCriterion: <<rubric_item>>"


def synthetic_grade_sample():
    return GRADER_TEMPLATE


def test_actual_official_template_association_uses_complete_messages_and_keeps_duplicate_indices():
    prompt = [{"role": "user", "content": "synthetic user"}]
    expected = official_criterion_messages(
        synthetic_grade_sample, prompt, "reply", ["same", "same"]
    )
    assert expected[0] == [
        {
            "role": "user",
            "content": (
                "Official synthetic: user: synthetic user\n\nassistant: reply\nCriterion: same"
            ),
        }
    ]
    response = SimpleNamespace(
        usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
    )
    meter = MeteredCompletions(SimpleNamespace(create=lambda **_: response), retain_evidence=True)
    meter.bind_criterion_messages(expected)
    meter.create(messages=expected[0])
    assert meter.evidence()[0]["criterion_indices"] == [0, 1]
    assert meter.evidence()[0]["response"]["input_tokens"] is None
    assert official_criterion_messages(object(), prompt, "reply", ["same"]) is None


def test_cancellation_drains_owned_grading_and_preserves_its_original_result(tmp_path, monkeypatch):
    import threading

    from skillev_private.evaluation import integrity_native_scoring

    entered, release = threading.Event(), threading.Event()

    def controlled_grade(candidate, private_case, settings, *, diagnostics, record_requests):
        # Lifecycle fixture only, not an actual model/rubric-quality assertion.
        entered.set()
        assert release.wait(3)
        record_requests([])
        diagnostics.update(status="completed")
        return 0.25, 0, {"model_request_attempts": 0.0}

    monkeypatch.setattr(integrity_native_scoring, "_grade_health", controlled_grade)

    async def cancel_after_start():
        task = asyncio.create_task(grader(tmp_path).grade("task", "fixed"))
        while not entered.is_set():
            await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_after_start())
    state = json.loads((tmp_path / "task.json").read_text())
    assert state["status"] == "completed"
    assert state["result"]["native_raw_score"] == 0.25
    assert state["result"]["binary_success"] is False
