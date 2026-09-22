import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from skillev_private.benchmarks.format_content_review import (
    FormatReviewedEvaluator,
    LunaFormatContentReviewer,
    wrap_training_evaluator,
)
from skillev_private.benchmarks.protocol_v13_training_sessions import _StaticEvaluator

from skillev.evaluation.external_judge_policy import EXTERNAL_JUDGE_MODEL
from skillev.rollout import NoTerminalSubmission, RolloutTermination, TerminalEvaluatorError
from skillev.rollout.environment import NoSubmissionReason, TerminalActionEvidence
from skillev.runtime.request_journal import DurableRequestJournal
from tests.benchmarks.test_protocol_v13_training_sessions import _record, _request


class Client:
    def __init__(self, verdict=None, error=None):
        self.calls = []
        self.verdict = verdict or {
            "format_only": True,
            "content_correct": True,
            "explanation": "Only the closing wrapper is missing; the full answer is present.",
        }
        self.error = error
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            id="fixture-response",
            model=EXTERNAL_JUDGE_MODEL,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=self.verdict
                        if isinstance(self.verdict, str)
                        else json.dumps(self.verdict),
                        refusal=None,
                    ),
                )
            ],
            usage=None,
        )

    def close(self):
        pass


def setup(tmp_path, client=None):
    record = _record()
    client = client or Client()
    journal = DurableRequestJournal(tmp_path / "requests.sqlite3")
    reviewer = LunaFormatContentReviewer(journal, lambda: client)
    native = _StaticEvaluator(record, record.input)
    evaluator = FormatReviewedEvaluator(
        native, record.input, "hotpotqa", record.output.target, reviewer
    )
    request = replace(
        _request(record.input.task_id, "unused"),
        termination=RolloutTermination.HORIZON_EXHAUSTED,
        evaluation_input=NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED),
        last_action=TerminalActionEvidence(
            8, '<tool_call>{"answer":"private answer"', "parse-error", "parse_error", "length"
        ),
    )
    return record, evaluator, request, client


def test_format_zero_can_gain_credit_without_rewriting_action_native_metrics_or_history(tmp_path):
    record, evaluator, request, client = setup(tmp_path)
    before = request.to_value()
    reward = asyncio.run(evaluator.evaluate(request))
    assert reward.value == 1
    assert reward.success
    review = reward.native_payload["format_content_review"]
    assert review["native_result"]["value"] == 0
    assert review["native_result"]["success"] is False
    assert reward.native_payload["public_metrics"]["answer-f1"] == 0
    assert request.to_value() == before
    assert "private answer" not in json.dumps(record.input.to_value())
    call = client.calls[0]
    assert (call["model"], call["reasoning_effort"]) == (EXTERNAL_JUDGE_MODEL, "medium")
    assert call["stream"] is False
    assert "tools" not in call
    assert json.loads(call["messages"][1]["content"])["candidate"] == request.last_action.text
    assert asyncio.run(evaluator.evaluate(request)).value == 1
    assert len(client.calls) == 1  # Restore the original settled review, never resample.


@pytest.mark.parametrize(("format_only", "correct"), [(False, True), (True, False), (False, False)])
def test_credit_requires_both_format_only_and_correct(tmp_path, format_only, correct):
    client = Client(
        {"format_only": format_only, "content_correct": correct, "explanation": "Reject."}
    )
    _, evaluator, request, _ = setup(tmp_path, client)
    reward = asyncio.run(evaluator.evaluate(request))
    assert reward.value == 0
    assert not reward.success
    assert reward.native_payload["format_content_review"]["approved"] is False


@pytest.mark.parametrize("answer", ["private answer", "wrong answer"])
def test_regular_admitted_answers_never_get_a_second_chance(tmp_path, answer):
    record, evaluator, _, client = setup(tmp_path)
    request = _request(record.input.task_id, answer)
    expected = asyncio.run(evaluator.delegate.evaluate(request))
    assert asyncio.run(evaluator.evaluate(request)) == expected
    assert not client.calls


def test_environment_failures_and_non_format_exhaustion_are_not_textual_successes(tmp_path):
    _, evaluator, request, client = setup(tmp_path)
    reward = asyncio.run(replace(evaluator, domain="alfworld").evaluate(request))
    assert reward.value == 0
    assert not client.calls
    request = replace(
        request,
        last_action=replace(
            request.last_action, parse_status="valid", observation_status="tool_error"
        ),
    )
    assert asyncio.run(evaluator.evaluate(request)).value == 0
    assert not client.calls


def test_stop_only_without_answer_is_not_sent_to_the_judge(tmp_path):
    _, evaluator, request, client = setup(tmp_path)
    request = replace(request, last_action=replace(request.last_action, text=""))
    assert asyncio.run(evaluator.evaluate(request)).value == 0
    assert not client.calls


@pytest.mark.parametrize("error", [TimeoutError("unknown outcome"), None])
def test_judge_failure_is_infrastructure_not_zero_and_is_not_retried(tmp_path, error):
    client = Client(verdict="not-json", error=error)
    _, evaluator, request, _ = setup(tmp_path, client)
    for _ in range(2):
        with pytest.raises(TerminalEvaluatorError):
            asyncio.run(evaluator.evaluate(request))
    assert len(client.calls) == 1


def test_native_infrastructure_error_never_enters_fallback(tmp_path):
    _, evaluator, request, client = setup(tmp_path)

    class Broken:
        async def evaluate(self, request):
            raise TerminalEvaluatorError("native scorer unavailable")

    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(replace(evaluator, delegate=Broken()).evaluate(request))
    assert not client.calls


def test_admission_schema_failure_can_be_reviewed(tmp_path):
    _, evaluator, request, _ = setup(tmp_path)
    request = replace(
        request,
        last_action=replace(
            request.last_action, parse_status="valid", observation_status="schema_invalid"
        ),
    )
    assert asyncio.run(evaluator.evaluate(request)).value == 1


def test_only_training_records_from_declared_step_receive_wrapper(tmp_path):
    record, evaluator, _, _ = setup(tmp_path)
    journal = evaluator.reviewer.journal
    for step, expected in ((80, False), (81, True), (82, True)):
        current = replace(
            record, episode=replace(record.episode, optimizer_step=step, block_position=step - 1)
        )
        wrapped = wrap_training_evaluator(evaluator.delegate, record.input, current, 81, journal)
        assert isinstance(wrapped, FormatReviewedEvaluator) is expected
    with pytest.raises(ValueError):
        wrap_training_evaluator(evaluator.delegate, record.input, object(), 81, journal)
