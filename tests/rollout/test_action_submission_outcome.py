import pytest

from skillev.diagnostics.action_submission import ActionSubmissionOutcome
from skillev.runtime.execution import ActionParseResult, ActionParseStatus


@pytest.mark.parametrize(
    ("text", "finish", "status", "expected"),
    [
        (
            "<tool_call><function=submit_answer>partial",
            "length",
            "parse-error",
            "incomplete-at-budget",
        ),
        (
            "<tool_call>x</tool_call><tool_call>y</tool_call>",
            "stop",
            "parse-error",
            "multiple-explicit-calls",
        ),
        ("I have finished.", "stop", "parse-error", "prose-without-call"),
        ("成", "stop", "parse-error", "prose-without-call"),
        ('{"name":"wrong"}', "stop", "schema-invalid", "unsupported-schema"),
    ],
)
def test_failure_classification_preserves_response_and_budget(text, finish, status, expected):
    parsed = ActionParseResult(ActionParseStatus(status), None, "public-code")
    outcome = ActionSubmissionOutcome.observe(
        text,
        parsed,
        finish_reason=finish,
        output_tokens=2048,
        action_token_cap=2048,
        turns_remaining=0,
    )
    assert outcome.carrier_status == expected
    assert outcome.public_feedback()["submitted"] is False
    assert outcome.public_feedback()["controller_turns_remaining"] == 0
    assert outcome.response_received
