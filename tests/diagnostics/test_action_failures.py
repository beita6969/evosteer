import pytest

from skillev.diagnostics.action_failures import classify_action_outcome


@pytest.mark.parametrize(
    ("text", "status", "expected"),
    [
        ("I will now open the drawer.", "parse-error", "natural-language-or-non-json-action"),
        ('{"code":"a\\q"}', "parse-error", "json-string-escaping"),
        ('{"code":"a\nb"}', "parse-error", "json-string-escaping"),
        ('{"kind":', "parse-error", "json-syntax"),
        ('{"wrong":1}', "schema-invalid", "json-schema-mismatch"),
    ],
)
def test_failure_classification_does_not_rewrite_text(text, status, expected):
    assert expected in classify_action_outcome(
        text,
        parse_status=status,
        finish_reason="length",
        action_kind=None,
        completed=False,
        observation_status="parse-error",
    )


def test_completion_and_interface_candidates_are_not_task_success_claims():
    kwargs = {
        "parse_status": "valid",
        "finish_reason": "stop",
        "action_kind": "complete",
        "completed": False,
        "observation_status": "success",
    }
    assert classify_action_outcome("{}", **kwargs) == ("complete-not-terminal",)
    labels = classify_action_outcome("{}", **kwargs, interface_matches=False)
    assert "declared-interface-mismatch" in labels
    assert "complete-not-terminal" in labels
    kwargs["completed"] = True
    assert classify_action_outcome("{}", **kwargs) == ("accepted-action",)


def test_classifier_consumes_actual_codec_wire_status():
    from skillev.rollout import StructuredJsonActionCodec

    for text, expected in [
        ("I finished the task.", "natural-language-or-non-json-action"),
        ('{"not_an_action": true}', "json-schema-mismatch"),
    ]:
        parsed = StructuredJsonActionCodec().parse(text)
        assert expected in classify_action_outcome(
            text,
            parse_status=parsed.status.value,
            finish_reason=None,
            action_kind=None,
            completed=False,
            observation_status="parse-error",
        )


def test_historical_record_without_terminal_flag_is_not_a_completion_bug():
    assert classify_action_outcome(
        "{}",
        parse_status="valid",
        finish_reason=None,
        action_kind="complete",
        completed=None,
        observation_status="success",
    ) == ("completion-terminal-state-unavailable",)


@pytest.mark.parametrize(
    ("text", "status", "expected"),
    [
        ("", "parse-error", "empty-stop"),
        ("<tool_call><function=run>", "parse-error", "unclosed-native-carrier"),
        (
            '<tool_call>{"name":"run","arguments":{"code":"bad\\q"}}</tool_call>',
            "schema-invalid",
            "json-string-escaping",
        ),
        (
            '<tool_call>{"name":"absent","arguments":{}}</tool_call>',
            "schema-invalid",
            "action-not-in-current-surface",
        ),
        (
            "<tool_call><function=read_skill><parameter=skill_id>absent</parameter></function></tool_call>",
            "schema-invalid",
            "action-not-in-current-surface",
        ),
        ("I will explain the task.", "parse-error", "non-carrier-action-text"),
    ],
)
def test_native_observable_labels_do_not_assume_json_or_intent(text, status, expected):
    labels = classify_action_outcome(
        text,
        parse_status=status,
        finish_reason="stop",
        action_kind=None,
        completed=False,
        observation_status="schema_invalid",
        action_wire="native-single-tool-call@3",
        available_native_names=("run", "read_skill"),
        visible_skill_ids=("visible",),
    )
    assert expected in labels
    assert "natural-language-or-non-json-action" not in labels
    assert "json-schema-mismatch" not in labels


def test_length_and_execution_and_terminal_failure_are_not_mutually_exclusive():
    labels = classify_action_outcome(
        "raw",
        parse_status="valid",
        finish_reason="length",
        action_kind="tool",
        completed=False,
        observation_status="tool_error",
        admitted=True,
        executed=True,
        terminal_success=False,
    )
    assert "output-length-limit" in labels
    assert "admitted-execution-unsuccessful" in labels
    assert "admitted-action-in-terminal-failed-trajectory" in labels
    unknown = classify_action_outcome(
        None,
        parse_status=None,
        finish_reason=None,
        action_kind=None,
        completed=None,
        observation_status=None,
        action_wire=None,
    )
    assert unknown == ("raw-action-unavailable",)
