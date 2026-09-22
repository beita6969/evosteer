from __future__ import annotations

import logging
from types import SimpleNamespace

from training import batch_inference
from training.environment import GenericTaskEnvironment


def _completion(content: str) -> SimpleNamespace:
    message = SimpleNamespace(
        content=content,
        reasoning_content=None,
        tool_calls=None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def _client(content: str) -> SimpleNamespace:
    completions = SimpleNamespace(create=lambda **_: _completion(content))
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_search_log_records_size_without_query_text(caplog, monkeypatch) -> None:
    private_query = "PRIVATE QUESTION PREFIX must never enter logs"
    environment = object.__new__(GenericTaskEnvironment)
    environment._task_type = "factual_qa"
    monkeypatch.setattr(environment, "_search_passages", lambda _: "private observation")
    caplog.set_level(logging.INFO, logger="training.environment")

    assert environment._handle_search({"query": private_query}) == "private observation"

    assert private_query not in caplog.text
    assert f"query_chars={len(private_query)}" in caplog.text


def test_supervisor_log_records_size_without_answer_text(caplog, monkeypatch) -> None:
    private_answer = "PRIVATE SUPERVISOR ANSWER must never enter logs"
    monkeypatch.setattr(batch_inference, "_get_client", lambda *_: _client(private_answer))
    caplog.set_level(logging.DEBUG, logger="training.batch_inference")

    content, tool_name, tool_args = batch_inference._supervisor_call_unpaused(
        [], "http://127.0.0.1:8005/v1", "model", [], "", 0.0, 10, False
    )

    assert (content, tool_name, tool_args) == (private_answer, None, None)
    assert private_answer not in caplog.text
    assert f"content_chars={len(private_answer)}" in caplog.text


def test_react_parse_failure_log_records_size_without_output(caplog, monkeypatch) -> None:
    private_output = "PRIVATE REACT OUTPUT must never enter logs"
    monkeypatch.setattr(batch_inference, "_get_client", lambda *_: _client(private_output))
    caplog.set_level(logging.WARNING, logger="training.batch_inference")

    content, action = batch_inference.react_call(
        "private prompt", "http://127.0.0.1:8005/v1", "model"
    )

    assert (content, action) == (private_output, None)
    assert private_output not in caplog.text
    assert f"content_chars={len(private_output)}" in caplog.text


def test_finish_length_retry_rechecks_full_budget(monkeypatch) -> None:
    first_message = SimpleNamespace(
        content="partial",
        reasoning_content=None,
        tool_calls=None,
    )
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="answer", arguments='{"response":"ok"}')
    )
    second_message = SimpleNamespace(
        content="",
        reasoning_content=None,
        tool_calls=[tool_call],
    )
    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=first_message, finish_reason="length")]),
        SimpleNamespace(choices=[SimpleNamespace(message=second_message, finish_reason="stop")]),
    ]

    class Policy:
        def __init__(self):
            self.calls = []

        def create_chat_completion(self, _client, **kwargs):
            self.calls.append(kwargs)
            return responses.pop(0)

    policy = Policy()
    monkeypatch.setattr(batch_inference, "_get_client", lambda *_: object())
    messages = [{"role": "user", "content": "safe"}]

    _, tool_name, tool_args = batch_inference._supervisor_call_unpaused(
        messages,
        "http://127.0.0.1:8005/v1",
        "model",
        [],
        "",
        0.0,
        10,
        False,
        policy,  # type: ignore[arg-type]
        60,
    )

    assert (tool_name, tool_args) == ("answer", {"response": "ok"})
    assert [call["request_kind"] for call in policy.calls] == [
        "supervisor",
        "supervisor_finish_retry",
    ]
    assert len(policy.calls[1]["messages"]) == len(messages) + 2
