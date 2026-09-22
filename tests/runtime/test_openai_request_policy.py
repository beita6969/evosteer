from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.executor.m_exec import MExec
from src.executor.openai_request_policy import (
    ContextBudgetExceeded,
    OpenAIRequestPolicy,
    PermanentRequestRejected,
    TransientRequestExhausted,
)
from training.environment import GenericTaskEnvironment
from training.gflownet_trainer import GFlowNetTrainer
from training.trajectory import Trajectory


class Tokenizer:
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return list(range(self.tokens + (3 if kwargs.get("tools") else 0)))


class MappingTokenizer(Tokenizer):
    def apply_chat_template(self, messages, **kwargs):
        token_ids = super().apply_chat_template(messages, **kwargs)
        return {"input_ids": [token_ids], "attention_mask": [[1] * len(token_ids)]}


class QwenToolCallTokenizer(Tokenizer):
    def apply_chat_template(self, messages, **kwargs):
        arguments = messages[-1]["tool_calls"][0]["function"]["arguments"]
        if not isinstance(arguments, dict):
            raise TypeError("Qwen tool-call arguments must be a mapping")
        return super().apply_chat_template(messages, **kwargs)


class Completions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class HTTPFailureError(RuntimeError):
    def __init__(self, status_code: int, message: str = "failure") -> None:
        self.status_code = status_code
        super().__init__(message)


def client(*outcomes):
    completions = Completions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def call(policy, fake_client, **overrides):
    kwargs = {
        "request_kind": "supervisor",
        "model": "model",
        "messages": [{"role": "user", "content": "safe"}],
        "tools": [{"type": "function", "function": {"name": "tool"}}],
        "max_tokens": 10,
        "temperature": 0.0,
        "enable_thinking": False,
        "extra_body": None,
    }
    kwargs.update(overrides)
    return policy.create_chat_completion(fake_client, **kwargs)


def test_tools_and_completion_reserve_are_in_budget() -> None:
    tokenizer = Tokenizer(87)
    policy = OpenAIRequestPolicy(tokenizer=tokenizer, context_limit=104, safety_tokens=4)
    fake_client, completions = client("ok")

    assert call(policy, fake_client) == "ok"
    assert len(completions.calls) == 1
    assert tokenizer.calls[0][1]["tools"]


def test_mapping_chat_template_counts_input_ids_not_mapping_keys() -> None:
    policy = OpenAIRequestPolicy(tokenizer=MappingTokenizer(87), context_limit=104, safety_tokens=4)
    fake_client, completions = client("ok")

    assert call(policy, fake_client) == "ok"
    assert len(completions.calls) == 1


def test_openai_tool_call_arguments_are_mapped_only_for_tokenizer() -> None:
    tokenizer = QwenToolCallTokenizer(10)
    policy = OpenAIRequestPolicy(tokenizer=tokenizer, context_limit=100)
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "safe"}',
                    },
                }
            ],
        }
    ]
    fake_client, completions = client("ok")

    call(policy, fake_client, messages=messages, tools=None)

    assert tokenizer.calls[0][0][-1]["tool_calls"][0]["function"]["arguments"] == {"query": "safe"}
    assert messages[-1]["tool_calls"][0]["function"]["arguments"] == '{"query": "safe"}'
    assert completions.calls[0]["messages"] == messages


def test_one_token_over_budget_never_sends_http() -> None:
    policy = OpenAIRequestPolicy(tokenizer=Tokenizer(88), context_limit=104, safety_tokens=4)
    fake_client, completions = client("unused")

    with pytest.raises(ContextBudgetExceeded) as caught:
        call(policy, fake_client)

    assert caught.value.source == "local_preflight"
    assert caught.value.budget.overflow_tokens == 1
    assert completions.calls == []


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_http_rejection_is_not_retried(status: int) -> None:
    policy = OpenAIRequestPolicy(tokenizer=Tokenizer(10), context_limit=100, max_retries=2)
    fake_client, completions = client(HTTPFailureError(status))

    with pytest.raises(PermanentRequestRejected):
        call(policy, fake_client, tools=None)

    assert len(completions.calls) == 1


def test_server_context_400_is_typed_and_not_retried() -> None:
    policy = OpenAIRequestPolicy(tokenizer=Tokenizer(10), context_limit=100, max_retries=2)
    fake_client, completions = client(HTTPFailureError(400, "maximum context length exceeded"))

    with pytest.raises(ContextBudgetExceeded) as caught:
        call(policy, fake_client, tools=None)

    assert caught.value.source == "server_rejection"
    assert len(completions.calls) == 1


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
def test_transient_http_failure_has_bounded_retries(status: int) -> None:
    delays = []
    policy = OpenAIRequestPolicy(
        tokenizer=Tokenizer(10),
        context_limit=100,
        max_retries=2,
        sleep=delays.append,
    )
    fake_client, completions = client(
        HTTPFailureError(status), HTTPFailureError(status), HTTPFailureError(status)
    )

    with pytest.raises(TransientRequestExhausted) as caught:
        call(policy, fake_client, tools=None)

    assert caught.value.attempts == 3
    assert len(completions.calls) == 3
    assert delays == [1.0, 2.0]


def test_context_termination_preserves_existing_and_tool_turn() -> None:
    environment = object.__new__(GenericTaskEnvironment)
    environment.epsilon_min = 0.1
    environment._messages = [{"role": "user", "content": "private"}]
    trajectory = Trajectory()
    existing = SimpleNamespace()
    trajectory.turns = [existing]  # type: ignore[list-item]
    budget = OpenAIRequestPolicy(tokenizer=Tokenizer(20), context_limit=20, safety_tokens=1).budget(
        messages=[{"role": "user", "content": "private"}],
        tools=None,
        completion_tokens=1,
        enable_thinking=False,
    )
    error = ContextBudgetExceeded(budget, request_kind="m_exec:plan", source="local_preflight")

    _, done, _ = environment.terminate_context_budget(
        trajectory,
        error,
        failed_step=2,
        content="tool call",
        tool_name="plan",
        tool_args={"instruction": "private"},
    )

    assert done is True
    assert trajectory.turns[0] is existing
    assert trajectory.turns[-1].action_type == "plan"
    assert trajectory.termination_reason == "context_budget_exceeded"
    assert trajectory.failed_step == 2


def test_trajectory_request_failure_metadata_round_trips() -> None:
    trajectory = Trajectory(
        termination_reason="context_budget_server_rejection",
        runtime_error_class="ContextBudgetExceeded",
        failed_step=3,
        request_prompt_tokens=30,
        request_completion_tokens=10,
        request_context_limit=32,
        request_overflow_tokens=8,
        request_error_source="server_rejection",
    )

    restored = Trajectory.from_dict(trajectory.to_dict())

    assert restored.termination_reason == trajectory.termination_reason
    assert restored.runtime_error_class == trajectory.runtime_error_class
    assert restored.failed_step == 3
    assert restored.request_overflow_tokens == 8
    assert restored.request_error_source == "server_rejection"


def test_formal_environment_does_not_apply_character_truncation() -> None:
    environment = GenericTaskEnvironment(
        m_exec=SimpleNamespace(), max_context_chars=0, max_obs_chars=0
    )
    question = {
        "question": "x" * 31_000,
        "answer": "unused",
        "task_type": "science_qa",
        "extra": {},
    }

    messages, _ = environment.reset(question)

    assert "[CONTEXT TRUNCATED]" not in messages[-1]["content"]
    assert len(messages[-1]["content"]) > 30_000


def test_formal_forward_context_overflow_is_not_silently_trimmed() -> None:
    trainer = object.__new__(GFlowNetTrainer)
    trainer.config = {"formal_runtime": True, "logprob_context_length": 8}
    trainer.device = "cpu"
    trainer.tokenizer = SimpleNamespace(encode=lambda value, **_kwargs: list(range(len(value))))

    with pytest.raises(RuntimeError, match="RecordedForwardContextOverflow"):
        trainer._prepare_turn_tokens("1234567", "89")


def test_mexec_disables_openai_sdk_retries(monkeypatch) -> None:
    observed = {}

    def fake_openai(**kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr("src.executor.m_exec.OpenAI", fake_openai)
    executor = MExec("http://127.0.0.1:8005/v1", request_timeout=17)

    assert executor.client is not None
    assert observed["max_retries"] == 0
    assert observed["timeout"] == 17
