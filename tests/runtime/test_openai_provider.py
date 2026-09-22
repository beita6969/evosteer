from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from skillev.contracts.canonical import canonical_json
from skillev.runtime.contracts import BudgetVector
from skillev.runtime.models import ModelRequest, TraceContext
from skillev.runtime.openai_provider import (
    HTTPRequest,
    HTTPResponse,
    OpenAICompatibleProvider,
    OpenAIProviderConfig,
    ProviderResponseError,
)


@dataclass(slots=True)
class FakeTransport:
    response: HTTPResponse
    requests: list[HTTPRequest] = field(default_factory=list)

    def send(self, request: HTTPRequest) -> HTTPResponse:
        self.requests.append(request)
        return self.response


def _response(
    *,
    choice_fields: dict[str, object] | None = None,
    prompt_tokens: int = 7,
    completion_tokens: int = 3,
) -> HTTPResponse:
    choice: dict[str, object] = {
        "finish_reason": "stop",
        "index": 0,
        "message": {
            "content": canonical_json({"kind": "complete", "value": 4}),
            "role": "assistant",
        },
    }
    if choice_fields is not None:
        choice.update(choice_fields)
    payload = {
        "choices": [choice],
        "id": "response-local",
        "model": "local-model",
        "usage": {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return HTTPResponse(200, canonical_json(payload).encode("utf-8"))


def _request() -> ModelRequest:
    return ModelRequest(
        prompt="Choose one structured action.",
        response_schema={
            "additionalProperties": False,
            "properties": {"kind": {"type": "string"}},
            "required": ["kind"],
            "type": "object",
        },
    )


def _trace() -> TraceContext:
    return TraceContext(
        run_id="run-local",
        attempt_id="attempt-local",
        invocation_id="invocation-local",
        turn=1,
    )


def _budget() -> BudgetVector:
    return BudgetVector(
        input_tokens=32,
        output_tokens=16,
        model_calls=1,
        agent_turns=1,
    )


def test_structured_request_parses_usage_and_explicit_action_span() -> None:
    secret = "local-test-token"
    transport = FakeTransport(
        _response(
            choice_fields={
                "action_token_span": {
                    "token_ids": [101, 102, 103],
                    "token_logprobs": [-0.1, -0.2, -0.3],
                }
            }
        )
    )
    provider = OpenAICompatibleProvider(
        OpenAIProviderConfig(
            endpoint_base="http://127.0.0.1:8000/v1",
            model="local-model",
            seed=17,
            max_output_tokens=8,
        ),
        transport=transport,
        bearer_token_provider=lambda: secret,
    )

    result = asyncio.run(provider.generate(_request(), budget=_budget(), trace_context=_trace()))

    assert result.input_tokens == 7
    assert result.output_tokens == 3
    assert result.action_token_ids == (101, 102, 103)
    assert result.action_token_logprobs == pytest.approx((-0.1, -0.2, -0.3))

    sent = transport.requests[0]
    body = json.loads(sent.body)
    assert body["seed"] == 17
    assert body["max_tokens"] == 8
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == _request().response_schema
    assert sent.headers["Authorization"] == f"Bearer {secret}"

    assert secret not in repr(sent)
    assert secret not in repr(provider)
    assert secret not in canonical_json(result.provider_metadata)


def test_missing_token_ids_does_not_fabricate_an_action_span() -> None:
    transport = FakeTransport(
        _response(
            choice_fields={
                "logprobs": {
                    "content": [
                        {"logprob": -0.1, "token": "{"},
                        {"logprob": -0.2, "token": "}"},
                    ]
                }
            },
            completion_tokens=2,
        )
    )
    provider = OpenAICompatibleProvider(
        OpenAIProviderConfig(
            endpoint_base="http://localhost:8000/v1",
            model="local-model",
        ),
        transport=transport,
    )

    result = asyncio.run(provider.generate(_request(), budget=_budget(), trace_context=_trace()))

    assert result.action_token_ids == ()
    assert result.action_token_logprobs == ()
    assert result.provider_metadata["action_span_source"] == "missing"


def test_completion_token_ids_do_not_imply_an_action_boundary() -> None:
    transport = FakeTransport(
        _response(
            choice_fields={
                "logprobs": {
                    "content": [
                        {"logprob": -0.4, "token": "{"},
                        {"logprob": -0.5, "token": "}"},
                    ]
                },
                "token_ids": [8, 9],
            },
            completion_tokens=2,
        )
    )
    provider = OpenAICompatibleProvider(
        OpenAIProviderConfig(
            endpoint_base="http://localhost:8000/v1",
            model="local-model",
        ),
        transport=transport,
    )

    result = asyncio.run(provider.generate(_request(), budget=_budget(), trace_context=_trace()))

    assert result.action_token_ids == ()
    assert result.action_token_logprobs == ()


def test_misaligned_explicit_action_span_is_rejected() -> None:
    transport = FakeTransport(
        _response(
            choice_fields={
                "action_token_span": {
                    "token_ids": [1, 2],
                    "token_logprobs": [-0.1],
                }
            }
        )
    )
    provider = OpenAICompatibleProvider(
        OpenAIProviderConfig(
            endpoint_base="http://localhost:8000/v1",
            model="local-model",
        ),
        transport=transport,
    )

    with pytest.raises(ProviderResponseError):
        asyncio.run(provider.generate(_request(), budget=_budget(), trace_context=_trace()))
