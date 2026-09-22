"""Synthetic credentials/HTTP only: no live endpoint or private dataset access."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from skillev_private.evaluation.external_judge_api import (
    make_external_judge_client,
    make_healthbench_external_sampler,
)

from skillev.evaluation.external_judge_policy import (
    DIRECT_JUDGE_API_BASE,
    DIRECT_JUDGE_MODEL,
    EXTERNAL_JUDGE_API_BASE,
    EXTERNAL_JUDGE_KEY_ENV,
    EXTERNAL_JUDGE_MODEL,
)
from skillev.evaluation.healthbench_luna_profile import luna_profile
from skillev.evaluation.healthbench_transport import HealthBenchTransportError


@pytest.fixture(autouse=True)
def isolated_route(monkeypatch):
    from skillev_private.evaluation import external_judge_failover as failover

    monkeypatch.setattr(failover, "PROCESS_ROUTE", failover.JudgeRouteState())


def _models():
    return httpx.Response(200, json={"object": "list", "data": []})


def test_supplied_client_config_uses_key_but_not_its_sol_default(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv(EXTERNAL_JUDGE_KEY_ENV, raising=False)
    monkeypatch.delenv(f"{EXTERNAL_JUDGE_KEY_ENV}_FILE", raising=False)
    path = tmp_path / ".config/student-api/client.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "base_url": EXTERNAL_JUDGE_API_BASE,
                "api_key": "synthetic-gateway-key",
                "model": "lab-gpt-5.6-sol",
            }
        )
    )
    calls = []

    def respond(request):
        if request.method == "GET":
            return _models()
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "id": "synthetic-response",
                "created_at": 0,
                "object": "response",
                "model": "lab-gpt-5.6-luna",
                "status": "completed",
                "output": [
                    {
                        "id": "message-1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"criteria_met": false}',
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
            },
        )

    client_class = openai.DefaultHttpxClient
    monkeypatch.setattr(
        openai,
        "DefaultHttpxClient",
        lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(respond)),
    )
    with make_external_judge_client() as client:
        sampler = make_healthbench_external_sampler(response_type=SimpleNamespace, client=client)
        response = sampler([{"role": "user", "content": "Synthetic JSON criterion"}])
    assert len(calls) == 1
    request = calls[0]
    assert str(request.url) == EXTERNAL_JUDGE_API_BASE + "/responses"
    assert request.headers["authorization"] == "Bearer synthetic-gateway-key"
    assert request.headers["user-agent"] == "student-api-client/1.0"
    body = json.loads(request.content)
    assert body["model"] == EXTERNAL_JUDGE_MODEL
    assert body["reasoning"] == {"effort": "medium"}
    assert body["max_output_tokens"] == 8000
    assert body["stream"] is False
    assert "tools" not in body
    assert response.response_metadata["returned_model"] == "lab-gpt-5.6-luna"
    assert response.response_metadata["judge_model"] == EXTERNAL_JUDGE_MODEL
    path.write_text(json.dumps({"base_url": "https://different.invalid/v1", "api_key": "fake"}))
    with pytest.raises(RuntimeError):
        make_external_judge_client()
    assert len(calls) == 1


@pytest.mark.parametrize("status", [None, 429, 500, 307])
def test_no_sdk_transport_or_upstream_rubric_retry(monkeypatch, status):
    monkeypatch.setenv(EXTERNAL_JUDGE_KEY_ENV, "synthetic-gateway-key")
    calls = []

    def fail(request):
        if request.method == "GET":
            return _models()
        calls.append(request)
        if status is None:
            raise httpx.ReadTimeout("synthetic", request=request)
        return httpx.Response(
            status,
            json={"error": {"message": "synthetic"}},
            headers={"location": "https://different.invalid/v1"},
        )

    client_class = openai.DefaultHttpxClient
    monkeypatch.setattr(
        openai,
        "DefaultHttpxClient",
        lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(fail)),
    )
    with make_external_judge_client() as client:
        sampler = make_healthbench_external_sampler(response_type=SimpleNamespace, client=client)
        messages = [{"role": "user", "content": "Synthetic JSON criterion"}]
        with pytest.raises((openai.APIError, HealthBenchTransportError)):
            sampler(messages)
        # A caller retrying the same rubric must not send another paid request.
        with pytest.raises(RuntimeError):
            sampler(messages)
    assert len(calls) == 1


def test_gateway_profile_rejects_retry_policy():
    with pytest.raises(ValueError):
        replace(luna_profile(), maximum_attempts=2)


def test_gateway_model_and_effort_are_not_taken_from_client_defaults():
    from skillev_private.evaluation.external_judge_api import ResponsesCompletionsAdapter

    calls = []
    adapter = ResponsesCompletionsAdapter(
        SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: calls.append(kw)))
    )
    for model, effort in [("lab-gpt-5.6-sol", "medium"), (EXTERNAL_JUDGE_MODEL, "high")]:
        with pytest.raises(ValueError):
            adapter.create(model=model, reasoning_effort=effort, stream=False)
    assert not calls


def _official_response(*, finish="stop", content='{"criteria_met":false}'):
    return httpx.Response(
        200,
        json={
            "id": "official-synthetic",
            "object": "chat.completion",
            "created": 0,
            "model": DIRECT_JUDGE_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish,
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        },
    )


def _mock_http(monkeypatch, callback):
    monkeypatch.setenv(EXTERNAL_JUDGE_KEY_ENV, "gateway-synthetic")
    monkeypatch.setenv("OPENAI_API_KEY", "official-synthetic")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://must-not-be-used.invalid/v1")
    client_class = openai.DefaultHttpxClient
    monkeypatch.setattr(
        openai,
        "DefaultHttpxClient",
        lambda **kwargs: client_class(
            **kwargs,
            transport=httpx.MockTransport(callback),
        ),
    )


@pytest.mark.parametrize("status", [None, 401, 403, 429, 503])
def test_unavailable_read_only_probe_routes_unsent_request_to_official(monkeypatch, status):
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            assert str(request.url) == EXTERNAL_JUDGE_API_BASE + "/models"
            assert request.headers["authorization"] == "Bearer gateway-synthetic"
            if status is None:
                raise httpx.ReadTimeout("synthetic", request=request)
            return httpx.Response(status, json={"error": {"message": "synthetic"}})
        assert str(request.url) == DIRECT_JUDGE_API_BASE + "/chat/completions"
        assert request.headers["authorization"] == "Bearer official-synthetic"
        return _official_response()

    _mock_http(monkeypatch, respond)
    with make_external_judge_client() as client:
        sampler = make_healthbench_external_sampler(response_type=SimpleNamespace, client=client)
        result = sampler([{"role": "user", "content": "first JSON criterion"}])
    assert [r.method for r in requests] == ["GET", "POST"]
    body = json.loads(requests[1].content)
    assert body["model"] == DIRECT_JUDGE_MODEL
    assert body["reasoning_effort"] == "medium"
    assert body["max_completion_tokens"] == 8000
    assert body["stream"] is False
    assert "tools" not in body
    assert json.loads(result.response_text)["criteria_met"] is False
    assert result.response_metadata["provider"] == "openai-official"
    assert result.response_metadata["provider_requested_model"] == DIRECT_JUDGE_MODEL
    assert result.response_metadata["returned_model"] == DIRECT_JUDGE_MODEL


def test_historical_gateway_only_route_never_switches_provider(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(503, json={"error": {"message": "synthetic"}})

    _mock_http(monkeypatch, respond)
    with make_external_judge_client(allow_official_fallback=False) as client:
        sampler = make_healthbench_external_sampler(response_type=SimpleNamespace, client=client)
        for text in ("first JSON criterion", "different JSON criterion"):
            with pytest.raises(HealthBenchTransportError):
                sampler([{"role": "user", "content": text}])
    assert len(requests) == 2
    assert all(request.method == "POST" for request in requests)
    assert all(request.url.host == "llm-gateway.example.org" for request in requests)


@pytest.mark.parametrize("status", [None, 403, 503])
def test_uncertain_gateway_post_is_not_replayed_but_next_rubric_uses_official(monkeypatch, status):
    requests = []

    def respond(request):
        if request.method == "GET":
            return _models()
        requests.append(request)
        if request.url.host == "llm-gateway.example.org":
            if status is None:
                raise httpx.ReadTimeout("synthetic", request=request)
            return httpx.Response(status, json={"error": {"message": "synthetic"}})
        return _official_response()

    _mock_http(monkeypatch, respond)
    with make_external_judge_client(allow_official_fallback=True) as client:
        sampler = make_healthbench_external_sampler(response_type=SimpleNamespace, client=client)
        uncertain = [{"role": "user", "content": "uncertain JSON criterion"}]
        with pytest.raises((openai.APIError, HealthBenchTransportError)):
            sampler(uncertain)
        assert len(requests) == 1  # no same-call failover/replay
        with pytest.raises(RuntimeError):
            sampler(uncertain)
        assert len(requests) == 1  # upstream rubric retry also suppressed
        result = sampler([{"role": "user", "content": "different JSON criterion"}])
        assert result.response_metadata["provider"] == "openai-official"
    assert len(requests) == 2
    assert json.loads(requests[1].content)["messages"][0]["content"] != uncertain[0]["content"]


def test_official_failure_is_not_retried_or_sent_back_to_gateway(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(503, json={"error": {"message": "synthetic"}})

    _mock_http(monkeypatch, respond)
    with make_external_judge_client(allow_official_fallback=True) as client:
        sampler = make_healthbench_external_sampler(response_type=SimpleNamespace, client=client)
        with pytest.raises(HealthBenchTransportError):
            sampler([{"role": "user", "content": "JSON criterion"}])
    assert [r.method for r in requests] == ["GET", "POST"]
    assert requests[-1].url.host == "api.openai.com"


def test_invalid_or_incomplete_judgement_does_not_select_another_provider():
    from skillev_private.evaluation.external_judge_failover import (
        FailoverCompletions,
        JudgeRouteState,
    )

    outcomes = iter([("stop", '{"criteria_met":false}'), ("length", ""), ("stop", "not JSON")])
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        finish, content = next(outcomes)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=finish,
                    message=SimpleNamespace(content=content),
                )
            ]
        )

    def forbidden():
        raise AssertionError("bad content is not provider unavailability")

    state = JudgeRouteState()
    route = FailoverCompletions(
        SimpleNamespace(create=create), probe=lambda: None, official_factory=forbidden, state=state
    )
    for _ in range(3):
        result = route.create(model=EXTERNAL_JUDGE_MODEL, stream=False)
        assert result.judge_provider == "flowsteer-third-party"
    assert len(calls) == 3
    assert not state.official
