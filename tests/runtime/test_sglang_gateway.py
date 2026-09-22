from __future__ import annotations

import asyncio
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import pytest

from skillev.contracts import JsonValue, canonical_json
from skillev.runtime.contracts import BudgetVector
from skillev.runtime.models import ModelRequest, TraceContext
from skillev.runtime.openai_provider import HTTPRequest, HTTPResponse
from skillev.runtime.sglang_gateway import (
    SGLangGateway,
    SGLangGatewayConfig,
    SGLangRole,
    UrllibSGLangControlTransport,
)


@dataclass(slots=True)
class FakeControlTransport:
    calls: list[tuple[str, str, object]] = field(default_factory=list)
    loaded_adapters: set[str] = field(default_factory=set)
    fail_next_load: bool = False

    def request(
        self,
        *,
        method: str,
        url: str,
        payload: object,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> tuple[int, JsonValue]:
        del timeout_seconds, max_response_bytes
        self.calls.append((method, url, payload))
        if url.endswith("/v1/models"):
            model_ids = ["skillflow-qwen35"]
            model_ids.extend(sorted(self.loaded_adapters))
            return 200, {"data": [{"id": model_id} for model_id in model_ids]}
        if url.endswith("/v1/chat/completions"):
            assert isinstance(payload, dict)
            if payload.get("model") not in self.loaded_adapters:
                return 404, {"error": "adapter absent"}
            return 200, {"choices": [{"message": {"content": "OK"}}]}
        if url.endswith("/load_lora_adapter"):
            if self.fail_next_load:
                self.fail_next_load = False
                return 500, {"success": False}
            assert isinstance(payload, dict)
            self.loaded_adapters.add(str(payload["lora_name"]))
        elif url.endswith("/unload_lora_adapter"):
            assert isinstance(payload, dict)
            self.loaded_adapters.discard(str(payload["lora_name"]))
        return 200, {"success": True}


@dataclass(slots=True)
class FakeModelTransport:
    requests: list[HTTPRequest] = field(default_factory=list)

    def send(self, request: HTTPRequest) -> HTTPResponse:
        self.requests.append(request)
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": canonical_json({"kind": "complete"}),
                        "role": "assistant",
                    },
                }
            ],
            "id": "response-local",
            "model": "served",
            "usage": {"completion_tokens": 2, "prompt_tokens": 4, "total_tokens": 6},
        }
        return HTTPResponse(200, canonical_json(payload).encode())


def _gateway() -> tuple[SGLangGateway, FakeControlTransport, FakeModelTransport]:
    control = FakeControlTransport()
    model = FakeModelTransport()
    gateway = SGLangGateway(
        SGLangGatewayConfig(
            endpoint_base="http://127.0.0.1:8005",
            base_model="skillflow-qwen35",
            supervisor_adapter="theta_step_",
        ),
        _control_transport=control,
        _model_transport=model,
    )
    return gateway, control, model


def _request() -> ModelRequest:
    return ModelRequest(
        prompt="Choose an action",
        response_schema={
            "additionalProperties": False,
            "properties": {"kind": {"type": "string"}},
            "required": ["kind"],
            "type": "object",
        },
    )


def _budget() -> BudgetVector:
    return BudgetVector(input_tokens=16, output_tokens=8, model_calls=1, agent_turns=1)


def _trace() -> TraceContext:
    return TraceContext("run", "attempt", "invocation", 1)


def test_executor_health_and_skill_creator_use_base_while_supervisor_uses_adapter() -> None:
    gateway, _, model = _gateway()

    for role in SGLangRole:
        response = asyncio.run(
            gateway.provider(role).generate(_request(), budget=_budget(), trace_context=_trace())
        )
        assert response.provider_metadata["sglang_role"] == role.value

    sent_models = [json.loads(request.body)["model"] for request in model.requests]
    assert sent_models == [
        "skillflow-qwen35",
        "skillflow-qwen35",
        "theta_step_",
        "skillflow-qwen35",
    ]


def test_health_and_adapter_generation_use_native_sglang_endpoints() -> None:
    gateway, control, _ = _gateway()

    assert asyncio.run(gateway.health()) == ("skillflow-qwen35",)
    generation = asyncio.run(
        gateway.swap_supervisor_adapter(
            adapter_path="/private/checkpoints/theta-step-1",
            adapter_revision="theta-step-1",
        )
    )

    assert generation.generation == 1
    assert generation.adapter_revision == "theta-step-1"
    assert control.calls[-4] == (
        "POST",
        "http://127.0.0.1:8005/load_lora_adapter",
        {
            "lora_name": "theta_step_theta-step-1",
            "lora_path": "/private/checkpoints/theta-step-1",
        },
    )


def test_existing_adapter_binding_only_verifies_and_does_not_mutate_server() -> None:
    gateway, control, _ = _gateway()
    control.loaded_adapters.add("theta_step_")

    generation = gateway.bind_existing_supervisor_adapter(adapter_revision="step-7")

    assert generation.adapter_name == "theta_step_"
    assert generation.adapter_revision == "step-7"
    assert [call[1].rsplit("/", 1)[-1] for call in control.calls] == ["health", "models"]


def test_restore_binds_an_identical_live_revision_without_loading_it_again() -> None:
    gateway, control, _ = _gateway()
    revision = "theta-step-7"
    adapter_name = gateway._adapter_name(revision)
    control.loaded_adapters.add(adapter_name)

    generation = gateway.restore_supervisor_adapter(
        adapter_path="/private/checkpoints/theta-step-7",
        adapter_revision=revision,
    )

    assert generation.adapter_name == adapter_name
    assert generation.adapter_revision == revision
    assert "load_lora_adapter" not in [call[1].rsplit("/", 1)[-1] for call in control.calls]


def test_non_json_http_failure_preserves_status_without_exposing_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise urllib.error.HTTPError(
            "http://127.0.0.1:8005/generate",
            500,
            "failure",
            {},
            io.BytesIO(b"private server diagnostic"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)

    status, value = UrllibSGLangControlTransport().request(
        method="POST",
        url="http://127.0.0.1:8005/generate",
        payload={"input_ids": [1]},
        timeout_seconds=1.0,
        max_response_bytes=1024,
    )

    assert status == 500
    assert value is None


def test_second_adapter_swap_verifies_before_unloading_previous_generation() -> None:
    gateway, control, _ = _gateway()
    asyncio.run(gateway.swap_supervisor_adapter(adapter_path="/private/a", adapter_revision="a"))
    asyncio.run(gateway.swap_supervisor_adapter(adapter_path="/private/b", adapter_revision="b"))

    assert [call[1].rsplit("/", 1)[-1] for call in control.calls] == [
        "load_lora_adapter",
        "health",
        "models",
        "completions",
        "load_lora_adapter",
        "health",
        "models",
        "completions",
        "unload_lora_adapter",
    ]


def test_failed_second_swap_restores_previous_adapter() -> None:
    gateway, control, _ = _gateway()
    first = asyncio.run(
        gateway.swap_supervisor_adapter(adapter_path="/private/a", adapter_revision="a")
    )
    control.fail_next_load = True

    try:
        asyncio.run(
            gateway.swap_supervisor_adapter(adapter_path="/private/b", adapter_revision="b")
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed adapter load must be reported")

    assert gateway.adapter_generation == first
    assert control.loaded_adapters == {"theta_step_a"}
    assert control.calls[-1][2] == {"lora_name": "theta_step_b"}


def test_prepared_swap_keeps_previous_adapter_until_commit_or_rollback() -> None:
    gateway, control, _ = _gateway()
    first = asyncio.run(
        gateway.swap_supervisor_adapter(adapter_path="/private/a", adapter_revision="a")
    )

    prepared = gateway.prepare_supervisor_adapter(
        adapter_path="/private/b",
        adapter_revision="b",
    )
    assert gateway.adapter_generation == prepared.candidate
    assert control.loaded_adapters == {"theta_step_a", "theta_step_b"}

    gateway.rollback_supervisor_adapter(prepared)
    assert gateway.adapter_generation == first
    assert control.loaded_adapters == {"theta_step_a"}

    prepared = gateway.prepare_supervisor_adapter(
        adapter_path="/private/c",
        adapter_revision="c",
    )
    committed = gateway.commit_supervisor_adapter(prepared)
    assert gateway.adapter_generation == committed
    assert control.loaded_adapters == {"theta_step_c"}
