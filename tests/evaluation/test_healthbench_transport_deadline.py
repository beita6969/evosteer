"""An exhausted HTTP budget must not start another full-length request."""

from types import SimpleNamespace

import pytest

from skillev.evaluation.healthbench_official import (
    ExternalJudgeTransport,
    HealthBenchTransportExhausted,
    QwenChatTransport,
    TransportRetryPolicy,
)


@pytest.mark.parametrize("transport_type", [QwenChatTransport, ExternalJudgeTransport])
@pytest.mark.parametrize("oversleep", [False, True])
def test_deadline_clamps_requests_and_is_checked_after_backoff(transport_type, oversleep):
    now = 0.0
    requests = []

    def create(**kwargs):
        nonlocal now
        requests.append(kwargs)
        now += kwargs["timeout"]
        raise TimeoutError("synthetic transport timeout")

    def sleep(delay):
        nonlocal now
        now += delay + int(oversleep)

    transport = transport_type(
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        model="fixture-base",
        retry_policy=TransportRetryPolicy(
            maximum_attempts=4,
            request_timeout_seconds=2,
            total_deadline_seconds=3,
            initial_backoff_seconds=0.5,
        ),
        transient_error_types=(TimeoutError,),
        sleeper=sleep,
        clock=lambda: now,
    )
    messages = [{"role": "user", "content": "Synthetic rubric"}]
    kwargs = {"messages": messages}
    if transport_type is QwenChatTransport:
        kwargs["response_format"] = {"type": "json_object"}
    with pytest.raises(HealthBenchTransportExhausted) as caught:
        transport.create(**kwargs)

    assert isinstance(caught.value.__cause__, TimeoutError)
    assert [request["timeout"] for request in requests] == ([2] if oversleep else [2, 0.5])
    assert all(request["messages"] == messages for request in requests)
    assert all(request["model"] == "fixture-base" for request in requests)
    assert all(request["response_format"] == {"type": "json_object"} for request in requests)
    if not oversleep:
        assert now == 3
