import asyncio
import json
import time

import httpx
from skillev_private.benchmarks.trusted_model_request_broker import TrustedModelRequestBroker

from skillev.training import AsyncResourceLimiter


def test_broker_leases_each_unchanged_rubric_request_and_drains(monkeypatch):
    seen = []
    limiter = AsyncResourceLimiter(1)

    def forward(self, body):
        seen.append(json.loads(body))
        time.sleep(0.005)
        return 200, b'{"result":"ok"}'

    monkeypatch.setattr(TrustedModelRequestBroker, "_forward", forward)
    requests = [
        {
            "model": "base",
            "messages": [{"role": "user", "content": str(i)}],
            "seed": 0,
            "temperature": 0,
            "max_tokens": 1024,
        }
        for i in range(6)
    ]

    async def run():
        async with TrustedModelRequestBroker(
            endpoint="http://127.0.0.1/v1", model="base", limiter=limiter, timeout=5
        ) as broker:

            def request(payload):
                with httpx.Client(transport=httpx.HTTPTransport(uds=broker.socket_path)) as client:
                    return client.post("http://localhost/v1/chat/completions", json=payload)

            replies = await asyncio.gather(*(asyncio.to_thread(request, p) for p in requests))
            assert all(reply.status_code == 200 for reply in replies)

    asyncio.run(run())
    assert sorted(seen, key=lambda p: p["messages"][0]["content"]) == requests
    assert limiter.timing.calls == 6
    assert limiter.timing.high_water_mark == 1


def test_cancelled_request_keeps_shared_capacity_until_http_thread_drains(monkeypatch):
    import threading

    import pytest

    started, release = threading.Event(), threading.Event()
    limiter = AsyncResourceLimiter(1)

    def forward(self, body):
        started.set()
        assert release.wait(5)
        return 200, b"{}"

    monkeypatch.setattr(TrustedModelRequestBroker, "_forward", forward)

    async def run():
        broker = TrustedModelRequestBroker(
            endpoint="http://localhost/v1", model="base", limiter=limiter, timeout=5
        )

        async def leased():
            async with limiter.lease():
                return await broker._forward_drained(b"{}")

        pending = asyncio.create_task(leased())
        assert await asyncio.to_thread(started.wait, 3)
        pending.cancel()
        await asyncio.sleep(0)
        assert limiter._active == 1
        assert not pending.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert limiter._active == 0

    asyncio.run(run())


def test_rubric_responses_restore_in_different_arrival_order_without_new_judging(
    tmp_path, monkeypatch
):
    from skillev.runtime.request_journal import DurableRequestJournal

    calls = []

    def forward(self, body):
        calls.append(body)
        return 200, b'{"choices":[{"message":{"content":"original"}}]}'

    monkeypatch.setattr(TrustedModelRequestBroker, "_forward", forward)
    journal = tmp_path / "requests.sqlite3"

    def broker():
        return TrustedModelRequestBroker(
            endpoint="http://judge/v1",
            model="base",
            limiter=AsyncResourceLimiter(2),
            timeout=5,
            task_id="batch1-trajectory2",
            request_journal=DurableRequestJournal(journal),
        )

    requests = [json.dumps({"model": "base", "messages": [str(i)]}).encode() for i in range(2)]
    first = broker()
    expected = [first._persisted_forward(body) for body in requests]
    restored = broker()
    assert [restored._persisted_forward(body) for body in reversed(requests)] == list(
        reversed(expected)
    )
    assert len(calls) == 2


def test_unknown_rubric_blocks_different_repair_prompt_and_new_process_case(tmp_path, monkeypatch):
    import pytest

    from skillev.runtime.request_journal import DurableRequestJournal, UnknownRequestOutcomeError

    calls = []

    def fail(self, body):
        calls.append(body)
        raise TimeoutError("response lost")

    monkeypatch.setattr(TrustedModelRequestBroker, "_forward", fail)
    path = tmp_path / "requests.sqlite3"

    def broker():
        return TrustedModelRequestBroker(
            endpoint="http://judge/v1",
            model="base",
            limiter=AsyncResourceLimiter(2),
            timeout=5,
            task_id="trajectory",
            request_journal=DurableRequestJournal(path),
        )

    first = broker()
    with pytest.raises(TimeoutError):
        first._persisted_forward(b'{"messages":["rubric"]}')
    with pytest.raises(UnknownRequestOutcomeError):
        first._persisted_forward(b'{"messages":["different repair prompt"]}')

    async def resume():
        with pytest.raises(UnknownRequestOutcomeError):
            async with broker():
                pytest.fail("cannot start another scorer for an unresolved case")

    asyncio.run(resume())
    assert len(calls) == 1
