"""Episode deadlines must release their own HTTP/GPU request, not wait for it."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from aiohttp import web

from skillev.rollout import EvaluationSGLangRolloutGenerator, ExternalSGLangRolloutConfig
from skillev.runtime.sglang_gateway import SGLangGatewayError
from tests.rollout.test_evaluation_sglang import _snapshot, _Tokenizer


@asynccontextmanager
async def serving_fixture(*, abort_status=200):
    arrived = asyncio.Event()
    release = asyncio.Event()
    disconnected = asyncio.Event()
    requests, aborts = [], []

    async def generate(request):
        requests.append(await request.json())
        arrived.set()
        while not release.is_set():
            if request.transport is None or request.transport.is_closing():
                disconnected.set()
                break
            await asyncio.sleep(0.005)
        return web.json_response({"output_ids": []})

    async def abort(request):
        aborts.append(await request.json())
        if abort_status == 200:
            release.set()
        return web.Response(status=abort_status)

    app = web.Application()
    app.router.add_post("/generate", generate)
    app.router.add_post("/abort_request", abort)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    endpoint = f"http://127.0.0.1:{runner.addresses[0][1]}"
    try:
        yield endpoint, arrived, release, disconnected, requests, aborts
    finally:
        release.set()
        await runner.cleanup()


@pytest.mark.parametrize("abort_status", [200, 503])
def test_episode_cancel_closes_http_and_aborts_only_its_request(abort_status):
    async def run():
        async with serving_fixture(abort_status=abort_status) as fixture:
            endpoint, arrived, release, disconnected, requests, aborts = fixture
            generator = EvaluationSGLangRolloutGenerator(
                ExternalSGLangRolloutConfig(endpoint, request_timeout_seconds=10),
                _Tokenizer(),
                _snapshot,
            )
            task = asyncio.create_task(
                generator._request({"input_ids": [1, 2], "sampling_params": {}, "stream": False})
            )
            await asyncio.wait_for(arrived.wait(), 2)
            loop = asyncio.get_running_loop()
            # Even the pre-fix blocking transport must be allowed to drain so
            # this regression test fails quickly rather than hanging pytest.
            fallback = loop.call_later(0.8, release.set)
            started = loop.time()
            try:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert loop.time() - started < 0.5
                assert len(requests) == len(aborts) == 1
                assert requests[0]["rid"]
                assert aborts[0]["rid"] == requests[0]["rid"]
                assert not aborts[0].get("abort_all", False)
                if abort_status != 200:
                    await asyncio.wait_for(disconnected.wait(), 0.5)
            finally:
                fallback.cancel()
                release.set()
                generator.close()

    asyncio.run(run())


def test_successful_transport_keeps_payload_and_never_aborts():
    async def run():
        async with serving_fixture() as fixture:
            endpoint, _, release, _, requests, aborts = fixture
            release.set()
            generator = EvaluationSGLangRolloutGenerator(
                ExternalSGLangRolloutConfig(endpoint), _Tokenizer(), _snapshot
            )
            payload = {"input_ids": [1, 2], "sampling_params": {"max_new_tokens": 8}}
            try:
                status, raw = await generator._request(payload)
                assert status == 200
                assert raw["output_ids"] == []
                assert all(requests[0][key] == value for key, value in payload.items())
                assert "rid" not in payload
                assert not aborts
            finally:
                generator.close()

    asyncio.run(run())


def test_transport_timeout_is_bounded_and_requests_its_own_abort():
    async def run():
        async with serving_fixture() as fixture:
            endpoint, _, _, _, requests, aborts = fixture
            generator = EvaluationSGLangRolloutGenerator(
                ExternalSGLangRolloutConfig(endpoint, request_timeout_seconds=0.05),
                _Tokenizer(),
                _snapshot,
                transport_maximum_attempts=1,
            )
            try:
                with pytest.raises(SGLangGatewayError):
                    await generator._request_with_retry({"input_ids": [1]})
                assert len(requests) == len(aborts) == 1
                assert requests[0]["rid"] == aborts[0]["rid"]
            finally:
                generator.close()

    asyncio.run(run())


def test_cancelling_queued_call_does_not_abort_an_active_sibling():
    async def run():
        async with serving_fixture() as fixture:
            endpoint, arrived, release, _, requests, aborts = fixture
            generator = EvaluationSGLangRolloutGenerator(
                ExternalSGLangRolloutConfig(endpoint, transport_worker_threads=1),
                _Tokenizer(),
                _snapshot,
            )
            active = asyncio.create_task(generator._request({"input_ids": [1]}))
            await asyncio.wait_for(arrived.wait(), 2)
            queued = asyncio.create_task(generator._request({"input_ids": [2]}))
            try:
                await asyncio.sleep(0)
                queued.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await queued
                assert len(requests) == 1
                assert not aborts
                assert not active.done()
                release.set()
                assert (await active)[0] == 200
            finally:
                release.set()
                await active
                generator.close()

    asyncio.run(run())
