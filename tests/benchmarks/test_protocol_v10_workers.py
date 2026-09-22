from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from skillev_private.benchmarks.protocol_v10_workers import (
    InMemoryWorkerTransport,
    PrivateJSONWorker,
    ProtocolV10WorkerError,
)

from skillev.contracts import JsonValue
from skillev.training import AsyncResourceLimiter


def test_private_json_workers_share_the_process_grader_limit(tmp_path: Path) -> None:
    limiter = AsyncResourceLimiter(2)

    async def respond(encoded: bytes) -> bytes:
        value = json.loads(encoded)
        await asyncio.sleep(0.02)
        return json.dumps({"request": value["request"]}).encode()

    transport = InMemoryWorkerTransport(respond)
    worker = PrivateJSONWorker(
        command=("in-memory-worker",),
        working_directory=tmp_path,
        timeout_seconds=2.0,
        process_limiter=limiter,
        transport=transport,
    )

    async def scenario() -> tuple[dict[str, JsonValue], ...]:
        return tuple(
            await asyncio.gather(*(worker.request({"request": index}) for index in range(6)))
        )

    results = asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))
    assert tuple(item["request"] for item in results) == tuple(range(6))
    assert len(transport.requests) == 6
    assert limiter.timing.calls == 6
    assert limiter.timing.high_water_mark == 2
    assert limiter.timing.queue_seconds > 0


def test_private_json_async_subprocess_wire_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def worker(code: str, *, timeout: float = 30.0, maximum: int = 1024) -> PrivateJSONWorker:
        return PrivateJSONWorker(
            command=(sys.executable, "-c", code),
            working_directory=tmp_path,
            timeout_seconds=timeout,
            max_response_bytes=maximum,
        )

    async def scenario() -> None:
        monkeypatch.setenv("PYTHONPATH", "parent-runtime-only")
        environment = await worker(
            "import json,os; print(json.dumps({'has_pythonpath': 'PYTHONPATH' in os.environ}))"
        ).request({"request": 0})
        assert environment == {"has_pythonpath": False}

        echoed = await worker(
            "import json,sys; value=json.loads(sys.stdin.readline()); "
            "print(json.dumps({'request': value['request']}))"
        ).request({"request": 7})
        assert echoed == {"request": 7}

        with pytest.raises(ProtocolV10WorkerError):
            await worker("raise SystemExit(2)").request({"request": 1})
        with pytest.raises(ProtocolV10WorkerError):
            await worker("print('not-json')").request({"request": 1})
        with pytest.raises(ProtocolV10WorkerError):
            await worker("print('x' * 128)", maximum=32).request({"request": 1})
        with pytest.raises(ProtocolV10WorkerError):
            await worker("import time; time.sleep(30)", timeout=0.05).request({"request": 1})

    asyncio.run(asyncio.wait_for(scenario(), timeout=120.0))
