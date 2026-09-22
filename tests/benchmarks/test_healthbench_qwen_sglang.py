from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from skillev_private.benchmarks.healthbench_qwen_sglang import (
    HEALTHBENCH_QWEN_VERIFIER,
    QwenSGLangHealthBenchGrader,
)
from skillev_private.benchmarks.protocol_v10_workers import (
    InMemoryWorkerTransport,
    PrivateJSONWorker,
    ProtocolV10WorkerError,
)

from skillev.training import AsyncResourceLimiter


def _grader(
    tmp_path: Path,
    response: bytes,
) -> tuple[QwenSGLangHealthBenchGrader, InMemoryWorkerTransport]:
    transport = InMemoryWorkerTransport(lambda _: response)
    grader = QwenSGLangHealthBenchGrader(
        worker=PrivateJSONWorker(
            command=("private-health-grader",),
            working_directory=tmp_path,
            timeout_seconds=2,
            process_limiter=AsyncResourceLimiter(2),
            transport=transport,
        ),
        private_cases={
            "health-1": {
                "grader_kind": "official-healthbench",
                "prompt": [{"role": "user", "content": "private prompt"}],
                "rubrics": [{"criterion": "private rubric", "points": -1}],
            }
        },
        model_request_limiter=AsyncResourceLimiter(12),
    )
    return grader, transport


def test_qwen_grader_returns_only_protocol_metrics_and_uses_shared_limits(
    tmp_path: Path,
) -> None:
    grader, transport = _grader(
        tmp_path,
        b'{"official_rubric_score":0.75,"triggered_negative_rubric_count":1}',
    )
    grade = asyncio.run(grader.grade("health-1", "candidate"))
    assert grade.official_rubric_score == 0.75
    assert grade.triggered_negative_rubric_count == 1
    assert grader.verifier_version == HEALTHBENCH_QWEN_VERIFIER
    assert grader.model_request_limiter.timing.calls == 1
    assert grader.worker.process_limiter is not None
    assert grader.worker.process_limiter.timing.calls == 1
    private_request = json.loads(transport.requests[0])
    assert private_request["private_case"]["rubrics"][0]["criterion"] == "private rubric"


def test_qwen_grader_fails_closed_on_malformed_worker_output(tmp_path: Path) -> None:
    grader, _ = _grader(tmp_path, b"not-json")
    with pytest.raises(ProtocolV10WorkerError):
        asyncio.run(grader.grade("health-1", "candidate"))


def test_broker_leases_only_actual_requests_and_does_not_nest_the_shared_slot(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from skillev_private.benchmarks.healthbench_qwen_sglang import (
        HealthBenchQwenGraderConfig,
        HealthBenchQwenVerifierIdentity,
    )

    grader, _ = _grader(tmp_path, b"{}")
    limiter = AsyncResourceLimiter(1)
    config = HealthBenchQwenGraderConfig(
        endpoint_base="http://localhost:1234",
        base_model="base",
        identity=HealthBenchQwenVerifierIdentity("base", "tokenizer", "sglang"),
    )
    grader = replace(grader, model_request_limiter=limiter, broker_config=config)

    async def worker_request(worker, payload):
        assert "--broker-socket" in worker.command
        assert limiter.timing.calls == 0  # setup/process work holds no model slot
        # Represents an actor progressing while the case worker has started.
        async with limiter.lease():
            pass
        return {"official_rubric_score": 0.75, "triggered_negative_rubric_count": 1}

    monkeypatch.setattr(PrivateJSONWorker, "request", worker_request)

    async def run():
        return await asyncio.wait_for(grader.grade("health-1", "candidate"), timeout=2)

    assert asyncio.run(run()).official_rubric_score == 0.75
    assert limiter.timing.calls == 1
