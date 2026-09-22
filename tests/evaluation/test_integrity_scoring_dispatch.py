"""Native scoring concurrency preserves finals, completed grades and judge identity."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from skillev_private.evaluation import integrity_runtime
from skillev_private.evaluation.integrity_replicas import ReplicaLoadBalancer

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.integrity_metric_schema import NATIVE_VERIFIER_VERSIONS
from skillev.evaluation.integrity_pipeline import run_paired
from skillev.evaluation.integrity_results import NativeScore
from skillev.evaluation.integrity_resume import EvaluationRunMode
from skillev.evaluation.sealed_candidates import CandidateReader
from skillev.evaluation.step0_integrity import InferenceArm
from tests.evaluation.test_step0_integrity_pipeline import Runtime, panel


def scoring_run(runtime, directory, concurrency=3):
    entries = tuple(
        PublicTaskView.from_record(str(index), "aime-2026", {"problem": "Synthetic task"})
        for index in range(7)
    )
    return asyncio.run(
        run_paired(
            runtime,
            replace(panel(), entries=entries),
            (InferenceArm("A2"),),
            run_id="scoring",
            directory=directory,
            scoring_concurrency=concurrency,
            canary=True,
            run_mode=EvaluationRunMode.SAME_RUN_RESUME
            if (directory / "frozen-plan-private.json").exists()
            else EvaluationRunMode.FORMAL_FRESH,
        )
    )


def test_scoring_is_bounded_and_waits_for_every_answer(tmp_path: Path) -> None:
    class ConcurrentScorer(Runtime):
        active = peak = 0

        async def score(self, reader, scope, benchmark):
            assert self.generated == 7
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0)
                return await super().score(reader, scope, benchmark)
            finally:
                self.active -= 1

    runtime = ConcurrentScorer()
    report = scoring_run(runtime, tmp_path)
    assert runtime.peak == 3
    assert runtime.active == 0
    assert runtime.generated == runtime.graded == 7
    assert report["benchmarks"]["aime-2026"]["count"] == 7
    resumed = Runtime()
    assert scoring_run(resumed, tmp_path) == report
    assert resumed.generated == resumed.graded == 0


def test_failed_grader_drains_paid_for_scores_without_admitting_more(tmp_path: Path) -> None:
    class FailedScorer(Runtime):
        def __init__(self):
            super().__init__()
            self.ready = asyncio.Event()
            self.failed = asyncio.Event()
            self.started = []

        async def score(self, reader, scope, benchmark):
            self.started.append(scope[2])
            if len(self.started) == 3:
                self.ready.set()
            if scope[2] == "0":
                await self.ready.wait()
                self.failed.set()
                raise RuntimeError("synthetic grader unavailable")
            await self.failed.wait()
            await asyncio.sleep(0)
            return await super().score(reader, scope, benchmark)

    runtime = FailedScorer()
    with pytest.raises(RuntimeError):
        scoring_run(runtime, tmp_path)
    assert len(runtime.started) == 3
    reader = CandidateReader(tmp_path / "candidates-private.sqlite")
    assert reader.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 7
    assert reader.connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 2
    reader.close()
    resumed = Runtime()
    scoring_run(resumed, tmp_path)
    assert resumed.generated == 0
    assert resumed.graded == 5


@pytest.mark.parametrize("external", [False, True])
def test_blind_grader_balances_only_equivalent_declared_replicas(monkeypatch, external) -> None:
    runtime = object.__new__(integrity_runtime.PrivateIntegrityRuntime)
    endpoints = [f"http://fixture-{index}" for index in range(3)]
    endpoint = "http://external-judge" if external else endpoints[0]
    settings = {"endpoint_base": endpoint, "model_route": "fixture-judge"}
    runtime.config = {"endpoints": endpoints, "scorers": {"healthbench": settings}}
    runtime.replicas = ReplicaLoadBalancer(3)
    runtime.observed = [
        {
            "model_info": {"model_path": "fixture-model"},
            "server_info": {"served_model_name": "fixture-judge"},
        }
        for _ in endpoints
    ]
    runtime.source = SimpleNamespace(targets={})
    runtime.sandbox = runtime.mbpp_sandbox = None
    records = []
    runtime.journal = SimpleNamespace(record=lambda *args, **kwargs: records.append((args, kwargs)))
    called = []

    async def score(reader, scope, benchmark, target, **kwargs):
        called.append(kwargs["settings"]["healthbench"]["endpoint_base"])
        await asyncio.sleep(0)
        return NativeScore(
            scope[2],
            benchmark,
            "qwen-local-rubric-score",
            1.0,
            secondary_metrics={"triggered-negative-rubric-count": 0.0},
            verifier_version=NATIVE_VERIFIER_VERSIONS[benchmark],
            grader_used=True,
        )

    monkeypatch.setattr(integrity_runtime, "score_native", score)

    async def batch():
        return await asyncio.gather(
            *(runtime.score(None, ("run", "A2", str(i)), "healthbench") for i in range(9))
        )

    assert len(asyncio.run(batch())) == 9
    assert Counter(called) == ({endpoint: 9} if external else dict.fromkeys(endpoints, 3))
    assert runtime.replicas.inflight == [0, 0, 0]
    assert runtime.config["scorers"]["healthbench"] == settings
    assert len(records) == (0 if external else 9)
    if not external:
        for observed in runtime.observed:
            observed["model_info"] = {"model_path": "different-model"}
        runtime.observed[0]["model_info"] = {"model_path": "fixture-model"}
        with pytest.raises(ValueError):
            asyncio.run(batch())
