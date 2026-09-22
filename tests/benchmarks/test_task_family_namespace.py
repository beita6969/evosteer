"""Catalog admissions keep calibration contexts benchmark-namespaced."""

from __future__ import annotations

import pytest
from skillev_private.benchmarks.catalog import PrivateBenchmarkWorkload

from skillev.benchmarks import (
    ALFWorldPublicItem,
    BenchmarkPublicItem,
    ScienceWorldPublicItem,
    SWEBenchVerifiedPublicCase,
    WebShopPublicItem,
)
from skillev.experiments import Benchmark
from skillev.rollout import RolloutTask
from skillev.training import RolloutSessionBundle


def test_public_adapters_reject_cross_benchmark_task_families() -> None:
    with pytest.raises(ValueError):
        BenchmarkPublicItem(
            benchmark_id="static-fixture",
            dataset_revision="fixture@1",
            split="test",
            task_id="static-fixture-1",
            task_family="other/family",
            query="Public fixture.",
            public_context={},
        )
    with pytest.raises(ValueError):
        WebShopPublicItem(
            dataset_revision="fixture@1",
            environment_snapshot_id="environment@1",
            split="test",
            task_id="webshop-fixture-1",
            task_family="other/family",
            query="Public fixture.",
            public_context={},
        )
    with pytest.raises(ValueError):
        ALFWorldPublicItem(
            dataset_revision="fixture@1",
            environment_snapshot_id="environment@1",
            split="test",
            task_id="alfworld-fixture-1",
            task_family="other/family",
            query="Public fixture.",
            public_context={},
            seed=1,
            max_steps=1,
        )
    with pytest.raises(ValueError):
        ScienceWorldPublicItem(
            dataset_revision="fixture@1",
            environment_snapshot_id="environment@1",
            split="test",
            task_id="scienceworld-fixture-1",
            task_family="other/family",
            query="Public fixture.",
            public_context={},
            seed=1,
            max_steps=1,
        )
    with pytest.raises(ValueError):
        SWEBenchVerifiedPublicCase(
            dataset_revision="fixture@1",
            split="test",
            instance_id="fixture__project-1",
            repo="fixture/project",
            version="1.0",
            base_commit="0123456789abcdef",
            problem_statement="Public fixture.",
            environment_image_id="oci://fixture@sha256:public",
            task_family="other/family",
            max_steps=1,
        )


class _UnusedSessionFactory:
    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        del task
        raise AssertionError("catalog admission must not create a session")


def test_private_catalog_rejects_raw_task_with_foreign_context_namespace() -> None:
    task = RolloutTask(
        task_id="webshop-fixture-1",
        environment_id="fixture:webshop",
        task_family="other/family",
        context_id="fixture",
        query="Public fixture.",
        available_tools=(),
        public_context={"benchmark_id": Benchmark.WEBSHOP.value},
    )

    with pytest.raises(ValueError):
        PrivateBenchmarkWorkload(
            benchmark=Benchmark.WEBSHOP,
            tasks=(task,),
            session_factory=_UnusedSessionFactory(),
        )
