"""Answer-free CPU contracts for the fixed private benchmark entrypoint."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest
from skillev_private.benchmarks.catalog import (
    PrivateBenchmarkCatalog,
    PrivateBenchmarkWorkload,
)
from skillev_private.benchmarks.curriculum import (
    PrivateFrozenTaskSequence,
    build_result_blind_sequence,
)
from skillev_private.experiments.attempt_supervisor import (
    PrivateBenchmarkAttemptSupervisor,
)
from skillev_private.experiments.attempt_worker import (
    run_private_benchmark_attempt_worker,
)
from skillev_private.experiments.benchmark_attempt_builders import (
    build_private_benchmark_attempt,
)
from skillev_private.experiments.benchmark_runtime import (
    PrivateBenchmarkRuntime,
    PrivateExternalCompletionManifest,
    PrivateExternalProcessRuntime,
    PrivateImplementationBuildDeployment,
    PrivateSWEHarnessRuntime,
    PrivateSWEImageBinding,
    PrivateSWEImagePin,
)

from skillev.experiments import BENCHMARK_SPECS, Benchmark, SchedulePurpose, TrainingUse
from skillev.rollout import RolloutTask


@dataclass(frozen=True, slots=True)
class _UnusedSessionFactory:
    """The curriculum must not instantiate environments while it selects IDs."""

    def create(self, task: RolloutTask) -> object:
        raise AssertionError(f"curriculum unexpectedly opened {task.task_id}")


def _training_benchmarks() -> tuple[Benchmark, ...]:
    return tuple(
        spec.benchmark for spec in BENCHMARK_SPECS if spec.training_use is TrainingUse.TRAINING_MIX
    )


def _task(benchmark: Benchmark, ordinal: int) -> RolloutTask:
    return RolloutTask(
        task_id=f"synthetic-{benchmark.value}-{ordinal}",
        environment_id=f"synthetic-{benchmark.value}",
        task_family=f"{benchmark.value}/synthetic",
        context_id=f"synthetic:{benchmark.value}",
        query="Synthetic public unit-test query.",
        available_tools=(),
        public_context={"benchmark_id": benchmark.value},
    )


def _catalog(*, omit: Benchmark | None = None) -> PrivateBenchmarkCatalog:
    workloads = tuple(
        PrivateBenchmarkWorkload(
            benchmark=benchmark,
            tasks=tuple(_task(benchmark, index) for index in range(512)),
            session_factory=_UnusedSessionFactory(),
        )
        for benchmark in _training_benchmarks()
        if benchmark is not omit
    )
    return PrivateBenchmarkCatalog(workloads=workloads)


def test_private_training_curriculum_is_result_blind_and_covers_every_iid_domain() -> None:
    catalog = _catalog()

    sequence = build_result_blind_sequence(
        catalog,
        purpose=SchedulePurpose.IID_TRAINING,
        task_count=512 * len(_training_benchmarks()),
    )

    assert sequence.identity.purpose is SchedulePurpose.IID_TRAINING
    assert tuple(item.benchmark for item in sequence.identity.benchmark_counts) == tuple(
        sorted(_training_benchmarks(), key=lambda item: item.value)
    )
    assert all(item.count == 512 for item in sequence.identity.benchmark_counts)
    assert Benchmark.AIME_2026 in {
        Benchmark(task.public_context["benchmark_id"]) for task in sequence.resolve(catalog)
    }
    assert tuple(
        Benchmark(task.public_context["benchmark_id"]) for task in sequence.resolve(catalog)
    ) == tuple(benchmark for benchmark in _training_benchmarks() for _ in range(512))
    assert (
        build_result_blind_sequence(
            catalog,
            purpose=SchedulePurpose.IID_TRAINING,
            task_count=512 * len(_training_benchmarks()),
        )
        == sequence
    )


def test_private_training_curriculum_rejects_a_missing_declared_domain() -> None:
    with pytest.raises(ValueError):
        build_result_blind_sequence(
            _catalog(omit=_training_benchmarks()[0]),
            purpose=SchedulePurpose.IID_TRAINING,
            task_count=512 * len(_training_benchmarks()),
        )


def test_private_frozen_sequence_rejects_task_identity_tampering() -> None:
    sequence = build_result_blind_sequence(
        _catalog(),
        purpose=SchedulePurpose.IID_TRAINING,
        task_count=512 * len(_training_benchmarks()),
    )
    value = sequence.to_value()
    task_ids = value["task_ids"]
    assert isinstance(task_ids, list)
    task_ids[0] = "synthetic-tampered-task"

    with pytest.raises(ValueError):
        PrivateFrozenTaskSequence.from_value(value)


def test_private_runtime_round_trips_without_touching_deployment_paths() -> None:
    source = "docker://registry.invalid/skillev/swe:unit-test"
    digest = "sha256:" + "a" * 64
    image_id = f"{source}@{digest}"
    runtime = PrivateBenchmarkRuntime(
        acquisition_lock_path=Path("/private/locks/acquisition.json"),
        acquisition_lock_content_hash="sha256:" + "b" * 64,
        official_process_deployment_path=Path("/private/process/deployment.json"),
        official_process_deployment_content_hash="sha256:" + "c" * 64,
        iid_semantic_selection_path=Path("/private/training/iid-selection.json"),
        iid_semantic_selection_content_hash="sha256:" + "e" * 64,
        implementation_build_deployment=PrivateImplementationBuildDeployment(
            source_tree_root=Path("/private/source"),
            public_wheel_path=Path("/private/wheels/skillev.whl"),
            private_evaluation_wheel_path=Path("/private/wheels/skillev-private-evaluation.whl"),
            lockfile_path=Path("/private/source/uv.lock"),
        ),
        swe=PrivateSWEHarnessRuntime(
            apptainer_executable=Path("/private/bin/apptainer"),
            apptainer_version="1.3.0",
            materializer_storage_root=Path("/private/cache/swe"),
            materializer_capacity_bytes=1024 * 1024,
            materializer_timeout_seconds=60.0,
            worker_host_path=Path("/private/bin/swe-worker"),
            worker_container_path="/opt/skillev/swe-worker",
            worker_size_bytes=1,
            worker_sha256="sha256:" + "d" * 64,
            harness_revision="0123456789abcdef0123456789abcdef01234567",
            harness_source_root=Path("/private/src/swe-harness"),
            work_root=Path("/private/work/swe"),
            images=(
                PrivateSWEImagePin(
                    environment_image_id=image_id,
                    source_reference=source,
                    manifest_digest=digest,
                ),
            ),
            bindings=(
                PrivateSWEImageBinding(
                    instance_id="synthetic-instance",
                    environment_image_id=image_id,
                ),
            ),
            task_family="synthetic/swe",
            max_steps=1,
        ),
        external_completion_manifests=tuple(
            PrivateExternalCompletionManifest(
                benchmark=benchmark,
                path=Path(f"/private/{benchmark.value}/cases.jsonl"),
                sha256="sha256:" + marker * 64,
            )
            for benchmark, marker in zip(
                (
                    Benchmark.BIRD_SQL,
                    Benchmark.MBPP_PLUS,
                    Benchmark.TABLEBENCH,
                    Benchmark.HUMANEVAL_PLUS,
                ),
                "abcd",
                strict=True,
            )
        ),
        external_process_runtimes=(
            PrivateExternalProcessRuntime(
                benchmark=Benchmark.APPWORLD,
                interpreter_path=Path("/private/appworld/bin/python"),
                source_root=Path("/private/appworld/source"),
                state_root=Path("/private/appworld/state"),
                request_timeout_seconds=120.0,
            ),
        ),
    )

    assert PrivateBenchmarkRuntime.from_value(runtime.to_value()) == runtime


def test_formal_private_worker_and_supervisor_have_no_public_worker_seam() -> None:
    assert tuple(inspect.signature(build_private_benchmark_attempt).parameters) == ("request",)
    worker_source = inspect.getsource(run_private_benchmark_attempt_worker)
    supervisor_source = inspect.getsource(PrivateBenchmarkAttemptSupervisor)

    assert "build_private_benchmark_attempt" in worker_source
    assert "skillev_private.experiments.attempt_worker" in supervisor_source
    assert "worker_module" not in supervisor_source
