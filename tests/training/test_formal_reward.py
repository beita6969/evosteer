from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import requests

from training import swe_bench_eval
from training.environment import GenericTaskEnvironment
from training.reward import _swe_bench_official_reward
from training.swe_bench_eval import (
    _cpu_subprocess_environment,
    _exec_run_with_tolerant_decode,
    _make_test_spec_with_retry,
)


def test_verified_dataset_preserves_instance_identity_for_official_test_spec(
    monkeypatch,
) -> None:
    row = {
        "instance_id": "public-instance",
        "repo": "owner/repo",
        "version": "1",
        "base_commit": "base",
        "environment_setup_commit": "environment",
        "patch": "reference patch",
        "test_patch": "test patch",
        "FAIL_TO_PASS": [],
        "PASS_TO_PASS": [],
    }
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_from_disk=lambda _: [row]),
    )
    previous = dict(swe_bench_eval._verified_cache)
    swe_bench_eval._verified_cache.clear()
    try:
        swe_bench_eval._load_verified_dataset()

        assert swe_bench_eval._verified_cache["public-instance"]["instance_id"] == "public-instance"
        assert (
            swe_bench_eval._verified_cache["public-instance"]["environment_setup_commit"]
            == "environment"
        )
    finally:
        swe_bench_eval._verified_cache.clear()
        swe_bench_eval._verified_cache.update(previous)


def test_parallel_official_evaluations_use_distinct_container_names(tmp_path, monkeypatch) -> None:
    run_ids = []

    class Client:
        def close(self) -> None:
            return None

    class RunEvaluation:
        RUN_EVALUATION_LOG_DIR = None
        exec_run_with_timeout = staticmethod(lambda *_args, **_kwargs: None)
        cleanup_container = staticmethod(lambda *_args, **_kwargs: None)

        @staticmethod
        def run_instance(test_spec, prediction, *_args, **kwargs):
            run_ids.append(_args[-1])
            instance_id = prediction["instance_id"]
            return None, {instance_id: {"resolved": True}}

    docker_module = ModuleType("docker")
    docker_module.from_env = lambda **_kwargs: Client()  # type: ignore[attr-defined]
    swebench_module = ModuleType("swebench")
    harness_module = ModuleType("swebench.harness")
    harness_module.run_evaluation = RunEvaluation  # type: ignore[attr-defined]
    test_spec_package = ModuleType("swebench.harness.test_spec")
    test_spec_module = ModuleType("swebench.harness.test_spec.test_spec")
    test_spec_module.make_test_spec = lambda instance, **_kwargs: instance  # type: ignore[attr-defined]
    for name, module in (
        ("docker", docker_module),
        ("swebench", swebench_module),
        ("swebench.harness", harness_module),
        ("swebench.harness.test_spec", test_spec_package),
        ("swebench.harness.test_spec.test_spec", test_spec_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("SWE_BENCH_EVALUATION_ROOT", str(tmp_path))
    monkeypatch.setenv("SWEBENCH_HARNESS_PATH", str(tmp_path))
    monkeypatch.setenv("SWE_BENCH_DOCKER_NAMESPACE", "swebench")
    verified = {"instance_id": "public-instance"}

    for _ in range(2):
        resolved, score, _ = swe_bench_eval._evaluate_patch_with_official_harness(
            instance_id="public-instance",
            model_patch="patch",
            verified=verified,
            timeout=60,
        )
        assert resolved is True
        assert score == 1.0

    assert len(run_ids) == 2
    assert run_ids[0] != run_ids[1]


def test_official_harness_test_output_tolerates_non_utf8_bytes() -> None:
    class API:
        @staticmethod
        def exec_create(_container_id, _cmd):
            return {"Id": "exec-id"}

        @staticmethod
        def exec_start(_exec_id, *, stream):
            assert stream is True
            return [b"test output: \x86"]

    container = SimpleNamespace(
        id="container-id",
        client=SimpleNamespace(api=API()),
    )

    output, timed_out, elapsed = _exec_run_with_tolerant_decode(container, "test", timeout=1)

    assert "test output" in output
    assert timed_out is False
    assert elapsed >= 0


def test_official_test_spec_retries_transient_metadata_fetch(monkeypatch) -> None:
    attempts = []
    delays = []

    def make_test_spec(verified, *, namespace):
        attempts.append((verified, namespace))
        if len(attempts) < 3:
            raise requests.exceptions.SSLError("transient metadata failure")
        return "test-spec"

    monkeypatch.setattr(swe_bench_eval.time, "sleep", delays.append)

    assert (
        _make_test_spec_with_retry(make_test_spec, {"instance_id": "public"}, "swebench")
        == "test-spec"
    )
    assert len(attempts) == 3
    assert delays == [1.0, 2.0]


def test_formal_swe_reward_uses_local_official_evaluator(monkeypatch) -> None:
    observed = {}

    def evaluate(instance_id, patch, *, extra):
        observed.update(instance_id=instance_id, patch=patch, extra=extra)
        return True, 1.0, "resolved=Y"

    monkeypatch.setenv("SKILLEV_FORMAL_RUNTIME", "1")
    monkeypatch.setattr("training.swe_bench_eval.evaluate_patch", evaluate)
    extra = {"instance_id": "private-instance"}

    assert _swe_bench_official_reward("private-patch", "unused", extra) == 1.0
    assert observed == {
        "extra": extra,
        "instance_id": "private-instance",
        "patch": "private-patch",
    }


def test_formal_swe_reward_fails_closed_on_local_infrastructure_error(monkeypatch) -> None:
    monkeypatch.setenv("SKILLEV_FORMAL_RUNTIME", "1")

    diagnostic = swe_bench_eval.SWEHarnessDiagnostic(
        classification="docker_permission_denied",
        phase="docker",
        retryable=False,
        run_id="private-run",
        instance_hash="safe-hash",
        log_relative_path=None,
        log_blake2b=None,
        test_output_present=False,
        report_present=False,
        container_exit_code=None,
        container_oom_killed=None,
        image_key_hash=None,
    )

    def fail(*_args, **_kwargs):
        raise swe_bench_eval.SWEHarnessInfrastructureError(diagnostic)

    monkeypatch.setattr(
        "training.swe_bench_eval.evaluate_patch",
        fail,
    )

    with pytest.raises(swe_bench_eval.SWEHarnessInfrastructureError):
        _swe_bench_official_reward("patch", "unused", {"instance_id": "private-instance"})


@pytest.mark.parametrize(
    ("log_text", "test_text", "classification", "phase", "retryable"),
    [
        ("APPLY_PATCH_FAIL", "", "patch_apply_failed", "patch_apply", False),
        ("", "Timeout error: test", "test_timeout", "test", False),
        ("BuildImageError", "", "image_build_failed", "image", False),
        (
            "BuildImageError: 404 Not Found (No such image)",
            "",
            "image_missing_or_pull_failed",
            "image",
            False,
        ),
        ("permission denied", "", "docker_permission_denied", "docker", False),
        ("DockerException: connection reset", "", "docker_transport_transient", "docker", True),
        ("manifest unknown", "", "image_missing_or_pull_failed", "image", False),
        ("ContainerError while create", "", "container_create_failed", "container", False),
        ("ContainerError while start", "", "container_start_failed", "container", False),
        ("", "tests ran", "grading_failed", "grading", False),
        ("", "", "report_missing_unknown", "report", False),
    ],
)
def test_missing_report_classification(
    log_text, test_text, classification, phase, retryable
) -> None:
    assert swe_bench_eval._classify_missing_report(log_text.encode(), test_text.encode()) == (
        classification,
        phase,
        retryable,
    )


def test_missing_report_candidate_failure_is_zero_score(tmp_path, monkeypatch) -> None:
    class Client:
        def close(self) -> None:
            return None

    class Spec:
        instance_image_key = "private-image"

    class RunEvaluation:
        RUN_EVALUATION_LOG_DIR = None
        exec_run_with_timeout = staticmethod(lambda *_args, **_kwargs: None)
        cleanup_container = staticmethod(lambda *_args, **_kwargs: None)

        @classmethod
        def run_instance(cls, _spec, prediction, *_args, **_kwargs):
            run_id = _args[-1]
            log_dir = (
                cls.RUN_EVALUATION_LOG_DIR
                / run_id
                / "skillflow-baseline"
                / prediction["instance_id"]
            )
            log_dir.mkdir(parents=True)
            (log_dir / "run_instance.log").write_text("APPLY_PATCH_FAIL")
            return None

    docker_module = ModuleType("docker")
    docker_module.from_env = lambda **_kwargs: Client()  # type: ignore[attr-defined]
    harness_module = ModuleType("swebench.harness")
    harness_module.run_evaluation = RunEvaluation  # type: ignore[attr-defined]
    test_spec_module = ModuleType("swebench.harness.test_spec.test_spec")
    test_spec_module.make_test_spec = lambda *_args, **_kwargs: Spec()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "docker", docker_module)
    monkeypatch.setitem(sys.modules, "swebench", ModuleType("swebench"))
    monkeypatch.setitem(sys.modules, "swebench.harness", harness_module)
    monkeypatch.setitem(sys.modules, "swebench.harness.test_spec", ModuleType("test_spec"))
    monkeypatch.setitem(sys.modules, "swebench.harness.test_spec.test_spec", test_spec_module)
    monkeypatch.setenv("SWE_BENCH_EVALUATION_ROOT", str(tmp_path))
    monkeypatch.setenv("SWEBENCH_HARNESS_PATH", str(tmp_path))
    monkeypatch.setenv("SWE_BENCH_DOCKER_NAMESPACE", "swebench")

    assert swe_bench_eval._evaluate_patch_with_official_harness(
        instance_id="private-instance",
        model_patch="private-patch",
        verified={"instance_id": "private-instance"},
        timeout=60,
    ) == (False, 0.0, "patch_apply_failed")


def test_infrastructure_error_string_does_not_include_private_payload() -> None:
    diagnostic = swe_bench_eval.SWEHarnessDiagnostic(
        classification="grading_failed",
        phase="grading",
        retryable=False,
        run_id="run",
        instance_hash="safe-hash",
        log_relative_path="private/path",
        log_blake2b="log-hash",
        test_output_present=True,
        report_present=False,
        container_exit_code=1,
        container_oom_killed=False,
        image_key_hash="image-hash",
    )
    message = str(swe_bench_eval.SWEHarnessInfrastructureError(diagnostic))
    assert "private/path" not in message
    assert "traceback" not in message.lower()
    assert "safe-hash" in message


def test_swe_repository_subprocesses_cannot_see_training_gpus(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    environment = _cpu_subprocess_environment()

    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_formal_swe_run_tests_uses_official_docker_evaluator(monkeypatch) -> None:
    environment = object.__new__(GenericTaskEnvironment)
    environment.formal_runtime = True
    environment._extra = {"instance_id": "public-instance"}
    environment._repo_path = "formal-worktree"
    environment._generate_workspace_diff = lambda: "diff --git a/a.py b/a.py\n"  # type: ignore[method-assign]
    monkeypatch.setattr(swe_bench_eval, "_load_verified_dataset", lambda: None)
    monkeypatch.setitem(
        swe_bench_eval._verified_cache,
        "public-instance",
        {"repo": "owner/repo", "version": "1"},
    )
    calls = []

    def evaluate(instance_id, model_patch, *, extra, timeout):
        calls.append((instance_id, model_patch, extra, timeout))
        return True, 1.0, "resolved"

    monkeypatch.setattr(swe_bench_eval, "evaluate_patch", evaluate)

    result = environment._run_tests_in_swe_env("pytest", "public-instance")

    assert result == "[run_tests] [PASSED] Official SWE-bench status: resolved"
    assert calls == [
        (
            "public-instance",
            "diff --git a/a.py b/a.py\n",
            {"instance_id": "public-instance"},
            900,
        )
    ]
