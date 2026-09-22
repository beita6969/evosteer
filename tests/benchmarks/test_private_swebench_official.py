from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from skillev_private.benchmarks.swebench import (
    PrivateSWEBenchSessionFactory,
    PrivateSWEBenchVerifiedCase,
    PrivateSWEVerifiedTruth,
    SWEVerifierRequest,
    SWEVerifierResult,
)
from skillev_private.benchmarks.swebench_grader import (
    PinnedSWEOfficialGrader,
    _OfficialHarnessBindings,
)
from skillev_private.benchmarks.swebench_official import (
    SWE_HARNESS_PROTOCOL_VERSION,
    ApptainerOfficialSWEVerifierFactory,
    ApptainerSWEHarnessProcess,
    ApptainerSWEWorkspaceBackendFactory,
    MaterializedSWEImage,
    OfficialSWEBenchVerifiedCaseConverter,
    PinnedSWEHarnessDeployment,
    PinnedSWEInstanceImageBinding,
    PinnedSWEOCIImage,
    SubprocessSWEProcessExecutor,
    SWEHarnessInfrastructureError,
    SWEImageMaterializer,
    SWEProcessResult,
)

from skillev.benchmarks import SWEBENCH_WORKSPACE_RESOURCE_ID, SWEBenchVerifiedPublicCase
from skillev.contracts import JsonValue, canonical_json, stable_hash
from skillev.rollout import (
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import ActionKind, BudgetVector, StructuredAction

PRIVATE_CANARY = "PRIVATE-OFFICIAL-SWE-CANARY"
IMAGE_DIGEST = f"sha256:{'a' * 64}"
IMAGE_SOURCE_REFERENCE = (
    "docker://docker.io/swebench/sweb.eval.x86_64.example_1776_project-1:latest"
)
IMAGE_ID = f"{IMAGE_SOURCE_REFERENCE}@{IMAGE_DIGEST}"
IMAGE_PULL_REFERENCE = (
    f"docker://docker.io/swebench/sweb.eval.x86_64.example_1776_project-1@{IMAGE_DIGEST}"
)
WORKER_CONTAINER_PATH = "/opt/skillev/bin/swebench-harness-worker"


def _case() -> PrivateSWEBenchVerifiedCase:
    public = SWEBenchVerifiedPublicCase(
        dataset_revision="princeton-nlp/swe-bench-verified@pinned",
        split="test",
        instance_id="example__project-1",
        repo="example/project",
        version="1.0",
        base_commit="0123456789abcdef",
        problem_statement="Correct the public behavior.",
        environment_image_id=IMAGE_ID,
        task_family="swe-bench-verified/software-engineering/python",
        max_steps=5,
    )
    return PrivateSWEBenchVerifiedCase(
        public,
        PrivateSWEVerifiedTruth(
            version="1.0",
            gold_patch=f"diff --git a/public.py b/public.py\n+{PRIVATE_CANARY}\n",
            test_patch=f"diff --git a/private_test.py b/private_test.py\n+{PRIVATE_CANARY}\n",
            fail_to_pass=(f"test_private::{PRIVATE_CANARY}",),
            pass_to_pass=(f"test_guard::{PRIVATE_CANARY}",),
        ),
    )


def _official_row() -> dict[str, object]:
    return {
        "repo": "example/project",
        "instance_id": "example__project-1",
        "base_commit": "2" * 40,
        "patch": f"diff --git a/public.py b/public.py\n+{PRIVATE_CANARY}\n",
        "test_patch": f"diff --git a/private_test.py b/private_test.py\n+{PRIVATE_CANARY}\n",
        "problem_statement": "Correct the public behavior.",
        "hints_text": "",
        "created_at": "2024-01-02T03:04:05Z",
        "version": "1.0",
        "FAIL_TO_PASS": json.dumps([f"test_private::{PRIVATE_CANARY}"]),
        "PASS_TO_PASS": json.dumps([f"test_guard::{PRIVATE_CANARY}"]),
        "environment_setup_commit": "3" * 40,
        "difficulty": "15 min - 1 hour",
    }


def _executable(path: Path, source: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o700)
    return path.resolve()


def _harness_source(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "official-swe-harness"

    def run_git(
        *arguments: str,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - fixed Git fixture commands
            ("git", *arguments),
            cwd=source,
            check=True,
            capture_output=capture_output,
            text=True,
        )

    if not source.exists():
        source.mkdir()
        (source / "README").write_text("pinned official fixture\n", encoding="utf-8")
        run_git("init", "-q")
        run_git("config", "user.email", "fixture@example.test")
        run_git("config", "user.name", "Fixture")
        run_git("add", "README")
        run_git("commit", "-q", "-m", "fixture")
    revision = run_git("rev-parse", "HEAD", capture_output=True).stdout.strip()
    return source.resolve(), revision


@dataclass(slots=True)
class _BoundedFakeMaterializer:
    """Small lease fake: one path is active only inside the context manager."""

    image_path: Path
    materializer_id: str = "bounded-fake-materializer@1"
    capacity_bytes: int = 1024 * 1024
    max_active_leases: int = 1
    returned_environment_image_id: str | None = None
    returned_manifest_digest: str | None = None
    remove_image_before_yield: bool = False
    lease_events: list[tuple[str, str]] = field(default_factory=list)
    leased_sources: list[PinnedSWEOCIImage] = field(default_factory=list)
    active_leases: int = 0
    peak_active_leases: int = 0

    @contextmanager
    def lease(self, image: PinnedSWEOCIImage) -> Iterator[MaterializedSWEImage]:
        self.leased_sources.append(image)
        self.lease_events.append(("enter", image.environment_image_id))
        if self.image_path.stat().st_size > self.capacity_bytes:
            raise AssertionError("fake materializer byte capacity was exceeded")
        self.active_leases += 1
        self.peak_active_leases = max(self.peak_active_leases, self.active_leases)
        if self.active_leases > self.max_active_leases:
            raise AssertionError("fake materializer capacity was exceeded")
        try:
            materialized = MaterializedSWEImage(
                environment_image_id=(
                    self.returned_environment_image_id or image.environment_image_id
                ),
                manifest_digest=self.returned_manifest_digest or image.manifest_digest,
                image_path=self.image_path,
            )
            if self.remove_image_before_yield:
                self.image_path.unlink()
            yield materialized
        finally:
            self.active_leases -= 1
            self.lease_events.append(("exit", image.environment_image_id))


def _deployment(
    tmp_path: Path,
    *,
    executable_source: str | None = None,
    materializer: SWEImageMaterializer | None = None,
) -> PinnedSWEHarnessDeployment:
    harness_source, harness_revision = _harness_source(tmp_path)
    launcher = _executable(
        tmp_path / "apptainer",
        executable_source or "#!/bin/sh\nexit 0\n",
    )
    image_path = tmp_path / "leased-instance.sif"
    image_path.write_bytes(b"ephemeral synthetic SIF bytes")
    worker_path = _executable(tmp_path / "swebench-harness-worker")
    worker_size, worker_sha256 = (
        worker_path.stat().st_size,
        f"sha256:{hashlib.sha256(worker_path.read_bytes()).hexdigest()}",
    )
    work_root = tmp_path / "work"
    work_root.mkdir()
    image = PinnedSWEOCIImage(
        environment_image_id=IMAGE_ID,
        source_reference=IMAGE_SOURCE_REFERENCE,
        manifest_digest=IMAGE_DIGEST,
    )
    return PinnedSWEHarnessDeployment(
        apptainer_executable=launcher,
        worker_host_path=worker_path,
        worker_container_path=WORKER_CONTAINER_PATH,
        worker_size_bytes=worker_size,
        worker_sha256=worker_sha256,
        harness_revision=harness_revision,
        harness_source_root=harness_source,
        work_root=work_root.resolve(),
        images=(image,),
        materializer=materializer or _BoundedFakeMaterializer(image_path.resolve()),
        request_timeout_seconds=5.0,
    )


def _converter(tmp_path: Path) -> OfficialSWEBenchVerifiedCaseConverter:
    return OfficialSWEBenchVerifiedCaseConverter(
        deployment=_deployment(tmp_path),
        image_bindings=(
            PinnedSWEInstanceImageBinding(
                instance_id="example__project-1",
                environment_image_id=IMAGE_ID,
            ),
        ),
        task_family="swe-bench-verified/software-engineering/python",
        max_steps=12,
    )


def test_official_converter_requires_swebench_context_namespace(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        replace(_converter(tmp_path), task_family="other/family")


def _response(request: dict[str, object], result: JsonValue) -> SWEProcessResult:
    return SWEProcessResult(
        0,
        json.dumps(
            {
                "deployment_id": request["deployment_id"],
                "environment_image_id": request["environment_image_id"],
                "harness_revision": request["harness_revision"],
                "protocol_version": SWE_HARNESS_PROTOCOL_VERSION,
                "request_id": request["request_id"],
                "result": result,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode(),
    )


@dataclass(slots=True)
class _RecordingExecutor:
    mode: str = "ok"
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = field(default_factory=list)

    def execute(
        self,
        argv: object,
        *,
        stdin: bytes,
        cwd: Path,
        timeout_seconds: float,
    ) -> SWEProcessResult:
        del cwd, timeout_seconds
        if not isinstance(argv, tuple):
            argv = tuple(cast(list[str], argv))
        argv = cast(tuple[str, ...], argv)
        request = cast(dict[str, object], json.loads(stdin))
        self.calls.append((argv, request))
        if self.mode == "nonzero":
            return SWEProcessResult(23, b"")
        if self.mode == "invalid-json":
            return SWEProcessResult(0, b"not-json")
        operation = request["operation"]
        payload = cast(dict[str, object], request["payload"])
        if operation == "workspace-initialize":
            assert set(payload) == {
                "base_commit",
                "instance_id",
                "max_steps",
                "repo",
                "test_command",
                "workspace_id",
            }
            result: JsonValue = {
                "base_commit": payload["base_commit"],
                "initialized": True,
                "instance_id": payload["instance_id"],
                "workspace_id": payload["workspace_id"],
            }
        elif operation == "workspace-execute":
            command = cast(dict[str, object], payload["command"])
            kind = command["kind"]
            result = {
                "budget_usage": BudgetVector(
                    tool_calls=1,
                    wall_time_milliseconds=7,
                ).to_value(),
                "public_observation": {"command": kind, "status": "completed"},
                "step_index": payload["step_index"],
                "terminal": kind == "submit_patch",
                "workspace_id": payload["workspace_id"],
            }
        elif operation == "verify":
            assert set(payload) == {
                "base_commit",
                "candidate_patch",
                "eval_script",
                "instance_id",
                "repo",
                "test_patch",
                "version",
            }
            assert PRIVATE_CANARY in canonical_json(payload)
            applied = payload["candidate_patch"] not in {None, "", "cannot-apply"}
            result = {
                "candidate_patch_applied": applied,
                "instance_id": payload["instance_id"],
                "test_output": "official raw pass output" if applied else "",
            }
        else:
            raise AssertionError(f"unhandled operation {operation}")
        return _response(request, result)


@dataclass(slots=True)
class _FakeOfficialGrader:
    harness_revision: str
    grader_id: str = "fake-official-grader@1"
    test_commands: list[str] = field(default_factory=list)
    scripts: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def test_command(self, public: SWEBenchVerifiedPublicCase) -> str:
        command = f"python -m pytest --version-tag={public.version}"
        self.test_commands.append(command)
        return command

    def eval_script(self, case: PrivateSWEBenchVerifiedCase) -> str:
        script = f"#!/bin/sh\necho {case.truth.version}\n"
        self.scripts.append(script)
        return script

    def grade(
        self,
        case: PrivateSWEBenchVerifiedCase,
        *,
        candidate_patch: str,
        test_output: str,
    ) -> SWEVerifierResult:
        assert candidate_patch
        self.outputs.append(test_output)
        passed = test_output == "official raw pass output"
        return SWEVerifierResult(
            resolved=passed,
            fail_to_pass_passed=1 if passed else 0,
            fail_to_pass_total=len(case.truth.fail_to_pass),
            pass_to_pass_passed=1 if passed else 0,
            pass_to_pass_total=len(case.truth.pass_to_pass),
        )


def _tool(name: str, arguments: JsonValue) -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.TOOL,
        name=name,
        arguments=arguments,
        resource_id=SWEBENCH_WORKSPACE_RESOURCE_ID,
    )


def _terminal_request(
    case: PrivateSWEBenchVerifiedCase,
    submission: JsonValue,
) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="swe-official-trajectory",
        task_id=case.public.task_id,
        termination=RolloutTermination.COMPLETED,
        evaluation_input=SubmittedTerminalValue(submission),
        public_transcript_hash=stable_hash({"public": "transcript"}),
    )


def test_pinned_oci_source_is_immutable_and_digest_qualified() -> None:
    source = PinnedSWEOCIImage(
        environment_image_id=IMAGE_ID,
        source_reference=IMAGE_SOURCE_REFERENCE,
        manifest_digest=IMAGE_DIGEST,
    )

    assert source.immutable_source_reference == IMAGE_PULL_REFERENCE
    assert source.platform_os == "linux"
    assert source.platform_architecture == "amd64"
    with pytest.raises(FrozenInstanceError):
        source.manifest_digest = f"sha256:{'b' * 64}"


def test_pinned_oci_source_requires_exact_digest_qualified_public_identity() -> None:
    with pytest.raises(ValueError):
        PinnedSWEOCIImage(
            environment_image_id="unqualified-image-alias",
            source_reference=IMAGE_SOURCE_REFERENCE,
            manifest_digest=IMAGE_DIGEST,
        )


@pytest.mark.parametrize(
    ("capacity_bytes", "max_active_leases"),
    [(0, 1), (1024, 0), (1024, 2)],
)
def test_deployment_requires_positive_capacity_and_one_active_lease(
    tmp_path: Path,
    capacity_bytes: int,
    max_active_leases: int,
) -> None:
    materializer = _BoundedFakeMaterializer(
        (tmp_path / "leased-instance.sif").resolve(),
        capacity_bytes=capacity_bytes,
        max_active_leases=max_active_leases,
    )

    with pytest.raises(ValueError):
        _deployment(tmp_path, materializer=materializer)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("materializer_id", "changed-materializer@2"),
        ("capacity_bytes", 2 * 1024 * 1024),
        ("max_active_leases", 2),
    ],
)
def test_materializer_policy_change_is_rejected_before_acquiring_a_lease(
    tmp_path: Path,
    field_name: str,
    replacement: object,
) -> None:
    leased_path = tmp_path / "leased-instance.sif"
    materializer = _BoundedFakeMaterializer(leased_path.resolve())
    deployment = _deployment(tmp_path, materializer=materializer)
    setattr(materializer, field_name, replacement)
    executor = _RecordingExecutor()
    process = ApptainerSWEHarnessProcess(deployment, executor)

    with pytest.raises(SWEHarnessInfrastructureError):
        ApptainerSWEWorkspaceBackendFactory(
            process, _FakeOfficialGrader(process.harness_revision)
        ).create(_case().public)

    assert materializer.lease_events == []
    assert executor.calls == []


@pytest.mark.parametrize("asset_name", ["apptainer", "worker"])
def test_host_executable_byte_drift_is_rejected_before_acquiring_a_lease(
    tmp_path: Path,
    asset_name: str,
) -> None:
    leased_path = tmp_path / "leased-instance.sif"
    materializer = _BoundedFakeMaterializer(leased_path.resolve())
    deployment = _deployment(tmp_path, materializer=materializer)
    asset_path = (
        deployment.apptainer_executable
        if asset_name == "apptainer"
        else deployment.worker_host_path
    )
    asset_path.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    asset_path.chmod(0o755)
    executor = _RecordingExecutor()
    process = ApptainerSWEHarnessProcess(deployment, executor)

    with pytest.raises(SWEHarnessInfrastructureError):
        ApptainerSWEWorkspaceBackendFactory(
            process, _FakeOfficialGrader(process.harness_revision)
        ).create(_case().public)

    assert materializer.lease_events == []
    assert executor.calls == []


def test_process_leases_every_request_and_has_no_static_verified_image_cache(
    tmp_path: Path,
) -> None:
    leased_path = tmp_path / "leased-instance.sif"
    materializer = _BoundedFakeMaterializer(leased_path.resolve())
    deployment = _deployment(tmp_path, materializer=materializer)
    executor = _RecordingExecutor()
    process = ApptainerSWEHarnessProcess(deployment, executor)

    for _ in range(2):
        ApptainerSWEWorkspaceBackendFactory(
            process, _FakeOfficialGrader(process.harness_revision)
        ).create(_case().public)

    assert not hasattr(process, "_verified_images")
    assert materializer.lease_events == [
        ("enter", IMAGE_ID),
        ("exit", IMAGE_ID),
        ("enter", IMAGE_ID),
        ("exit", IMAGE_ID),
    ]
    assert materializer.active_leases == 0
    assert materializer.peak_active_leases == 1
    assert len(executor.calls) == 2


def test_workspace_and_verifier_factories_reject_another_grader_revision(
    tmp_path: Path,
) -> None:
    process = ApptainerSWEHarnessProcess(_deployment(tmp_path), _RecordingExecutor())
    grader = _FakeOfficialGrader("f" * 40)

    with pytest.raises(ValueError):
        ApptainerSWEWorkspaceBackendFactory(process, grader)
    with pytest.raises(ValueError):
        ApptainerOfficialSWEVerifierFactory(process, grader)


@pytest.mark.parametrize(
    ("returned_environment_image_id", "returned_manifest_digest", "remove_image"),
    [
        ("different-environment-image", None, False),
        (None, f"sha256:{'b' * 64}", False),
        (None, None, True),
    ],
)
def test_materialized_lease_must_match_source_identity_and_exist(
    tmp_path: Path,
    returned_environment_image_id: str | None,
    returned_manifest_digest: str | None,
    remove_image: bool,
) -> None:
    leased_path = tmp_path / "leased-instance.sif"
    materializer = _BoundedFakeMaterializer(
        leased_path.resolve(),
        returned_environment_image_id=returned_environment_image_id,
        returned_manifest_digest=returned_manifest_digest,
        remove_image_before_yield=remove_image,
    )
    deployment = _deployment(tmp_path, materializer=materializer)
    executor = _RecordingExecutor()
    process = ApptainerSWEHarnessProcess(deployment, executor)

    with pytest.raises(SWEHarnessInfrastructureError):
        ApptainerSWEWorkspaceBackendFactory(
            process, _FakeOfficialGrader(process.harness_revision)
        ).create(_case().public)

    assert executor.calls == []
    assert materializer.active_leases == 0
    assert materializer.lease_events == [
        ("enter", IMAGE_ID),
        ("exit", IMAGE_ID),
    ]


def test_materialized_lease_is_released_after_worker_process_failure(tmp_path: Path) -> None:
    leased_path = tmp_path / "leased-instance.sif"
    materializer = _BoundedFakeMaterializer(leased_path.resolve())
    deployment = _deployment(tmp_path, materializer=materializer)
    process = ApptainerSWEHarnessProcess(deployment, _RecordingExecutor("nonzero"))

    with pytest.raises(SWEHarnessInfrastructureError):
        ApptainerSWEWorkspaceBackendFactory(
            process, _FakeOfficialGrader(process.harness_revision)
        ).create(_case().public)

    assert materializer.active_leases == 0
    assert materializer.lease_events == [
        ("enter", IMAGE_ID),
        ("exit", IMAGE_ID),
    ]


def test_official_row_converter_keeps_all_verifier_truth_private(tmp_path: Path) -> None:
    row = _official_row()

    case = _converter(tmp_path).convert(
        row,
        dataset_revision="princeton-nlp/swe-bench-verified@frozen",
        split="test",
    )

    task_wire = canonical_json(case.public.to_rollout_task().to_value())
    assert case.public.problem_statement == row["problem_statement"]
    assert case.public.environment_image_id == IMAGE_ID
    assert case.public.max_steps == 12
    assert case.public.version == row["version"]
    assert case.truth.version == case.public.version
    assert case.truth.gold_patch == row["patch"]
    assert case.truth.test_patch == row["test_patch"]
    assert case.truth.fail_to_pass == (f"test_private::{PRIVATE_CANARY}",)
    assert case.truth.pass_to_pass == (f"test_guard::{PRIVATE_CANARY}",)
    assert PRIVATE_CANARY not in task_wire
    assert PRIVATE_CANARY not in repr(case)
    for private_field in ("patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"):
        assert row[private_field] not in task_wire


def test_pinned_official_grader_uses_official_spec_command_script_and_report(
    tmp_path: Path,
) -> None:
    case = _case()
    source_root, revision = _harness_source(tmp_path)
    work_root = tmp_path / "grader-work"
    work_root.mkdir()
    spec_inputs: list[dict[str, object]] = []
    report_outputs: list[str] = []

    def make_test_spec(instance: object) -> object:
        assert isinstance(instance, dict)
        spec_inputs.append(instance)
        return SimpleNamespace(eval_script="#!/bin/bash\necho official-eval\n")

    def get_eval_report(**kwargs: object) -> object:
        log_path = kwargs["test_log_path"]
        assert isinstance(log_path, Path)
        report_outputs.append(log_path.read_text(encoding="utf-8"))
        assert kwargs["include_tests_status"] is True
        assert kwargs["prediction"] == {
            "instance_id": case.public.instance_id,
            "model_name_or_path": "skillev",
            "model_patch": "candidate patch",
        }
        return {
            case.public.instance_id: {
                "resolved": True,
                "tests_status": {
                    "FAIL_TO_PASS": {
                        "failure": [],
                        "success": list(case.truth.fail_to_pass),
                    },
                    "PASS_TO_PASS": {
                        "failure": [],
                        "success": list(case.truth.pass_to_pass),
                    },
                },
            }
        }

    bindings = _OfficialHarnessBindings(
        make_test_spec=make_test_spec,
        get_eval_report=get_eval_report,
        repo_version_specs={
            case.public.repo: {
                case.public.version: {
                    "test_cmd": ["old command", "python -m pytest"],
                }
            }
        },
        key_instance_id="instance_id",
        key_model="model_name_or_path",
        key_prediction="model_patch",
        fail_to_pass="FAIL_TO_PASS",
        pass_to_pass="PASS_TO_PASS",
    )
    grader = PinnedSWEOfficialGrader(
        harness_source_root=source_root,
        harness_revision=revision,
        work_root=work_root.resolve(),
        bindings_loader=lambda _: bindings,
    )

    assert grader.test_command(case.public) == "python -m pytest"
    assert grader.eval_script(case) == "#!/bin/bash\necho official-eval\n"
    result = grader.grade(
        case,
        candidate_patch="candidate patch",
        test_output="raw official output\n",
    )

    assert result.resolved
    assert report_outputs == ["raw official output\n"]
    assert spec_inputs == [
        {
            "FAIL_TO_PASS": list(case.truth.fail_to_pass),
            "PASS_TO_PASS": list(case.truth.pass_to_pass),
            "base_commit": case.public.base_commit,
            "instance_id": case.public.instance_id,
            "repo": case.public.repo,
            "test_patch": case.truth.test_patch,
            "version": case.public.version,
        },
        {
            "FAIL_TO_PASS": list(case.truth.fail_to_pass),
            "PASS_TO_PASS": list(case.truth.pass_to_pass),
            "base_commit": case.public.base_commit,
            "instance_id": case.public.instance_id,
            "repo": case.public.repo,
            "test_patch": case.truth.test_patch,
            "version": case.public.version,
        },
    ]
    assert not tuple(work_root.iterdir())


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("base_commit", "not-a-full-commit"),
        ("FAIL_TO_PASS", ["not-json-text"]),
        ("PASS_TO_PASS", '{"not":"an-array"}'),
        ("problem_statement", 3),
        ("repo", "another/project"),
    ],
)
def test_official_row_converter_rejects_malformed_exact_rows(
    tmp_path: Path,
    field_name: str,
    replacement: object,
) -> None:
    row = _official_row()
    row[field_name] = replacement

    with pytest.raises((TypeError, ValueError)):
        _converter(tmp_path).convert(
            row,
            dataset_revision="princeton-nlp/swe-bench-verified@frozen",
            split="test",
        )


def test_official_row_converter_rejects_field_drift_and_unbound_instances(
    tmp_path: Path,
) -> None:
    converter = _converter(tmp_path)
    extra = {**_official_row(), "unexpected": "field"}
    missing = _official_row()
    del missing["difficulty"]
    unbound = _official_row()
    unbound["instance_id"] = "example__project-2"

    for row in (extra, missing, unbound):
        with pytest.raises(ValueError):
            converter.convert(
                row,
                dataset_revision="princeton-nlp/swe-bench-verified@frozen",
                split="test",
            )
    with pytest.raises(ValueError):
        converter.convert(
            _official_row(),
            dataset_revision="princeton-nlp/swe-bench-verified@frozen",
            split="validation",
        )


def test_production_factories_use_pinned_apptainer_and_keep_truth_verifier_only(
    tmp_path: Path,
) -> None:
    case = _case()
    leased_path = tmp_path / "leased-instance.sif"
    materializer = _BoundedFakeMaterializer(leased_path.resolve())
    deployment = _deployment(tmp_path, materializer=materializer)
    executor = _RecordingExecutor()
    process = ApptainerSWEHarnessProcess(deployment, executor)
    grader = _FakeOfficialGrader(process.harness_revision)
    sessions = PrivateSWEBenchSessionFactory(
        (case,),
        ApptainerSWEWorkspaceBackendFactory(process, grader),
        ApptainerOfficialSWEVerifierFactory(process, grader),
    )

    first = sessions.create(case.public.to_rollout_task())
    written = asyncio.run(
        first.environment.execute(
            _tool("write", {"content": "public = 2\n", "path": "src/public.py"}),
            step_index=1,
        )
    )
    submitted = asyncio.run(
        first.environment.execute(
            _tool("submit_patch", {"patch": "diff --git a/src/public.py b/src/public.py\n"}),
            step_index=2,
        )
    )
    reward = asyncio.run(
        first.evaluator.evaluate(_terminal_request(case, submitted.terminal_submission))
    )

    assert written.budget_usage == BudgetVector(tool_calls=1, wall_time_milliseconds=7)
    assert reward.value == 1.0
    assert reward.success
    assert PRIVATE_CANARY not in canonical_json(case.public.to_rollout_task().to_value())
    non_verifier_requests = tuple(
        request for _, request in executor.calls if request["operation"] != "verify"
    )
    assert non_verifier_requests
    assert all(PRIVATE_CANARY not in canonical_json(request) for request in non_verifier_requests)
    verifier_requests = tuple(
        request for _, request in executor.calls if request["operation"] == "verify"
    )
    assert len(verifier_requests) == 1
    assert PRIVATE_CANARY in canonical_json(verifier_requests[0])
    initialize_requests = tuple(
        request for _, request in executor.calls if request["operation"] == "workspace-initialize"
    )
    assert [
        cast(dict[str, object], request["payload"])["test_command"]
        for request in initialize_requests
    ] == grader.test_commands
    verifier_payload = cast(dict[str, object], verifier_requests[0]["payload"])
    assert verifier_payload["eval_script"] == grader.scripts[0]
    assert verifier_payload["version"] == case.public.version
    assert grader.outputs == ["official raw pass output"]

    argv = executor.calls[0][0]
    assert argv[:5] == (
        str(deployment.apptainer_executable),
        "exec",
        "--cleanenv",
        "--containall",
        "--no-home",
    )
    assert f"{deployment.work_root}:/skillev/work" in argv
    assert f"{deployment.worker_host_path}:{deployment.worker_container_path}:ro" in argv
    assert argv[-2:] == (
        str(leased_path.resolve()),
        deployment.worker_container_path,
    )
    assert all("docker" not in argument.lower() for argument in argv)
    assert materializer.active_leases == 0
    assert materializer.peak_active_leases == 1
    assert len(materializer.leased_sources) == len(executor.calls)
    assert all(source is deployment.images[0] for source in materializer.leased_sources)
    assert materializer.lease_events == [
        event for _ in executor.calls for event in (("enter", IMAGE_ID), ("exit", IMAGE_ID))
    ]


@pytest.mark.parametrize("mode", ["nonzero", "invalid-json"])
def test_harness_process_failure_is_infrastructure_not_a_fake_reward(
    tmp_path: Path,
    mode: str,
) -> None:
    case = _case()
    process = ApptainerSWEHarnessProcess(_deployment(tmp_path), _RecordingExecutor(mode))
    grader = _FakeOfficialGrader(process.harness_revision)
    evaluator = PrivateSWEBenchSessionFactory(
        (case,),
        ApptainerSWEWorkspaceBackendFactory(process, grader),
        ApptainerOfficialSWEVerifierFactory(process, grader),
    )

    if mode == "nonzero":
        with pytest.raises(SWEHarnessInfrastructureError):
            evaluator.create(case.public.to_rollout_task())
        return

    with pytest.raises(SWEHarnessInfrastructureError):
        evaluator.create(case.public.to_rollout_task())


@pytest.mark.parametrize(
    ("termination", "candidate_patch"),
    [
        (RolloutTermination.HORIZON_EXHAUSTED, None),
        (RolloutTermination.COMPLETED, ""),
        (RolloutTermination.COMPLETED, "cannot-apply"),
    ],
)
def test_absent_empty_or_unapplied_candidate_is_unresolved_without_grading(
    tmp_path: Path,
    termination: RolloutTermination,
    candidate_patch: str | None,
) -> None:
    case = _case()
    process = ApptainerSWEHarnessProcess(_deployment(tmp_path), _RecordingExecutor())
    grader = _FakeOfficialGrader(process.harness_revision)
    verifier = ApptainerOfficialSWEVerifierFactory(process, grader).create(case)

    result = asyncio.run(
        verifier.verify(
            SWEVerifierRequest(
                instance_id=case.public.instance_id,
                repo=case.public.repo,
                base_commit=case.public.base_commit,
                environment_image_id=case.public.environment_image_id,
                termination=termination,
                candidate_patch=candidate_patch,
            )
        )
    )

    assert not result.resolved
    assert result.fail_to_pass_passed == 0
    assert result.pass_to_pass_passed == 0
    assert grader.outputs == []


def test_verifier_process_failure_becomes_terminal_infrastructure_error(
    tmp_path: Path,
) -> None:
    case = _case()
    executor = _RecordingExecutor()
    process = ApptainerSWEHarnessProcess(_deployment(tmp_path), executor)
    grader = _FakeOfficialGrader(process.harness_revision)
    verifier = ApptainerOfficialSWEVerifierFactory(process, grader).create(case)
    evaluator = (
        PrivateSWEBenchSessionFactory(
            (case,),
            ApptainerSWEWorkspaceBackendFactory(process, grader),
            ApptainerOfficialSWEVerifierFactory(process, grader),
        )
        .create(case.public.to_rollout_task())
        .evaluator
    )
    del verifier
    executor.mode = "nonzero"

    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(evaluator.evaluate(_terminal_request(case, {"patch": "public patch"})))


def test_worker_bytes_are_verified_before_materialization_or_worker_request(
    tmp_path: Path,
) -> None:
    case = _case()
    leased_path = tmp_path / "leased-instance.sif"
    materializer = _BoundedFakeMaterializer(leased_path.resolve())
    deployment = _deployment(tmp_path, materializer=materializer)
    deployment.worker_host_path.write_bytes(b"changed worker bytes")
    executor = _RecordingExecutor()
    process = ApptainerSWEHarnessProcess(deployment, executor)

    with pytest.raises(SWEHarnessInfrastructureError):
        ApptainerSWEWorkspaceBackendFactory(
            process, _FakeOfficialGrader(process.harness_revision)
        ).create(case.public)
    assert executor.calls == []
    assert materializer.lease_events == []


_FAKE_APPTAINER = r"""#!/usr/bin/env python3
import json
import sys

request = json.loads(sys.stdin.buffer.read())
payload = request["payload"]
if request["operation"] != "workspace-initialize":
    raise SystemExit(9)
result = {
    "base_commit": payload["base_commit"],
    "initialized": True,
    "instance_id": payload["instance_id"],
    "workspace_id": payload["workspace_id"],
}
response = {
    "deployment_id": request["deployment_id"],
    "environment_image_id": request["environment_image_id"],
    "harness_revision": request["harness_revision"],
    "protocol_version": request["protocol_version"],
    "request_id": request["request_id"],
    "result": result,
}
sys.stdout.write(json.dumps(response, separators=(",", ":"), sort_keys=True))
"""


def test_real_subprocess_executor_runs_tiny_fake_apptainer_protocol(tmp_path: Path) -> None:
    case = _case()
    deployment = _deployment(tmp_path, executable_source=_FAKE_APPTAINER)
    process = ApptainerSWEHarnessProcess(deployment, SubprocessSWEProcessExecutor())

    backend = ApptainerSWEWorkspaceBackendFactory(
        process, _FakeOfficialGrader(process.harness_revision)
    ).create(case.public)

    assert backend.instance_id == case.public.instance_id
    assert backend.environment_id == case.public.environment_id


def test_real_subprocess_executor_reports_missing_process_as_infrastructure(
    tmp_path: Path,
) -> None:
    executor = SubprocessSWEProcessExecutor()
    with pytest.raises(SWEHarnessInfrastructureError):
        executor.execute(
            (str(tmp_path / "missing-executable"),),
            stdin=b"{}",
            cwd=tmp_path,
            timeout_seconds=1.0,
        )
