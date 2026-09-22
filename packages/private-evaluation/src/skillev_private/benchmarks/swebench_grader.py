"""Revision-pinned host-side bindings to the official SWE-bench grader.

The container worker executes only the exact evaluation script produced here
and returns raw output.  Official report parsing stays on the private host, so
test identities and grading semantics never enter the model-facing package.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from skillev.benchmarks.swebench import SWEBenchVerifiedPublicCase
from skillev.contracts import stable_hash

from .swebench import (
    PrivateSWEBenchVerifiedCase,
    SWEVerifierInfrastructureError,
    SWEVerifierResult,
)

_REVISION_LENGTH = 40


class SWEHarnessInfrastructureError(SWEVerifierInfrastructureError):
    """The pinned Apptainer or official-harness process failed."""


def _text(value: object, *, field_name: str, allow_empty: bool = False) -> str:
    if type(value) is not str or "\x00" in value or (not allow_empty and not value.strip()):
        raise ValueError(f"{field_name} has invalid text")
    return value


def _git_revision(value: object) -> str:
    revision = _text(value, field_name="harness_revision").lower()
    if len(revision) != _REVISION_LENGTH or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("harness_revision must be a full 40-character Git commit")
    return revision


def _existing_absolute_path(value: Path, *, field_name: str, directory: bool) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{field_name} must be an absolute Path")
    resolved = value.resolve(strict=True)
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"{field_name} must identify an existing {kind}")
    return resolved


def _pinned_git_source(value: Path, *, revision: str) -> Path:
    root = _existing_absolute_path(value, field_name="harness_source_root", directory=True)
    try:
        head = subprocess.run(  # noqa: S603 - fixed Git metadata query
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(  # noqa: S603 - fixed Git metadata query
            ("git", "-C", str(root), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("harness_source_root must be a readable Git checkout") from error
    if head != revision or dirty:
        raise ValueError("harness_source_root differs from its pinned clean revision")
    return root


@dataclass(frozen=True, slots=True)
class _OfficialHarnessBindings:
    make_test_spec: Callable[[Mapping[str, object]], object]
    get_eval_report: Callable[..., object]
    repo_version_specs: Mapping[str, object]
    key_instance_id: str
    key_model: str
    key_prediction: str
    fail_to_pass: str
    pass_to_pass: str


def _module_from_source(module: object, source_root: Path, *, label: str) -> None:
    raw_path = getattr(module, "__file__", None)
    if type(raw_path) is not str:
        raise SWEHarnessInfrastructureError(f"official {label} module has no source path")
    module_path = Path(raw_path).resolve(strict=True)
    if not module_path.is_relative_to(source_root):
        raise SWEHarnessInfrastructureError(f"official {label} module is not pinned source")


def _load_official_harness_bindings(source_root: Path) -> _OfficialHarnessBindings:
    source_text = str(source_root)
    sys.path.insert(0, source_text)
    try:
        importlib.invalidate_caches()
        test_spec_module = importlib.import_module("swebench.harness.test_spec.test_spec")
        grading_module = importlib.import_module("swebench.harness.grading")
        constants_module = importlib.import_module("swebench.harness.constants")
    except (ImportError, OSError) as error:
        raise SWEHarnessInfrastructureError(
            "pinned official SWE-bench grader could not be imported"
        ) from error
    finally:
        sys.path.pop(0)
    _module_from_source(test_spec_module, source_root, label="test-spec")
    _module_from_source(grading_module, source_root, label="grading")
    _module_from_source(constants_module, source_root, label="constants")
    make_test_spec = getattr(test_spec_module, "make_test_spec", None)
    get_eval_report = getattr(grading_module, "get_eval_report", None)
    repo_version_specs = getattr(constants_module, "MAP_REPO_VERSION_TO_SPECS", None)
    constants = tuple(
        getattr(constants_module, name, None)
        for name in (
            "KEY_INSTANCE_ID",
            "KEY_MODEL",
            "KEY_PREDICTION",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
        )
    )
    if not callable(make_test_spec) or not callable(get_eval_report):
        raise SWEHarnessInfrastructureError("pinned official grader entrypoints are absent")
    if not isinstance(repo_version_specs, Mapping):
        raise SWEHarnessInfrastructureError("pinned official repository specs are absent")
    if any(type(value) is not str or not value for value in constants):
        raise SWEHarnessInfrastructureError("pinned official grader constants are absent")
    return _OfficialHarnessBindings(
        make_test_spec=cast(Callable[[Mapping[str, object]], object], make_test_spec),
        get_eval_report=cast(Callable[..., object], get_eval_report),
        repo_version_specs=cast(Mapping[str, object], repo_version_specs),
        key_instance_id=cast(str, constants[0]),
        key_model=cast(str, constants[1]),
        key_prediction=cast(str, constants[2]),
        fail_to_pass=cast(str, constants[3]),
        pass_to_pass=cast(str, constants[4]),
    )


class SWEOfficialGrader(Protocol):
    @property
    def harness_revision(self) -> str: ...

    @property
    def grader_id(self) -> str: ...

    def test_command(self, public: SWEBenchVerifiedPublicCase) -> str: ...

    def eval_script(self, case: PrivateSWEBenchVerifiedCase) -> str: ...

    def grade(
        self,
        case: PrivateSWEBenchVerifiedCase,
        *,
        candidate_patch: str,
        test_output: str,
    ) -> SWEVerifierResult: ...


@dataclass(frozen=True, slots=True)
class PinnedSWEOfficialGrader:
    """Generate and grade only through one clean, pinned harness checkout."""

    harness_source_root: Path = field(repr=False)
    harness_revision: str
    work_root: Path = field(repr=False)
    bindings_loader: Callable[[Path], _OfficialHarnessBindings] = field(
        default=_load_official_harness_bindings,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        revision = _git_revision(self.harness_revision)
        source = _pinned_git_source(self.harness_source_root, revision=revision)
        work = _existing_absolute_path(self.work_root, field_name="work_root", directory=True)
        object.__setattr__(self, "harness_revision", revision)
        object.__setattr__(self, "harness_source_root", source)
        object.__setattr__(self, "work_root", work)

    @property
    def grader_id(self) -> str:
        return stable_hash(
            {
                "harness_revision": self.harness_revision,
                "harness_source_root": str(self.harness_source_root),
            }
        )

    def test_command(self, public: SWEBenchVerifiedPublicCase) -> str:
        if not isinstance(public, SWEBenchVerifiedPublicCase):
            raise TypeError("official grader requires a public SWE case")
        bindings = self.bindings_loader(self.harness_source_root)
        raw_repo_specs = bindings.repo_version_specs.get(public.repo)
        if not isinstance(raw_repo_specs, Mapping):
            raise SWEHarnessInfrastructureError("official grader omitted repository specs")
        raw_version_specs = raw_repo_specs.get(public.version)
        if not isinstance(raw_version_specs, Mapping):
            raise SWEHarnessInfrastructureError("official grader omitted version specs")
        raw_command = raw_version_specs.get("test_cmd")
        if isinstance(raw_command, list):
            if not raw_command:
                raise SWEHarnessInfrastructureError("official grader test command is empty")
            raw_command = raw_command[-1]
        try:
            return _text(raw_command, field_name="official test_cmd")
        except ValueError as error:
            raise SWEHarnessInfrastructureError(
                "official grader test command is invalid"
            ) from error

    def eval_script(self, case: PrivateSWEBenchVerifiedCase) -> str:
        spec = self._test_spec(case)
        try:
            return _text(getattr(spec, "eval_script", None), field_name="official eval_script")
        except ValueError as error:
            raise SWEHarnessInfrastructureError("official grader eval script is invalid") from error

    def grade(
        self,
        case: PrivateSWEBenchVerifiedCase,
        *,
        candidate_patch: str,
        test_output: str,
    ) -> SWEVerifierResult:
        patch = _text(candidate_patch, field_name="candidate_patch")
        output = _text(test_output, field_name="test_output", allow_empty=True)
        bindings = self.bindings_loader(self.harness_source_root)
        spec = self._make_test_spec(bindings, case)
        prediction = {
            bindings.key_instance_id: case.public.instance_id,
            bindings.key_model: "skillev",
            bindings.key_prediction: patch,
        }
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.work_root,
                prefix=".official-swe-test-output-",
            ) as stream:
                stream.write(output)
                stream.flush()
                raw_report = bindings.get_eval_report(
                    test_spec=spec,
                    prediction=prediction,
                    test_log_path=Path(stream.name),
                    include_tests_status=True,
                )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise SWEHarnessInfrastructureError(
                "pinned official SWE-bench grader failed"
            ) from error
        return self._result(case, raw_report, bindings)

    def _test_spec(self, case: PrivateSWEBenchVerifiedCase) -> object:
        if not isinstance(case, PrivateSWEBenchVerifiedCase):
            raise TypeError("official grader requires a private SWE case")
        return self._make_test_spec(self.bindings_loader(self.harness_source_root), case)

    @staticmethod
    def _make_test_spec(
        bindings: _OfficialHarnessBindings,
        case: PrivateSWEBenchVerifiedCase,
    ) -> object:
        try:
            return bindings.make_test_spec(PinnedSWEOfficialGrader._instance(case))
        except (RuntimeError, TypeError, ValueError) as error:
            raise SWEHarnessInfrastructureError(
                "pinned official SWE-bench test spec failed"
            ) from error

    @staticmethod
    def _instance(case: PrivateSWEBenchVerifiedCase) -> dict[str, object]:
        return {
            "FAIL_TO_PASS": list(case.truth.fail_to_pass),
            "PASS_TO_PASS": list(case.truth.pass_to_pass),
            "base_commit": case.public.base_commit,
            "instance_id": case.public.instance_id,
            "repo": case.public.repo,
            "test_patch": case.truth.test_patch,
            "version": case.public.version,
        }

    @staticmethod
    def _result(
        case: PrivateSWEBenchVerifiedCase,
        raw_report: object,
        bindings: _OfficialHarnessBindings,
    ) -> SWEVerifierResult:
        if not isinstance(raw_report, dict):
            raise SWEHarnessInfrastructureError("official grader report is not an object")
        instance_report = raw_report.get(case.public.instance_id)
        if not isinstance(instance_report, dict):
            raise SWEHarnessInfrastructureError("official grader omitted the instance report")
        statuses = instance_report.get("tests_status")
        if not isinstance(statuses, dict):
            raise SWEHarnessInfrastructureError("official grader omitted test statuses")

        def passed(category: str, expected: tuple[str, ...]) -> int:
            value = statuses.get(category)
            if not isinstance(value, dict):
                raise SWEHarnessInfrastructureError("official grader omitted a test category")
            successes = value.get("success")
            failures = value.get("failure")
            if not isinstance(successes, list) or not isinstance(failures, list):
                raise SWEHarnessInfrastructureError("official grader test status is invalid")
            observed = successes + failures
            if any(type(item) is not str for item in observed):
                raise SWEHarnessInfrastructureError("official grader test identity is invalid")
            if len(set(observed)) != len(observed) or set(observed) != set(expected):
                raise SWEHarnessInfrastructureError(
                    "official grader status differs from pinned test identities"
                )
            return len(successes)

        fail_passed = passed(bindings.fail_to_pass, case.truth.fail_to_pass)
        pass_passed = passed(bindings.pass_to_pass, case.truth.pass_to_pass)
        result = SWEVerifierResult(
            resolved=(
                fail_passed == len(case.truth.fail_to_pass)
                and pass_passed == len(case.truth.pass_to_pass)
            ),
            fail_to_pass_passed=fail_passed,
            fail_to_pass_total=len(case.truth.fail_to_pass),
            pass_to_pass_passed=pass_passed,
            pass_to_pass_total=len(case.truth.pass_to_pass),
        )
        if instance_report.get("resolved") is not result.resolved:
            raise SWEHarnessInfrastructureError(
                "official grader resolved flag differs from official test statuses"
            )
        return result


__all__ = [
    "PinnedSWEOfficialGrader",
    "SWEHarnessInfrastructureError",
    "SWEOfficialGrader",
]
