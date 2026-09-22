"""Private SWE-bench Verified verifier and rollout-session composition.

Hidden patches and test identities are retained in this private wheel.  The
workspace receives only the public case, while the official verifier receives
an answer-free candidate request after rollout termination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from skillev.benchmarks.swebench import (
    SWEBENCH_VERIFIED_BENCHMARK_ID,
    SWEBenchVerifiedPublicCase,
    SWEBenchWorkspaceEnvironment,
    SWEWorkspaceBackend,
)
from skillev.contracts import SuccessRule, TerminalReward
from skillev.rollout import (
    NoTerminalSubmission,
    RolloutTask,
    RolloutTermination,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.training import RolloutSessionBundle

from .terminal_inputs import no_submission_reward, submitted_value


def _text(value: object, *, field_name: str, allow_empty: bool = False) -> str:
    if type(value) is not str or "\x00" in value or (not allow_empty and not value.strip()):
        raise ValueError(f"{field_name} has invalid text")
    return value


@dataclass(frozen=True, slots=True)
class PrivateSWEVerifiedTruth:
    """Official verifier truth; never projected into a rollout task."""

    version: str = field(repr=False)
    gold_patch: str = field(repr=False)
    test_patch: str = field(repr=False)
    fail_to_pass: tuple[str, ...] = field(repr=False)
    pass_to_pass: tuple[str, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _text(self.version, field_name="version")
        _text(self.gold_patch, field_name="gold_patch")
        _text(self.test_patch, field_name="test_patch")
        for field_name, tests in (
            ("fail_to_pass", self.fail_to_pass),
            ("pass_to_pass", self.pass_to_pass),
        ):
            if not isinstance(tests, tuple):
                raise TypeError(f"{field_name} must be a tuple")
            if any(type(test) is not str or not test.strip() for test in tests):
                raise ValueError(f"{field_name} contains an invalid test identity")
            if len(set(tests)) != len(tests):
                raise ValueError(f"{field_name} test identities must be unique")


@dataclass(frozen=True, slots=True)
class PrivateSWEBenchVerifiedCase:
    public: SWEBenchVerifiedPublicCase
    truth: PrivateSWEVerifiedTruth = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.public, SWEBenchVerifiedPublicCase):
            raise TypeError("private SWE-bench case requires a public case")
        if not isinstance(self.truth, PrivateSWEVerifiedTruth):
            raise TypeError("private SWE-bench case requires official verifier truth")
        if self.truth.version != self.public.version:
            raise ValueError("private SWE-bench version differs from its public case")


@dataclass(frozen=True, slots=True)
class SWEVerifierRequest:
    """Answer-free candidate sent to the injected official verifier."""

    instance_id: str
    repo: str
    base_commit: str
    environment_image_id: str
    termination: RolloutTermination
    candidate_patch: str | None

    def __post_init__(self) -> None:
        for field_name in ("instance_id", "repo", "base_commit", "environment_image_id"):
            _text(getattr(self, field_name), field_name=field_name)
        if not isinstance(self.termination, RolloutTermination):
            raise TypeError("SWE verifier termination is invalid")
        if self.termination is RolloutTermination.COMPLETED:
            if self.candidate_patch is None:
                raise ValueError("completed SWE verification requires a candidate patch")
            _text(self.candidate_patch, field_name="candidate_patch", allow_empty=True)
        elif self.candidate_patch is not None:
            raise ValueError("horizon-exhausted SWE verification cannot carry a patch")


@dataclass(frozen=True, slots=True)
class SWEVerifierResult:
    """Explicit official harness outcome without hidden test identities."""

    resolved: bool
    fail_to_pass_passed: int
    fail_to_pass_total: int
    pass_to_pass_passed: int
    pass_to_pass_total: int

    def __post_init__(self) -> None:
        if type(self.resolved) is not bool:
            raise TypeError("SWE verifier resolved flag must be boolean")
        for passed_name, total_name in (
            ("fail_to_pass_passed", "fail_to_pass_total"),
            ("pass_to_pass_passed", "pass_to_pass_total"),
        ):
            passed = getattr(self, passed_name)
            total = getattr(self, total_name)
            if type(passed) is not int or type(total) is not int:
                raise TypeError("SWE verifier test counts must be integers")
            if not 0 <= passed <= total:
                raise ValueError("SWE verifier test counts are inconsistent")
        expected_resolved = (
            self.fail_to_pass_passed == self.fail_to_pass_total
            and self.pass_to_pass_passed == self.pass_to_pass_total
        )
        if self.resolved is not expected_resolved:
            raise ValueError("SWE verifier resolved flag disagrees with test counts")


class SWEVerifierInfrastructureError(RuntimeError):
    """Known official-harness outage; it is not an agent outcome."""


class OfficialSWEVerifierBackend(Protocol):
    @property
    def instance_id(self) -> str: ...

    @property
    def environment_id(self) -> str: ...

    @property
    def verifier_version(self) -> str: ...

    async def verify(self, request: SWEVerifierRequest) -> SWEVerifierResult: ...


class SWEWorkspaceBackendFactory(Protocol):
    """Create a workspace from public identity only."""

    def create(self, public: SWEBenchVerifiedPublicCase) -> SWEWorkspaceBackend: ...


class OfficialSWEVerifierFactory(Protocol):
    """Create a verifier inside the private truth boundary."""

    def create(self, case: PrivateSWEBenchVerifiedCase) -> OfficialSWEVerifierBackend: ...


@dataclass(slots=True)
class PrivateSWEBenchTerminalEvaluator:
    public: SWEBenchVerifiedPublicCase
    truth: PrivateSWEVerifiedTruth = field(repr=False)
    verifier: OfficialSWEVerifierBackend = field(repr=False)

    def __post_init__(self) -> None:
        if self.verifier.instance_id != self.public.instance_id:
            raise ValueError("SWE verifier belongs to another instance")
        if self.verifier.environment_id != self.public.environment_id:
            raise ValueError("SWE verifier belongs to another pinned environment")
        _text(self.verifier.verifier_version, field_name="verifier_version")

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.public.task_id:
            raise TerminalEvaluatorError("terminal request reached another SWE-bench task")
        if isinstance(request.evaluation_input, NoTerminalSubmission):
            return no_submission_reward(
                request,
                native_metric_name="swe-bench-verified-resolved",
                native_payload={
                    "benchmark_id": SWEBENCH_VERIFIED_BENCHMARK_ID,
                    "dataset_revision": self.public.dataset_revision,
                    "fail_to_pass_passed": 0,
                    "fail_to_pass_total": len(self.truth.fail_to_pass),
                    "pass_to_pass_passed": 0,
                    "pass_to_pass_total": len(self.truth.pass_to_pass),
                    "resolved": False,
                    "split": self.public.split,
                },
                environment_id=self.public.environment_id,
                verifier_version=self.verifier.verifier_version,
            )
        candidate_patch = _candidate_patch(request)
        verifier_request = SWEVerifierRequest(
            instance_id=self.public.instance_id,
            repo=self.public.repo,
            base_commit=self.public.base_commit,
            environment_image_id=self.public.environment_image_id,
            termination=request.termination,
            candidate_patch=candidate_patch,
        )
        try:
            result = await self.verifier.verify(verifier_request)
        except SWEVerifierInfrastructureError as error:
            raise TerminalEvaluatorError("SWE-bench verifier infrastructure failed") from error
        if not isinstance(result, SWEVerifierResult):
            raise TerminalEvaluatorError("SWE verifier returned an incompatible result")
        if result.fail_to_pass_total != len(self.truth.fail_to_pass):
            raise TerminalEvaluatorError(
                "SWE verifier FAIL_TO_PASS total differs from pinned truth"
            )
        if result.pass_to_pass_total != len(self.truth.pass_to_pass):
            raise TerminalEvaluatorError(
                "SWE verifier PASS_TO_PASS total differs from pinned truth"
            )
        reward = 1.0 if result.resolved else 0.0
        return TerminalReward(
            value=reward,
            success=result.resolved,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name="swe-bench-verified-resolved",
            native_payload={
                "benchmark_id": SWEBENCH_VERIFIED_BENCHMARK_ID,
                "dataset_revision": self.public.dataset_revision,
                "fail_to_pass_passed": result.fail_to_pass_passed,
                "fail_to_pass_total": result.fail_to_pass_total,
                "pass_to_pass_passed": result.pass_to_pass_passed,
                "pass_to_pass_total": result.pass_to_pass_total,
                "resolved": result.resolved,
                "split": self.public.split,
            },
            environment_id=self.public.environment_id,
            verifier_version=self.verifier.verifier_version,
        )


def _candidate_patch(request: TerminalEvaluationRequest) -> str | None:
    submission = submitted_value(request)
    if not isinstance(submission, dict) or set(submission) != {"patch"}:
        raise ValueError("completed SWE request must carry the exact patch submission shape")
    return _text(submission["patch"], field_name="candidate_patch", allow_empty=True)


@dataclass(slots=True)
class PrivateSWEBenchSessionFactory:
    cases: tuple[PrivateSWEBenchVerifiedCase, ...]
    workspace_factory: SWEWorkspaceBackendFactory
    verifier_factory: OfficialSWEVerifierFactory

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("private SWE-bench session factory requires cases")
        if any(not isinstance(case, PrivateSWEBenchVerifiedCase) for case in self.cases):
            raise TypeError("private SWE-bench session factory cases are invalid")
        task_ids = tuple(case.public.task_id for case in self.cases)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("private SWE-bench cases must have unique task identities")
        if not callable(getattr(self.workspace_factory, "create", None)):
            raise TypeError("SWE workspace factory must implement create")
        if not callable(getattr(self.verifier_factory, "create", None)):
            raise TypeError("SWE verifier factory must implement create")

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        if not isinstance(task, RolloutTask):
            raise TypeError("SWE-bench session creation requires RolloutTask")
        matches = tuple(case for case in self.cases if case.public.task_id == task.task_id)
        if len(matches) != 1:
            raise ValueError("public task has no unique private SWE-bench case")
        case = matches[0]
        if task != case.public.to_rollout_task():
            raise ValueError("public SWE-bench task projection differs from its private case")
        workspace = self.workspace_factory.create(case.public)
        verifier = self.verifier_factory.create(case)
        return RolloutSessionBundle(
            environment=SWEBenchWorkspaceEnvironment(case.public, workspace),
            evaluator=PrivateSWEBenchTerminalEvaluator(case.public, case.truth, verifier),
            retrieved_skills=(),
        )


__all__ = [
    "OfficialSWEVerifierBackend",
    "OfficialSWEVerifierFactory",
    "PrivateSWEBenchSessionFactory",
    "PrivateSWEBenchTerminalEvaluator",
    "PrivateSWEBenchVerifiedCase",
    "PrivateSWEVerifiedTruth",
    "SWEVerifierInfrastructureError",
    "SWEVerifierRequest",
    "SWEVerifierResult",
    "SWEWorkspaceBackendFactory",
]
