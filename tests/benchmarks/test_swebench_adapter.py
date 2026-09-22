from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import cast

import pytest
from skillev_private.benchmarks import (
    PrivateSWEBenchTerminalEvaluator,
    PrivateSWEBenchVerifiedCase,
    PrivateSWEVerifiedTruth,
    SWEVerifierInfrastructureError,
    SWEVerifierRequest,
    SWEVerifierResult,
)

from skillev.benchmarks import (
    SWEBENCH_WORKSPACE_RESOURCE_ID,
    OrderedSWEBenchVerifiedTaskProvider,
    SWEBenchVerifiedPublicCase,
    SWEBenchWorkspaceEnvironment,
    SWECommandKind,
    SWEWorkspaceCommand,
    SWEWorkspacePublicStep,
)
from skillev.contracts import JsonValue, stable_hash
from skillev.rollout import (
    NoSubmissionReason,
    NoTerminalSubmission,
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import ActionKind, BudgetVector, StructuredAction

PRIVATE_CANARY = "PRIVATE-SWEBENCH-VERIFIER-CANARY"


class SyntheticWorkspaceError(RuntimeError):
    pass


class UnknownVerifierError(RuntimeError):
    pass


@dataclass(slots=True)
class _SyntheticWorkspace:
    public: SWEBenchVerifiedPublicCase
    environment_override: str | None = None
    max_steps_override: int | None = None
    files: dict[str, str] = field(default_factory=lambda: {"src/public.py": "value = 1\n"})
    calls: list[SWEWorkspaceCommand] = field(default_factory=list)
    submitted_patch: str | None = None

    @property
    def instance_id(self) -> str:
        return cast(str, self.public.instance_id)

    @property
    def environment_id(self) -> str:
        return self.environment_override or cast(str, self.public.environment_id)

    @property
    def max_steps(self) -> int:
        return self.max_steps_override or cast(int, self.public.max_steps)

    async def execute(
        self,
        command: SWEWorkspaceCommand,
        *,
        step_index: int,
    ) -> SWEWorkspacePublicStep:
        if command.kind is SWECommandKind.TEST and command.arguments["target"] == "explode":
            raise SyntheticWorkspaceError("synthetic container failed")
        if command.kind is SWECommandKind.TEST and command.arguments["target"] == "bad-budget":
            return SWEWorkspacePublicStep(
                public_observation={"status": "measured-without-tool-call"},
                terminal=False,
                budget_usage=BudgetVector(),
            )

        self.calls.append(command)
        public: dict[str, JsonValue]
        if command.kind is SWECommandKind.READ:
            path = cast(str, command.arguments["path"])
            public = {"content": self.files.get(path), "path": path}
        elif command.kind is SWECommandKind.WRITE:
            path = cast(str, command.arguments["path"])
            content = cast(str, command.arguments["content"])
            self.files[path] = content
            public = {"bytes_written": len(content.encode()), "path": path}
        elif command.kind is SWECommandKind.SEARCH:
            path = cast(str, command.arguments["path"])
            query = cast(str, command.arguments["query"])
            matches = sorted(name for name, text in self.files.items() if query in text)
            public = {"matches": matches, "path": path}
        elif command.kind is SWECommandKind.TEST:
            public = {
                "exit_code": 0,
                "summary": "public selected tests passed",
                "target": command.arguments["target"],
            }
        else:
            self.submitted_patch = cast(str, command.arguments["patch"])
            public = {"patch_bytes": len(self.submitted_patch.encode()), "status": "submitted"}
        return SWEWorkspacePublicStep(
            public_observation={"step_index": step_index, **public},
            terminal=command.kind is SWECommandKind.SUBMIT_PATCH,
            budget_usage=BudgetVector(tool_calls=1, wall_time_milliseconds=2),
        )


@dataclass(slots=True)
class _SyntheticWorkspaceFactory:
    workspaces: list[_SyntheticWorkspace] = field(default_factory=list)

    def create(self, public: SWEBenchVerifiedPublicCase) -> _SyntheticWorkspace:
        workspace = _SyntheticWorkspace(public)
        self.workspaces.append(workspace)
        return workspace


@dataclass(slots=True)
class _SyntheticVerifier:
    public: SWEBenchVerifiedPublicCase
    truth: PrivateSWEVerifiedTruth = field(repr=False)
    mode: str = "success"
    requests: list[SWEVerifierRequest] = field(default_factory=list)

    @property
    def instance_id(self) -> str:
        return cast(str, self.public.instance_id)

    @property
    def environment_id(self) -> str:
        return cast(str, self.public.environment_id)

    @property
    def verifier_version(self) -> str:
        return "synthetic-official-swe-verifier@1"

    async def verify(self, request: SWEVerifierRequest) -> SWEVerifierResult:
        self.requests.append(request)
        if self.mode == "infrastructure":
            raise SWEVerifierInfrastructureError(f"private harness unavailable: {PRIVATE_CANARY}")
        if self.mode == "unknown":
            raise UnknownVerifierError("unexpected verifier implementation failure")
        resolved = self.mode == "success" and request.candidate_patch is not None
        return SWEVerifierResult(
            resolved=resolved,
            fail_to_pass_passed=len(self.truth.fail_to_pass) if resolved else 0,
            fail_to_pass_total=len(self.truth.fail_to_pass),
            pass_to_pass_passed=len(self.truth.pass_to_pass) if resolved else 0,
            pass_to_pass_total=len(self.truth.pass_to_pass),
        )


@dataclass(slots=True)
class _SyntheticVerifierFactory:
    mode: str = "success"
    verifiers: list[_SyntheticVerifier] = field(default_factory=list)

    def create(self, case: PrivateSWEBenchVerifiedCase) -> _SyntheticVerifier:
        verifier = _SyntheticVerifier(case.public, case.truth, self.mode)
        self.verifiers.append(verifier)
        return verifier


def _public_case(*, max_steps: int = 5) -> SWEBenchVerifiedPublicCase:
    return SWEBenchVerifiedPublicCase(
        dataset_revision="swebench-verified-fixture@1",
        split="test",
        instance_id="synthetic__repo-1",
        repo="synthetic/repo",
        version="1.0",
        base_commit="0123456789abcdef",
        problem_statement="Correct the public arithmetic behavior.",
        environment_image_id="oci://synthetic/swebench@sha256:public",
        task_family="swe-bench-verified/software-engineering/python",
        max_steps=max_steps,
    )


def _private_case(*, max_steps: int = 5) -> PrivateSWEBenchVerifiedCase:
    return PrivateSWEBenchVerifiedCase(
        _public_case(max_steps=max_steps),
        PrivateSWEVerifiedTruth(
            version="1.0",
            gold_patch=f"diff --git a/src/public.py b/src/public.py\n+{PRIVATE_CANARY}\n",
            test_patch=f"diff --git a/private_test.py b/private_test.py\n+{PRIVATE_CANARY}\n",
            fail_to_pass=(f"test_private_regression::{PRIVATE_CANARY}",),
            pass_to_pass=(f"test_private_guard::{PRIVATE_CANARY}",),
        ),
    )


def _tool(name: str, arguments: JsonValue) -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.TOOL,
        name=name,
        arguments=arguments,
        resource_id=SWEBENCH_WORKSPACE_RESOURCE_ID,
    )


def _request(
    public: SWEBenchVerifiedPublicCase,
    *,
    termination: RolloutTermination,
    submission: JsonValue,
) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="synthetic-swe-trajectory",
        task_id=public.task_id,
        termination=termination,
        evaluation_input=(
            SubmittedTerminalValue(submission)
            if termination is RolloutTermination.COMPLETED
            else NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED)
        ),
        public_transcript_hash=stable_hash({"public": "synthetic transcript"}),
    )


def test_swebench_public_identity_and_strict_command_projection() -> None:
    case = _public_case()
    task = case.to_rollout_task()
    provider = OrderedSWEBenchVerifiedTaskProvider((case,))

    assert provider.next_task() == task
    assert task.task_id == case.instance_id
    assert task.query == case.problem_statement
    assert task.environment_id == case.environment_id
    with pytest.raises(RuntimeError):
        provider.next_task()

    assert SWEWorkspaceCommand(SWECommandKind.READ, {"path": "src/public.py"})
    assert SWEWorkspaceCommand(SWECommandKind.SEARCH, {"path": ".", "query": "value"})
    with pytest.raises(ValueError):
        SWEWorkspaceCommand(SWECommandKind.READ, {"path": "../private"})
    with pytest.raises(ValueError):
        SWEWorkspaceCommand(SWECommandKind.TEST, {"command": "arbitrary shell"})


def test_invalid_agent_action_is_data_without_hidden_retry() -> None:
    public = _public_case()
    backend = _SyntheticWorkspace(public)
    environment = SWEBenchWorkspaceEnvironment(public, backend)
    invalid = _tool("read", {"path": "../escape"})

    observation = asyncio.run(environment.execute(invalid, step_index=1))

    assert observation.observation_status == "tool_error"
    assert observation.budget_usage.tool_calls == 1
    assert not backend.calls


def test_horizon_exhaustion_is_verified_as_explicit_no_patch() -> None:
    case = _private_case()
    verifier = _SyntheticVerifier(case.public, case.truth, mode="failure")
    evaluator = PrivateSWEBenchTerminalEvaluator(case.public, case.truth, verifier)

    reward = asyncio.run(
        evaluator.evaluate(
            _request(
                case.public,
                termination=RolloutTermination.HORIZON_EXHAUSTED,
                submission=None,
            )
        )
    )

    assert reward.value == 0.0
    assert verifier.requests == []
    assert reward.native_metric_name == "swe-bench-verified-resolved"
    assert "public_metrics" not in reward.native_payload


def test_verifier_typed_infrastructure_and_unknown_errors_stay_distinct() -> None:
    case = _private_case()
    infra = PrivateSWEBenchTerminalEvaluator(
        case.public,
        case.truth,
        _SyntheticVerifier(case.public, case.truth, mode="infrastructure"),
    )
    unknown = PrivateSWEBenchTerminalEvaluator(
        case.public,
        case.truth,
        _SyntheticVerifier(case.public, case.truth, mode="unknown"),
    )
    request = _request(
        case.public,
        termination=RolloutTermination.COMPLETED,
        submission={"patch": "public candidate"},
    )

    with pytest.raises(TerminalEvaluatorError) as captured:
        asyncio.run(infra.evaluate(request))
    assert PRIVATE_CANARY not in str(captured.value)
    with pytest.raises(UnknownVerifierError):
        asyncio.run(unknown.evaluate(request))


def test_submit_patch_is_terminal_and_step_limit_is_pinned() -> None:
    public = _public_case(max_steps=1)
    backend = _SyntheticWorkspace(public)
    environment = SWEBenchWorkspaceEnvironment(public, backend)
    submitted = asyncio.run(
        environment.execute(
            _tool("submit_patch", {"patch": "public patch"}),
            step_index=1,
        )
    )
    assert submitted.terminal_submission == {"patch": "public patch"}
    with pytest.raises(ValueError):
        asyncio.run(
            environment.execute(
                _tool("read", {"path": "src/public.py"}),
                step_index=2,
            )
        )
