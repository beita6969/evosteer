"""Trusted Protocol 10 adapters for static, rubric, OJ, and AppWorld outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from skillev.contracts import JsonValue, normalize_json
from skillev.experiments.protocol_v10 import BenchmarkV10
from skillev.rollout import (
    NoTerminalSubmission,
    TerminalEvaluationRequest,
    TerminalEvaluator,
    TerminalEvaluatorError,
)

from .protocol_v10_evaluator import ProtocolV10NativeResult
from .static import PrivateStaticBenchmarkCase, StaticScoringRule
from .terminal_inputs import submitted_value


def _submitted_answer(request: TerminalEvaluationRequest) -> str | None:
    if isinstance(request.evaluation_input, NoTerminalSubmission):
        return None
    submitted = normalize_json(submitted_value(request))
    if not isinstance(submitted, dict) or set(submitted) != {"answer"}:
        raise TerminalEvaluatorError("completion has incompatible fields")
    answer = submitted["answer"]
    if type(answer) is not str:
        raise TerminalEvaluatorError("completion answer must be text")
    return answer


@dataclass(frozen=True, slots=True)
class ProtocolV10StaticBackend:
    """Official QA/integer scoring with separate native reward and success."""

    case: PrivateStaticBenchmarkCase
    benchmark: BenchmarkV10
    verifier_version: str
    environment_id: str | None = None

    def __post_init__(self) -> None:
        if self.case.public.benchmark_id != self.benchmark.value:
            raise ValueError("static case and Protocol 10 benchmark differ")
        expected = {
            BenchmarkV10.HOTPOT_QA: StaticScoringRule.TOKEN_F1,
            BenchmarkV10.TRIVIA_QA: StaticScoringRule.TOKEN_F1,
            BenchmarkV10.AIME_2026: StaticScoringRule.INTEGER,
        }.get(self.benchmark)
        if expected is None or self.case.target.scoring_rule is not expected:
            raise ValueError("static backend does not implement this benchmark rule")
        if not self.verifier_version.strip():
            raise ValueError("static verifier version must be non-empty")
        if self.environment_id is not None and not self.environment_id.strip():
            raise ValueError("static environment override must be non-empty")

    async def evaluate_native(
        self,
        request: TerminalEvaluationRequest,
    ) -> ProtocolV10NativeResult:
        if request.task_id != self.case.public.task_id:
            raise TerminalEvaluatorError("terminal request reached another static case")
        answer = _submitted_answer(request)
        if answer is None:
            score = exact = 0.0
        else:
            score = self.case.target.score(answer)
            exact = self.case.target.exact_match(answer)
        fields: dict[str, JsonValue]
        if self.benchmark in {BenchmarkV10.HOTPOT_QA, BenchmarkV10.TRIVIA_QA}:
            fields = {"answer-exact-match": exact, "answer-f1": score}
        else:
            fields = {"accuracy": score}
        return ProtocolV10NativeResult(
            task_id=request.task_id,
            benchmark=self.benchmark,
            native_fields=fields,
            environment_id=self.environment_id or self.case.public.environment_id,
            verifier_version=self.verifier_version,
        )


@dataclass(frozen=True, slots=True)
class HealthBenchGrade:
    official_rubric_score: float
    triggered_negative_rubric_count: int
    grader_cost: dict[str, float] | None = None
    criterion_ledger: dict[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.official_rubric_score, bool)
            or not isinstance(self.official_rubric_score, int | float)
            or not math.isfinite(float(self.official_rubric_score))
        ):
            raise ValueError("HealthBench rubric score must be finite")
        if (
            type(self.triggered_negative_rubric_count) is not int
            or self.triggered_negative_rubric_count < 0
        ):
            raise ValueError("HealthBench negative-rubric count must be non-negative")


class HealthBenchOfficialGrader(Protocol):
    """Pinned simple-evals grader; rubric truth never crosses this boundary."""

    @property
    def verifier_version(self) -> str: ...

    async def grade(self, task_id: str, candidate_answer: str) -> HealthBenchGrade: ...


@dataclass(frozen=True, slots=True)
class HealthBenchNativeBackend:
    task_id: str
    environment_id: str
    grader: HealthBenchOfficialGrader

    async def evaluate_native(
        self,
        request: TerminalEvaluationRequest,
    ) -> ProtocolV10NativeResult:
        if request.task_id != self.task_id:
            raise TerminalEvaluatorError("terminal request reached another HealthBench case")
        answer = _submitted_answer(request)
        if answer is None:
            grade = HealthBenchGrade(0.0, 0)
        else:
            try:
                grade = await self.grader.grade(self.task_id, answer)
            except Exception as error:
                raise TerminalEvaluatorError("HealthBench grader failed") from error
        return ProtocolV10NativeResult(
            task_id=self.task_id,
            benchmark=BenchmarkV10.HEALTHBENCH,
            native_fields={
                "official-rubric-score": float(grade.official_rubric_score),
                "triggered-negative-rubric-count": grade.triggered_negative_rubric_count,
            },
            environment_id=self.environment_id,
            verifier_version=self.grader.verifier_version,
        )


@dataclass(frozen=True, slots=True)
class SpreadsheetBenchGrade:
    passed_case_count: int
    total_case_count: int

    def __post_init__(self) -> None:
        if type(self.passed_case_count) is not int or type(self.total_case_count) is not int:
            raise TypeError("spreadsheet case counts must be integers")
        if self.total_case_count <= 0 or not 0 <= self.passed_case_count <= self.total_case_count:
            raise ValueError("spreadsheet case counts are inconsistent")

    @property
    def all_cases_pass(self) -> bool:
        return self.passed_case_count == self.total_case_count


class SpreadsheetBenchOfficialOJ(Protocol):
    @property
    def verifier_version(self) -> str: ...

    async def grade(self, task_id: str, submitted_workbook: JsonValue) -> SpreadsheetBenchGrade: ...


@dataclass(frozen=True, slots=True)
class SpreadsheetBenchNativeBackend:
    task_id: str
    environment_id: str
    oj: SpreadsheetBenchOfficialOJ

    async def evaluate_native(
        self,
        request: TerminalEvaluationRequest,
    ) -> ProtocolV10NativeResult:
        if request.task_id != self.task_id:
            raise TerminalEvaluatorError("terminal request reached another spreadsheet case")
        if isinstance(request.evaluation_input, NoTerminalSubmission):
            passed = False
        else:
            try:
                grade = await self.oj.grade(self.task_id, submitted_value(request))
            except Exception as error:
                raise TerminalEvaluatorError("SpreadsheetBench official OJ failed") from error
            passed = grade.all_cases_pass
        return ProtocolV10NativeResult(
            task_id=self.task_id,
            benchmark=BenchmarkV10.SPREADSHEETBENCH,
            native_fields={"all-workbook-cases-pass": float(passed)},
            environment_id=self.environment_id,
            verifier_version=self.oj.verifier_version,
        )


@dataclass(frozen=True, slots=True)
class AppWorldTaskGrade:
    task_goal_completion: float
    task_success: bool
    scenario_id: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.task_goal_completion, bool)
            or not isinstance(self.task_goal_completion, int | float)
            or not math.isfinite(float(self.task_goal_completion))
            or not 0.0 <= self.task_goal_completion <= 1.0
        ):
            raise ValueError("AppWorld TGC must lie in [0, 1]")
        if type(self.task_success) is not bool or not self.scenario_id.strip():
            raise ValueError("AppWorld task grade is incomplete")


class AppWorldOfficialOutcome(Protocol):
    @property
    def verifier_version(self) -> str: ...

    async def grade(self, task_id: str) -> AppWorldTaskGrade: ...


@dataclass(frozen=True, slots=True)
class ExistingAppWorldTerminalEvaluatorNativeBackend:
    """Preserve official per-task TGC while adding its public scenario group.

    The official AppWorld process evaluator already owns the private task and
    returns the trusted task-goal-completion scalar.  Protocol 10 additionally
    needs the answer-free scenario identity so final evaluation can aggregate
    SGC only after every variation has completed.  The scenario is taken from
    the frozen public task, never inferred from evaluator completion order.
    """

    task_id: str
    scenario_id: str
    expected_scenario_task_count: int
    evaluator: TerminalEvaluator

    def __post_init__(self) -> None:
        if (
            not self.task_id.strip()
            or not self.scenario_id.strip()
            or type(self.expected_scenario_task_count) is not int
            or self.expected_scenario_task_count < 1
        ):
            raise ValueError("AppWorld task and scenario identities must be non-empty")
        if not callable(getattr(self.evaluator, "evaluate", None)):
            raise TypeError("AppWorld existing evaluator is incompatible")

    async def evaluate_native(
        self,
        request: TerminalEvaluationRequest,
    ) -> ProtocolV10NativeResult:
        if request.task_id != self.task_id:
            raise TerminalEvaluatorError("terminal request reached another AppWorld task")
        reward = await self.evaluator.evaluate(request)
        return ProtocolV10NativeResult(
            task_id=self.task_id,
            benchmark=BenchmarkV10.APPWORLD,
            native_fields={
                "official-task-success": float(reward.success),
                "scenario-expected-task-count": self.expected_scenario_task_count,
                "scenario-id": self.scenario_id,
                "task-goal-completion": reward.value,
            },
            environment_id=reward.environment_id,
            verifier_version=reward.verifier_version,
        )


@dataclass(frozen=True, slots=True)
class AppWorldNativeBackend:
    task_id: str
    scenario_id: str
    environment_id: str
    outcome: AppWorldOfficialOutcome

    async def evaluate_native(
        self,
        request: TerminalEvaluationRequest,
    ) -> ProtocolV10NativeResult:
        if request.task_id != self.task_id:
            raise TerminalEvaluatorError("terminal request reached another AppWorld case")
        if not self.scenario_id.strip():
            raise TerminalEvaluatorError("AppWorld scenario identity is unavailable")
        if isinstance(request.evaluation_input, NoTerminalSubmission):
            grade = AppWorldTaskGrade(0.0, False, self.scenario_id)
        else:
            try:
                grade = await self.outcome.grade(self.task_id)
            except Exception as error:
                raise TerminalEvaluatorError("AppWorld official evaluator failed") from error
        if grade.scenario_id != self.scenario_id:
            raise TerminalEvaluatorError("AppWorld evaluator returned another scenario")
        return ProtocolV10NativeResult(
            task_id=self.task_id,
            benchmark=BenchmarkV10.APPWORLD,
            native_fields={
                "official-task-success": float(grade.task_success),
                "scenario-id": grade.scenario_id,
                "task-goal-completion": float(grade.task_goal_completion),
            },
            environment_id=self.environment_id,
            verifier_version=self.outcome.verifier_version,
        )


@dataclass(frozen=True, slots=True)
class AppWorldScenarioAggregate:
    scenario_id: str
    task_count: int
    average_task_goal_completion: float
    scenario_goal_completion: float


@dataclass(slots=True)
class AppWorldScenarioAccumulator:
    """Emit SGC only after every registered variation has been evaluated."""

    scenario_id: str
    expected_task_ids: tuple[str, ...]
    _grades: dict[str, AppWorldTaskGrade] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.scenario_id.strip() or not self.expected_task_ids:
            raise ValueError("AppWorld scenario registration is incomplete")
        if len(set(self.expected_task_ids)) != len(self.expected_task_ids):
            raise ValueError("AppWorld scenario task IDs must be unique")

    def record(self, task_id: str, grade: AppWorldTaskGrade) -> None:
        if task_id not in self.expected_task_ids or task_id in self._grades:
            raise ValueError("AppWorld scenario received an unexpected or duplicate task")
        if grade.scenario_id != self.scenario_id:
            raise ValueError("AppWorld task belongs to another scenario")
        self._grades[task_id] = grade

    def aggregate(self) -> AppWorldScenarioAggregate:
        if set(self._grades) != set(self.expected_task_ids):
            raise RuntimeError("AppWorld SGC requires the complete scenario group")
        ordered = tuple(self._grades[task_id] for task_id in self.expected_task_ids)
        return AppWorldScenarioAggregate(
            scenario_id=self.scenario_id,
            task_count=len(ordered),
            average_task_goal_completion=math.fsum(
                float(item.task_goal_completion) for item in ordered
            )
            / len(ordered),
            scenario_goal_completion=float(all(item.task_success for item in ordered)),
        )


__all__ = [
    "AppWorldNativeBackend",
    "AppWorldOfficialOutcome",
    "AppWorldScenarioAccumulator",
    "AppWorldScenarioAggregate",
    "AppWorldTaskGrade",
    "ExistingAppWorldTerminalEvaluatorNativeBackend",
    "HealthBenchGrade",
    "HealthBenchNativeBackend",
    "HealthBenchOfficialGrader",
    "ProtocolV10StaticBackend",
    "SpreadsheetBenchGrade",
    "SpreadsheetBenchNativeBackend",
    "SpreadsheetBenchOfficialOJ",
]
