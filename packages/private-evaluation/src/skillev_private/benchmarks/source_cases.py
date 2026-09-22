"""Verifier-only case types for benchmarks without a safe static-text scorer.

These objects deliberately have no public serialization method.  Their
``public`` member is the sole model-facing projection; executable tests,
worked solutions, gold browser actions, explanations, and canaries remain in
the private evaluation wheel or private run memory.
"""

from __future__ import annotations

from dataclasses import dataclass

from skillev.benchmarks import BenchmarkPublicItem

from .static import PrivateStaticBenchmarkCase, PrivateStaticTarget, StaticScoringRule


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if type(value) is not str or "\x00" in value:
        raise ValueError(f"{field} must be text without NUL")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _same_identity(public: BenchmarkPublicItem, task_id: str) -> None:
    if public.task_id != task_id:
        raise ValueError("public benchmark item and private target identities differ")


@dataclass(frozen=True, slots=True)
class PrivateGPQADiamondTarget:
    """Private GPQA verifier material, including the dataset canary."""

    task_id: str
    correct_option: str
    explanation: str
    canary_string: str

    def __post_init__(self) -> None:
        _text(self.task_id, field="GPQA task_id")
        if self.correct_option not in {"A", "B", "C", "D"}:
            raise ValueError("GPQA correct_option must be A, B, C, or D")
        _text(self.explanation, field="GPQA explanation")
        _text(self.canary_string, field="GPQA canary_string")

    def to_static_target(self) -> PrivateStaticTarget:
        """Return only the answer material needed by the trusted evaluator."""

        return PrivateStaticTarget(
            task_id=self.task_id,
            scoring_rule=StaticScoringRule.OPTION,
            accepted_answers=(self.correct_option,),
        )


@dataclass(frozen=True, slots=True)
class PrivateGPQADiamondCase:
    public: BenchmarkPublicItem
    target: PrivateGPQADiamondTarget

    def __post_init__(self) -> None:
        _same_identity(self.public, self.target.task_id)

    def to_static_case(self) -> PrivateStaticBenchmarkCase:
        return PrivateStaticBenchmarkCase(self.public, self.target.to_static_target())


@dataclass(frozen=True, slots=True)
class PrivateHumanEvalTarget:
    """HumanEval solution and tests; never execute these in-process."""

    task_id: str
    canonical_solution: str
    test: str
    entry_point: str

    def __post_init__(self) -> None:
        _text(self.task_id, field="HumanEval task_id")
        _text(self.canonical_solution, field="HumanEval canonical_solution")
        _text(self.test, field="HumanEval test")
        _text(self.entry_point, field="HumanEval entry_point")


@dataclass(frozen=True, slots=True)
class PrivateHumanEvalCase:
    public: BenchmarkPublicItem
    target: PrivateHumanEvalTarget

    def __post_init__(self) -> None:
        _same_identity(self.public, self.target.task_id)


@dataclass(frozen=True, slots=True)
class PrivateMathTarget:
    """MATH worked solution and its strictly parsed final boxed expression."""

    task_id: str
    solution: str
    boxed_answer: str

    def __post_init__(self) -> None:
        _text(self.task_id, field="MATH task_id")
        _text(self.solution, field="MATH solution")
        _text(self.boxed_answer, field="MATH boxed_answer")


@dataclass(frozen=True, slots=True)
class PrivateMathCase:
    public: BenchmarkPublicItem
    target: PrivateMathTarget

    def __post_init__(self) -> None:
        _same_identity(self.public, self.target.task_id)


@dataclass(frozen=True, slots=True)
class PrivateMind2WebStepTarget:
    """One gold Mind2Web step over an answer-free candidate projection."""

    task_id: str
    annotation_id: str
    action_uid: str
    operation: str
    original_operation: str
    value: str
    positive_backend_node_ids: tuple[str, ...]
    action_repr: str

    def __post_init__(self) -> None:
        for field in ("task_id", "annotation_id", "action_uid", "operation"):
            _text(getattr(self, field), field=f"Mind2Web {field}")
        _text(self.original_operation, field="Mind2Web original_operation")
        _text(self.value, field="Mind2Web value", allow_empty=True)
        _text(self.action_repr, field="Mind2Web action_repr")
        if self.operation not in {"CLICK", "TYPE", "SELECT"}:
            raise ValueError("Mind2Web operation must be CLICK, TYPE, or SELECT")
        for node_id in self.positive_backend_node_ids:
            _text(node_id, field="Mind2Web positive backend_node_id")
        if len(set(self.positive_backend_node_ids)) != len(self.positive_backend_node_ids):
            raise ValueError("Mind2Web positive backend node identities must be unique")


@dataclass(frozen=True, slots=True)
class PrivateMind2WebStepCase:
    public: BenchmarkPublicItem
    target: PrivateMind2WebStepTarget

    def __post_init__(self) -> None:
        _same_identity(self.public, self.target.task_id)


__all__ = [
    "PrivateGPQADiamondCase",
    "PrivateGPQADiamondTarget",
    "PrivateHumanEvalCase",
    "PrivateHumanEvalTarget",
    "PrivateMathCase",
    "PrivateMathTarget",
    "PrivateMind2WebStepCase",
    "PrivateMind2WebStepTarget",
]
