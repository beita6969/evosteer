"""Answer-blind code extraction and terminal outcome taxonomy for Protocol 14."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from skillev.evaluation.direct_baseline.parsing import ParseStatus, parse_python_source


class CodeExtractionStatus(StrEnum):
    VALID = "valid"
    EMPTY_FINAL = "empty-final"
    REASONING_ONLY = "reasoning-only"
    INCOMPLETE_FENCE = "incomplete-fence"
    AMBIGUOUS_FENCES = "ambiguous-fences"
    SYNTAX_ERROR = "syntax-error"
    MISSING_REQUIRED_SYMBOL = "missing-required-symbol"


@dataclass(frozen=True, slots=True)
class EvalPlusProtocolProfile:
    source_repository: str
    source_revision: str
    package_version: str
    mbpp_plus_dataset_version: str
    mbpp_plus_task_count: int
    humaneval_plus_dataset_version: str
    humaneval_plus_task_count: int
    candidate_count_per_task: int
    success_rule: str
    maximum_memory_bytes: int
    minimum_time_limit_seconds: float
    ground_truth_time_limit_factor: float
    workers: int
    total_timeout_seconds: int

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.source_repository,
                self.source_revision,
                self.package_version,
                self.mbpp_plus_dataset_version,
                self.humaneval_plus_dataset_version,
            )
        ):
            raise ValueError("EvalPlus source identity is incomplete")
        if (
            self.package_version != "0.2.0"
            or self.mbpp_plus_dataset_version != "v0.2.0"
            or self.mbpp_plus_task_count != 378
            or self.humaneval_plus_dataset_version != "v0.1.9"
            or self.humaneval_plus_task_count != 164
            or self.candidate_count_per_task != 1
            or self.success_rule != "base-and-plus"
            or self.maximum_memory_bytes != 4 * 1024**3
        ):
            raise ValueError("EvalPlus formal protocol differs from the frozen source lane")
        if (
            self.minimum_time_limit_seconds <= 0
            or self.ground_truth_time_limit_factor <= 0
            or self.workers <= 0
            or self.total_timeout_seconds <= 0
        ):
            raise ValueError("EvalPlus sandbox controls must be positive")


def load_evalplus_protocol_profile(path: Path) -> EvalPlusProtocolProfile:
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict) or set(root) != {
        "format",
        "source_repository",
        "source_revision",
        "package_version",
        "mbpp_plus_dataset_version",
        "mbpp_plus_task_count",
        "humaneval_plus_dataset_version",
        "humaneval_plus_task_count",
        "candidate_count_per_task",
        "success_rule",
        "sandbox",
    }:
        raise ValueError("EvalPlus protocol config fields differ")
    if root["format"] != "skillev-protocol14-evalplus@1":
        raise ValueError("EvalPlus protocol config format differs")
    sandbox = root["sandbox"]
    if not isinstance(sandbox, dict) or set(sandbox) != {
        "process_per_task",
        "maximum_memory_bytes",
        "minimum_time_limit_seconds",
        "ground_truth_time_limit_factor",
        "temporary_working_directory",
        "network_disabled_by_reliability_guard",
        "workers",
        "total_timeout_seconds",
    }:
        raise ValueError("EvalPlus sandbox config fields differ")
    required_true = (
        sandbox["process_per_task"],
        sandbox["temporary_working_directory"],
        sandbox["network_disabled_by_reliability_guard"],
    )
    if any(value is not True for value in required_true):
        raise ValueError("EvalPlus sandbox isolation controls must be enabled")
    return EvalPlusProtocolProfile(
        source_repository=_text(root["source_repository"], "source repository"),
        source_revision=_text(root["source_revision"], "source revision"),
        package_version=_text(root["package_version"], "package version"),
        mbpp_plus_dataset_version=_text(root["mbpp_plus_dataset_version"], "MBPP+ version"),
        mbpp_plus_task_count=_integer(root["mbpp_plus_task_count"], "MBPP+ count"),
        humaneval_plus_dataset_version=_text(
            root["humaneval_plus_dataset_version"], "HumanEval+ version"
        ),
        humaneval_plus_task_count=_integer(root["humaneval_plus_task_count"], "HumanEval+ count"),
        candidate_count_per_task=_integer(root["candidate_count_per_task"], "candidate count"),
        success_rule=_text(root["success_rule"], "success rule"),
        maximum_memory_bytes=_integer(sandbox["maximum_memory_bytes"], "memory limit"),
        minimum_time_limit_seconds=_number(
            sandbox["minimum_time_limit_seconds"], "minimum time limit"
        ),
        ground_truth_time_limit_factor=_number(
            sandbox["ground_truth_time_limit_factor"], "time limit factor"
        ),
        workers=_integer(sandbox["workers"], "workers"),
        total_timeout_seconds=_integer(sandbox["total_timeout_seconds"], "deadline"),
    )


@dataclass(frozen=True, slots=True)
class CodeExtractionResult:
    status: CodeExtractionStatus
    source: str | None
    required_symbols: tuple[str, ...]
    present_symbols: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.status is CodeExtractionStatus.VALID


def classify_python_completion(
    *,
    final_text: str,
    reasoning_text: str | None,
    required_symbols: tuple[str, ...],
) -> CodeExtractionResult:
    """Classify one completion without tests, repair, or reasoning-channel salvage."""

    if not final_text.strip():
        status = (
            CodeExtractionStatus.REASONING_ONLY
            if reasoning_text is not None and reasoning_text.strip()
            else CodeExtractionStatus.EMPTY_FINAL
        )
        return CodeExtractionResult(status, None, required_symbols, ())
    if final_text.count("```") % 2:
        return CodeExtractionResult(
            CodeExtractionStatus.INCOMPLETE_FENCE, None, required_symbols, ()
        )
    parsed = parse_python_source(final_text)
    if parsed.status is ParseStatus.AMBIGUOUS:
        return CodeExtractionResult(
            CodeExtractionStatus.AMBIGUOUS_FENCES, None, required_symbols, ()
        )
    if parsed.status is not ParseStatus.EXTRACTED or parsed.value is None:
        return CodeExtractionResult(CodeExtractionStatus.EMPTY_FINAL, None, required_symbols, ())
    try:
        tree = ast.parse(parsed.value)
    except SyntaxError:
        return CodeExtractionResult(
            CodeExtractionStatus.SYNTAX_ERROR, parsed.value, required_symbols, ()
        )
    present = tuple(
        dict.fromkeys(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        )
    )
    if any(symbol not in present for symbol in required_symbols):
        return CodeExtractionResult(
            CodeExtractionStatus.MISSING_REQUIRED_SYMBOL,
            parsed.value,
            required_symbols,
            present,
        )
    return CodeExtractionResult(CodeExtractionStatus.VALID, parsed.value, required_symbols, present)


class CodeOutcomeKind(StrEnum):
    SCORED_SUCCESS = "scored-success"
    SCORED_FAILURE = "scored-failure"
    CANDIDATE_INVALID = "candidate-invalid"
    GENERATION_INFRASTRUCTURE = "generation-infrastructure"
    SCORER_INFRASTRUCTURE = "scorer-infrastructure"


@dataclass(frozen=True, slots=True)
class CodeTerminalOutcome:
    kind: CodeOutcomeKind
    extraction_status: CodeExtractionStatus | None
    base_passed: bool | None
    plus_passed: bool | None

    def __post_init__(self) -> None:
        if self.kind is CodeOutcomeKind.SCORED_SUCCESS:
            if self.extraction_status is not CodeExtractionStatus.VALID:
                raise ValueError("successful code outcome requires valid extraction")
            if self.base_passed is not True or self.plus_passed is not True:
                raise ValueError("successful code outcome requires Base and Plus pass")
        elif self.kind is CodeOutcomeKind.SCORED_FAILURE:
            if self.extraction_status is not CodeExtractionStatus.VALID:
                raise ValueError("scored code failure requires valid extraction")
            if self.base_passed is None or self.plus_passed is None:
                raise ValueError("scored code failure requires Base and Plus verdicts")
            if self.base_passed and self.plus_passed:
                raise ValueError("passing code cannot be a scored failure")
        elif self.kind is CodeOutcomeKind.CANDIDATE_INVALID:
            if self.extraction_status in {None, CodeExtractionStatus.VALID}:
                raise ValueError("invalid candidate requires an extraction failure")
            if self.base_passed is not None or self.plus_passed is not None:
                raise ValueError("invalid candidate must not receive test verdicts")
        elif any(
            value is not None
            for value in (self.extraction_status, self.base_passed, self.plus_passed)
        ):
            raise ValueError("infrastructure outcome must not masquerade as a verdict")

    @property
    def definitive(self) -> bool:
        return self.kind not in {
            CodeOutcomeKind.GENERATION_INFRASTRUCTURE,
            CodeOutcomeKind.SCORER_INFRASTRUCTURE,
        }

    @property
    def success(self) -> bool:
        return self.kind is CodeOutcomeKind.SCORED_SUCCESS


def project_evalplus_outcomes(
    *,
    full_task_ids: Sequence[str],
    panel_task_ids: Sequence[str],
    outcomes: Mapping[str, CodeTerminalOutcome],
    expected_full_count: int = 378,
    expected_panel_count: int = 128,
) -> tuple[CodeTerminalOutcome, ...]:
    """Project the frozen panel from the same full-run outputs, without selection."""

    if (
        len(full_task_ids) != expected_full_count
        or len(set(full_task_ids)) != expected_full_count
        or len(panel_task_ids) != expected_panel_count
        or len(set(panel_task_ids)) != expected_panel_count
    ):
        raise ValueError("EvalPlus full or panel task identity is incomplete")
    if set(outcomes) != set(full_task_ids):
        raise ValueError("EvalPlus outcomes do not exactly cover the full source population")
    if not set(panel_task_ids).issubset(full_task_ids):
        raise ValueError("EvalPlus panel escapes the full source population")
    return tuple(outcomes[task_id] for task_id in panel_task_ids)


def evalplus_composite_percent(humaneval_plus: float, mbpp_plus: float) -> float:
    """Published cross-check only; never a standalone MBPP+ target."""

    if any(
        isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 100
        for value in (humaneval_plus, mbpp_plus)
    ):
        raise ValueError("EvalPlus component percentages must lie in [0, 100]")
    return (float(humaneval_plus) + float(mbpp_plus)) / 2.0


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    return float(value)


__all__ = [
    "CodeExtractionResult",
    "CodeExtractionStatus",
    "CodeOutcomeKind",
    "CodeTerminalOutcome",
    "EvalPlusProtocolProfile",
    "classify_python_completion",
    "evalplus_composite_percent",
    "load_evalplus_protocol_profile",
    "project_evalplus_outcomes",
]
