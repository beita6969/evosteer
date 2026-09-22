"""Protocol 10 terminal projection over trusted native benchmark evidence.

This module is the single boundary where a benchmark's native result becomes
the scalar TTB reward and Bernoulli observation required by ``idea.tex``.
Targets, rubrics, tests, and per-item evidence stay behind the injected
private backend and are never added to the rollout transcript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from skillev.contracts import JsonValue, SuccessRule, TerminalReward, normalize_json
from skillev.experiments.protocol_v10 import (
    BenchmarkProtocolV10,
    BenchmarkV10,
    ProtocolV10Error,
)
from skillev.rollout import TerminalEvaluationRequest, TerminalEvaluator, TerminalEvaluatorError


@dataclass(frozen=True, slots=True)
class ProtocolV10NativeResult:
    """Answer-free trusted metrics for one completed evaluation attempt."""

    task_id: str
    benchmark: BenchmarkV10
    native_fields: dict[str, JsonValue]
    environment_id: str
    verifier_version: str

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.environment_id.strip():
            raise ValueError("native result identities must be non-empty")
        if not self.verifier_version.strip():
            raise ValueError("native result verifier version must be non-empty")
        normalized = normalize_json(self.native_fields)
        if not isinstance(normalized, dict) or not normalized:
            raise ValueError("native result fields must be a non-empty JSON object")
        object.__setattr__(self, "native_fields", normalized)


class ProtocolV10NativeBackend(Protocol):
    """Private benchmark implementation that may inspect verifier truth."""

    async def evaluate_native(
        self,
        request: TerminalEvaluationRequest,
    ) -> ProtocolV10NativeResult: ...


@dataclass(frozen=True, slots=True)
class ExistingTerminalEvaluatorNativeBackend:
    """Expose a binary/native-score official evaluator under Protocol 10 names."""

    evaluator: TerminalEvaluator
    benchmark: BenchmarkV10
    reward_field: str
    success_field: str
    source_environment_id: str
    environment_id: str

    def __post_init__(self) -> None:
        if not callable(getattr(self.evaluator, "evaluate", None)):
            raise TypeError("existing terminal evaluator is incompatible")
        if not self.reward_field.strip() or not self.success_field.strip():
            raise ValueError("Protocol 10 native field names must be non-empty")
        if not self.source_environment_id.strip() or not self.environment_id.strip():
            raise ValueError("Protocol 10 environment identities must be non-empty")

    async def evaluate_native(
        self,
        request: TerminalEvaluationRequest,
    ) -> ProtocolV10NativeResult:
        reward = await self.evaluator.evaluate(request)
        if reward.environment_id != self.source_environment_id:
            raise TerminalEvaluatorError("adapted evaluator returned another source environment")
        return ProtocolV10NativeResult(
            task_id=request.task_id,
            benchmark=self.benchmark,
            native_fields={
                self.reward_field: reward.value,
                self.success_field: float(reward.success),
            },
            environment_id=self.environment_id,
            verifier_version=reward.verifier_version,
        )


@dataclass(frozen=True, slots=True)
class ProtocolV10TerminalEvaluator:
    """Apply the frozen benchmark-specific R and Y projections exactly once."""

    protocol: BenchmarkProtocolV10
    backend: ProtocolV10NativeBackend

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, BenchmarkProtocolV10):
            raise TypeError("Protocol 10 evaluator requires a benchmark protocol")
        if not callable(getattr(self.backend, "evaluate_native", None)):
            raise TypeError("Protocol 10 evaluator backend is incompatible")

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        try:
            result = await self.backend.evaluate_native(request)
        except TerminalEvaluatorError:
            raise
        except Exception as error:
            raise TerminalEvaluatorError("Protocol 10 trusted evaluator backend failed") from error
        if result.task_id != request.task_id:
            raise TerminalEvaluatorError("Protocol 10 evaluator returned another task")
        if result.benchmark is not self.protocol.benchmark:
            raise TerminalEvaluatorError("Protocol 10 evaluator returned another benchmark")
        try:
            reward = self.protocol.reward_projection.project(result.native_fields)
            success = self.protocol.success_projection.project(result.native_fields)
        except ProtocolV10Error as error:
            raise TerminalEvaluatorError("Protocol 10 native evidence is incomplete") from error
        payload = normalize_json(
            {
                "benchmark_id": result.benchmark.value,
                "native_fields": result.native_fields,
            }
        )
        if not isinstance(payload, dict):  # pragma: no cover - literal mapping
            raise TypeError("Protocol 10 native payload must be an object")
        return TerminalReward(
            value=reward,
            success=success,
            success_rule=SuccessRule.TRUSTED_NATIVE_PROJECTION,
            success_threshold=None,
            native_metric_name=self.protocol.reward_projection.source,
            native_payload=payload,
            environment_id=result.environment_id,
            verifier_version=result.verifier_version,
        )


__all__ = [
    "ExistingTerminalEvaluatorNativeBackend",
    "ProtocolV10NativeBackend",
    "ProtocolV10NativeResult",
    "ProtocolV10TerminalEvaluator",
]
