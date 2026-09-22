"""Typed preflight receipts for Protocol 11 benchmark deployments."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.experiments.protocol_v11 import ACTIVE_BENCHMARKS_V11, BenchmarkV11


@dataclass(frozen=True, slots=True)
class ProtocolV11DeploymentReceipt:
    benchmark: BenchmarkV11
    environment_profile: str
    evaluator_profile: str
    preflight_passed: bool

    def __post_init__(self) -> None:
        if not self.environment_profile.strip() or not self.evaluator_profile.strip():
            raise ValueError("deployment profiles must be non-empty")


def require_all_deployments(
    receipts: tuple[ProtocolV11DeploymentReceipt, ...],
) -> None:
    if tuple(item.benchmark for item in receipts) != ACTIVE_BENCHMARKS_V11:
        raise ValueError("Protocol 11 deployment receipts differ from the current catalog")
    if not all(item.preflight_passed for item in receipts):
        raise RuntimeError("Protocol 11 evaluator/environment preflight is incomplete")


__all__ = ["ProtocolV11DeploymentReceipt", "require_all_deployments"]
