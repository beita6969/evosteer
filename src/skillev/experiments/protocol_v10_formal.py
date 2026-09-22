"""Frozen scientific run shape and method identities for Protocol 10.

This module defines experiment science only.  Deployment-only rollout
concurrency and physical GPU bindings deliberately live outside this identity.
The execution gate was opened only after the private populations, evaluators,
fail-closed distributed composition root, resume/OOM evidence, nine-domain
coverage, and bounded 22-step lifecycles were complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan

from .protocol_v10 import (
    PROTOCOL_V10_FORMAT,
    PROTOCOL_V10_SEED,
    ActiveBenchmarkProtocolV10,
    FormalMethodV10,
    ProtocolV10Error,
)

PROTOCOL_V10_FORMAL_EXPERIMENT_FORMAT: Final = "skillev-protocol-v10-formal-experiment@1"
PROTOCOL_V10_BATCH_SIZE: Final = 16
PROTOCOL_V10_PHASE_SEARCH_STEPS: Final = 272
PROTOCOL_V10_CLOSURE_STEPS: Final = 16
PROTOCOL_V10_MAXIMUM_CYCLES: Final = 2
PROTOCOL_V10_TOTAL_STEPS: Final = 288
PROTOCOL_V10_TOTAL_EPISODES: Final = 4_608


class FormalApplicationV10(StrEnum):
    """Closed executable semantics; none is an alias for another method."""

    EXACT_SKILLFLOW_BASELINE = "exact-skillflow-baseline"
    BAYESIAN_IMPROVE_FULL = "bayesian-improve-full"
    BAYESIAN_IMPROVE_NO_CALIBRATION = "bayesian-improve-no-calibration"


@dataclass(frozen=True, slots=True)
class FormalMethodBindingV10:
    method: FormalMethodV10
    application: FormalApplicationV10

    def __post_init__(self) -> None:
        expected = {
            FormalMethodV10.SKILLFLOW_BASELINE: FormalApplicationV10.EXACT_SKILLFLOW_BASELINE,
            FormalMethodV10.BAYESIAN_IMPROVE_FULL: FormalApplicationV10.BAYESIAN_IMPROVE_FULL,
            FormalMethodV10.BAYESIAN_IMPROVE_NO_CALIBRATION: (
                FormalApplicationV10.BAYESIAN_IMPROVE_NO_CALIBRATION
            ),
        }
        if self.application is not expected[self.method]:
            raise ProtocolV10Error("formal method is bound to another application semantics")


@dataclass(frozen=True, slots=True)
class ProtocolV10FormalExperimentSpec:
    protocol_format: str
    seed: int
    batch_size: int
    phase_search_steps: int
    closure_steps: int
    maximum_cycles: int
    total_steps: int
    total_episodes: int
    methods: tuple[FormalMethodBindingV10, ...]
    state_flow_estimator: str
    skill_outcome_estimator: str
    skill_flow_weight: str
    closure_semantics: str
    authoring_backend: str
    executable: bool
    format: str = PROTOCOL_V10_FORMAL_EXPERIMENT_FORMAT

    def __post_init__(self) -> None:
        if self.format != PROTOCOL_V10_FORMAL_EXPERIMENT_FORMAT:
            raise ProtocolV10Error("formal experiment format is unsupported")
        if self.protocol_format != PROTOCOL_V10_FORMAT or self.seed != PROTOCOL_V10_SEED:
            raise ProtocolV10Error("formal experiment differs from Protocol 10")
        if (
            self.batch_size != PROTOCOL_V10_BATCH_SIZE
            or self.phase_search_steps != PROTOCOL_V10_PHASE_SEARCH_STEPS
            or self.closure_steps != PROTOCOL_V10_CLOSURE_STEPS
            or self.maximum_cycles != PROTOCOL_V10_MAXIMUM_CYCLES
            or self.total_steps != PROTOCOL_V10_TOTAL_STEPS
            or self.total_episodes != PROTOCOL_V10_TOTAL_EPISODES
        ):
            raise ProtocolV10Error("formal experiment training shape is not frozen Protocol 10")
        if self.phase_search_steps + self.closure_steps != self.total_steps:
            raise ProtocolV10Error("formal experiment step partitions are inconsistent")
        if self.batch_size * self.total_steps != self.total_episodes:
            raise ProtocolV10Error("formal experiment does not consume all 4,608 episodes")
        if tuple(binding.method for binding in self.methods) != tuple(FormalMethodV10):
            raise ProtocolV10Error("formal method bindings differ from Protocol 10")
        if len({binding.application for binding in self.methods}) != len(self.methods):
            raise ProtocolV10Error("formal methods must have distinct executable semantics")
        expected_semantics = (
            "single-observed-history-prefix@1",
            "trajectory-terminal-success-shared-by-invoked-skills@1",
            "mean-normalized-state-flow@1",
            "post-treatment-closure-tail-no-new-phase@1",
            "external-sglang-base-model-authoring@1",
        )
        actual_semantics = (
            self.state_flow_estimator,
            self.skill_outcome_estimator,
            self.skill_flow_weight,
            self.closure_semantics,
            self.authoring_backend,
        )
        if actual_semantics != expected_semantics:
            raise ProtocolV10Error("formal method semantics differ from the frozen interpretation")

    @property
    def run_plan(self) -> ExactAttemptRunPlan:
        return ExactAttemptRunPlan(
            phase_search_steps=self.phase_search_steps,
            closure_steps=self.closure_steps,
            maximum_cycles=self.maximum_cycles,
        )

    def binding(self, method: FormalMethodV10) -> FormalMethodBindingV10:
        if not isinstance(method, FormalMethodV10):
            raise TypeError("formal method must be a Protocol 10 method")
        return next(binding for binding in self.methods if binding.method is method)

    def require_execution_ready(self, protocol: ActiveBenchmarkProtocolV10) -> None:
        if protocol.format != self.protocol_format or protocol.seed != self.seed:
            raise ProtocolV10Error("formal experiment and benchmark protocol disagree")
        protocol.require_execution_ready()
        if not self.executable:
            raise ProtocolV10Error("Protocol 10 formal experiment runtime gate is not open")


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ProtocolV10Error(f"{label} must be a mapping")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolV10Error(f"{label} must be non-empty text")
    return value


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ProtocolV10Error(f"{label} must be an integer")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ProtocolV10Error(f"{label} must be boolean")
    return value


def parse_protocol_v10_formal_experiment(value: object) -> ProtocolV10FormalExperimentSpec:
    root = _mapping(value, label="formal experiment")
    training = _mapping(root.get("training"), label="formal training")
    semantics = _mapping(root.get("method_semantics"), label="method semantics")
    gate = _mapping(root.get("execution_gate"), label="formal execution gate")
    method_rows = root.get("methods")
    if not isinstance(method_rows, list):
        raise ProtocolV10Error("formal methods must be a sequence")
    methods = tuple(
        FormalMethodBindingV10(
            method=FormalMethodV10(_text(row.get("method"), label="formal method")),
            application=FormalApplicationV10(
                _text(row.get("application"), label="formal application")
            ),
        )
        for row in (_mapping(item, label="formal method binding") for item in method_rows)
    )
    return ProtocolV10FormalExperimentSpec(
        format=_text(root.get("format"), label="formal experiment format"),
        protocol_format=_text(root.get("protocol_format"), label="benchmark protocol format"),
        seed=_integer(root.get("seed"), label="formal seed"),
        batch_size=_integer(training.get("batch_size"), label="batch size"),
        phase_search_steps=_integer(training.get("phase_search_steps"), label="phase-search steps"),
        closure_steps=_integer(training.get("closure_steps"), label="closure steps"),
        maximum_cycles=_integer(training.get("maximum_cycles"), label="maximum cycles"),
        total_steps=_integer(training.get("total_steps"), label="total steps"),
        total_episodes=_integer(training.get("total_episodes"), label="total episodes"),
        methods=methods,
        state_flow_estimator=_text(
            semantics.get("state_flow_estimator"), label="state-flow estimator"
        ),
        skill_outcome_estimator=_text(
            semantics.get("skill_outcome_estimator"), label="skill outcome estimator"
        ),
        skill_flow_weight=_text(semantics.get("skill_flow_weight"), label="skill flow weight"),
        closure_semantics=_text(semantics.get("closure_semantics"), label="closure semantics"),
        authoring_backend=_text(semantics.get("authoring_backend"), label="authoring backend"),
        executable=_boolean(gate.get("executable"), label="formal execution gate"),
    )


def load_protocol_v10_formal_experiment(path: Path) -> ProtocolV10FormalExperimentSpec:
    import yaml

    return parse_protocol_v10_formal_experiment(yaml.safe_load(path.read_text(encoding="utf-8")))


__all__ = [
    "PROTOCOL_V10_BATCH_SIZE",
    "PROTOCOL_V10_CLOSURE_STEPS",
    "PROTOCOL_V10_FORMAL_EXPERIMENT_FORMAT",
    "PROTOCOL_V10_MAXIMUM_CYCLES",
    "PROTOCOL_V10_PHASE_SEARCH_STEPS",
    "PROTOCOL_V10_TOTAL_EPISODES",
    "PROTOCOL_V10_TOTAL_STEPS",
    "FormalApplicationV10",
    "FormalMethodBindingV10",
    "ProtocolV10FormalExperimentSpec",
    "load_protocol_v10_formal_experiment",
    "parse_protocol_v10_formal_experiment",
]
