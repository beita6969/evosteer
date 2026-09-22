"""Small, public known-truth oracle for pipeline debugging only.

The planted specification and caller-supplied seed are intentionally public,
so results from this module are never admissible as formal evaluation evidence.
Formal generators and their seeds live in the private environment package.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import StrEnum


class DebugEndpointFamily(StrEnum):
    GAUSSIAN = "GAUSSIAN"
    BERNOULLI_LOGIT = "BERNOULLI_LOGIT"
    LOG_NORMAL = "LOG_NORMAL"
    POISSON_LOG = "POISSON_LOG"


@dataclass(frozen=True, slots=True)
class DebugEndpointSpec:
    endpoint_id: str
    family: DebugEndpointFamily
    baseline: float
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.endpoint_id:
            raise ValueError("debug endpoint identity is required")
        if not all(math.isfinite(value) for value in (self.baseline, self.scale)):
            raise ValueError("debug endpoint parameters must be finite")
        if self.scale <= 0:
            raise ValueError("debug endpoint scale must be positive")


@dataclass(frozen=True, slots=True)
class DebugOperator:
    operator_id: str
    activation_probability: float
    adherence_probability: float
    endpoint_effects: tuple[tuple[str, float], ...]
    context_effects: tuple[tuple[str, str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.operator_id:
            raise ValueError("debug operator identity is required")
        if not 0 <= self.activation_probability <= 1:
            raise ValueError("activation probability must lie in [0, 1]")
        if not 0 <= self.adherence_probability <= 1:
            raise ValueError("adherence probability must lie in [0, 1]")
        if self.endpoint_effects != tuple(sorted(self.endpoint_effects)):
            raise ValueError("debug endpoint effects must be sorted")
        if self.context_effects != tuple(sorted(self.context_effects)):
            raise ValueError("debug context effects must be sorted")
        if not all(math.isfinite(item[-1]) for item in self.endpoint_effects):
            raise ValueError("debug endpoint effects must be finite")
        if not all(math.isfinite(item[-1]) for item in self.context_effects):
            raise ValueError("debug context effects must be finite")

    def effect(self, endpoint_id: str, context: str) -> float:
        main = dict(self.endpoint_effects).get(endpoint_id, 0.0)
        contextual = {
            (endpoint, label): value for endpoint, label, value in self.context_effects
        }.get((endpoint_id, context), 0.0)
        return main + contextual


@dataclass(frozen=True, slots=True)
class DebugInteraction:
    left_operator: str
    right_operator: str
    endpoint_effects: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.left_operator >= self.right_operator:
            raise ValueError("interaction operators must be canonically ordered")
        if self.endpoint_effects != tuple(sorted(self.endpoint_effects)):
            raise ValueError("interaction endpoint effects must be sorted")


@dataclass(frozen=True, slots=True)
class DebugFactorDiagnostic:
    operator_id: str
    requested: bool
    activated: bool
    adhered: bool


@dataclass(frozen=True, slots=True)
class DebugOracleDraw:
    endpoint_values: tuple[tuple[str, float], ...]
    factor_diagnostics: tuple[DebugFactorDiagnostic, ...]

    def value(self, endpoint_id: str) -> float:
        try:
            return dict(self.endpoint_values)[endpoint_id]
        except KeyError as error:
            raise KeyError(f"unknown debug endpoint: {endpoint_id}") from error

    @property
    def terminal_reward(self) -> float:
        """Return a bounded terminal reward for exercising TTB plumbing."""

        if "success" in dict(self.endpoint_values):
            return min(1.0, max(0.0, self.value("success")))
        quality = self.value("quality")
        return 1.0 / (1.0 + math.exp(-quality))


@dataclass(frozen=True, slots=True)
class DebugSyntheticOracle:
    endpoints: tuple[DebugEndpointSpec, ...]
    operators: tuple[DebugOperator, ...]
    interactions: tuple[DebugInteraction, ...]
    contexts: tuple[str, ...]
    formal_evaluation: bool = False

    def __post_init__(self) -> None:
        if self.formal_evaluation:
            raise ValueError("the public debug oracle cannot be used for formal evaluation")
        if not self.endpoints or not self.operators:
            raise ValueError("debug oracle requires endpoints and operators")
        if len({item.endpoint_id for item in self.endpoints}) != len(self.endpoints):
            raise ValueError("debug endpoint identities must be unique")
        if len({item.operator_id for item in self.operators}) != len(self.operators):
            raise ValueError("debug operator identities must be unique")
        if self.contexts != tuple(sorted(set(self.contexts))):
            raise ValueError("debug contexts must be unique and sorted")

    def draw(
        self,
        *,
        assignment: tuple[tuple[str, int], ...],
        context: str,
        seed: int,
    ) -> DebugOracleDraw:
        if context not in self.contexts:
            raise KeyError("unknown debug context")
        requested = dict(assignment)
        known = {item.operator_id for item in self.operators}
        if set(requested) != known or any(level not in {0, 1} for level in requested.values()):
            raise ValueError("assignment must give one binary level for every debug operator")
        if assignment != tuple(sorted(assignment)):
            raise ValueError("debug assignment must be sorted")

        rng = random.Random(seed)  # noqa: S311 - deterministic public debug fixture
        diagnostics: list[DebugFactorDiagnostic] = []
        effective: dict[str, int] = {}
        by_id = {item.operator_id: item for item in self.operators}
        for operator_id, level in assignment:
            operator = by_id[operator_id]
            activated = bool(level) and rng.random() < operator.activation_probability
            adhered = activated and rng.random() < operator.adherence_probability
            effective[operator_id] = int(adhered)
            diagnostics.append(DebugFactorDiagnostic(operator_id, bool(level), activated, adhered))

        values: list[tuple[str, float]] = []
        for endpoint in self.endpoints:
            predictor = endpoint.baseline
            predictor += sum(
                effective[item.operator_id] * item.effect(endpoint.endpoint_id, context)
                for item in self.operators
            )
            for interaction in self.interactions:
                predictor += (
                    effective[interaction.left_operator]
                    * effective[interaction.right_operator]
                    * dict(interaction.endpoint_effects).get(endpoint.endpoint_id, 0.0)
                )
            values.append(
                (
                    endpoint.endpoint_id,
                    _draw_endpoint(endpoint.family, predictor, endpoint.scale, rng),
                )
            )
        return DebugOracleDraw(tuple(sorted(values)), tuple(diagnostics))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _poisson(rng: random.Random, rate: float) -> int:
    if rate <= 0:
        return 0
    if rate > 40:
        return max(0, round(rng.gauss(rate, math.sqrt(rate))))
    threshold = math.exp(-rate)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def _draw_endpoint(
    family: DebugEndpointFamily,
    predictor: float,
    scale: float,
    rng: random.Random,
) -> float:
    if family is DebugEndpointFamily.GAUSSIAN:
        return rng.gauss(predictor, scale)
    if family is DebugEndpointFamily.BERNOULLI_LOGIT:
        return float(rng.random() < _sigmoid(predictor))
    if family is DebugEndpointFamily.LOG_NORMAL:
        return math.exp(rng.gauss(predictor, scale))
    if family is DebugEndpointFamily.POISSON_LOG:
        return float(_poisson(rng, math.exp(predictor)))
    raise AssertionError(f"unhandled debug endpoint family: {family}")


def default_debug_oracle() -> DebugSyntheticOracle:
    """Known-truth fixture with main, contextual, and interaction effects."""

    operators = (
        DebugOperator(
            "contextual",
            0.9,
            0.9,
            (("quality", 0.0), ("success", 0.0)),
            (("quality", "physical", -0.4), ("quality", "social", 0.7)),
        ),
        DebugOperator(
            "negative",
            0.9,
            0.9,
            (("quality", -0.8), ("success", -0.7)),
        ),
        DebugOperator(
            "positive",
            0.9,
            0.9,
            (("quality", 0.9), ("success", 0.8)),
        ),
        DebugOperator(
            "synergy-a",
            0.9,
            0.9,
            (("quality", 0.1), ("success", 0.0)),
        ),
        DebugOperator(
            "synergy-b",
            0.9,
            0.9,
            (("quality", 0.1), ("success", 0.0)),
        ),
    )
    return DebugSyntheticOracle(
        endpoints=(
            DebugEndpointSpec("cost", DebugEndpointFamily.LOG_NORMAL, 0.0, 0.15),
            DebugEndpointSpec("failures", DebugEndpointFamily.POISSON_LOG, -0.3, 1.0),
            DebugEndpointSpec("quality", DebugEndpointFamily.GAUSSIAN, 0.0, 0.25),
            DebugEndpointSpec("success", DebugEndpointFamily.BERNOULLI_LOGIT, 0.0, 1.0),
        ),
        operators=operators,
        interactions=(
            DebugInteraction(
                "synergy-a",
                "synergy-b",
                (("quality", 0.8), ("success", 0.6)),
            ),
        ),
        contexts=("physical", "social"),
    )


__all__ = [
    "DebugEndpointFamily",
    "DebugEndpointSpec",
    "DebugFactorDiagnostic",
    "DebugInteraction",
    "DebugOperator",
    "DebugOracleDraw",
    "DebugSyntheticOracle",
    "default_debug_oracle",
]
