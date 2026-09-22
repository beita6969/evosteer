"""Small standard-library estimators shared by aggregate experiment metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

_CONFIDENCE_LEVEL = 0.95
_NORMAL_95 = 1.959963984540054
_MOMENT_TOLERANCE = 1e-12


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite number")
    return normalized


def _nonnegative_count(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _nonempty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A finite closed 95% confidence interval."""

    lower: float
    upper: float
    confidence_level: float = _CONFIDENCE_LEVEL

    def __post_init__(self) -> None:
        lower = _finite(self.lower, field="confidence interval lower bound")
        upper = _finite(self.upper, field="confidence interval upper bound")
        level = _finite(self.confidence_level, field="confidence level")
        if lower > upper:
            raise ValueError("confidence interval lower bound cannot exceed upper bound")
        if level != _CONFIDENCE_LEVEL:
            raise ValueError("aggregate experiment intervals are fixed at 95%")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "confidence_level", level)


@dataclass(frozen=True, slots=True)
class BinomialRate:
    """A count-backed rate with a Wilson score interval."""

    numerator: int
    denominator: int
    value: float
    interval: ConfidenceInterval

    def __post_init__(self) -> None:
        numerator = _nonnegative_count(self.numerator, field="rate numerator")
        denominator = _nonnegative_count(self.denominator, field="rate denominator")
        if denominator == 0 or numerator > denominator:
            raise ValueError("binomial rate requires 0 <= numerator <= positive denominator")
        value = _finite(self.value, field="rate value")
        if value != numerator / denominator:
            raise ValueError("rate value must equal numerator / denominator")
        if not isinstance(self.interval, ConfidenceInterval):
            raise TypeError("interval must be a ConfidenceInterval")
        if not self.interval.lower <= value <= self.interval.upper:
            raise ValueError("rate value must lie inside its confidence interval")
        object.__setattr__(self, "value", value)

    @classmethod
    def from_counts(cls, numerator: int, denominator: int) -> BinomialRate:
        """Construct the rate and its bounded Wilson interval."""

        numerator = _nonnegative_count(numerator, field="rate numerator")
        denominator = _nonnegative_count(denominator, field="rate denominator")
        if denominator == 0 or numerator > denominator:
            raise ValueError("binomial rate requires 0 <= numerator <= positive denominator")
        value = numerator / denominator
        z_squared = _NORMAL_95**2
        scale = 1.0 + z_squared / denominator
        center = (value + z_squared / (2.0 * denominator)) / scale
        radius = (
            _NORMAL_95
            * math.sqrt(value * (1.0 - value) / denominator + z_squared / (4.0 * denominator**2))
            / scale
        )
        return cls(
            numerator=numerator,
            denominator=denominator,
            value=value,
            interval=ConfidenceInterval(
                lower=(0.0 if numerator == 0 else max(0.0, center - radius)),
                upper=(1.0 if numerator == denominator else min(1.0, center + radius)),
            ),
        )


@dataclass(frozen=True, slots=True)
class MeanEstimate:
    """A mean and normal 95% interval reconstructed from aggregate moments."""

    sample_count: int
    mean: float
    interval: ConfidenceInterval

    def __post_init__(self) -> None:
        count = _nonnegative_count(self.sample_count, field="mean sample_count")
        if count == 0:
            raise ValueError("a mean estimate requires at least one sample")
        mean = _finite(self.mean, field="mean")
        if not isinstance(self.interval, ConfidenceInterval):
            raise TypeError("interval must be a ConfidenceInterval")
        if not self.interval.lower <= mean <= self.interval.upper:
            raise ValueError("mean must lie inside its confidence interval")
        object.__setattr__(self, "mean", mean)

    @classmethod
    def from_moments(
        cls,
        *,
        count: int,
        total: float,
        squared_total: float,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
    ) -> MeanEstimate:
        """Construct a bounded estimate from aggregate-only moments."""

        return _bounded_mean_from_moments(
            count=count,
            total=total,
            squared_total=squared_total,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )


def _bounded_mean_from_moments(
    *,
    count: int,
    total: float,
    squared_total: float,
    lower_bound: float,
    upper_bound: float,
) -> MeanEstimate:
    if count <= 0:
        raise ValueError("a bounded mean requires a positive count")
    mean = total / count
    if not lower_bound <= mean <= upper_bound:
        raise ValueError("aggregate mean lies outside its declared bounds")
    minimum_squared_total = total * total / count
    if squared_total + _MOMENT_TOLERANCE < minimum_squared_total:
        raise ValueError("aggregate squared total violates the moment lower bound")
    centered = squared_total - minimum_squared_total
    if centered < 0.0:
        if centered < -_MOMENT_TOLERANCE:
            raise ValueError("aggregate moments imply a negative variance")
        centered = 0.0
    sample_variance = centered / (count - 1) if count > 1 else 0.0
    standard_error = math.sqrt(sample_variance / count)
    return MeanEstimate(
        sample_count=count,
        mean=mean,
        interval=ConfidenceInterval(
            lower=max(lower_bound, mean - _NORMAL_95 * standard_error),
            upper=min(upper_bound, mean + _NORMAL_95 * standard_error),
        ),
    )


@dataclass(frozen=True, slots=True)
class DifferenceEstimate:
    """After-minus-before difference with an interval from the two estimates."""

    value: float
    interval: ConfidenceInterval

    def __post_init__(self) -> None:
        value = _finite(self.value, field="difference")
        if not isinstance(self.interval, ConfidenceInterval):
            raise TypeError("interval must be a ConfidenceInterval")
        if not self.interval.lower <= value <= self.interval.upper:
            raise ValueError("difference must lie inside its confidence interval")
        object.__setattr__(self, "value", value)

    @classmethod
    def between(cls, after: MeanEstimate, before: MeanEstimate) -> DifferenceEstimate:
        """Return the aggregate after-minus-before estimate."""

        return _difference(after, before)


def _difference(after: MeanEstimate, before: MeanEstimate) -> DifferenceEstimate:
    return DifferenceEstimate(
        value=after.mean - before.mean,
        interval=ConfidenceInterval(
            lower=after.interval.lower - before.interval.upper,
            upper=after.interval.upper - before.interval.lower,
        ),
    )


__all__ = [
    "BinomialRate",
    "ConfidenceInterval",
    "DifferenceEstimate",
    "MeanEstimate",
]
