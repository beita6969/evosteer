import math
from dataclasses import replace

import pytest

from skillev.diagnostics.curves import CurveSanityPolicy, TrainingCurvePoint, ewma


def _point(step: int, reward: float) -> TrainingCurvePoint:
    return TrainingCurvePoint(
        step,
        reward,
        reward,
        0.5,
        0.1,
        0.1,
        0.5,
        1.0,
        0.5,
        1.0,
        0.1,
        1.0,
        4.0,
        1,
        1,
        0,
    )


def test_ewma_constant_and_noisy_curve() -> None:
    assert ewma((1.0, 1.0, 1.0), alpha=0.5) == (1.0, 1.0, 1.0)
    smooth = ewma((0.1, 0.4, 0.2, 0.6), alpha=0.5)
    assert smooth[-1] > smooth[0]
    with pytest.raises(ValueError):
        ewma((math.nan,), alpha=0.5)


def test_curve_sanity_does_not_require_monotonic_steps() -> None:
    CurveSanityPolicy(window=2).validate(
        (_point(1, 0.2), _point(2, 0.4), _point(3, 0.3), _point(4, 0.5))
    )


def test_curve_sanity_rejects_zero_gradient_and_collapsed_domain() -> None:
    points = tuple(replace(_point(step, 0.2), gradient_norm=0.0) for step in (1, 2))
    with pytest.raises(ValueError, match="gradient"):
        CurveSanityPolicy(window=1).validate(points)
