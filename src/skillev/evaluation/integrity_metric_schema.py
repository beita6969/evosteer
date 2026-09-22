"""Current six-IID and six-OOD metrics, plus explicitly historical schemas.

The schema is deliberately about native evaluator evidence, not the bounded
``TerminalReward`` projection used by trajectory balance.  In particular,
HealthBench keeps its per-item raw score here, including legitimate negative
values; clipping happens only after the complete planned panel is aggregated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

from .external_judge_policy import (
    DIRECT_OMNI_JUDGE_VERIFIER,
    GATEWAY_OMNI_JUDGE_VERIFIER,
    LEGACY_OMNI_JUDGE_METRIC,
    LEGACY_OMNI_JUDGE_VERIFIER,
    OMNI_JUDGE_METRIC,
    OMNI_JUDGE_VERIFIER,
)
from .livemedbench import DIRECT_VERIFIER as LIVEMEDBENCH_DIRECT_VERIFIER
from .livemedbench import GATEWAY_VERIFIER as LIVEMEDBENCH_GATEWAY_VERIFIER
from .livemedbench import METRIC as LIVEMEDBENCH_METRIC
from .livemedbench import VERIFIER as LIVEMEDBENCH_VERIFIER

if TYPE_CHECKING:
    from .integrity_results import NativeScore


UNSPECIFIED_VERIFIER: Final = "unspecified"


@dataclass(frozen=True, slots=True)
class MetricSchema:
    """One benchmark's native primary metric and required secondary evidence."""

    primary: str
    lower: float | None
    upper: float
    binary: bool
    required_secondary: tuple[str, ...]
    binary_secondary: tuple[str, ...] = ()


METRIC_SCHEMAS: Final[Mapping[str, MetricSchema]] = MappingProxyType(
    {
        "hotpotqa": MetricSchema(
            "answer-f1",
            0.0,
            1.0,
            False,
            ("answer-exact-match", "answer-f1"),
            ("answer-exact-match",),
        ),
        "triviaqa": MetricSchema(
            "answer-f1",
            0.0,
            1.0,
            False,
            ("answer-exact-match", "answer-f1"),
            ("answer-exact-match",),
        ),
        "aime-2026": MetricSchema("accuracy", 0.0, 1.0, True, ()),
        "healthbench": MetricSchema(
            "qwen-local-rubric-score",
            None,
            1.0,
            False,
            ("triggered-negative-rubric-count",),
        ),
        "webshop": MetricSchema("native-reward", 0.0, 1.0, False, ("success",), ("success",)),
        "alfworld": MetricSchema("success", 0.0, 1.0, True, ("success",), ("success",)),
        "mbpp-plus": MetricSchema(
            "base-plus-pass-at-1",
            0.0,
            1.0,
            True,
            ("base-pass", "plus-pass"),
            ("base-pass", "plus-pass"),
        ),
        "humaneval": MetricSchema("pass-at-1", 0.0, 1.0, True, ()),
        "musique": MetricSchema(
            "answer-f1",
            0.0,
            1.0,
            False,
            ("answer-exact-match", "answer-f1"),
            ("answer-exact-match",),
        ),
        "nq-open": MetricSchema(
            "answer-exact-match",
            0.0,
            1.0,
            True,
            ("answer-exact-match", "answer-f1"),
            ("answer-exact-match",),
        ),
        "omni-math": MetricSchema(OMNI_JUDGE_METRIC, 0.0, 1.0, True, ()),
        "gpqa-diamond-bioorganic": MetricSchema("accuracy", 0.0, 1.0, True, ()),
        "math-hard": MetricSchema("accuracy", 0.0, 1.0, True, ()),
        "livemedbench": MetricSchema(
            LIVEMEDBENCH_METRIC, None, 1.0, False, ("triggered-negative-rubric-count",)
        ),
        "scienceworld": MetricSchema(
            "native-final-score",
            None,
            1.0,
            False,
            ("success", "zero-clipped-learning-reward"),
            ("success",),
        ),
        "livecodebench": MetricSchema("pass-at-1", 0.0, 1.0, True, ()),
        "apps-introductory": MetricSchema("pass-at-1", 0.0, 1.0, True, ()),
    }
)


# This is a production contract, not a best-effort display label.  Runtimes
# freeze a copy in their evaluator controls and pass that concrete value to
# aggregation; no score may use an inferred or unspecified verifier identity.
NATIVE_VERIFIER_VERSIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "hotpotqa": "hotpotqa-native-em-f1@integrity1",
        "triviaqa": "triviaqa-native-em-f1@integrity1",
        "aime-2026": "aime-2026-native-integrity@1",
        "healthbench": "healthbench-qwen3.5-9b-sglang-temperature-0.5@2",
        "webshop": "webshop-native-integrity@1",
        "alfworld": "alfworld-native-integrity@1",
        "mbpp-plus": "mbpp-plus-native-default-time-independent-lanes@2",
        "humaneval": "humaneval-public-context@2-original-isolated@2",
        "musique": "musique-ans-official-answer-em-f1@ood2",
        "nq-open": "nq-open-fid-unicode-answer-em-f1@ood2",
        "omni-math": OMNI_JUDGE_VERIFIER,
        "gpqa-diamond-bioorganic": "gpqa-diamond-frozen-choice-exact@ood3",
        "math-hard": "math-hard-level5-math-verify-0.9.0-final-answer@ood1",
        "livemedbench": LIVEMEDBENCH_VERIFIER,
        "scienceworld": "scienceworld-official-native-final-score@ood3",
        "livecodebench": "livecodebench-official-configured-timeout@ood2",
        "apps-introductory": "apps-official-script-entry-configured-timeout@ood3",
    }
)


def require_expected_verifier(expected_verifier: str) -> str:
    """Accept only a concrete, frozen native verifier identity."""

    if (
        type(expected_verifier) is not str
        or not expected_verifier.strip()
        or expected_verifier == UNSPECIFIED_VERIFIER
    ):
        raise ValueError("native aggregation requires an explicit verifier version")
    return expected_verifier


def _require_finite_number(value: object, *, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    numeric = float(cast(int | float, value))
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _validate_fraction(value: object, *, label: str, lower: float, upper: float) -> float:
    numeric = _require_finite_number(value, label=label)
    if not lower <= numeric <= upper:
        raise ValueError(f"{label} is outside its declared fraction range")
    return numeric


def _validate_secondary_metrics(score: NativeScore, spec: MetricSchema) -> None:
    secondary = score.secondary_metrics
    if not isinstance(secondary, dict) or set(secondary) != set(spec.required_secondary):
        raise ValueError("native secondary metrics differ from the declared schema")
    for name in spec.required_secondary:
        value = secondary[name]
        if name == "triggered-negative-rubric-count":
            numeric = _require_finite_number(value, label=name)
            if numeric < 0 or not numeric.is_integer():
                raise ValueError("negative rubric count must be a nonnegative integer")
        else:
            numeric = _validate_fraction(value, label=name, lower=0.0, upper=1.0)
            if name in spec.binary_secondary and numeric not in (0.0, 1.0):
                raise ValueError("binary secondary metric must be zero or one")


def _validate_cross_metric_relations(score: NativeScore) -> None:
    secondary = score.secondary_metrics
    if score.benchmark in {"hotpotqa", "triviaqa", "musique", "nq-open"}:
        if score.value != secondary[score.metric]:
            raise ValueError("QA primary differs from its declared native metric")
    elif score.benchmark in {"webshop", "scienceworld"}:
        if secondary["success"] != float(score.value == 1.0):
            raise ValueError("interactive success must equal native reward == 1")
        if (
            score.benchmark == "scienceworld"
            and score.metric == "native-final-score"
            and secondary["zero-clipped-learning-reward"] != max(0.0, score.value)
        ):
            raise ValueError("ScienceWorld learning reward differs from its explicit projection")
    elif score.benchmark == "alfworld":
        if score.value != secondary["success"]:
            raise ValueError("ALFWorld primary must equal native success")
    elif score.benchmark == "mbpp-plus":
        expected = float(secondary["base-pass"] == 1.0 and secondary["plus-pass"] == 1.0)
        if score.value != expected:
            raise ValueError("MBPP+ primary must require both Base and Plus passes")


def validate_native_score(
    score: NativeScore,
    *,
    expected_verifier: str | None = None,
) -> None:
    """Validate one fully declared native score before it can be reported.

    ``expected_verifier`` is supplied by the frozen evaluator controls at an
    aggregation boundary.  Construction still requires the benchmark's own
    declared version so unbound records cannot be persisted as new results.
    """

    try:
        spec = METRIC_SCHEMAS[score.benchmark]
        declared_verifier = NATIVE_VERIFIER_VERSIONS[score.benchmark]
    except KeyError as error:
        raise ValueError("native score benchmark has no declared metric schema") from error
    if score.benchmark == "omni-math" and score.verifier_version in {
        LEGACY_OMNI_JUDGE_VERIFIER,
        DIRECT_OMNI_JUDGE_VERIFIER,
        GATEWAY_OMNI_JUDGE_VERIFIER,
    }:
        metric = (
            LEGACY_OMNI_JUDGE_METRIC
            if score.verifier_version == LEGACY_OMNI_JUDGE_VERIFIER
            else OMNI_JUDGE_METRIC
        )
        spec = MetricSchema(metric, 0.0, 1.0, True, ())
        declared_verifier = score.verifier_version
    if score.benchmark == "livemedbench" and score.verifier_version in {
        LIVEMEDBENCH_DIRECT_VERIFIER,
        LIVEMEDBENCH_GATEWAY_VERIFIER,
    }:
        declared_verifier = score.verifier_version
    if (
        score.benchmark == "gpqa-diamond-bioorganic"
        and score.verifier_version == "gpqa-diamond-frozen-choice-exact@ood1"
    ):
        # Old choice-parser scores stay readable; frozen comparisons still
        # reject mixing them with the new Markdown/labelled-option condition.
        declared_verifier = score.verifier_version
    if (
        score.benchmark == "scienceworld"
        and score.verifier_version == "scienceworld-official-final-score-with-raw@ood2"
    ):
        # Historical clipped results remain readable, never relabelled as raw.
        spec = MetricSchema("native-reward", 0.0, 1.0, False, ("success",), ("success",))
        declared_verifier = score.verifier_version
    if (
        score.benchmark == "humaneval"
        and score.verifier_version == "humaneval-native-public-scaffold@2"
    ):
        # Historical scores remain explicitly legacy; frozen comparison still rejects a mixture.
        declared_verifier = score.verifier_version
    from .healthbench_luna_profile import (
        DIRECT_METRIC,
        DIRECT_VERIFIER,
        GATEWAY_VERIFIER,
        METRIC,
        VERIFIER,
    )

    if score.benchmark == "healthbench" and score.verifier_version in {
        VERIFIER,
        DIRECT_VERIFIER,
        GATEWAY_VERIFIER,
    }:
        metric = DIRECT_METRIC if score.verifier_version == DIRECT_VERIFIER else METRIC
        spec = MetricSchema(metric, None, 1.0, False, ("triggered-negative-rubric-count",))
        declared_verifier = score.verifier_version
    if score.metric != spec.primary:
        raise ValueError("native score metric differs from the declared primary metric")
    if score.verifier_version != declared_verifier:
        raise ValueError("native score verifier differs from its declared benchmark verifier")
    if expected_verifier is not None and score.verifier_version != require_expected_verifier(
        expected_verifier
    ):
        raise ValueError("native score verifier differs from the frozen evaluator")

    value = _require_finite_number(score.value, label="native score")
    if value > spec.upper or (spec.lower is not None and value < spec.lower):
        raise ValueError("native score is outside its declared units")
    if spec.binary and value not in (0.0, 1.0):
        raise ValueError("native binary metric must be zero or one")
    _validate_secondary_metrics(score, spec)
    _validate_cross_metric_relations(score)

    if score.benchmark in {"healthbench", "livemedbench"}:
        if type(score.grader_used) is not bool:
            raise ValueError("Rubric scores must state whether the rubric grader was used")
        if score.grader_used is False and (
            score.value != 0.0 or score.secondary_metrics["triggered-negative-rubric-count"] != 0.0
        ):
            raise ValueError("an ungraded rubric candidate must have explicit zero evidence")
    elif score.grader_used is not None:
        raise ValueError("only rubric-based native scores may report rubric-grader use")


def aggregate_secondary_metrics(scores: Sequence[NativeScore]) -> dict[str, float]:
    """Return complete secondary means without intersecting away missing columns."""

    if not scores:
        raise ValueError("cannot aggregate secondary metrics from an empty panel")
    expected_columns = set(scores[0].secondary_metrics)
    if any(set(score.secondary_metrics) != expected_columns for score in scores[1:]):
        raise ValueError("native secondary metric columns are incomplete")
    return {
        name: sum(score.secondary_metrics[name] for score in scores) / len(scores)
        for name in sorted(expected_columns)
    }


__all__ = [
    "METRIC_SCHEMAS",
    "NATIVE_VERIFIER_VERSIONS",
    "UNSPECIFIED_VERIFIER",
    "MetricSchema",
    "aggregate_secondary_metrics",
    "require_expected_verifier",
    "validate_native_score",
]
