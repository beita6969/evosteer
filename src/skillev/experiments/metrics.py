"""Answer-free aggregate metrics for SKILLEV experiments.

The inputs in this module are already-aggregated observations.  They contain no
task text, answer, trajectory identifier, or per-example payload, and none of
the result objects provides a persistence format.  This keeps benchmark truth
on the evaluator side while still making the scientific summaries reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Final

from skillev.contracts import EvolutionActionType
from skillev.runtime import RetrievalInclusionReason

from ._statistics import (
    _MOMENT_TOLERANCE,
    BinomialRate,
    ConfidenceInterval,
    DifferenceEstimate,
    MeanEstimate,
    _bounded_mean_from_moments,
    _difference,
    _finite,
    _nonempty_text,
    _nonnegative_count,
)


class SamplingPeriod(StrEnum):
    """Closed comparison sides for trajectory sampling."""

    BEFORE_TRAINING = "before-training"
    AFTER_TRAINING = "after-training"


@dataclass(frozen=True, slots=True)
class RewardBucket:
    """One pre-registered reward interval; only the final interval includes 1."""

    bucket_id: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        _nonempty_text(self.bucket_id, field="reward bucket_id")
        lower = _finite(self.lower, field="reward bucket lower")
        upper = _finite(self.upper, field="reward bucket upper")
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("reward bucket bounds must satisfy 0 <= lower < upper <= 1")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class RewardBucketScheme:
    """A contiguous, complete, pre-registered partition of ``[0, 1]``."""

    buckets: tuple[RewardBucket, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.buckets, tuple) or not self.buckets:
            raise ValueError("reward buckets must be a non-empty tuple")
        if any(not isinstance(bucket, RewardBucket) for bucket in self.buckets):
            raise TypeError("reward buckets contain an incompatible value")
        if len({bucket.bucket_id for bucket in self.buckets}) != len(self.buckets):
            raise ValueError("reward bucket IDs must be unique")
        if self.buckets[0].lower != 0.0 or self.buckets[-1].upper != 1.0:
            raise ValueError("reward buckets must cover the complete [0, 1] support")
        for previous, current in zip(self.buckets, self.buckets[1:], strict=False):
            if previous.upper != current.lower:
                raise ValueError("reward buckets must be ordered, contiguous, and non-overlapping")


FROZEN_REWARD_BUCKET_SCHEME: Final = RewardBucketScheme(
    (
        RewardBucket("lower-reward", 0.0, 0.5),
        RewardBucket("high-reward", 0.5, 1.0),
    )
)


@dataclass(frozen=True, slots=True)
class RewardBucketObservation:
    """Aggregate reward moments for one pre-registered bucket and period."""

    period: SamplingPeriod
    bucket_id: str
    sample_count: int
    reward_sum: float
    reward_squared_sum: float

    def __post_init__(self) -> None:
        if not isinstance(self.period, SamplingPeriod):
            raise TypeError("period must be a SamplingPeriod")
        _nonempty_text(self.bucket_id, field="reward observation bucket_id")
        count = _nonnegative_count(self.sample_count, field="reward sample_count")
        total = _finite(self.reward_sum, field="reward_sum")
        squared_total = _finite(self.reward_squared_sum, field="reward_squared_sum")
        if total < 0.0 or squared_total < 0.0:
            raise ValueError("reward aggregate moments cannot be negative")
        if count == 0 and (total != 0.0 or squared_total != 0.0):
            raise ValueError("an empty reward bucket must have zero moments")
        if count > 0:
            if total > count or squared_total > count:
                raise ValueError("reward moments cannot exceed the [0, 1] support")
            if squared_total + _MOMENT_TOLERANCE < total * total / count:
                raise ValueError("reward squared sum violates the moment lower bound")
        object.__setattr__(self, "reward_sum", total)
        object.__setattr__(self, "reward_squared_sum", squared_total)


@dataclass(frozen=True, slots=True)
class RewardBucketComparison:
    """Before/after aggregate quality for one reward bucket."""

    bucket: RewardBucket
    before_count: int
    after_count: int
    before_share: BinomialRate
    after_share: BinomialRate
    before_reward: MeanEstimate | None
    after_reward: MeanEstimate | None
    reward_change: DifferenceEstimate | None
    sample_share_change: float

    def __post_init__(self) -> None:
        if not isinstance(self.bucket, RewardBucket):
            raise TypeError("bucket must be a RewardBucket")
        _nonnegative_count(self.before_count, field="before_count")
        _nonnegative_count(self.after_count, field="after_count")
        share_change = _finite(self.sample_share_change, field="sample_share_change")
        if share_change != self.after_share.value - self.before_share.value:
            raise ValueError("sample_share_change must equal after share minus before share")
        if (self.before_reward is None) != (self.before_count == 0):
            raise ValueError("before_reward presence must match before_count")
        if (self.after_reward is None) != (self.after_count == 0):
            raise ValueError("after_reward presence must match after_count")
        if self.reward_change is not None and (
            self.before_reward is None or self.after_reward is None
        ):
            raise ValueError("reward_change requires means on both comparison sides")
        if (
            self.reward_change is None
            and self.before_reward is not None
            and self.after_reward is not None
        ):
            raise ValueError("reward_change is required when both bucket means exist")
        object.__setattr__(self, "sample_share_change", share_change)


@dataclass(frozen=True, slots=True)
class RewardBehaviorSummary:
    """Reward-proportional sampling behavior before and after training."""

    before_count: int
    after_count: int
    before_reward: MeanEstimate
    after_reward: MeanEstimate
    reward_change: DifferenceEstimate
    buckets: tuple[RewardBucketComparison, ...]


def reward_proportional_trajectory_behavior(
    scheme: RewardBucketScheme,
    observations: tuple[RewardBucketObservation, ...],
) -> RewardBehaviorSummary:
    """Compare aggregate trajectory quality over a pre-registered reward partition."""

    if not isinstance(scheme, RewardBucketScheme):
        raise TypeError("scheme must be a RewardBucketScheme")
    if not isinstance(observations, tuple):
        raise TypeError("observations must be a tuple")
    expected_keys = {
        (period, bucket.bucket_id) for period in SamplingPeriod for bucket in scheme.buckets
    }
    by_key: dict[tuple[SamplingPeriod, str], RewardBucketObservation] = {}
    bucket_by_id = {bucket.bucket_id: bucket for bucket in scheme.buckets}
    for observation in observations:
        if not isinstance(observation, RewardBucketObservation):
            raise TypeError("observations contain an incompatible value")
        key = (observation.period, observation.bucket_id)
        if key in by_key:
            raise ValueError("reward observations must contain one aggregate per period/bucket")
        bucket = bucket_by_id.get(observation.bucket_id)
        if bucket is None:
            raise ValueError("reward observation refers to an unregistered bucket")
        if observation.sample_count:
            mean = observation.reward_sum / observation.sample_count
            if not bucket.lower <= mean <= bucket.upper:
                raise ValueError(
                    "reward bucket aggregate mean lies outside its registered interval"
                )
        by_key[key] = observation
    if set(by_key) != expected_keys:
        raise ValueError("reward observations must cover every registered period/bucket pair")

    totals: dict[SamplingPeriod, tuple[int, float, float]] = {}
    for period in SamplingPeriod:
        period_observations = tuple(by_key[(period, bucket.bucket_id)] for bucket in scheme.buckets)
        count = sum(observation.sample_count for observation in period_observations)
        if count == 0:
            raise ValueError("each sampling period requires at least one trajectory")
        totals[period] = (
            count,
            math.fsum(observation.reward_sum for observation in period_observations),
            math.fsum(observation.reward_squared_sum for observation in period_observations),
        )

    before_count, before_sum, before_squared = totals[SamplingPeriod.BEFORE_TRAINING]
    after_count, after_sum, after_squared = totals[SamplingPeriod.AFTER_TRAINING]
    before_reward = _bounded_mean_from_moments(
        count=before_count,
        total=before_sum,
        squared_total=before_squared,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    after_reward = _bounded_mean_from_moments(
        count=after_count,
        total=after_sum,
        squared_total=after_squared,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    comparisons: list[RewardBucketComparison] = []
    for bucket in scheme.buckets:
        before = by_key[(SamplingPeriod.BEFORE_TRAINING, bucket.bucket_id)]
        after = by_key[(SamplingPeriod.AFTER_TRAINING, bucket.bucket_id)]
        before_bucket_mean = (
            _bounded_mean_from_moments(
                count=before.sample_count,
                total=before.reward_sum,
                squared_total=before.reward_squared_sum,
                lower_bound=bucket.lower,
                upper_bound=bucket.upper,
            )
            if before.sample_count
            else None
        )
        after_bucket_mean = (
            _bounded_mean_from_moments(
                count=after.sample_count,
                total=after.reward_sum,
                squared_total=after.reward_squared_sum,
                lower_bound=bucket.lower,
                upper_bound=bucket.upper,
            )
            if after.sample_count
            else None
        )
        before_share = BinomialRate.from_counts(before.sample_count, before_count)
        after_share = BinomialRate.from_counts(after.sample_count, after_count)
        comparisons.append(
            RewardBucketComparison(
                bucket=bucket,
                before_count=before.sample_count,
                after_count=after.sample_count,
                before_share=before_share,
                after_share=after_share,
                before_reward=before_bucket_mean,
                after_reward=after_bucket_mean,
                reward_change=(
                    _difference(after_bucket_mean, before_bucket_mean)
                    if before_bucket_mean is not None and after_bucket_mean is not None
                    else None
                ),
                sample_share_change=after_share.value - before_share.value,
            )
        )
    return RewardBehaviorSummary(
        before_count=before_count,
        after_count=after_count,
        before_reward=before_reward,
        after_reward=after_reward,
        reward_change=_difference(after_reward, before_reward),
        buckets=tuple(comparisons),
    )


@dataclass(frozen=True, slots=True)
class ProbabilityBinScheme:
    """Pre-registered reliability-bin edges spanning ``[0, 1]``."""

    edges: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.edges, tuple) or len(self.edges) < 2:
            raise ValueError("probability-bin edges must be a tuple with at least two values")
        normalized = tuple(_finite(edge, field="probability-bin edge") for edge in self.edges)
        if normalized[0] != 0.0 or normalized[-1] != 1.0:
            raise ValueError("probability bins must span [0, 1]")
        if any(left >= right for left, right in pairwise(normalized)):
            raise ValueError("probability-bin edges must be strictly increasing")
        object.__setattr__(self, "edges", normalized)


@dataclass(frozen=True, slots=True)
class CalibrationPredictionObservation:
    """One identity-free prequential prediction and its observed outcome."""

    posterior_mean: float
    lower_confidence_bound: float
    outcome: bool

    def __post_init__(self) -> None:
        mean = _finite(self.posterior_mean, field="posterior_mean")
        if not 0.0 < mean < 1.0:
            raise ValueError("posterior_mean must lie strictly inside (0, 1) for finite NLL")
        lower_bound = _finite(self.lower_confidence_bound, field="lower_confidence_bound")
        if type(self.outcome) is not bool:
            raise TypeError("calibration prediction outcome must be a boolean")
        object.__setattr__(self, "posterior_mean", mean)
        object.__setattr__(self, "lower_confidence_bound", lower_bound)


@dataclass(frozen=True, slots=True)
class CalibrationCellAssessment:
    """Identity-free validation evidence for one unique posterior cell."""

    mean_prequential_lcb: float
    validation_count: int
    success_count: int
    final_posterior_evidence: float

    def __post_init__(self) -> None:
        mean_lcb = _finite(self.mean_prequential_lcb, field="mean_prequential_lcb")
        validation_count = _nonnegative_count(
            self.validation_count,
            field="validation_count",
        )
        success_count = _nonnegative_count(self.success_count, field="success_count")
        evidence = _finite(
            self.final_posterior_evidence,
            field="final_posterior_evidence",
        )
        if validation_count == 0:
            raise ValueError("calibration cell assessment must contain validation evidence")
        if success_count > validation_count:
            raise ValueError("calibration cell successes cannot exceed validation count")
        if evidence < 0.0:
            raise ValueError("final posterior evidence cannot be negative")
        object.__setattr__(self, "mean_prequential_lcb", mean_lcb)
        object.__setattr__(self, "final_posterior_evidence", evidence)


@dataclass(frozen=True, slots=True)
class ReliabilityBinMetric:
    """One non-empty reliability-diagram bin."""

    lower: float
    upper: float
    prediction_count: int
    mean_predicted_probability: float
    empirical_success: BinomialRate
    absolute_gap: float


@dataclass(frozen=True, slots=True)
class BayesianCalibrationSummary:
    """Aggregate posterior reliability and context-cell coverage metrics."""

    possible_cell_count: int
    observed_cell_count: int
    prediction_count: int
    brier_score: float
    negative_log_likelihood: float
    expected_calibration_error: float
    lcb_coverage: BinomialRate
    unobserved_cell_rate: BinomialRate
    low_evidence_observed_rate: BinomialRate
    sparse_cell_rate: BinomialRate
    reliability: tuple[ReliabilityBinMetric, ...]


def _probability_bin_index(value: float, edges: tuple[float, ...]) -> int:
    for index, upper in enumerate(edges[1:]):
        if value < upper or (upper == 1.0 and value <= upper):
            return index
    raise ValueError("probability does not belong to the registered binning")


def bayesian_calibration_metrics(
    predictions: tuple[CalibrationPredictionObservation, ...],
    *,
    assessments: tuple[CalibrationCellAssessment, ...],
    possible_cell_count: int,
    low_evidence_threshold: float,
    reliability_bins: ProbabilityBinScheme,
) -> BayesianCalibrationSummary:
    """Compute prequential proper scores and unique-cell sparsity."""

    if not isinstance(predictions, tuple) or not predictions:
        raise ValueError("calibration predictions must be a non-empty tuple")
    if any(not isinstance(item, CalibrationPredictionObservation) for item in predictions):
        raise TypeError("calibration predictions contain an incompatible value")
    if not isinstance(assessments, tuple) or not assessments:
        raise ValueError("calibration cell assessments must be a non-empty tuple")
    if any(not isinstance(item, CalibrationCellAssessment) for item in assessments):
        raise TypeError("calibration cell assessments contain an incompatible value")
    possible_cells = _nonnegative_count(possible_cell_count, field="possible_cell_count")
    if possible_cells == 0 or possible_cells < len(assessments):
        raise ValueError("possible_cell_count must cover all assessed cells")
    evidence_threshold = _finite(low_evidence_threshold, field="low_evidence_threshold")
    if evidence_threshold < 0.0:
        raise ValueError("low_evidence_threshold cannot be negative")
    assessment_prediction_count = sum(item.validation_count for item in assessments)
    if assessment_prediction_count != len(predictions):
        raise ValueError("cell assessment counts must equal the prediction count")
    assessment_success_count = sum(item.success_count for item in assessments)
    if assessment_success_count != sum(item.outcome for item in predictions):
        raise ValueError("cell assessment successes must equal prediction outcomes")
    if not isinstance(reliability_bins, ProbabilityBinScheme):
        raise TypeError("reliability_bins must be a ProbabilityBinScheme")

    prediction_count = len(predictions)
    brier_total = math.fsum(
        (float(item.outcome) - item.posterior_mean) ** 2 for item in predictions
    )
    nll_total = math.fsum(
        -math.log(item.posterior_mean) if item.outcome else -math.log1p(-item.posterior_mean)
        for item in predictions
    )
    covered_cells = sum(
        item.success_count / item.validation_count >= item.mean_prequential_lcb
        for item in assessments
    )
    observed_cells = len(assessments)
    unobserved_cells = possible_cells - observed_cells
    low_evidence_cells = sum(
        item.final_posterior_evidence <= evidence_threshold for item in assessments
    )

    grouped: dict[int, list[CalibrationPredictionObservation]] = {}
    for item in predictions:
        index = _probability_bin_index(item.posterior_mean, reliability_bins.edges)
        grouped.setdefault(index, []).append(item)
    reliability: list[ReliabilityBinMetric] = []
    ece_terms: list[float] = []
    for index in sorted(grouped):
        items = grouped[index]
        bin_prediction_count = len(items)
        bin_successes = sum(item.outcome for item in items)
        mean_prediction = math.fsum(item.posterior_mean for item in items) / bin_prediction_count
        empirical = BinomialRate.from_counts(bin_successes, bin_prediction_count)
        gap = abs(mean_prediction - empirical.value)
        ece_terms.append(bin_prediction_count / prediction_count * gap)
        reliability.append(
            ReliabilityBinMetric(
                lower=reliability_bins.edges[index],
                upper=reliability_bins.edges[index + 1],
                prediction_count=bin_prediction_count,
                mean_predicted_probability=mean_prediction,
                empirical_success=empirical,
                absolute_gap=gap,
            )
        )
    return BayesianCalibrationSummary(
        possible_cell_count=possible_cells,
        observed_cell_count=observed_cells,
        prediction_count=prediction_count,
        brier_score=brier_total / prediction_count,
        negative_log_likelihood=nll_total / prediction_count,
        expected_calibration_error=math.fsum(ece_terms),
        lcb_coverage=BinomialRate.from_counts(covered_cells, observed_cells),
        unobserved_cell_rate=BinomialRate.from_counts(
            unobserved_cells,
            possible_cells,
        ),
        low_evidence_observed_rate=BinomialRate.from_counts(
            low_evidence_cells,
            observed_cells,
        ),
        sparse_cell_rate=BinomialRate.from_counts(
            unobserved_cells + low_evidence_cells,
            possible_cells,
        ),
        reliability=tuple(reliability),
    )


class EvolutionErrorKind(StrEnum):
    """Closed scientific interpretation of an erroneous Phi action."""

    RETAIN_REGRESSION = "retain-compress-regression"
    REFINE_INEFFECTIVE = "refine-ineffective"
    SPLIT_OVERSEGMENTED = "split-oversegmented"
    PRUNE_FALSE_POSITIVE = "prune-false-positive"
    GENERATE_UNUSED = "generate-unused"


_ERROR_KIND_BY_ACTION = {
    EvolutionActionType.RETAIN_COMPRESS: EvolutionErrorKind.RETAIN_REGRESSION,
    EvolutionActionType.REFINE: EvolutionErrorKind.REFINE_INEFFECTIVE,
    EvolutionActionType.SPLIT: EvolutionErrorKind.SPLIT_OVERSEGMENTED,
    EvolutionActionType.PRUNE: EvolutionErrorKind.PRUNE_FALSE_POSITIVE,
    EvolutionActionType.GENERATE: EvolutionErrorKind.GENERATE_UNUSED,
}


@dataclass(frozen=True, slots=True)
class EvolutionErrorObservation:
    """Aggregate evaluated and erroneous counts for one closed action type."""

    action_type: EvolutionActionType
    evaluated_count: int
    error_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.action_type, EvolutionActionType):
            raise TypeError("action_type must be an EvolutionActionType")
        evaluated = _nonnegative_count(self.evaluated_count, field="evaluated_count")
        errors = _nonnegative_count(self.error_count, field="error_count")
        if errors > evaluated:
            raise ValueError("error_count cannot exceed evaluated_count")


@dataclass(frozen=True, slots=True)
class EvolutionActionErrorMetric:
    """One action's aggregate error count and rate, if it was evaluated."""

    action_type: EvolutionActionType
    error_kind: EvolutionErrorKind
    evaluated_count: int
    error_count: int
    error_rate: BinomialRate | None


@dataclass(frozen=True, slots=True)
class EvolutionErrorSummary:
    """Aggregate error evolution rate across the five-action vocabulary."""

    evaluated_count: int
    error_count: int
    error_rate: BinomialRate
    by_action: tuple[EvolutionActionErrorMetric, ...]


def evolution_error_rates(
    observations: tuple[EvolutionErrorObservation, ...],
) -> EvolutionErrorSummary:
    """Compute per-action and overall erroneous evolution rates."""

    if not isinstance(observations, tuple) or not observations:
        raise ValueError("evolution observations must be a non-empty tuple")
    by_action: dict[EvolutionActionType, EvolutionErrorObservation] = {}
    for item in observations:
        if not isinstance(item, EvolutionErrorObservation):
            raise TypeError("evolution observations contain an incompatible value")
        if item.action_type in by_action:
            raise ValueError("evolution observations must contain at most one row per action")
        by_action[item.action_type] = item
    total_evaluated = sum(item.evaluated_count for item in observations)
    total_errors = sum(item.error_count for item in observations)
    if total_evaluated == 0:
        raise ValueError("at least one evolution action must have been evaluated")
    metrics = tuple(
        EvolutionActionErrorMetric(
            action_type=action_type,
            error_kind=_ERROR_KIND_BY_ACTION[action_type],
            evaluated_count=item.evaluated_count,
            error_count=item.error_count,
            error_rate=(
                BinomialRate.from_counts(item.error_count, item.evaluated_count)
                if item.evaluated_count
                else None
            ),
        )
        for action_type, item in sorted(by_action.items(), key=lambda pair: pair[0].value)
    )
    return EvolutionErrorSummary(
        evaluated_count=total_evaluated,
        error_count=total_errors,
        error_rate=BinomialRate.from_counts(total_errors, total_evaluated),
        by_action=metrics,
    )


@dataclass(frozen=True, slots=True)
class RetrievalFunnelObservation:
    """Aggregate counts for one retrieval-inclusion cohort."""

    inclusion_reason: RetrievalInclusionReason
    included_count: int
    invoked_count: int
    flow_evidenced_count: int
    posterior_evidenced_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.inclusion_reason, RetrievalInclusionReason):
            raise TypeError("inclusion_reason must be a RetrievalInclusionReason")
        counts = (
            _nonnegative_count(self.included_count, field="included_count"),
            _nonnegative_count(self.invoked_count, field="invoked_count"),
            _nonnegative_count(self.flow_evidenced_count, field="flow_evidenced_count"),
            _nonnegative_count(
                self.posterior_evidenced_count,
                field="posterior_evidenced_count",
            ),
        )
        if any(left < right for left, right in pairwise(counts)):
            raise ValueError("retrieval funnel counts must be monotonically non-increasing")


@dataclass(frozen=True, slots=True)
class RetrievalFunnelMetric:
    """Conditional and end-to-end rates for one inclusion cohort."""

    inclusion_reason: RetrievalInclusionReason | None
    included_count: int
    invoked_count: int
    flow_evidenced_count: int
    posterior_evidenced_count: int
    invocation_given_inclusion: BinomialRate | None
    flow_given_invocation: BinomialRate | None
    posterior_given_flow: BinomialRate | None
    posterior_given_inclusion: BinomialRate | None


@dataclass(frozen=True, slots=True)
class RetrievalFunnelSummary:
    """Inclusion-to-posterior funnel, split by inclusion provenance and overall."""

    overall: RetrievalFunnelMetric
    by_inclusion_reason: tuple[RetrievalFunnelMetric, ...]


def _funnel_metric(
    *,
    reason: RetrievalInclusionReason | None,
    included: int,
    invoked: int,
    flowed: int,
    posterior: int,
) -> RetrievalFunnelMetric:
    return RetrievalFunnelMetric(
        inclusion_reason=reason,
        included_count=included,
        invoked_count=invoked,
        flow_evidenced_count=flowed,
        posterior_evidenced_count=posterior,
        invocation_given_inclusion=(
            BinomialRate.from_counts(invoked, included) if included else None
        ),
        flow_given_invocation=(BinomialRate.from_counts(flowed, invoked) if invoked else None),
        posterior_given_flow=(BinomialRate.from_counts(posterior, flowed) if flowed else None),
        posterior_given_inclusion=(
            BinomialRate.from_counts(posterior, included) if included else None
        ),
    )


def retrieval_evidence_funnel(
    observations: tuple[RetrievalFunnelObservation, ...],
) -> RetrievalFunnelSummary:
    """Aggregate inclusion -> invocation -> flow -> posterior evidence rates."""

    if not isinstance(observations, tuple) or not observations:
        raise ValueError("retrieval funnel observations must be a non-empty tuple")
    by_reason: dict[RetrievalInclusionReason, RetrievalFunnelObservation] = {}
    for item in observations:
        if not isinstance(item, RetrievalFunnelObservation):
            raise TypeError("retrieval funnel observations contain an incompatible value")
        if item.inclusion_reason in by_reason:
            raise ValueError("retrieval funnel cohorts must have unique inclusion reasons")
        by_reason[item.inclusion_reason] = item
    included = sum(item.included_count for item in observations)
    invoked = sum(item.invoked_count for item in observations)
    flowed = sum(item.flow_evidenced_count for item in observations)
    posterior = sum(item.posterior_evidenced_count for item in observations)
    if included == 0:
        raise ValueError("retrieval funnel requires at least one included skill")
    cohort_metrics = tuple(
        _funnel_metric(
            reason=reason,
            included=item.included_count,
            invoked=item.invoked_count,
            flowed=item.flow_evidenced_count,
            posterior=item.posterior_evidenced_count,
        )
        for reason, item in sorted(by_reason.items(), key=lambda pair: pair[0].value)
    )
    return RetrievalFunnelSummary(
        overall=_funnel_metric(
            reason=None,
            included=included,
            invoked=invoked,
            flowed=flowed,
            posterior=posterior,
        ),
        by_inclusion_reason=cohort_metrics,
    )


class ResourceRateKind(StrEnum):
    """Closed set of scale-normalized resource rates."""

    ROLLOUTS_PER_SECOND = "rollouts-per-second"
    TRAINED_TRAJECTORIES_PER_ROLLOUT = "trained-trajectories-per-rollout"
    GENERATED_TOKENS_PER_ROLLOUT = "generated-tokens-per-rollout"
    EVENT_BYTES_PER_ROLLOUT = "event-bytes-per-rollout"
    POSTERIOR_CELLS_PER_TRAINED_TRAJECTORY = "posterior-cells-per-trained-trajectory"
    AUTHORING_SECONDS_PER_EVOLUTION_ACTION = "authoring-seconds-per-evolution-action"
    LIBRARY_CHANGES_PER_OPTIMIZER_STEP = "library-changes-per-optimizer-step"


@dataclass(frozen=True, slots=True)
class ResourceScaleObservation:
    """One answer-free aggregate resource window."""

    rollout_count: int
    trained_trajectory_count: int
    optimizer_step_count: int
    generated_token_count: int
    event_log_bytes: int
    observer_peak_bytes: int
    posterior_cells_created: int
    evolution_action_count: int
    library_change_count: int
    rollout_seconds: float
    authoring_seconds: float

    def __post_init__(self) -> None:
        for field_name in (
            "rollout_count",
            "trained_trajectory_count",
            "optimizer_step_count",
            "generated_token_count",
            "event_log_bytes",
            "observer_peak_bytes",
            "posterior_cells_created",
            "evolution_action_count",
            "library_change_count",
        ):
            _nonnegative_count(getattr(self, field_name), field=field_name)
        rollout_seconds = _finite(self.rollout_seconds, field="rollout_seconds")
        authoring_seconds = _finite(self.authoring_seconds, field="authoring_seconds")
        if rollout_seconds < 0.0 or authoring_seconds < 0.0:
            raise ValueError("resource durations cannot be negative")
        if self.trained_trajectory_count > self.rollout_count:
            raise ValueError("trained trajectories cannot exceed completed rollouts")
        if self.rollout_count == 0 and (self.generated_token_count != 0 or rollout_seconds != 0.0):
            raise ValueError("zero-rollout windows cannot report tokens or rollout time")
        if self.rollout_count > 0 and rollout_seconds == 0.0:
            raise ValueError("non-empty rollout windows require positive rollout time")
        if self.evolution_action_count == 0 and authoring_seconds != 0.0:
            raise ValueError("authoring time requires at least one evolution action")
        if self.library_change_count > self.evolution_action_count:
            raise ValueError("library changes cannot exceed evolution actions")
        object.__setattr__(self, "rollout_seconds", rollout_seconds)
        object.__setattr__(self, "authoring_seconds", authoring_seconds)


@dataclass(frozen=True, slots=True)
class ResourceTotals:
    """Summed counts and durations plus the maximum observer footprint."""

    observation_count: int
    rollout_count: int
    trained_trajectory_count: int
    optimizer_step_count: int
    generated_token_count: int
    event_log_bytes: int
    observer_peak_bytes: int
    posterior_cells_created: int
    evolution_action_count: int
    library_change_count: int
    rollout_seconds: float
    authoring_seconds: float


@dataclass(frozen=True, slots=True)
class ResourceRate:
    """A finite ratio of two aggregate resource totals."""

    kind: ResourceRateKind
    numerator: float
    denominator: float
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ResourceRateKind):
            raise TypeError("kind must be a ResourceRateKind")
        numerator = _finite(self.numerator, field="resource rate numerator")
        denominator = _finite(self.denominator, field="resource rate denominator")
        value = _finite(self.value, field="resource rate value")
        if numerator < 0.0 or denominator <= 0.0:
            raise ValueError(
                "resource rate requires a non-negative numerator and positive denominator"
            )
        if value != numerator / denominator:
            raise ValueError("resource rate value must equal numerator / denominator")
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class ResourceScaleSummary:
    """Aggregate resource totals and every rate with an observed denominator."""

    totals: ResourceTotals
    rates: tuple[ResourceRate, ...]


def _resource_rate(
    kind: ResourceRateKind,
    numerator: int | float,
    denominator: int | float,
) -> ResourceRate:
    normalized_numerator = float(numerator)
    normalized_denominator = float(denominator)
    return ResourceRate(
        kind=kind,
        numerator=normalized_numerator,
        denominator=normalized_denominator,
        value=normalized_numerator / normalized_denominator,
    )


def resource_scale_summary(
    observations: tuple[ResourceScaleObservation, ...],
) -> ResourceScaleSummary:
    """Summarize cost and scale without retaining any trajectory-level record."""

    if not isinstance(observations, tuple) or not observations:
        raise ValueError("resource observations must be a non-empty tuple")
    if any(not isinstance(item, ResourceScaleObservation) for item in observations):
        raise TypeError("resource observations contain an incompatible value")
    totals = ResourceTotals(
        observation_count=len(observations),
        rollout_count=sum(item.rollout_count for item in observations),
        trained_trajectory_count=sum(item.trained_trajectory_count for item in observations),
        optimizer_step_count=sum(item.optimizer_step_count for item in observations),
        generated_token_count=sum(item.generated_token_count for item in observations),
        event_log_bytes=sum(item.event_log_bytes for item in observations),
        observer_peak_bytes=max(item.observer_peak_bytes for item in observations),
        posterior_cells_created=sum(item.posterior_cells_created for item in observations),
        evolution_action_count=sum(item.evolution_action_count for item in observations),
        library_change_count=sum(item.library_change_count for item in observations),
        rollout_seconds=math.fsum(item.rollout_seconds for item in observations),
        authoring_seconds=math.fsum(item.authoring_seconds for item in observations),
    )
    rates: list[ResourceRate] = []
    if totals.rollout_seconds:
        rates.append(
            _resource_rate(
                ResourceRateKind.ROLLOUTS_PER_SECOND,
                totals.rollout_count,
                totals.rollout_seconds,
            )
        )
    if totals.rollout_count:
        rates.extend(
            (
                _resource_rate(
                    ResourceRateKind.TRAINED_TRAJECTORIES_PER_ROLLOUT,
                    totals.trained_trajectory_count,
                    totals.rollout_count,
                ),
                _resource_rate(
                    ResourceRateKind.GENERATED_TOKENS_PER_ROLLOUT,
                    totals.generated_token_count,
                    totals.rollout_count,
                ),
                _resource_rate(
                    ResourceRateKind.EVENT_BYTES_PER_ROLLOUT,
                    totals.event_log_bytes,
                    totals.rollout_count,
                ),
            )
        )
    if totals.trained_trajectory_count:
        rates.append(
            _resource_rate(
                ResourceRateKind.POSTERIOR_CELLS_PER_TRAINED_TRAJECTORY,
                totals.posterior_cells_created,
                totals.trained_trajectory_count,
            )
        )
    if totals.evolution_action_count:
        rates.append(
            _resource_rate(
                ResourceRateKind.AUTHORING_SECONDS_PER_EVOLUTION_ACTION,
                totals.authoring_seconds,
                totals.evolution_action_count,
            )
        )
    if totals.optimizer_step_count:
        rates.append(
            _resource_rate(
                ResourceRateKind.LIBRARY_CHANGES_PER_OPTIMIZER_STEP,
                totals.library_change_count,
                totals.optimizer_step_count,
            )
        )
    return ResourceScaleSummary(
        totals=totals,
        rates=tuple(sorted(rates, key=lambda rate: rate.kind.value)),
    )


__all__ = [
    "FROZEN_REWARD_BUCKET_SCHEME",
    "BayesianCalibrationSummary",
    "BinomialRate",
    "CalibrationCellAssessment",
    "CalibrationPredictionObservation",
    "ConfidenceInterval",
    "DifferenceEstimate",
    "EvolutionActionErrorMetric",
    "EvolutionErrorKind",
    "EvolutionErrorObservation",
    "EvolutionErrorSummary",
    "MeanEstimate",
    "ProbabilityBinScheme",
    "ReliabilityBinMetric",
    "ResourceRate",
    "ResourceRateKind",
    "ResourceScaleObservation",
    "ResourceScaleSummary",
    "ResourceTotals",
    "RetrievalFunnelMetric",
    "RetrievalFunnelObservation",
    "RetrievalFunnelSummary",
    "RewardBehaviorSummary",
    "RewardBucket",
    "RewardBucketComparison",
    "RewardBucketObservation",
    "RewardBucketScheme",
    "SamplingPeriod",
    "bayesian_calibration_metrics",
    "evolution_error_rates",
    "resource_scale_summary",
    "retrieval_evidence_funnel",
    "reward_proportional_trajectory_behavior",
]
