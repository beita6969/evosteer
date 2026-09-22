from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields

import pytest

from skillev.contracts import EvolutionActionType
from skillev.experiments import (
    BinomialRate,
    EvolutionErrorObservation,
    ProbabilityBinScheme,
    ResourceRateKind,
    ResourceScaleObservation,
    RetrievalFunnelObservation,
    RewardBucket,
    RewardBucketObservation,
    RewardBucketScheme,
    SamplingPeriod,
    bayesian_calibration_metrics,
    evolution_error_rates,
    resource_scale_summary,
    retrieval_evidence_funnel,
    reward_proportional_trajectory_behavior,
)
from skillev.experiments.metrics import (
    CalibrationCellAssessment,
    CalibrationPredictionObservation,
)
from skillev.runtime import RetrievalInclusionReason


def _reward_scheme() -> RewardBucketScheme:
    return RewardBucketScheme(
        buckets=(
            RewardBucket(bucket_id="low", lower=0.0, upper=0.5),
            RewardBucket(bucket_id="high", lower=0.5, upper=1.0),
        )
    )


def _reward_observations() -> tuple[RewardBucketObservation, ...]:
    return (
        RewardBucketObservation(
            period=SamplingPeriod.BEFORE_TRAINING,
            bucket_id="low",
            sample_count=3,
            reward_sum=0.6,
            reward_squared_sum=0.14,
        ),
        RewardBucketObservation(
            period=SamplingPeriod.BEFORE_TRAINING,
            bucket_id="high",
            sample_count=1,
            reward_sum=0.6,
            reward_squared_sum=0.36,
        ),
        RewardBucketObservation(
            period=SamplingPeriod.AFTER_TRAINING,
            bucket_id="low",
            sample_count=1,
            reward_sum=0.4,
            reward_squared_sum=0.16,
        ),
        RewardBucketObservation(
            period=SamplingPeriod.AFTER_TRAINING,
            bucket_id="high",
            sample_count=3,
            reward_sum=2.4,
            reward_squared_sum=1.94,
        ),
    )


def test_wilson_rate_matches_hand_calculation() -> None:
    rate = BinomialRate.from_counts(5, 10)

    assert rate.value == 0.5
    assert rate.interval.lower == pytest.approx(0.236593090512564)
    assert rate.interval.upper == pytest.approx(0.763406909487436)
    assert rate.interval.confidence_level == 0.95


@pytest.mark.parametrize(("numerator", "expected"), [(0, 0.0), (10, 1.0)])
def test_wilson_rate_contains_exact_boundary_probabilities(
    numerator: int,
    expected: float,
) -> None:
    rate = BinomialRate.from_counts(numerator, 10)

    assert rate.value == expected
    assert rate.interval.lower <= expected <= rate.interval.upper


def test_reward_proportional_behavior_detects_shift_to_high_reward_bucket() -> None:
    summary = reward_proportional_trajectory_behavior(
        _reward_scheme(),
        _reward_observations(),
    )

    assert summary.before_count == 4
    assert summary.after_count == 4
    assert summary.before_reward.mean == pytest.approx(0.3)
    assert summary.after_reward.mean == pytest.approx(0.7)
    assert summary.reward_change.value == pytest.approx(0.4)
    low, high = summary.buckets
    assert low.bucket.bucket_id == "low"
    assert high.bucket.bucket_id == "high"
    assert high.before_share.value == 0.25
    assert high.after_share.value == 0.75
    assert high.sample_share_change == 0.5
    assert high.reward_change is not None
    assert high.reward_change.value == pytest.approx(0.2)


def test_reward_behavior_requires_exact_registered_aggregate_grid() -> None:
    observations = _reward_observations()

    with pytest.raises(ValueError):
        reward_proportional_trajectory_behavior(_reward_scheme(), observations[:-1])
    with pytest.raises(ValueError):
        reward_proportional_trajectory_behavior(
            _reward_scheme(),
            (*observations, observations[0]),
        )


@pytest.mark.parametrize(
    ("kwargs"),
    [
        {
            "period": SamplingPeriod.BEFORE_TRAINING,
            "bucket_id": "low",
            "sample_count": 1,
            "reward_sum": math.nan,
            "reward_squared_sum": 0.0,
        },
        {
            "period": SamplingPeriod.BEFORE_TRAINING,
            "bucket_id": "low",
            "sample_count": 2,
            "reward_sum": 1.0,
            "reward_squared_sum": 0.1,
        },
        {
            "period": SamplingPeriod.BEFORE_TRAINING,
            "bucket_id": "low",
            "sample_count": 0,
            "reward_sum": 0.1,
            "reward_squared_sum": 0.01,
        },
    ],
)
def test_reward_observations_reject_nonfinite_or_impossible_moments(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        RewardBucketObservation(**kwargs)  # type: ignore[arg-type]


def _calibration_observations() -> tuple[CalibrationPredictionObservation, ...]:
    return tuple(
        CalibrationPredictionObservation(
            posterior_mean=mean,
            lower_confidence_bound=lower,
            outcome=outcome,
        )
        for mean, lower, successes in ((0.2, 0.1, 2), (0.8, 0.7, 6))
        for outcome in (True,) * successes + (False,) * (10 - successes)
    )


def _calibration_assessments() -> tuple[CalibrationCellAssessment, ...]:
    return (
        CalibrationCellAssessment(
            mean_prequential_lcb=0.1,
            validation_count=10,
            success_count=2,
            final_posterior_evidence=1.0,
        ),
        CalibrationCellAssessment(
            mean_prequential_lcb=0.7,
            validation_count=10,
            success_count=6,
            final_posterior_evidence=5.0,
        ),
    )


def test_bayesian_reliability_proper_scores_coverage_and_sparsity() -> None:
    summary = bayesian_calibration_metrics(
        _calibration_observations(),
        assessments=_calibration_assessments(),
        possible_cell_count=4,
        low_evidence_threshold=1.0,
        reliability_bins=ProbabilityBinScheme(edges=(0.0, 0.5, 1.0)),
    )

    expected_nll = (
        -2 * math.log(0.2) - 8 * math.log(0.8) - 6 * math.log(0.8) - 4 * math.log(0.2)
    ) / 20
    assert summary.prediction_count == 20
    assert summary.brier_score == pytest.approx(0.22)
    assert summary.negative_log_likelihood == pytest.approx(expected_nll)
    assert summary.expected_calibration_error == pytest.approx(0.1)
    assert summary.lcb_coverage.value == 0.5
    assert summary.unobserved_cell_rate.value == 0.5
    assert summary.low_evidence_observed_rate.value == 0.5
    assert summary.sparse_cell_rate.value == 0.75
    assert tuple(metric.prediction_count for metric in summary.reliability) == (10, 10)
    assert summary.reliability[1].empirical_success.value == 0.6


@pytest.mark.parametrize("posterior_mean", [0.0, 1.0, math.inf, math.nan])
def test_calibration_observation_rejects_nonfinite_nll_probabilities(
    posterior_mean: float,
) -> None:
    with pytest.raises(ValueError):
        CalibrationPredictionObservation(
            posterior_mean=posterior_mean,
            lower_confidence_bound=0.0,
            outcome=True,
        )


def test_calibration_metrics_bind_assessment_counts_to_prediction_points() -> None:
    with pytest.raises(ValueError):
        bayesian_calibration_metrics(
            _calibration_observations(),
            assessments=(
                CalibrationCellAssessment(
                    mean_prequential_lcb=0.1,
                    validation_count=19,
                    success_count=8,
                    final_posterior_evidence=1.0,
                ),
            ),
            possible_cell_count=1,
            low_evidence_threshold=1.0,
            reliability_bins=ProbabilityBinScheme(edges=(0.0, 1.0)),
        )


def test_calibration_metrics_require_possible_population_to_cover_assessments() -> None:
    with pytest.raises(ValueError):
        bayesian_calibration_metrics(
            _calibration_observations(),
            assessments=_calibration_assessments(),
            possible_cell_count=1,
            low_evidence_threshold=1.0,
            reliability_bins=ProbabilityBinScheme(edges=(0.0, 1.0)),
        )


def test_evolution_error_rates_use_closed_action_semantics() -> None:
    summary = evolution_error_rates(
        (
            EvolutionErrorObservation(
                action_type=EvolutionActionType.RETAIN_COMPRESS,
                evaluated_count=10,
                error_count=1,
            ),
            EvolutionErrorObservation(
                action_type=EvolutionActionType.GENERATE,
                evaluated_count=5,
                error_count=2,
            ),
            EvolutionErrorObservation(
                action_type=EvolutionActionType.PRUNE,
                evaluated_count=0,
                error_count=0,
            ),
        )
    )

    assert summary.evaluated_count == 15
    assert summary.error_count == 3
    assert summary.error_rate.value == 0.2
    by_action = {metric.action_type: metric for metric in summary.by_action}
    assert by_action[EvolutionActionType.GENERATE].error_kind.value == "generate-unused"
    assert by_action[EvolutionActionType.GENERATE].error_rate is not None
    assert by_action[EvolutionActionType.GENERATE].error_rate.value == 0.4
    assert by_action[EvolutionActionType.PRUNE].error_rate is None


def test_evolution_error_rates_reject_duplicates_and_impossible_counts() -> None:
    item = EvolutionErrorObservation(
        action_type=EvolutionActionType.REFINE,
        evaluated_count=1,
        error_count=0,
    )
    with pytest.raises(ValueError):
        evolution_error_rates((item, item))
    with pytest.raises(ValueError):
        EvolutionErrorObservation(
            action_type=EvolutionActionType.SPLIT,
            evaluated_count=1,
            error_count=2,
        )


def test_retrieval_funnel_reports_each_transition_and_overall_rate() -> None:
    summary = retrieval_evidence_funnel(
        (
            RetrievalFunnelObservation(
                inclusion_reason=RetrievalInclusionReason.APPLICABILITY_MATCH,
                included_count=30,
                invoked_count=15,
                flow_evidenced_count=12,
                posterior_evidenced_count=7,
            ),
        )
    )

    overall = summary.overall
    assert overall.included_count == 30
    assert overall.invocation_given_inclusion is not None
    assert overall.invocation_given_inclusion.value == 0.5
    assert overall.flow_given_invocation is not None
    assert overall.flow_given_invocation.value == 0.8
    assert overall.posterior_given_flow is not None
    assert overall.posterior_given_flow.value == pytest.approx(7 / 12)
    assert overall.posterior_given_inclusion is not None
    assert overall.posterior_given_inclusion.value == pytest.approx(7 / 30)
    assert tuple(metric.inclusion_reason for metric in summary.by_inclusion_reason) == (
        RetrievalInclusionReason.APPLICABILITY_MATCH,
    )


def test_retrieval_funnel_rejects_nonmonotone_or_duplicate_cohorts() -> None:
    with pytest.raises(ValueError):
        RetrievalFunnelObservation(
            inclusion_reason=RetrievalInclusionReason.APPLICABILITY_MATCH,
            included_count=1,
            invoked_count=2,
            flow_evidenced_count=1,
            posterior_evidenced_count=0,
        )
    empty = RetrievalFunnelObservation(
        inclusion_reason=RetrievalInclusionReason.APPLICABILITY_MATCH,
        included_count=0,
        invoked_count=0,
        flow_evidenced_count=0,
        posterior_evidenced_count=0,
    )
    with pytest.raises(ValueError):
        retrieval_evidence_funnel((empty, empty))


def _resource_observations() -> tuple[ResourceScaleObservation, ...]:
    return (
        ResourceScaleObservation(
            rollout_count=10,
            trained_trajectory_count=8,
            optimizer_step_count=2,
            generated_token_count=1_000,
            event_log_bytes=2_000,
            observer_peak_bytes=300,
            posterior_cells_created=4,
            evolution_action_count=2,
            library_change_count=1,
            rollout_seconds=5.0,
            authoring_seconds=4.0,
        ),
        ResourceScaleObservation(
            rollout_count=5,
            trained_trajectory_count=4,
            optimizer_step_count=1,
            generated_token_count=400,
            event_log_bytes=1_000,
            observer_peak_bytes=500,
            posterior_cells_created=2,
            evolution_action_count=1,
            library_change_count=1,
            rollout_seconds=5.0,
            authoring_seconds=2.0,
        ),
    )


def test_resource_summary_reports_sums_peak_and_closed_rates() -> None:
    summary = resource_scale_summary(_resource_observations())

    assert summary.totals.observation_count == 2
    assert summary.totals.rollout_count == 15
    assert summary.totals.generated_token_count == 1_400
    assert summary.totals.observer_peak_bytes == 500
    rates = {rate.kind: rate.value for rate in summary.rates}
    assert rates[ResourceRateKind.ROLLOUTS_PER_SECOND] == 1.5
    assert rates[ResourceRateKind.TRAINED_TRAJECTORIES_PER_ROLLOUT] == 0.8
    assert rates[ResourceRateKind.GENERATED_TOKENS_PER_ROLLOUT] == pytest.approx(1400 / 15)
    assert rates[ResourceRateKind.EVENT_BYTES_PER_ROLLOUT] == 200.0
    assert rates[ResourceRateKind.POSTERIOR_CELLS_PER_TRAINED_TRAJECTORY] == 0.5
    assert rates[ResourceRateKind.AUTHORING_SECONDS_PER_EVOLUTION_ACTION] == 2.0
    assert rates[ResourceRateKind.LIBRARY_CHANGES_PER_OPTIMIZER_STEP] == pytest.approx(2 / 3)


def test_resource_observation_rejects_fabricated_denominators_and_nonfinite_time() -> None:
    with pytest.raises(ValueError):
        ResourceScaleObservation(
            rollout_count=0,
            trained_trajectory_count=0,
            optimizer_step_count=0,
            generated_token_count=1,
            event_log_bytes=0,
            observer_peak_bytes=0,
            posterior_cells_created=0,
            evolution_action_count=0,
            library_change_count=0,
            rollout_seconds=0.0,
            authoring_seconds=0.0,
        )
    with pytest.raises(ValueError):
        ResourceScaleObservation(
            rollout_count=1,
            trained_trajectory_count=1,
            optimizer_step_count=1,
            generated_token_count=1,
            event_log_bytes=1,
            observer_peak_bytes=1,
            posterior_cells_created=1,
            evolution_action_count=1,
            library_change_count=1,
            rollout_seconds=math.inf,
            authoring_seconds=1.0,
        )


def test_observation_dtos_are_frozen_and_contain_no_benchmark_truth_fields() -> None:
    observation_types = (
        RewardBucketObservation,
        CalibrationPredictionObservation,
        CalibrationCellAssessment,
        EvolutionErrorObservation,
        RetrievalFunnelObservation,
        ResourceScaleObservation,
    )
    prohibited = {"answer", "question", "task_text", "native_payload", "trajectory_id"}
    for observation_type in observation_types:
        assert prohibited.isdisjoint(field.name for field in fields(observation_type))
        assert not hasattr(observation_type, "to_value")
        assert not hasattr(observation_type, "from_value")

    item = _reward_observations()[0]
    with pytest.raises(FrozenInstanceError):
        item.sample_count = 100  # type: ignore[misc]


def test_experiment_metrics_import_does_not_load_model_dependencies() -> None:
    code = """
import sys
import skillev.experiments
loaded = {name.partition('.')[0] for name in sys.modules}
for forbidden in ('torch', 'transformers', 'peft', 'tokenizers'):
    if forbidden in loaded:
        raise SystemExit(f'{forbidden} was loaded')
"""
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and argument vector
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
