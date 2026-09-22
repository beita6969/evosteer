"""Exact panel joins and paired controls; performance is not an integrity gate."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import cast

from .input_metric_contracts import CONTRACTS
from .integrity_metric_schema import (
    aggregate_secondary_metrics,
    require_expected_verifier,
    validate_native_score,
)
from .step0_integrity import (
    InferenceArm,
    InterventionCounts,
    SkillMode,
    validate_paired_intervention,
)
from .step0_receipts import (
    PairedBootstrapInterval,
    mcnemar_exact_two_sided,
    paired_bootstrap_interval,
    paired_outcome_counts,
)


class EvaluationStatus(StrEnum):
    SCORED = "scored"
    CANDIDATE_FAILURE = "candidate-failure"
    INFRASTRUCTURE_FAILURE = "infrastructure-failure"


@dataclass(frozen=True, slots=True)
class NativeScore:
    task_id: str
    benchmark: str
    metric: str
    value: float
    status: EvaluationStatus = EvaluationStatus.SCORED
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    verifier_version: str = field(kw_only=True)
    scorer_cost: dict[str, float] = field(default_factory=dict)
    grader_used: bool | None = field(default=None, kw_only=True)
    failure_kind: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        # Retired native records stay readable; FrozenPanel guards new runs
        # against benchmarks outside the current seven-domain catalog.
        if self.benchmark not in CONTRACTS or not self.task_id or not self.metric:
            raise ValueError("native score identity is invalid")
        if type(self.value) not in (int, float) or not math.isfinite(float(self.value)):
            raise ValueError("native score must be finite")
        object.__setattr__(self, "value", float(self.value))
        if not isinstance(self.status, EvaluationStatus):
            raise TypeError("native evaluation status must be explicit")
        # Native negative simulator outcomes are SCORED, not candidate failures.
        if self.status is EvaluationStatus.CANDIDATE_FAILURE and self.value != 0:
            raise ValueError("a candidate failure must contribute zero")
        if self.failure_kind is not None and (
            type(self.failure_kind) is not str
            or not self.failure_kind.strip()
            or self.status is EvaluationStatus.SCORED
        ):
            raise ValueError("native failure kind requires an explicit failure status")
        if not isinstance(self.scorer_cost, dict):
            raise TypeError("scorer costs must be a mapping")
        if any(
            type(name) is not str
            or type(value) not in (int, float)
            or not math.isfinite(float(value))
            or value < 0
            for name, value in self.scorer_cost.items()
        ):
            raise ValueError("scorer costs must be finite and non-negative")
        validate_native_score(self)

    @classmethod
    def from_value(cls, value: dict[str, object]) -> NativeScore:
        raw = value["value"]
        metrics = value.get("secondary_metrics")
        cost = value.get("scorer_cost", {})
        grader_used = value.get("grader_used")
        failure_kind = value.get("failure_kind")
        if grader_used is not None and type(grader_used) is not bool:
            raise TypeError("stored grader use must be boolean or absent")
        if failure_kind is not None and type(failure_kind) is not str:
            raise TypeError("stored native failure kind must be text or absent")
        if (
            type(raw) not in (int, float)
            or not isinstance(metrics, dict)
            or not isinstance(cost, dict)
        ):
            raise TypeError("stored native score has invalid numeric fields")
        if any(
            type(key) is not str or type(item) not in (int, float) for key, item in metrics.items()
        ):
            raise TypeError("stored native secondary metrics have invalid numeric fields")
        if any(
            type(key) is not str or type(item) not in (int, float) for key, item in cost.items()
        ):
            raise TypeError("stored native scorer costs have invalid numeric fields")
        return cls(
            task_id=str(value["task_id"]),
            benchmark=str(value["benchmark"]),
            metric=str(value["metric"]),
            value=float(cast(int | float, raw)),
            status=EvaluationStatus(str(value["status"])),
            secondary_metrics={key: float(item) for key, item in metrics.items()},
            verifier_version=str(value["verifier_version"]),
            scorer_cost={key: float(item) for key, item in cost.items()},
            grader_used=grader_used,
            failure_kind=failure_kind,
        )


def exact_panel_join(
    expected: tuple[str, ...], scores: tuple[NativeScore, ...], *, expected_verifier: str
) -> tuple[NativeScore, ...]:
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("frozen panel must be nonempty and unique")
    require_expected_verifier(expected_verifier)
    observed = [score.task_id for score in scores]
    if len(set(observed)) != len(observed) or set(observed) != set(expected):
        raise ValueError("result IDs do not join exactly to the frozen panel")
    if any(score.status is EvaluationStatus.INFRASTRUCTURE_FAILURE for score in scores):
        raise RuntimeError("panel is incomplete because of infrastructure failure")
    if len({(score.benchmark, score.metric) for score in scores}) != 1:
        raise ValueError("a native aggregate cannot mix benchmarks or metrics")
    for score in scores:
        validate_native_score(score, expected_verifier=expected_verifier)
    by_id = {score.task_id: score for score in scores}
    return tuple(by_id[task_id] for task_id in expected)


def native_mean(scores: tuple[NativeScore, ...]) -> float:
    if not scores:
        raise ValueError("cannot aggregate an empty panel")
    value = sum(score.value for score in scores) / len(scores)
    return (
        max(0.0, min(1.0, value))
        if scores[0].benchmark in {"healthbench", "livemedbench"}
        else value
    )


def scienceworld_terminal_metrics(outcomes: list[dict[str, object]]) -> dict[str, object]:
    """Report final native scores separately from bounded learning rewards.

    No peak-score substitution, done-as-success, or omission from the denominator.
    A missing native score remains unknown, not an inferred zero or partial mean.
    """
    raw = [row.get("native_final_score") for row in outcomes]
    known = [
        float(value)
        for value in raw
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value <= 100
    ]
    complete = bool(raw) and len(known) == len(raw)
    return {
        "count": len(raw),
        "missing_native_scores": len(raw) - len(known),
        "native_final_score_mean": sum(known) / len(raw) if complete else None,
        "clipped_reward_mean": sum(max(0, value) / 100 for value in known) / len(raw)
        if complete
        else None,
        "success_100": sum(value == 100 for value in known) / len(raw) if complete else None,
        "success_70": sum(value >= 70 for value in known) / len(raw) if complete else None,
        "negative_final_scores": sum(value < 0 for value in known),
    }


def native_score_diagnostics(
    scores: tuple[NativeScore, ...], environment: dict[str, object]
) -> dict[str, object]:
    """Answer-free reporting; never change native scores or infer missing split labels."""
    result: dict[str, object] = {
        "status_counts": {
            status.value: sum(row.status is status for row in scores) for status in EvaluationStatus
        },
        "verifier_versions": sorted({row.verifier_version for row in scores}),
        "failure_kinds": dict(
            Counter(
                row.failure_kind or "unspecified"
                for row in scores
                if row.status is not EvaluationStatus.SCORED
            )
        ),
    }
    if scores and scores[0].benchmark == "alfworld":
        # The frozen environment descriptor carries source split metadata. It
        # was available to the coordinator but omitted from the final summary.
        cases = environment.get("cases", {})
        groups: dict[str, list[NativeScore]] = {"valid_seen": [], "valid_unseen": []}
        unknown = 0
        for score in scores:
            descriptor = cases.get(score.task_id, {}) if isinstance(cases, dict) else {}
            case = descriptor.get("case", {}) if isinstance(descriptor, dict) else {}
            payload = case.get("payload", {}) if isinstance(case, dict) else {}
            split = payload.get("split") if isinstance(payload, dict) else None
            if isinstance(split, str) and split in groups:
                groups[split].append(score)
            else:
                unknown += 1
        result["splits"] = {
            name: {
                "count": len(rows),
                "successes": sum(int(row.value) for row in rows),
                "value": sum(row.value for row in rows) / len(rows) if rows else None,
            }
            for name, rows in groups.items()
        }
        result["split_metadata_missing_count"] = unknown
    return result


def native_interval(
    a: tuple[NativeScore, ...], b: tuple[NativeScore, ...]
) -> PairedBootstrapInterval:
    if a[0].benchmark not in {"healthbench", "livemedbench"}:
        return paired_bootstrap_interval(
            tuple(item.value for item in a), tuple(item.value for item in b)
        )
    # Resample pairs, then apply the SAME panel clipping as the reported estimand.
    generator = random.Random(0)  # noqa: S311 -- statistical resampling only
    estimates = []
    size, resamples = len(a), 10_000
    for _ in range(resamples):
        indices = tuple(generator.randrange(size) for _ in range(size))
        estimates.append(
            native_mean(tuple(b[i] for i in indices)) - native_mean(tuple(a[i] for i in indices))
        )
    estimates.sort()
    return PairedBootstrapInterval(
        native_mean(b) - native_mean(a), estimates[250], estimates[9750], 0.95, resamples
    )


@dataclass(frozen=True, slots=True)
class ExecutionControls:
    """Expanded runtime values. Opaque receipt IDs and caller equality flags are not accepted.

    public_inputs and environment/tool descriptors can contain private content;
    keep these records in the run's private directory, never in public reports.
    """

    model: dict[str, object]
    tokenizer: dict[str, object]
    service: dict[str, object]
    public_inputs: tuple[tuple[str, str], ...]
    tools: dict[str, object]
    environment: dict[str, object]
    evaluator: dict[str, object]
    parser: dict[str, object]
    sampling: dict[str, object]
    budgets: dict[str, object]
    skills: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("model", "tokenizer", "service", "evaluator", "parser", "sampling", "budgets"):
            value = getattr(self, name)
            if (
                not isinstance(value, dict)
                or not value
                or all(isinstance(item, bool) for item in value.values())
            ):
                raise ValueError("paired controls require expanded runtime descriptors")
        ids = [task_id for task_id, _ in self.public_inputs]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("paired public inputs must include every ordered task once")

    def differences(self, other: ExecutionControls) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.__dataclass_fields__
            if getattr(self, name) != getattr(other, name)
        )


@dataclass(frozen=True, slots=True)
class ExposureRecord:
    population_id: str
    development_exposed: bool
    provenance: str
    prior_use: str


def _require_model_policy(model: dict[str, object], arm: InferenceArm) -> None:
    policy = model.get("policy")
    if not isinstance(policy, dict) or not policy.get("snapshot_id"):
        raise ValueError("policy comparisons require the actual serving policy descriptor")
    route = model.get("adapter_route_sent")
    if arm.optimizer_steps:
        checkpoint = model.get("checkpoint")
        if (
            policy.get("snapshot_id") != arm.policy_id
            or not isinstance(route, str)
            or not route.strip()
            or policy.get("forward_adapter_version") in {None, "adapter-free"}
            or not isinstance(checkpoint, dict)
            or checkpoint.get("optimizer_steps") != arm.optimizer_steps
            or checkpoint.get("forward_version") != policy.get("forward_adapter_version")
        ):
            raise ValueError("trained arm does not match its served forward checkpoint")
    elif (
        route is not None
        or policy.get("forward_adapter_version") != "adapter-free"
        or model.get("checkpoint") is not None
    ):
        raise ValueError("Step-0 must actually use the adapter-free policy")


def require_paired_controls(
    left: ExecutionControls,
    right: ExecutionControls,
    left_arm: InferenceArm,
    right_arm: InferenceArm,
) -> str:
    """Allow only the declared forward policy to differ, not its frozen backbone or settings."""
    intervention = validate_paired_intervention(left_arm, right_arm)
    if left_arm.optimizer_steps or right_arm.optimizer_steps:
        _require_model_policy(left.model, left_arm)
        _require_model_policy(right.model, right_arm)
    differences = set(left.differences(right))
    for controls, arm in ((left, left_arm), (right, right_arm)):
        if arm.skill_mode is SkillMode.OFF and controls.skills:
            raise ValueError("no-skill controls cannot carry a skill library")
        if arm.skill_mode is SkillMode.LIBRARY and (
            controls.skills.get("library_id") != arm.skill_library_id
            or controls.skills.get("retrieval_rule") != arm.skill_retrieval_rule
            or not isinstance(controls.skills.get("state"), dict)
        ):
            raise ValueError("skill controls must contain the selected actual library state")
    if intervention == "skill-access":
        differences.discard("skills")
    if intervention == "forward-policy":
        variable_fields = {"policy", "adapter_route_sent", "checkpoint"}
        fixed_left = {key: value for key, value in left.model.items() if key not in variable_fields}
        fixed_right = {
            key: value for key, value in right.model.items() if key not in variable_fields
        }
        policy_left = cast(dict[str, object], left.model["policy"])
        policy_right = cast(dict[str, object], right.model["policy"])
        fixed_policy = {"snapshot_id", "forward_adapter_version"}
        if (
            not fixed_left.get("observed")
            or fixed_left != fixed_right
            or {key: value for key, value in policy_left.items() if key not in fixed_policy}
            != {key: value for key, value in policy_right.items() if key not in fixed_policy}
        ):
            raise ValueError("weight-only comparisons require the same observed frozen backbone")
        differences.discard("model")
    if differences:
        raise ValueError(f"paired runtime controls differ: {', '.join(sorted(differences))}")
    return intervention


@dataclass(frozen=True, slots=True)
class PublicationStatus:
    integrity_status: str
    execution_status: str
    communication_status: str
    performance_goal_status: str
    external_reference_status: str

    @property
    def publishable(self) -> bool:
        return self.integrity_status == "pass" and self.execution_status == "complete"


def compare_paired(
    expected: tuple[str, ...],
    left: tuple[NativeScore, ...],
    right: tuple[NativeScore, ...],
    *,
    left_controls: ExecutionControls,
    right_controls: ExecutionControls,
    left_arm: InferenceArm,
    right_arm: InferenceArm,
    left_counts: InterventionCounts,
    right_counts: InterventionCounts,
    left_expected_verifier: str,
    right_expected_verifier: str,
) -> dict[str, object]:
    intervention = require_paired_controls(left_controls, right_controls, left_arm, right_arm)
    if tuple(task_id for task_id, _ in left_controls.public_inputs) != expected:
        raise ValueError("actual run population differs from the frozen comparison panel")
    left_counts.require_arm(left_arm)
    right_counts.require_arm(right_arm)
    if left_expected_verifier != right_expected_verifier:
        raise ValueError("paired native comparisons require the same frozen verifier")
    a = exact_panel_join(expected, left, expected_verifier=left_expected_verifier)
    b = exact_panel_join(expected, right, expected_verifier=right_expected_verifier)
    if a[0].benchmark != b[0].benchmark or a[0].metric != b[0].metric:
        raise ValueError("paired native metrics differ")
    interval = native_interval(a, b)
    result: dict[str, object] = {
        "intervention": intervention,
        "count": len(expected),
        "left": native_mean(a),
        "right": native_mean(b),
        "delta": native_mean(b) - native_mean(a),
        "paired_interval": interval,
        "multiple_comparisons": (
            "eight benchmark comparisons; intervals are descriptive, not family-wise corrected; "
            "binary McNemar tests additionally report Bonferroni p over eight comparisons"
        ),
        "integrity_status": "pass",
        "execution_status": "complete",
        "communication_status": "reported-separately-from-native-task-success",
        "performance_goal_status": "improved"
        if native_mean(b) > native_mean(a)
        else "not-improved",
        "external_reference_status": "not-a-matched-control",
    }
    if a[0].benchmark in {"aime-2026", "alfworld", "mbpp-plus", "humaneval"}:
        counts = paired_outcome_counts(
            tuple(bool(x.value) for x in a), tuple(bool(x.value) for x in b)
        )
        p_value = mcnemar_exact_two_sided(counts)
        result.update(
            {
                "paired_outcomes": asdict(counts),
                "mcnemar_p": p_value,
                "mcnemar_bonferroni_p": min(1.0, 8 * p_value),
            }
        )
    left_secondary = aggregate_secondary_metrics(a)
    right_secondary = aggregate_secondary_metrics(b)
    if set(left_secondary) != set(right_secondary):
        raise ValueError("paired native secondary metrics differ")
    result["secondary_metrics"] = {
        name: {"left": left_secondary[name], "right": right_secondary[name]}
        for name in sorted(left_secondary)
    }
    return result
