"""Read-only metrics from complete source events, never from partial rollouts.

Native scores, terminal success and action structure are different quantities.
Only batch aggregates are eligible for the default external exporter; source
questions, action text and native verifier payloads never leave this module.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import cast

from skillev.contracts import JsonValue
from skillev.evaluation.format_content_review import FORMAT_REVIEW_PROFILE, format_review_metrics
from skillev.rollout.codec import codec_for_initial_meta
from skillev.runtime.execution import ActionParseStatus

FORMAT = "skillev-training-metrics@1"
NATIVE_METRICS = {
    "hotpotqa": ("answer-exact-match", "answer-f1"),
    "triviaqa": ("answer-exact-match", "answer-f1"),
    "aime-2026": ("accuracy",),
    "healthbench": ("qwen-local-rubric-score", "luna-medium-api-rubric-score"),
    "mbpp-plus": ("base-pass", "plus-pass", "base-plus-pass@1"),
    "humaneval": ("passed",),
    "alfworld": (),
}
DENOMINATORS: dict[str, JsonValue] = {
    "batch_success_fraction": "terminal success_count / trajectory_count",
    "reward_mean": "sum native projected reward / trajectory_count",
    "action_structure_valid_fraction": "structure-valid actions / all sampled actions",
    "first_turn_structure_valid_fraction": "structure-valid first actions / trajectory_count",
    "raw_delta_sq_mean": "sum delta**2 / trajectory_count (phase diagnostic, not loss)",
    "ttb_loss": "sum (delta / horizon)**2 / trajectory_count",
    "unique_source_question_count": "distinct known benchmark/population/source tuples",
    "admitted_count": "explicitly admitted actions; null unless every committed edge is assessed",
    "executed_count": "calls sent to environment.execute, including measured tool errors/timeouts",
    "execution_returned_success_count": "execute calls returning success status, NOT task progress",
    "accepted_submission_count": "explicit completion accepted for evaluation, NOT correct answers",
    "environment_terminal_count": "executed actions ending environment episodes, NOT task success",
    "terminal_evidence_record_count": (
        "trajectories with explicit assessments for every committed action; coverage, not success"
    ),
    "valid_terminal_record_count": (
        "trajectory OR of accepted_submission/environment_terminal; null if any record lacks "
        "complete committed-action evidence, independent of reward/success"
    ),
    "assessed_action_count": "committed edges matched to explicit execution assessments",
    "native_metrics": "per-domain native score mean; null if any member lacks the score",
}


def _object(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("metrics source must contain an object")
    return value


def _objects(value: JsonValue) -> list[dict[str, JsonValue]]:
    if not isinstance(value, list):
        raise ValueError("metrics source must contain an array")
    return [_object(item) for item in value]


def _number(value: JsonValue) -> float:
    if type(value) not in (int, float) or not math.isfinite(cast(float, value)):
        raise ValueError("metrics source must contain a finite number")
    return float(cast(float, value))


def _text(value: JsonValue) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("metrics identity must be nonempty text")
    return value


def _count(value: JsonValue) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("committed step/horizon/token count must be positive")
    return value


def _success(record: dict[str, JsonValue]) -> bool:
    value = _object(record["reward"])["success"]
    if type(value) is not bool:
        raise ValueError("terminal success must be the native boolean label")
    return value


@dataclass(frozen=True)
class TrainingMetricsSnapshot:
    run_id: str
    condition_id: str
    batch_id: str
    commit_id: str
    sampled_policy_id: str
    sampled_policy_step: int
    optimizer_step: int
    committed_at: str
    metrics: dict[str, JsonValue]
    native_metrics_by_domain: dict[str, JsonValue]
    telemetry: dict[str, JsonValue] | None = None

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.run_id, self.batch_id, self.commit_id

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": FORMAT if self.telemetry is None else "skillev-training-metrics@2",
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "batch_id": self.batch_id,
            "commit_id": self.commit_id,
            "sampled_policy_id": self.sampled_policy_id,
            "sampled_policy_step": self.sampled_policy_step,
            "optimizer_step": self.optimizer_step,
            "committed_at": self.committed_at,
            "metrics": self.metrics,
            "native_metrics_by_domain": self.native_metrics_by_domain,
            "denominator_definitions": DENOMINATORS,
            "smoothing": "none",
            **({"telemetry": self.telemetry} if self.telemetry is not None else {}),
        }

    def wandb_values(self) -> dict[str, JsonValue]:
        """Exclude per-domain results (B28 has only one source question/domain)."""
        return {
            "optimizer_step": self.optimizer_step,
            "sampled_policy_step": self.sampled_policy_step,
            "source_commit_id": self.commit_id,
            **{f"train/{name}": value for name, value in self.metrics.items()},
        }

    @classmethod
    def from_event(
        cls, event: dict[str, JsonValue], *, condition_id: str
    ) -> TrainingMetricsSnapshot:
        if event.get("event_type") != "training_step_committed":
            raise ValueError("metrics require a complete training commit, not a preview")
        payload = _object(event["payload"])
        records = _objects(payload["records"])
        stats = _object(payload["stats"])
        residuals = _objects(stats["residuals"])
        ids = [_text(record["trajectory_id"]) for record in records]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("metrics require a unique nonempty complete trajectory population")
        if ids != [item["trajectory_id"] for item in residuals]:
            raise ValueError("residual order differs from the committed population")
        step = _count(payload["optimizer_step"])
        valid = parsed = first_valid = action_count = action_tokens = 0
        delta_sq: list[float] = []
        losses: list[float] = []
        horizons: list[int] = []
        sources: set[tuple[str, str, str]] = set()
        source_unknown = 0
        groups: dict[str, list[dict[str, JsonValue]]] = defaultdict(list)
        for record, residual in zip(records, residuals, strict=True):
            codec = codec_for_initial_meta(
                _object(_object(record.get("initial_context", {})).get("meta", {}))
            )
            actions = _objects(record["steps"])
            horizon = _count(residual["horizon"])
            if len(actions) != horizon:
                raise ValueError("committed action count differs from the residual horizon")
            horizons.append(horizon)
            delta = _number(residual["delta"])
            delta_sq.append(delta**2)
            losses.append((delta / horizon) ** 2)
            for index, action in enumerate(actions):
                status = codec.parse(_text(action["action_text"])).status
                structural = status is ActionParseStatus.VALID
                valid += structural
                parsed += status is not ActionParseStatus.PARSE_ERROR
                first_valid += structural and index == 0
                action_count += 1
                action_tokens += _count(action["action_token_count"])
            native = _object(_object(record["reward"])["native_payload"])
            domain = str(native.get("benchmark_id", "unknown"))
            groups[domain].append(record)
            source = native.get("training_evidence_source")
            if isinstance(source, dict) and all(
                isinstance(source.get(key), str)
                for key in ("benchmark_id", "population_id", "source_question_id")
            ):
                sources.add(
                    cast(
                        tuple[str, str, str],
                        tuple(
                            source[key]
                            for key in ("benchmark_id", "population_id", "source_question_id")
                        ),
                    )
                )
            else:
                source_unknown += 1
        count = len(records)
        rewards = [_number(_object(record["reward"])["value"]) for record in records]
        reward_mean, loss = math.fsum(rewards) / count, math.fsum(losses) / count
        if not math.isclose(loss, _number(stats["batch_loss"]), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("source loss differs from the complete residual population")
        if not math.isclose(reward_mean, _number(stats["mean_reward"]), abs_tol=1e-12):
            raise ValueError("source reward differs from the complete trajectory population")
        success_count = sum(_success(record) for record in records)
        return cls(
            run_id=_text(event["run_id"]),
            condition_id=_text(condition_id),
            batch_id=_text(payload["batch_id"]),
            commit_id=_text(event["event_id"]),
            sampled_policy_id=_text(payload["policy_snapshot_before"]),
            # The source event is exactly one optimizer update after its pinned policy.
            # Do not guess a version by parsing an opaque adapter name.
            sampled_policy_step=step - 1,
            optimizer_step=step,
            committed_at=_text(event["occurred_at"]),
            metrics={
                "trajectory_count": count,
                "success_count": success_count,
                "batch_success_fraction": success_count / count,
                "reward_sum": math.fsum(rewards),
                "reward_mean": reward_mean,
                **(format_review_metrics(records) if FORMAT_REVIEW_PROFILE in condition_id else {}),
                "ttb_loss": loss,
                **_optimization_metrics(payload),
                "raw_delta_sq_mean": math.fsum(delta_sq) / count,
                "horizon_mean": math.fsum(horizons) / count,
                "horizon_max": max(horizons),
                "action_count": action_count,
                "action_content_tokens": action_tokens,
                "parsed_count": parsed,
                "structural_valid_count": valid,
                "parse_error_count": action_count - parsed,
                "schema_invalid_count": parsed - valid,
                "first_turn_valid_count": first_valid,
                "action_structure_valid_fraction": valid / action_count,
                "first_turn_structure_valid_fraction": first_valid / count,
                "admitted_count": None,
                "executed_count": None,
                "unique_source_question_count": len(sources),
                "unknown_source_trajectory_count": source_unknown,
                "positive_delta_count": sum(_number(r["delta"]) > 0 for r in residuals),
                "negative_delta_count": sum(_number(r["delta"]) < 0 for r in residuals),
                "log_z_mean": math.fsum(_number(r["log_z"]) for r in residuals) / count,
                "sum_forward_mean": math.fsum(_number(r["sum_forward"]) for r in residuals) / count,
                "sum_backward_mean": math.fsum(_number(r["sum_backward"]) for r in residuals)
                / count,
            },
            native_metrics_by_domain={
                name: _native_summary(name, rows) for name, rows in sorted(groups.items())
            },
        )


def _native_summary(domain: str, records: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    summary: dict[str, JsonValue] = {
        "trajectory_count": len(records),
        "success_fraction": sum(_success(r) for r in records) / len(records),
        "reward_mean": math.fsum(_number(_object(r["reward"])["value"]) for r in records)
        / len(records),
    }
    if any(
        "format_content_review" in _object(_object(r["reward"])["native_payload"]) for r in records
    ):
        summary.update(format_review_metrics(records))
    for name in NATIVE_METRICS.get(domain, ()):
        if name == "luna-medium-api-rubric-score" and not any(
            "luna" in str(_object(r["reward"]).get("verifier_version", ""))
            or (
                isinstance(
                    public := _object(_object(r["reward"])["native_payload"]).get("public_metrics"),
                    dict,
                )
                and name in public
            )
            for r in records
        ):
            # Do not change the schema of commits using another frozen judge
            # when an old run is resumed/backfilled with a newer reader.
            continue
        values = []
        for record in records:
            native = _object(_object(record["reward"])["native_payload"])
            public = native.get("public_metrics", {})
            value = public.get(name) if isinstance(public, dict) else None
            if value is None and name == "passed":
                passed = native.get("passed")
                value = int(passed) if type(passed) is bool else None
            if value is not None:
                values.append(_number(value))
        summary[name] = math.fsum(values) / len(records) if len(values) == len(records) else None
        summary[f"{name}/observed_count"] = len(values)
    return summary


def _optimization_metrics(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    report = payload.get("report")
    diagnostics = report.get("optimization_diagnostics") if isinstance(report, dict) else None
    if not isinstance(diagnostics, dict):
        return {}
    return {
        "stability_loss": diagnostics.get("stability_loss"),
        "total_loss": diagnostics.get("total_loss"),
        "gradient_groups": diagnostics.get("gradient_groups"),
    }
