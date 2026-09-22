"""Answer-free evidence records connecting flow diagnostics to calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Self, cast

from skillev.contracts import JsonValue, normalize_json

EVIDENCE_SCHEMA_VERSION: Final = "skillev-evidence@1"


class EvidenceSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def _text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _optional_finite(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric or null")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    event_id: str
    run_id: str
    split: EvidenceSplit
    seed: int
    train_step: int
    question_id: str
    task_type: str
    trajectory_id: str
    skill_uid: str | None
    skill_version_id: str | None
    action_index: int | None
    verified_outcome: float
    verifier_name: str
    verifier_passed: bool
    failure_mode: str | None
    reward: float
    r_tilde: float
    log_i: float | None
    state_log_flow: float | None
    skill_marginal_log_flow: float | None
    ttb_residual: float | None
    token_count: int
    turn_count: int
    latency_ms: float
    context_features: Mapping[str, JsonValue]
    evidence_weight: float
    created_at: str
    schema_version: str = field(default=EVIDENCE_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported evidence schema version")
        for field_name in (
            "event_id",
            "run_id",
            "question_id",
            "task_type",
            "trajectory_id",
            "verifier_name",
            "created_at",
        ):
            _text(getattr(self, field_name), field_name=field_name)
        if not isinstance(self.split, EvidenceSplit):
            raise TypeError("split must be EvidenceSplit")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if type(self.train_step) is not int or self.train_step < 0:
            raise ValueError("train_step must be non-negative")
        if (self.skill_uid is None) != (self.skill_version_id is None):
            raise ValueError("skill UID and version ID must either both be present or both be null")
        for field_name in ("skill_uid", "skill_version_id", "failure_mode"):
            value = getattr(self, field_name)
            if value is not None:
                _text(value, field_name=field_name)
        if self.action_index is not None and (
            type(self.action_index) is not int or self.action_index < 0
        ):
            raise ValueError("action_index must be non-negative or null")
        if type(self.verifier_passed) is not bool:
            raise TypeError("verifier_passed must be boolean")
        for field_name in ("verified_outcome", "reward"):
            value = _optional_finite(getattr(self, field_name), field_name=field_name)
            if value is None or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must lie in [0, 1]")
            object.__setattr__(self, field_name, value)
        r_tilde = _optional_finite(self.r_tilde, field_name="r_tilde")
        if r_tilde is None or r_tilde <= 0:
            raise ValueError("r_tilde must be positive")
        object.__setattr__(self, "r_tilde", r_tilde)
        for field_name in (
            "log_i",
            "state_log_flow",
            "skill_marginal_log_flow",
            "ttb_residual",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_finite(getattr(self, field_name), field_name=field_name),
            )
        for field_name in ("token_count", "turn_count"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        for field_name in ("latency_ms", "evidence_weight"):
            value = _optional_finite(getattr(self, field_name), field_name=field_name)
            minimum_ok = value is not None and value >= 0
            if field_name == "evidence_weight":
                minimum_ok = value is not None and value > 0
            if not minimum_ok:
                raise ValueError(f"{field_name} is outside its supported range")
            object.__setattr__(self, field_name, value)
        normalized_context = normalize_json(dict(self.context_features))
        if not isinstance(normalized_context, dict):
            raise TypeError("context_features must normalize to an object")
        object.__setattr__(self, "context_features", normalized_context)

    @property
    def may_update_posterior(self) -> bool:
        return self.split is EvidenceSplit.TRAIN and self.verifier_passed

    def to_value(self) -> dict[str, JsonValue]:
        value = normalize_json(
            {
                "action_index": self.action_index,
                "context_features": dict(self.context_features),
                "created_at": self.created_at,
                "event_id": self.event_id,
                "evidence_weight": self.evidence_weight,
                "failure_mode": self.failure_mode,
                "latency_ms": self.latency_ms,
                "log_i": self.log_i,
                "question_id": self.question_id,
                "r_tilde": self.r_tilde,
                "reward": self.reward,
                "run_id": self.run_id,
                "schema_version": self.schema_version,
                "seed": self.seed,
                "skill_marginal_log_flow": self.skill_marginal_log_flow,
                "skill_uid": self.skill_uid,
                "skill_version_id": self.skill_version_id,
                "split": self.split.value,
                "state_log_flow": self.state_log_flow,
                "task_type": self.task_type,
                "token_count": self.token_count,
                "train_step": self.train_step,
                "trajectory_id": self.trajectory_id,
                "ttb_residual": self.ttb_residual,
                "turn_count": self.turn_count,
                "verified_outcome": self.verified_outcome,
                "verifier_name": self.verifier_name,
                "verifier_passed": self.verifier_passed,
            }
        )
        if not isinstance(value, dict):
            raise TypeError("evidence record must normalize to an object")
        return cast(dict[str, JsonValue], value)

    @classmethod
    def from_value(cls, value: object) -> Self:
        normalized = normalize_json(value)
        if not isinstance(normalized, dict):
            raise TypeError("evidence record must be an object")
        expected = {
            "action_index",
            "context_features",
            "created_at",
            "event_id",
            "evidence_weight",
            "failure_mode",
            "latency_ms",
            "log_i",
            "question_id",
            "r_tilde",
            "reward",
            "run_id",
            "schema_version",
            "seed",
            "skill_marginal_log_flow",
            "skill_uid",
            "skill_version_id",
            "split",
            "state_log_flow",
            "task_type",
            "token_count",
            "train_step",
            "trajectory_id",
            "ttb_residual",
            "turn_count",
            "verified_outcome",
            "verifier_name",
            "verifier_passed",
        }
        if set(normalized) != expected:
            raise ValueError("evidence record has incompatible fields")
        context = normalized["context_features"]
        if not isinstance(context, dict):
            raise TypeError("context_features must be an object")
        verifier_passed = normalized["verifier_passed"]
        if type(verifier_passed) is not bool:
            raise TypeError("verifier_passed must be boolean")
        return cls(
            event_id=str(normalized["event_id"]),
            run_id=str(normalized["run_id"]),
            split=EvidenceSplit(str(normalized["split"])),
            seed=int(str(normalized["seed"])),
            train_step=int(str(normalized["train_step"])),
            question_id=str(normalized["question_id"]),
            task_type=str(normalized["task_type"]),
            trajectory_id=str(normalized["trajectory_id"]),
            skill_uid=None if normalized["skill_uid"] is None else str(normalized["skill_uid"]),
            skill_version_id=(
                None
                if normalized["skill_version_id"] is None
                else str(normalized["skill_version_id"])
            ),
            action_index=(
                None if normalized["action_index"] is None else int(str(normalized["action_index"]))
            ),
            verified_outcome=float(str(normalized["verified_outcome"])),
            verifier_name=str(normalized["verifier_name"]),
            verifier_passed=verifier_passed,
            failure_mode=(
                None if normalized["failure_mode"] is None else str(normalized["failure_mode"])
            ),
            reward=float(str(normalized["reward"])),
            r_tilde=float(str(normalized["r_tilde"])),
            log_i=None if normalized["log_i"] is None else float(str(normalized["log_i"])),
            state_log_flow=(
                None
                if normalized["state_log_flow"] is None
                else float(str(normalized["state_log_flow"]))
            ),
            skill_marginal_log_flow=(
                None
                if normalized["skill_marginal_log_flow"] is None
                else float(str(normalized["skill_marginal_log_flow"]))
            ),
            ttb_residual=(
                None
                if normalized["ttb_residual"] is None
                else float(str(normalized["ttb_residual"]))
            ),
            token_count=int(str(normalized["token_count"])),
            turn_count=int(str(normalized["turn_count"])),
            latency_ms=float(str(normalized["latency_ms"])),
            context_features=context,
            evidence_weight=float(str(normalized["evidence_weight"])),
            created_at=str(normalized["created_at"]),
            schema_version=str(normalized["schema_version"]),
        )


__all__ = ["EVIDENCE_SCHEMA_VERSION", "EvidenceRecord", "EvidenceSplit"]
