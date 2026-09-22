"""EvoSteer histories, distinct from the historical hindsight-TTB contracts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import cast

from .canonical import canonical_json, stable_hash
from .evosteer_risk import TrajectoryRiskAssessment

EVOSTEER_FORMAT = "evosteer-history@2"
TRAJECTORY_SOURCES = frozenset(
    {"current", "natural_reference", "paired_treatment", "paired_control"}
)


def _text(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be {'a' if allow_empty else 'a nonempty'} string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _canonical_object(value: object, name: str) -> dict[str, object]:
    text = _text(value, name)
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or canonical_json(parsed) != text:
        raise ValueError(f"{name} must be a canonical JSON object")
    return cast(dict[str, object], parsed)


def _tokens(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} must be a nonempty immutable token tuple")
    if any(type(token) is not int or token < 0 for token in value):
        raise ValueError(f"{name} contains an invalid token")
    return cast(tuple[int, ...], value)


def _wire_object(value: object, names: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError(f"{label} has incompatible fields")
    return cast(dict[str, object], value)


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _float_tuple(value: object, name: str) -> tuple[float, ...]:
    return tuple(_number(item, name) for item in _array(value, name))


def _token_array(value: object, name: str) -> tuple[int, ...]:
    return _tokens(tuple(_array(value, name)), name)


@dataclass(frozen=True, slots=True)
class EvoTask:
    """Public task only. Answer keys belong to the injected terminal evaluator."""

    task_id: str
    family: str
    prompt: str
    reset_id: str = "initial"
    environment_config_id: str = "static@1"
    prior_rate: float | None = None
    prior_count: float = 0.0

    def __post_init__(self) -> None:
        if any(
            not isinstance(v, str) or not v.strip()
            for v in (
                self.task_id,
                self.family,
                self.prompt,
                self.reset_id,
                self.environment_config_id,
            )
        ):
            raise ValueError("task fields must be nonempty strings")
        if _number(self.prior_count, "prior_count") < 0:
            raise ValueError("prior count cannot be negative")
        if self.prior_rate is None:
            if self.prior_count != 0:
                raise ValueError("positive prior count requires a prior rate")
        elif not 0 <= _number(self.prior_rate, "prior_rate") <= 1:
            raise ValueError("prior rate must be in [0, 1]")

    @classmethod
    def from_value(cls, value: object) -> EvoTask:
        item = _wire_object(
            value,
            {
                "task_id",
                "family",
                "prompt",
                "reset_id",
                "environment_config_id",
                "prior_rate",
                "prior_count",
            },
            "task",
        )
        return cls(
            task_id=_text(item["task_id"], "task_id"),
            family=_text(item["family"], "family"),
            prompt=_text(item["prompt"], "prompt"),
            reset_id=_text(item["reset_id"], "reset_id"),
            environment_config_id=_text(item["environment_config_id"], "environment_config_id"),
            prior_rate=None
            if item["prior_rate"] is None
            else _number(item["prior_rate"], "prior_rate"),
            prior_count=_number(item["prior_count"], "prior_count"),
        )

    @property
    def identity(self) -> str:
        return str(stable_hash(asdict(self)))


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Exact scored action and the public pre-action state that generated it.

    Legal token paths include a forced terminator. They define the identical
    masked distribution for sampling and both teacher-forced scores. Encoder
    features are frozen reference outputs, never hidden evaluator information.
    """

    state_id: str
    state_json: str
    action_json: str
    prompt_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    legal_token_paths: tuple[tuple[int, ...], ...]
    features: tuple[float, ...]
    reference_encoding: tuple[float, ...]
    value_estimate: float
    forced: bool = False

    def __post_init__(self) -> None:
        _text(self.state_id, "state_id")
        state = _canonical_object(self.state_json, "state_json")
        if self.state_id != stable_hash(state):
            raise ValueError("state identity differs from its recorded full public history")
        if state.get("stopped") is True:
            raise ValueError("a stopped state cannot generate another orchestration decision")
        _canonical_object(self.action_json, "action_json")
        _tokens(self.prompt_ids, "prompt_ids")
        _tokens(self.action_token_ids, "action_token_ids")
        if not isinstance(self.legal_token_paths, tuple) or not self.legal_token_paths:
            raise ValueError("legal token paths must be an immutable nonempty tuple")
        for path in self.legal_token_paths:
            _tokens(path, "legal token path")
        if self.action_token_ids not in self.legal_token_paths:
            raise ValueError("executed action is outside its recorded legal support")
        if len(set(self.legal_token_paths)) != len(self.legal_token_paths):
            raise ValueError("duplicate legal token paths")
        leaves = set(self.legal_token_paths)
        if any(path[:i] in leaves for path in leaves for i in range(1, len(path))):
            raise ValueError("legal token paths must be prefix-free")
        if not isinstance(self.features, tuple) or len(self.features) != 30:
            raise ValueError("expected 30 public execution features")
        if not isinstance(self.reference_encoding, tuple):
            raise ValueError(
                "reference encoding must be an immutable tuple, or empty when deferred"
            )
        for value in (*self.features, *self.reference_encoding):
            _number(value, "state feature")
        if not 0 <= _number(self.value_estimate, "value_estimate") <= 1:
            raise ValueError("invalid reference value estimate")
        if type(self.forced) is not bool:
            raise ValueError("forced must be boolean")

    @property
    def support_id(self) -> str:
        return str(stable_hash(self.legal_token_paths))

    def to_value(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(canonical_json(asdict(self))))

    @classmethod
    def from_value(cls, value: dict[str, object]) -> DecisionRecord:
        item = _wire_object(
            value,
            {
                "state_id",
                "state_json",
                "action_json",
                "prompt_ids",
                "action_token_ids",
                "legal_token_paths",
                "features",
                "reference_encoding",
                "value_estimate",
                "forced",
            },
            "decision",
        )
        if type(item["forced"]) is not bool:
            raise ValueError("forced must be boolean")
        return cls(
            state_id=_text(item["state_id"], "state_id"),
            state_json=_text(item["state_json"], "state_json"),
            action_json=_text(item["action_json"], "action_json"),
            prompt_ids=_token_array(item["prompt_ids"], "prompt_ids"),
            action_token_ids=_token_array(item["action_token_ids"], "action_token_ids"),
            legal_token_paths=tuple(
                _token_array(path, "legal token path")
                for path in _array(item["legal_token_paths"], "legal_token_paths")
            ),
            features=_float_tuple(item["features"], "features"),
            reference_encoding=_float_tuple(item["reference_encoding"], "reference_encoding"),
            value_estimate=_number(item["value_estimate"], "value_estimate"),
            forced=item["forced"],
        )


@dataclass(frozen=True, slots=True)
class EvoTrajectory:
    sample_id: str
    batch_id: str
    task: EvoTask
    source: str
    behavior_policy_id: str
    reference_id: str
    executor_id: str
    menu_id: str
    value_snapshot_id: str
    statistics_context: str
    decisions: tuple[DecisionRecord, ...]
    terminal_state_json: str
    reward: float
    output: str
    usage_json: str
    pair_id: str | None = None
    candidate_id: str | None = None
    format: str = EVOSTEER_FORMAT
    risk: TrajectoryRiskAssessment | None = None

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "batch_id",
            "behavior_policy_id",
            "reference_id",
            "executor_id",
            "menu_id",
            "value_snapshot_id",
            "statistics_context",
        ):
            _text(getattr(self, name), name)
        _text(self.source, "source")
        if self.format != EVOSTEER_FORMAT or self.source not in TRAJECTORY_SOURCES:
            raise ValueError("unsupported EvoSteer trajectory identity")
        if not isinstance(self.task, EvoTask):
            raise ValueError("trajectory requires a typed public task")
        if (
            not isinstance(self.decisions, tuple)
            or not self.decisions
            or any(not isinstance(step, DecisionRecord) for step in self.decisions)
            or not 0 <= _number(self.reward, "reward") <= 1
        ):
            raise ValueError("complete trajectory and bounded terminal reward required")
        _text(self.output, "output", allow_empty=True)
        terminal_state = _canonical_object(self.terminal_state_json, "terminal_state_json")
        if terminal_state.get("stopped") is not True:
            raise ValueError("complete trajectories require an explicitly stopped terminal state")
        if (
            _canonical_object(self.decisions[-1].action_json, "terminal action").get("kind")
            != "STOP"
        ):
            raise ValueError("complete trajectories must end with the STOP action")
        usage = _canonical_object(self.usage_json, "usage_json")
        if any(type(value) is not int or value < 0 for value in usage.values()):
            raise ValueError("usage entries must be nonnegative integer resource counts")
        paired = self.source.startswith("paired_")
        if paired and (self.pair_id is None or self.candidate_id is None):
            raise ValueError("paired provenance is incomplete")
        if not paired and (self.pair_id is not None or self.candidate_id is not None):
            raise ValueError("natural trajectories cannot carry paired provenance")
        if paired:
            _text(self.pair_id, "pair_id")
            _text(self.candidate_id, "candidate_id")
        if paired and (not self.decisions[0].forced or any(s.forced for s in self.decisions[1:])):
            raise ValueError("paired first action must be recorded as forced")
        if not paired and any(step.forced for step in self.decisions):
            raise ValueError("natural trajectories cannot contain forced actions")
        if len({step.state_id for step in self.decisions}) != len(self.decisions):
            raise ValueError("history states must have unique identities")
        if self.source != "current" and self.behavior_policy_id != self.reference_id:
            raise ValueError("reference trajectories must name the frozen behavior policy")
        if self.risk is not None:
            if not isinstance(self.risk, TrajectoryRiskAssessment):
                raise ValueError("risk must be an explicitly assessed typed record")
            if self.risk.evidence_id != self.risk_evidence_id:
                raise ValueError("risk assessment belongs to different executed evidence")

    @property
    def risk_evidence_id(self) -> str:
        """Bind the entire executed record without a circular risk field."""
        value = asdict(self)
        value.pop("risk")
        return str(stable_hash(value))

    @property
    def identity(self) -> str:
        return str(stable_hash(self.to_value()))

    def to_value(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(canonical_json(asdict(self))))

    @classmethod
    def from_value(cls, value: dict[str, object]) -> EvoTrajectory:
        item = _wire_object(
            value,
            {
                "sample_id",
                "batch_id",
                "task",
                "source",
                "behavior_policy_id",
                "reference_id",
                "executor_id",
                "menu_id",
                "value_snapshot_id",
                "statistics_context",
                "decisions",
                "terminal_state_json",
                "reward",
                "output",
                "usage_json",
                "pair_id",
                "candidate_id",
                "format",
                "risk",
            },
            "trajectory",
        )
        return cls(
            sample_id=_text(item["sample_id"], "sample_id"),
            batch_id=_text(item["batch_id"], "batch_id"),
            task=EvoTask.from_value(item["task"]),
            source=_text(item["source"], "source"),
            behavior_policy_id=_text(item["behavior_policy_id"], "behavior_policy_id"),
            reference_id=_text(item["reference_id"], "reference_id"),
            executor_id=_text(item["executor_id"], "executor_id"),
            menu_id=_text(item["menu_id"], "menu_id"),
            value_snapshot_id=_text(item["value_snapshot_id"], "value_snapshot_id"),
            statistics_context=_text(item["statistics_context"], "statistics_context"),
            decisions=tuple(
                DecisionRecord.from_value(cast(dict[str, object], row))
                for row in _array(item["decisions"], "decisions")
            ),
            terminal_state_json=_text(item["terminal_state_json"], "terminal_state_json"),
            reward=_number(item["reward"], "reward"),
            output=_text(item["output"], "output", allow_empty=True),
            usage_json=_text(item["usage_json"], "usage_json"),
            pair_id=None if item["pair_id"] is None else _text(item["pair_id"], "pair_id"),
            candidate_id=None
            if item["candidate_id"] is None
            else _text(item["candidate_id"], "candidate_id"),
            format=_text(item["format"], "format"),
            risk=None
            if item["risk"] is None
            else TrajectoryRiskAssessment.from_value(item["risk"]),
        )
