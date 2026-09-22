"""Deterministic residual-and-entropy phase detector for the full method."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise
from typing import TypeAlias

from skillev.contracts import (
    EntropyObservation,
    JsonValue,
    PhaseTransitionEvent,
    PhaseTriggerRule,
    WindowStats,
    stable_hash,
)
from skillev.diagnostics import (
    BatchDiagnostics,
    ComparableResidualWindows,
    DiagnosticsConfig,
    InsufficientResidualWindow,
    ZeroPreviousResidual,
)

from .config import EvolutionConfig


@dataclass(frozen=True, slots=True)
class DetectorBatchRecord:
    diagnostic: BatchDiagnostics
    skill_invocation_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        expected = tuple(
            (skill.skill_id, skill.invoking_edge_count)
            for skill in sorted(self.diagnostic.skill_flows, key=lambda item: item.skill_id)
        )
        if self.skill_invocation_counts != expected:
            raise ValueError("detector skill counts differ from the diagnostic")

    @classmethod
    def from_diagnostic(cls, diagnostic: BatchDiagnostics) -> DetectorBatchRecord:
        return cls(
            diagnostic=diagnostic,
            skill_invocation_counts=tuple(
                (skill.skill_id, skill.invoking_edge_count)
                for skill in sorted(diagnostic.skill_flows, key=lambda item: item.skill_id)
            ),
        )

    @property
    def batch_id(self) -> str:
        return self.diagnostic.batch_id

    @property
    def optimizer_step(self) -> int:
        return self.diagnostic.optimizer_step

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "diagnostic": self.diagnostic.to_value(),
            "skill_invocation_counts": [list(item) for item in self.skill_invocation_counts],
        }

    @classmethod
    def from_value(cls, value: object) -> DetectorBatchRecord:
        if not isinstance(value, dict) or set(value) != {
            "diagnostic",
            "skill_invocation_counts",
        }:
            raise ValueError("DetectorBatchRecord has incompatible fields")
        raw_counts = value["skill_invocation_counts"]
        if not isinstance(raw_counts, list):
            raise TypeError("detector skill counts must be an array")
        counts: list[tuple[str, int]] = []
        for item in raw_counts:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or type(item[1]) is not int
                or item[1] < 0
            ):
                raise ValueError("detector skill count entry is invalid")
            counts.append((item[0], item[1]))
        return cls(
            diagnostic=BatchDiagnostics.from_value(value["diagnostic"]),
            skill_invocation_counts=tuple(counts),
        )


@dataclass(frozen=True, slots=True)
class InsufficientEntropyWindow:
    required_observations: int
    observed_observations: int
    kind: str = "insufficient"

    def __post_init__(self) -> None:
        if type(self.required_observations) is not int or self.required_observations < 1:
            raise ValueError("required entropy observations must be positive")
        if (
            type(self.observed_observations) is not int
            or not 0 <= self.observed_observations < self.required_observations
        ):
            raise ValueError("observed entropy count must be insufficient")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind,
            "observed_observations": self.observed_observations,
            "required_observations": self.required_observations,
        }


@dataclass(frozen=True, slots=True)
class EntropyWindow:
    observations: tuple[EntropyObservation, ...]
    kind: str = "available"

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError("available entropy window requires observations")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind,
            "observations": [item.to_value() for item in self.observations],
        }


EntropyEvidence: TypeAlias = InsufficientEntropyWindow | EntropyWindow


class PhaseStatus(StrEnum):
    SEARCHING = "searching"
    DETECTED = "detected"
    NO_OP_WAITING = "no-op-waiting-for-fresh-evidence"


@dataclass(frozen=True, slots=True)
class AwaitingDetectorSegment:
    expected_library_version: str
    kind: str = "awaiting"

    def __post_init__(self) -> None:
        if not self.expected_library_version.strip():
            raise ValueError("expected detector library version must be non-empty")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "expected_library_version": self.expected_library_version,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class NoOpPhaseClosure:
    phase_event_id: str
    optimizer_step: int
    reason: str
    batch_id: str

    def __post_init__(self) -> None:
        if (
            not self.phase_event_id
            or not self.reason
            or not self.batch_id
            or self.optimizer_step < 1
        ):
            raise ValueError("no-op closure requires its phase, step and reason")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "phase_event_id": self.phase_event_id,
            "optimizer_step": self.optimizer_step,
            "reason": self.reason,
            "batch_id": self.batch_id,
        }


@dataclass(frozen=True, slots=True)
class ActiveDetectorSegment:
    library_version: str
    batch_records: tuple[DetectorBatchRecord, ...]
    entropy_evidence: EntropyEvidence
    phase_already_triggered: bool
    cursor: int
    no_op_closure: NoOpPhaseClosure | None = None
    kind: str = "active"

    def __post_init__(self) -> None:
        if not self.library_version.strip():
            raise ValueError("library_version must be non-empty")
        if type(self.cursor) is not int or self.cursor < 1:
            raise ValueError("detector cursor must be positive")
        if not self.batch_records:
            raise ValueError("active detector state requires records")
        if self.batch_records[-1].optimizer_step != self.cursor:
            raise ValueError("detector cursor differs from its last record")
        if any(
            item.diagnostic.library_version != self.library_version for item in self.batch_records
        ):
            raise ValueError("detector records cross library segments")
        steps = tuple(item.optimizer_step for item in self.batch_records)
        if tuple(sorted(set(steps))) != steps:
            raise ValueError("detector record steps must be strictly increasing")
        if type(self.phase_already_triggered) is not bool:
            raise TypeError("phase_already_triggered must be boolean")
        if self.no_op_closure is not None and (
            not self.phase_already_triggered or self.no_op_closure.optimizer_step > self.cursor
        ):
            raise ValueError("no-op closure does not belong to a detected phase")

    @property
    def phase_status(self) -> PhaseStatus:
        if self.no_op_closure is not None:
            return PhaseStatus.NO_OP_WAITING
        return PhaseStatus.DETECTED if self.phase_already_triggered else PhaseStatus.SEARCHING

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "no_op_closure": None if self.no_op_closure is None else self.no_op_closure.to_value(),
            "cursor": self.cursor,
            "entropy_evidence": self.entropy_evidence.to_value(),
            "kind": self.kind,
            "library_version": self.library_version,
            "batch_records": [item.to_value() for item in self.batch_records],
            "phase_already_triggered": self.phase_already_triggered,
        }


DetectorRuntimeState: TypeAlias = AwaitingDetectorSegment | ActiveDetectorSegment


def detector_state_from_value(value: object) -> DetectorRuntimeState:
    if not isinstance(value, dict):
        raise TypeError("detector runtime state must be an object")
    kind = value.get("kind")
    if kind == "awaiting":
        if set(value) != {"expected_library_version", "kind"}:
            raise ValueError("AwaitingDetectorSegment has incompatible fields")
        version = value["expected_library_version"]
        if not isinstance(version, str):
            raise TypeError("expected detector library version must be text")
        return AwaitingDetectorSegment(version)
    if kind == "active":
        if set(value) != {
            "no_op_closure",
            "batch_records",
            "cursor",
            "entropy_evidence",
            "kind",
            "library_version",
            "phase_already_triggered",
        }:
            raise ValueError("ActiveDetectorSegment has incompatible fields")
        records = value["batch_records"]
        entropy = value["entropy_evidence"]
        cursor = value["cursor"]
        library_version = value["library_version"]
        triggered = value["phase_already_triggered"]
        if not isinstance(records, list):
            raise TypeError("detector records must be an array")
        if type(cursor) is not int or not isinstance(library_version, str):
            raise TypeError("detector state identity has incompatible types")
        if type(triggered) is not bool:
            raise TypeError("detector triggered state must be boolean")
        return ActiveDetectorSegment(
            no_op_closure=None
            if value["no_op_closure"] is None
            else NoOpPhaseClosure(**value["no_op_closure"]),
            library_version=library_version,
            batch_records=tuple(DetectorBatchRecord.from_value(item) for item in records),
            entropy_evidence=_entropy_evidence_from_value(entropy),
            phase_already_triggered=triggered,
            cursor=cursor,
        )
    raise ValueError("unsupported detector runtime state kind")


def detector_library_version(state: DetectorRuntimeState) -> str:
    match state:
        case AwaitingDetectorSegment(expected_library_version=version):
            return version
        case ActiveDetectorSegment(library_version=version):
            return version
    from typing import assert_never

    assert_never(state)


def _entropy_evidence_from_value(value: object) -> EntropyEvidence:
    if not isinstance(value, dict):
        raise TypeError("entropy evidence must be an object")
    kind = value.get("kind")
    if kind == "insufficient" and set(value) == {
        "kind",
        "observed_observations",
        "required_observations",
    }:
        required = value["required_observations"]
        observed = value["observed_observations"]
        if type(required) is not int or type(observed) is not int:
            raise TypeError("entropy evidence counts must be integers")
        return InsufficientEntropyWindow(required, observed)
    if kind == "available" and set(value) == {"kind", "observations"}:
        observations = value["observations"]
        if not isinstance(observations, list):
            raise TypeError("entropy observations must be an array")
        return EntropyWindow(tuple(EntropyObservation.from_value(item) for item in observations))
    raise ValueError("unsupported entropy evidence kind")


class NoPhaseTransitionReason(StrEnum):
    SLOT_DISABLED = "slot-does-not-allow-phase-detection"
    NO_OP_WAITING = "no-op-waiting-for-fresh-evidence"
    INSUFFICIENT_RESIDUAL_WINDOW = "insufficient-residual-window"
    ZERO_PREVIOUS_RESIDUAL = "zero-previous-residual"
    RESIDUAL_NOT_STAGNANT = "residual-not-stagnant"
    INSUFFICIENT_ENTROPY_HISTORY = "insufficient-entropy-history"
    ENTROPY_NOT_STRICTLY_DECREASING = "entropy-not-strictly-decreasing"
    PHASE_ALREADY_TRIGGERED = "phase-already-triggered"
    NO_INVOCATION_COVERAGE = "no-invocation-coverage"


@dataclass(frozen=True, slots=True)
class NoPhaseTransition:
    next_state: DetectorRuntimeState
    reason: NoPhaseTransitionReason


@dataclass(frozen=True, slots=True)
class PhaseTransitionDetected:
    next_state: ActiveDetectorSegment
    event: PhaseTransitionEvent
    triggering_window: tuple[BatchDiagnostics, ...]


DetectorObservation: TypeAlias = NoPhaseTransition | PhaseTransitionDetected


class PhaseTransitionDetector:
    """Fold committed diagnostics into one exact library-segment cursor."""

    def __init__(
        self,
        *,
        evolution_config: EvolutionConfig,
        diagnostics_config: DiagnosticsConfig,
        state: DetectorRuntimeState,
    ) -> None:
        self._evolution_config = evolution_config
        self._diagnostics_config = diagnostics_config
        self._state = state

    @classmethod
    def fresh(
        cls,
        *,
        evolution_config: EvolutionConfig,
        diagnostics_config: DiagnosticsConfig,
        library_version: str,
    ) -> PhaseTransitionDetector:
        return cls(
            evolution_config=evolution_config,
            diagnostics_config=diagnostics_config,
            state=AwaitingDetectorSegment(library_version),
        )

    @classmethod
    def from_runtime_state(
        cls,
        *,
        evolution_config: EvolutionConfig,
        diagnostics_config: DiagnosticsConfig,
        expected_library_version: str,
        state: DetectorRuntimeState,
    ) -> PhaseTransitionDetector:
        if detector_library_version(state) != expected_library_version:
            raise ValueError("detector snapshot belongs to another library")
        return cls(
            evolution_config=evolution_config,
            diagnostics_config=diagnostics_config,
            state=state,
        )

    @property
    def state(self) -> DetectorRuntimeState:
        return self._state

    def reset_for_library(
        self,
        old_library_version: str,
        new_library_version: str,
    ) -> None:
        self.commit_reset(
            self.preview_reset(
                old_version=old_library_version,
                new_version=new_library_version,
            )
        )

    def preview_reset(
        self,
        *,
        old_version: str,
        new_version: str,
    ) -> AwaitingDetectorSegment:
        if detector_library_version(self._state) != old_version:
            raise ValueError("detector reset old library identity differs")
        if new_version == old_version:
            raise ValueError("detector reset requires a new library version")
        return AwaitingDetectorSegment(new_version)

    def commit_reset(self, state: AwaitingDetectorSegment) -> None:
        self._state = state

    def observe(self, diagnostic: BatchDiagnostics) -> DetectorObservation:
        """Compatibility operation that previews and commits one observation."""

        observation = self.preview_observation(diagnostic)
        self.commit_observation(observation)
        return observation

    def preview_observation(
        self, diagnostic: BatchDiagnostics, *, cold_start_supported: bool = False
    ) -> DetectorObservation:
        """Compute the next detector state without mutating the live segment."""

        if diagnostic.config != self._diagnostics_config:
            raise ValueError("diagnostic config differs from detector config")

        previous_state = self._state
        match previous_state:
            case AwaitingDetectorSegment(expected_library_version=expected):
                if diagnostic.library_version != expected:
                    raise ValueError("detector received a batch for another library")
                prior_records: tuple[DetectorBatchRecord, ...] = ()
                already_triggered = False
            case ActiveDetectorSegment(
                library_version=expected,
                batch_records=prior_records,
                phase_already_triggered=already_triggered,
            ):
                if diagnostic.library_version != expected:
                    raise ValueError("detector library changed without an explicit reset")
                if diagnostic.optimizer_step <= previous_state.cursor:
                    raise ValueError("diagnostic steps must be strictly increasing")
            case _:
                from typing import assert_never

                assert_never(previous_state)

        record = DetectorBatchRecord.from_diagnostic(diagnostic)
        record_limit = max(
            self._evolution_config.entropy_window
            + self._evolution_config.required_consecutive_drops,
            2 * self._diagnostics_config.window_size,
        )
        records = (*prior_records, record)[-record_limit:]
        closure = (
            previous_state.no_op_closure
            if isinstance(previous_state, ActiveDetectorSegment)
            else None
        )
        # Consume evidence, not the library. Both residual windows and the entire
        # entropy comparison must consist of genuinely newer committed batches.
        # Count actual records rather than a step-number gap (closure slots need
        # not have been observed). No thresholds, posterior or Z are changed.
        if (
            closure is not None
            and len(records) == record_limit
            and all(item.optimizer_step > closure.optimizer_step for item in records)
        ):
            already_triggered = False
            closure = None
        entropy_series = _entropy_series_from_records(
            records,
            window_size=self._evolution_config.entropy_window,
            retained_points=self._evolution_config.required_consecutive_drops + 1,
        )
        required_points = self._evolution_config.required_consecutive_drops + 1
        entropy_evidence: EntropyEvidence
        if len(entropy_series) < required_points:
            entropy_evidence = InsufficientEntropyWindow(
                required_observations=required_points,
                observed_observations=len(entropy_series),
            )
        else:
            entropy_evidence = EntropyWindow(entropy_series[-required_points:])
        next_state = ActiveDetectorSegment(
            no_op_closure=closure,
            library_version=diagnostic.library_version,
            batch_records=records,
            entropy_evidence=entropy_evidence,
            phase_already_triggered=already_triggered,
            cursor=diagnostic.optimizer_step,
        )
        if already_triggered:
            return NoPhaseTransition(
                next_state,
                (
                    NoPhaseTransitionReason.NO_OP_WAITING
                    if next_state.no_op_closure is not None
                    else NoPhaseTransitionReason.PHASE_ALREADY_TRIGGERED
                ),
            )

        residual = diagnostic.residual_window
        match residual:
            case InsufficientResidualWindow():
                return NoPhaseTransition(
                    next_state,
                    NoPhaseTransitionReason.INSUFFICIENT_RESIDUAL_WINDOW,
                )
            case ZeroPreviousResidual():
                return NoPhaseTransition(
                    next_state,
                    NoPhaseTransitionReason.ZERO_PREVIOUS_RESIDUAL,
                )
            case ComparableResidualWindows(stagnant=False):
                return NoPhaseTransition(
                    next_state,
                    NoPhaseTransitionReason.RESIDUAL_NOT_STAGNANT,
                )
            case ComparableResidualWindows(stagnant=True):
                pass
            case _:
                raise TypeError("unsupported residual-window evidence")

        cold_start = self._evolution_config.cold_start is not None and cold_start_supported
        entropy_met = _entropy_condition_met(
            entropy_series,
            required_consecutive_drops=self._evolution_config.required_consecutive_drops,
        )
        if isinstance(entropy_evidence, InsufficientEntropyWindow) and not cold_start:
            return NoPhaseTransition(
                next_state,
                NoPhaseTransitionReason.INSUFFICIENT_ENTROPY_HISTORY,
            )
        if not cold_start and any(
            item.total_invocations == 0 for item in entropy_series[-required_points:]
        ):
            # Empty windows serialize entropy=0 for historical reporting, but
            # have no empirical distribution. Losing all calls is not an entropy
            # decrease. Keep the explicitly declared legacy extension separate.
            return NoPhaseTransition(next_state, NoPhaseTransitionReason.NO_INVOCATION_COVERAGE)
        if not entropy_met and not cold_start:
            return NoPhaseTransition(
                next_state,
                NoPhaseTransitionReason.ENTROPY_NOT_STRICTLY_DECREASING,
            )

        window_size = self._diagnostics_config.window_size
        triggering_records = records[-2 * window_size :]
        if len(triggering_records) != 2 * window_size:
            raise RuntimeError("residual evidence exists without its exact batch window")
        event = _phase_event_from(
            diagnostic=diagnostic,
            residual=residual,
            entropy_series=entropy_series,
            records=triggering_records,
            evolution_config=self._evolution_config,
            trigger_rule=(
                PhaseTriggerRule.RESIDUAL_AND_ENTROPY
                if entropy_met
                else PhaseTriggerRule.ZERO_COVERAGE_COLD_START
            ),
        )
        triggered_state = ActiveDetectorSegment(
            library_version=next_state.library_version,
            batch_records=next_state.batch_records,
            entropy_evidence=next_state.entropy_evidence,
            phase_already_triggered=True,
            cursor=next_state.cursor,
        )
        return PhaseTransitionDetected(
            next_state=triggered_state,
            event=event,
            triggering_window=tuple(item.diagnostic for item in triggering_records),
        )

    def commit_observation(self, observation: DetectorObservation) -> None:
        if not isinstance(observation, NoPhaseTransition | PhaseTransitionDetected):
            raise TypeError("detector commit requires a previewed observation")
        if detector_library_version(observation.next_state) != detector_library_version(
            self._state
        ):
            raise ValueError("detector observation belongs to an old library segment")
        if isinstance(self._state, ActiveDetectorSegment) and (
            not isinstance(observation.next_state, ActiveDetectorSegment)
            or observation.next_state.cursor <= self._state.cursor
        ):
            raise ValueError("detector observation was prepared from an old state")
        self._state = observation.next_state

    def close_no_op(self, phase: PhaseTransitionEvent, reason: str) -> None:
        self._state = close_no_op_segment(self._state, phase, reason)


def close_no_op_segment(
    state: DetectorRuntimeState, phase: PhaseTransitionEvent, reason: str
) -> ActiveDetectorSegment:
    """Consume this phase cutoff, without permanently closing its library.

    A no-op changes neither diagnostics, posterior, library nor Z. The persisted
    cutoff prevents reuse until a complete fresh comparison has accumulated.
    """

    if (
        not isinstance(state, ActiveDetectorSegment)
        or not state.phase_already_triggered
        or state.library_version != phase.library_version
        or state.cursor != phase.triggered_at_step
        or state.no_op_closure is not None
    ):
        raise ValueError("no-op resolution requires the current unresolved phase")
    return replace(
        state,
        no_op_closure=NoOpPhaseClosure(
            phase.event_id, phase.triggered_at_step, reason, state.batch_records[-1].batch_id
        ),
    )


def _entropy_series_from_records(
    records: tuple[DetectorBatchRecord, ...],
    *,
    window_size: int,
    retained_points: int,
) -> tuple[EntropyObservation, ...]:
    if len(records) < window_size:
        return ()
    start = max(window_size, len(records) - retained_points + 1)
    return tuple(
        entropy
        for end in range(start, len(records) + 1)
        if (entropy := _current_entropy(records[:end], window_size=window_size)) is not None
    )[-retained_points:]


def _current_entropy(
    records: tuple[DetectorBatchRecord, ...],
    *,
    window_size: int,
) -> EntropyObservation | None:
    if len(records) < window_size:
        return None
    counts: Counter[str] = Counter()
    for record in records[-window_size:]:
        counts.update(dict(record.skill_invocation_counts))
    positive = {skill_id: count for skill_id, count in counts.items() if count > 0}
    total = sum(positive.values())
    if total == 0:
        return EntropyObservation(
            window_end_step=records[-1].optimizer_step,
            entropy=0.0,
            total_invocations=0,
            distinct_skills=0,
        )
    probabilities = tuple(positive[key] / total for key in sorted(positive))
    return EntropyObservation(
        window_end_step=records[-1].optimizer_step,
        entropy=-math.fsum(probability * math.log(probability) for probability in probabilities),
        total_invocations=total,
        distinct_skills=len(positive),
    )


def _entropy_condition_met(
    series: tuple[EntropyObservation, ...],
    *,
    required_consecutive_drops: int,
) -> bool:
    required_points = required_consecutive_drops + 1
    if len(series) < required_points:
        return False
    return all(
        current.entropy < previous.entropy
        for previous, current in pairwise(series[-required_points:])
    )


def _window_stats(
    records: tuple[DetectorBatchRecord, ...],
    *,
    mean_squared_residual: float,
) -> WindowStats:
    return WindowStats(
        start_optimizer_step=records[0].optimizer_step,
        end_optimizer_step=records[-1].optimizer_step,
        batch_count=len(records),
        mean_squared_residual=mean_squared_residual,
        member_batch_ids=tuple(record.batch_id for record in records),
    )


def _phase_event_from(
    *,
    diagnostic: BatchDiagnostics,
    residual: ComparableResidualWindows,
    entropy_series: tuple[EntropyObservation, ...],
    records: tuple[DetectorBatchRecord, ...],
    evolution_config: EvolutionConfig,
    trigger_rule: PhaseTriggerRule = PhaseTriggerRule.RESIDUAL_AND_ENTROPY,
) -> PhaseTransitionEvent:
    window_size = residual.window_size
    previous = _window_stats(
        records[:window_size],
        mean_squared_residual=residual.previous_window_mean_delta_squared,
    )
    current = _window_stats(
        records[window_size:],
        mean_squared_residual=residual.current_window_mean_delta_squared,
    )
    return PhaseTransitionEvent(
        event_id=(
            "cold-start:" if trigger_rule is PhaseTriggerRule.ZERO_COVERAGE_COLD_START else ""
        )
        + stable_hash(
            {
                "current_batch_ids": list(current.member_batch_ids),
                "library_version": diagnostic.library_version,
                "previous_batch_ids": list(previous.member_batch_ids),
                "triggered_at_step": diagnostic.optimizer_step,
            }
        ),
        triggered_at_step=diagnostic.optimizer_step,
        library_version=diagnostic.library_version,
        previous_window=previous,
        current_window=current,
        relative_improvement=residual.relative_improvement,
        rho=residual.rho,
        residual_condition_met=True,
        entropy_series=entropy_series,
        trigger_rule=trigger_rule,
        required_consecutive_drops=evolution_config.required_consecutive_drops,
        entropy_condition_met=_entropy_condition_met(
            entropy_series, required_consecutive_drops=evolution_config.required_consecutive_drops
        ),
        triggered=True,
    )


__all__ = [
    "ActiveDetectorSegment",
    "AwaitingDetectorSegment",
    "DetectorBatchRecord",
    "DetectorObservation",
    "DetectorRuntimeState",
    "EntropyEvidence",
    "EntropyWindow",
    "InsufficientEntropyWindow",
    "NoPhaseTransition",
    "NoPhaseTransitionReason",
    "PhaseTransitionDetected",
    "PhaseTransitionDetector",
    "detector_library_version",
    "detector_state_from_value",
]
