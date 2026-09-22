"""Residual-only phase detector for the declared single-condition arm."""

from __future__ import annotations

from skillev.contracts import (
    PhaseTransitionEvent,
    PhaseTriggerRule,
    WindowStats,
    stable_hash,
)
from skillev.diagnostics import (
    BatchDiagnostics,
    DiagnosticsConfig,
    InsufficientResidualWindow,
    ZeroPreviousResidual,
)
from skillev.evolution.detector import (
    ActiveDetectorSegment,
    AwaitingDetectorSegment,
    DetectorBatchRecord,
    DetectorObservation,
    DetectorRuntimeState,
    InsufficientEntropyWindow,
    NoPhaseTransition,
    NoPhaseTransitionReason,
    PhaseTransitionDetected,
    detector_library_version,
)


class ResidualOnlyPhaseDetector:
    def __init__(
        self,
        diagnostics_config: DiagnosticsConfig,
        *,
        state: DetectorRuntimeState,
    ) -> None:
        self._config = diagnostics_config
        self._state = state

    @classmethod
    def fresh(
        cls,
        diagnostics_config: DiagnosticsConfig,
        *,
        library_version: str,
    ) -> ResidualOnlyPhaseDetector:
        return cls(
            diagnostics_config,
            state=AwaitingDetectorSegment(library_version),
        )

    @classmethod
    def from_runtime_state(
        cls,
        *,
        diagnostics_config: DiagnosticsConfig,
        expected_library_version: str,
        state: DetectorRuntimeState,
    ) -> ResidualOnlyPhaseDetector:
        if detector_library_version(state) != expected_library_version:
            raise ValueError("residual-only detector snapshot belongs to another library")
        return cls(diagnostics_config, state=state)

    @property
    def state(self) -> DetectorRuntimeState:
        return self._state

    def observe(self, diagnostic: BatchDiagnostics) -> DetectorObservation:
        observation = self.preview_observation(diagnostic)
        self.commit_observation(observation)
        return observation

    def close_no_op(self, phase: PhaseTransitionEvent, reason: str) -> None:
        from skillev.evolution.detector import close_no_op_segment

        self._state = close_no_op_segment(self._state, phase, reason)

    def preview_observation(self, diagnostic: BatchDiagnostics) -> DetectorObservation:
        if diagnostic.config != self._config:
            raise ValueError("diagnostic configuration differs")
        previous = self._state
        if isinstance(previous, AwaitingDetectorSegment):
            if diagnostic.library_version != previous.expected_library_version:
                raise ValueError("residual-only detector received another library")
            records: tuple[DetectorBatchRecord, ...] = ()
            already_triggered = False
        else:
            if diagnostic.optimizer_step <= previous.cursor:
                raise ValueError("diagnostic steps must increase")
            if diagnostic.library_version != previous.library_version:
                raise ValueError("residual-only detector requires an explicit segment reset")
            records = previous.batch_records
            already_triggered = previous.phase_already_triggered
        records = (*records, DetectorBatchRecord.from_diagnostic(diagnostic))[
            -2 * self._config.window_size :
        ]
        next_state = ActiveDetectorSegment(
            no_op_closure=previous.no_op_closure
            if isinstance(previous, ActiveDetectorSegment)
            else None,
            library_version=diagnostic.library_version,
            batch_records=records,
            entropy_evidence=InsufficientEntropyWindow(1, 0),
            phase_already_triggered=already_triggered,
            cursor=diagnostic.optimizer_step,
        )
        if already_triggered:
            return NoPhaseTransition(
                next_state,
                NoPhaseTransitionReason.PHASE_ALREADY_TRIGGERED,
            )
        residual = diagnostic.residual_window
        if isinstance(residual, InsufficientResidualWindow):
            return NoPhaseTransition(
                next_state,
                NoPhaseTransitionReason.INSUFFICIENT_RESIDUAL_WINDOW,
            )
        if isinstance(residual, ZeroPreviousResidual):
            return NoPhaseTransition(
                next_state,
                NoPhaseTransitionReason.ZERO_PREVIOUS_RESIDUAL,
            )
        if not residual.stagnant:
            return NoPhaseTransition(
                next_state,
                NoPhaseTransitionReason.RESIDUAL_NOT_STAGNANT,
            )
        if len(records) != 2 * self._config.window_size:
            raise RuntimeError("residual evidence exists without its exact window")
        previous_records = records[: self._config.window_size]
        current_records = records[self._config.window_size :]
        previous_window = _window(
            previous_records,
            mean_squared_residual=residual.previous_window_mean_delta_squared,
        )
        current_window = _window(
            current_records,
            mean_squared_residual=residual.current_window_mean_delta_squared,
        )
        event = PhaseTransitionEvent(
            event_id=stable_hash(
                {
                    "arm": "residual-only",
                    "current_batch_ids": list(current_window.member_batch_ids),
                    "library_version": diagnostic.library_version,
                    "previous_batch_ids": list(previous_window.member_batch_ids),
                    "triggered_at_step": diagnostic.optimizer_step,
                }
            ),
            triggered_at_step=diagnostic.optimizer_step,
            library_version=diagnostic.library_version,
            previous_window=previous_window,
            current_window=current_window,
            relative_improvement=residual.relative_improvement,
            rho=residual.rho,
            residual_condition_met=True,
            entropy_series=(),
            trigger_rule=PhaseTriggerRule.RESIDUAL_ONLY,
            required_consecutive_drops=1,
            entropy_condition_met=False,
            triggered=True,
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
            triggering_window=tuple(item.diagnostic for item in records),
        )

    def commit_observation(self, observation: DetectorObservation) -> None:
        if not isinstance(observation, NoPhaseTransition | PhaseTransitionDetected):
            raise TypeError("residual-only detector commit requires a previewed observation")
        self._state = observation.next_state

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
        current = (
            self._state.expected_library_version
            if isinstance(self._state, AwaitingDetectorSegment)
            else self._state.library_version
        )
        if current != old_version:
            raise ValueError("residual-only detector reset old library differs")
        if new_version == old_version:
            raise ValueError("residual-only detector reset requires a new library")
        return AwaitingDetectorSegment(new_version)

    def commit_reset(self, state: AwaitingDetectorSegment) -> None:
        self._state = state


def _window(
    records: tuple[DetectorBatchRecord, ...],
    *,
    mean_squared_residual: float,
) -> WindowStats:
    return WindowStats(
        start_optimizer_step=records[0].optimizer_step,
        end_optimizer_step=records[-1].optimizer_step,
        batch_count=len(records),
        mean_squared_residual=mean_squared_residual,
        member_batch_ids=tuple(item.batch_id for item in records),
    )


__all__ = ["ResidualOnlyPhaseDetector"]
