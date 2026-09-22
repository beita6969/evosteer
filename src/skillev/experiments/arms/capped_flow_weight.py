"""The capped-flow-weight arm, isolated from full calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from skillev.calibration import CalibrationConfig, CalibrationEngine, flow_weights
from skillev.contracts import PosteriorCellState
from skillev.diagnostics import TrajectoryFlowDiagnostic


@dataclass(frozen=True, slots=True)
class CappedFlowWeightConfig:
    cap: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.cap, bool)
            or not isinstance(self.cap, int | float)
            or not math.isfinite(float(self.cap))
            or float(self.cap) <= 0.0
        ):
            raise ValueError("cap must be finite and positive")
        object.__setattr__(self, "cap", float(self.cap))


def capped_flow_weights_for(
    flows: tuple[TrajectoryFlowDiagnostic, ...],
    *,
    cap: float,
) -> dict[tuple[str, int], float]:
    """Apply the preregistered cap to the live full-method weights."""

    checked = CappedFlowWeightConfig(cap=cap)
    return {key: min(value, checked.cap) for key, value in flow_weights(flows).items()}


def capped_flow_weights(
    flows: tuple[TrajectoryFlowDiagnostic, ...],
    config: CappedFlowWeightConfig,
) -> dict[tuple[str, int], float]:
    """Compatibility spelling for the frozen public arm helper."""

    return capped_flow_weights_for(flows, cap=config.cap)


class CappedFlowCalibrationEngine(CalibrationEngine):
    """Use the declared capped kernel without changing full-core calibration."""

    def __init__(
        self,
        config: CalibrationConfig,
        cap_config: CappedFlowWeightConfig,
    ) -> None:
        super().__init__(config)
        self.cap_config = cap_config

    @classmethod
    def from_runtime_state(
        cls,
        config: CalibrationConfig,
        cells: tuple[PosteriorCellState, ...],
    ) -> CappedFlowCalibrationEngine:
        """Hydrate with the sole preregistered cap used by this arm.

        The public arm protocol fixes the cap, so restoration cannot accept a
        caller-controlled replacement value.
        """

        from skillev.experiments.protocol import CappedFlowWeightArmProtocol

        restored = CalibrationEngine.from_runtime_state(config, cells)
        engine = cls(
            config,
            CappedFlowWeightConfig(cap=CappedFlowWeightArmProtocol.CAP),
        )
        engine._cells = dict(restored._cells)
        return engine

    def _flow_weights(
        self,
        flows: tuple[TrajectoryFlowDiagnostic, ...],
    ) -> Mapping[tuple[str, int], float]:
        return capped_flow_weights_for(flows, cap=self.cap_config.cap)


__all__ = [
    "CappedFlowCalibrationEngine",
    "CappedFlowWeightConfig",
    "capped_flow_weights",
    "capped_flow_weights_for",
]
