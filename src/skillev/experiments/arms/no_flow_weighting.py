"""The no-flow-weighting arm's sole alternate calibration kernel."""

from collections.abc import Mapping

from skillev.calibration import CalibrationConfig, CalibrationEngine
from skillev.diagnostics import TrajectoryFlowDiagnostic


def unit_flow_weights_for(
    flows: tuple[TrajectoryFlowDiagnostic, ...],
) -> dict[tuple[str, int], float]:
    return {
        (trajectory.trajectory_id, edge.step_index): 1.0
        for trajectory in flows
        for edge in trajectory.edges
        if edge.invoked_skill_ids
    }


class UnitFlowCalibrationEngine(CalibrationEngine):
    """Change only the edge-weight kernel; preserve posterior order and prior."""

    def __init__(self, config: CalibrationConfig) -> None:
        super().__init__(config)

    def _flow_weights(
        self,
        flows: tuple[TrajectoryFlowDiagnostic, ...],
    ) -> Mapping[tuple[str, int], float]:
        return unit_flow_weights_for(flows)


__all__ = ["UnitFlowCalibrationEngine", "unit_flow_weights_for"]
