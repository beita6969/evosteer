"""Audited terminal state handed to the private frozen-evaluation boundary."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.contracts import PosteriorCellState
from skillev.experiments.evolution_progress import TrainingEvolutionCounts
from skillev.runtime import SkillLibraryState


@dataclass(frozen=True, slots=True)
class AuditedFinalTrainingState:
    """Source-recomputed library and calibration state at a successful run end.

    This is intentionally an in-memory audit result, not a public result
    serialization.  Frozen evaluation receives it only through the private
    constructor that also verifies the final checkpoint artifact.
    """

    library: SkillLibraryState
    calibration_cells: tuple[PosteriorCellState, ...]
    final_policy_snapshot_id: str
    final_optimizer_step: int
    evolution_counts: TrainingEvolutionCounts

    def __post_init__(self) -> None:
        if not isinstance(self.library, SkillLibraryState):
            raise TypeError("audited final state requires SkillLibraryState")
        if not isinstance(self.calibration_cells, tuple) or any(
            not isinstance(cell, PosteriorCellState) for cell in self.calibration_cells
        ):
            raise TypeError("audited final state has invalid posterior cells")
        if len({cell.z.cell_key(cell.skill_id) for cell in self.calibration_cells}) != len(
            self.calibration_cells
        ):
            raise ValueError("audited final state repeats posterior cells")
        if type(self.final_policy_snapshot_id) is not str or not self.final_policy_snapshot_id:
            raise ValueError("audited final state policy snapshot is invalid")
        if type(self.final_optimizer_step) is not int or self.final_optimizer_step < 0:
            raise ValueError("audited final state optimizer step is invalid")
        if not isinstance(self.evolution_counts, TrainingEvolutionCounts):
            raise TypeError("audited final state evolution counts are invalid")


__all__ = ["AuditedFinalTrainingState"]
