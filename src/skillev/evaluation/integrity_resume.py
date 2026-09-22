"""Same-run execution modes and read-only historical diagnostics.

Historical answers keep their original identity. There is deliberately no SQL
copy/import path into a new run, and no generation or grading callback here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .sealed_candidates import CandidateReader, FinalCandidate


class EvaluationRunMode(StrEnum):
    FORMAL_FRESH = "formal-fresh"
    SAME_RUN_RESUME = "same-run-resume"
    HISTORICAL_DIAGNOSTIC = "historical-diagnostic"


def require_execution_mode(
    mode: EvaluationRunMode, *, directory: Path, importing_other_run: bool
) -> None:
    if not isinstance(mode, EvaluationRunMode):
        raise TypeError("evaluation execution mode must be explicit")
    if importing_other_run:
        raise ValueError("new evaluation runs cannot import another run's candidates")
    if mode is EvaluationRunMode.HISTORICAL_DIAGNOSTIC:
        raise ValueError("historical diagnostics use the read-only history API, not generation")
    existing = (directory / "frozen-plan-private.json").exists()
    if mode is EvaluationRunMode.FORMAL_FRESH and existing:
        raise ValueError("existing evaluation needs explicit same-run-resume")
    database = directory / "candidates-private.sqlite"
    if mode is EvaluationRunMode.FORMAL_FRESH and database.exists():
        reader = CandidateReader(database)
        try:
            if (
                reader.connection.execute(
                    "SELECT 1 FROM candidates UNION ALL SELECT 1 FROM trace "
                    "UNION ALL SELECT 1 FROM executions UNION ALL SELECT 1 FROM scores "
                    "UNION ALL SELECT 1 FROM run_configuration LIMIT 1"
                ).fetchone()
                is not None
            ):
                raise ValueError("formal-fresh requires an unused candidate journal")
        finally:
            reader.close()
    if mode is EvaluationRunMode.SAME_RUN_RESUME and not existing:
        raise ValueError("same-run resume requires its original frozen plan")


@dataclass(frozen=True, slots=True)
class HistoricalArmSnapshot:
    origin_run_id: str
    arm_id: str
    candidates: tuple[FinalCandidate, ...]
    definitive_scores: tuple[dict[str, object] | None, ...]
    status: str = "historical-diagnostic-unverified; not a new evaluation"


def read_historical_arm(source_directory: Path, *, arm_id: str) -> HistoricalArmSnapshot:
    """Read all retained answers and scores, never rewrite provenance or regrade."""
    plan = json.loads((source_directory / "frozen-plan-private.json").read_text())
    origin = str(plan["run_id"])
    reader = CandidateReader(source_directory / "candidates-private.sqlite")
    try:
        candidates = tuple(
            FinalCandidate(**json.loads(payload))
            for (payload,) in reader.connection.execute(
                "SELECT payload FROM candidates WHERE run_id=? AND arm_id=? ORDER BY episode_id",
                (origin, arm_id),
            )
        )
        if any(
            candidate.run_id != origin or candidate.arm_id != arm_id for candidate in candidates
        ):
            raise ValueError("historical candidate identity disagrees with its stored scope")
        return HistoricalArmSnapshot(
            origin,
            arm_id,
            candidates,
            tuple(reader.stored_score((origin, arm_id, row.episode_id)) for row in candidates),
        )
    finally:
        reader.close()
