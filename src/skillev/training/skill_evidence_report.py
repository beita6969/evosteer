"""Export skill evidence from one saved runtime state, without loading a model.

This private report contains library/source IDs and document descriptions; do not
publish it as a benchmark result. It neither updates cells nor invents missing
historical input visibility. No optimizer or sampling API is called.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillev.contracts import JsonValue
from skillev.runtime import FullRuntimeExecutionState, RuntimeSnapshot, SkillLibrary

from .calibration_reporting import posterior_evidence_composition, prequential_calibration
from .coverage_reporting import batch_coverage
from .skill_lifecycle import skill_lifecycle_report


def report_from_checkpoint(runtime_state: Path) -> dict[str, JsonValue]:
    if not (runtime_state.parent / "COMPLETE").is_file():
        raise ValueError("skill report requires a complete saved checkpoint")
    snapshot = RuntimeSnapshot.from_value(json.loads(runtime_state.read_text(encoding="utf-8")))
    state = snapshot.execution_state
    if not isinstance(state, FullRuntimeExecutionState):
        raise ValueError("skill posterior report requires full-method runtime state")
    provenance = state.projections.posterior_provenance
    committed = provenance.batches[-1].optimizer_step if provenance.batches else 0
    if (
        committed != snapshot.optimizer_step
        or state.run_cursor.completed_training_steps != committed
    ):
        raise ValueError("skill evidence and saved checkpoint have different commit boundaries")
    return {
        "format": "skillev-saved-skill-evidence-report@1",
        "experiment_id": snapshot.experiment_id,
        "optimizer_step": snapshot.optimizer_step,
        "library_version": state.library.current_version,
        "skill_lifecycle": skill_lifecycle_report(SkillLibrary(state.library), provenance),
        "skill_coverage": [batch_coverage(batch) for batch in provenance.batches],
        "prequential_calibration": prequential_calibration(provenance),
        "posterior_evidence_composition": posterior_evidence_composition(provenance),
        "interpretation": "read-only-saved-evidence; no-causal-efficacy-or-natural-trigger-claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = report_from_checkpoint(args.runtime_state)
    # Each output names one immutable checkpoint report. Never overwrite old evidence.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
