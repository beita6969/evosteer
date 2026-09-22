"""Install completed, read-only T0 evidence without collecting replacement answers."""

from __future__ import annotations

import json
from pathlib import Path

from skillev.training.quality_gate import ProtocolProbe, evaluate_quality
from skillev.training.quality_monitor import QualityCheckpointStop

from .quality_panel import FixedQualityPanel


def import_initial_quality_probe(
    source: Path,
    *,
    monitor: QualityCheckpointStop,
    panel: FixedQualityPanel,
    optimizer_step: int,
    policy_snapshot_id: str,
) -> ProtocolProbe:
    """First installation binds the actual initial policy; later resumes only reuse it.

    Compare original bytes, not a rewritten projection. A later policy cannot
    authorize installing previously unbound T0 evidence. The import note records
    the initial binding, not a replacement snapshot identity inside the probe.
    """
    raw = source.read_bytes()
    probe = ProtocolProbe(**json.loads(raw))
    if (
        type(probe.policy_step) is not int
        or probe.policy_step != 0
        or type(probe.source_question_count) is not int
        or optimizer_step < 0
    ):
        raise ValueError("initial quality evidence requires integer T0/source coordinates")
    if (panel.panel_id, panel.condition_id) != (
        monitor.policy.panel_id,
        monitor.policy.condition_id,
    ):
        raise ValueError("initial quality import requires the declared panel and policy")
    if probe.source_question_count != len({slot.canonical_source for slot in panel.slots}):
        raise ValueError("initial quality source count differs from the bound panel")
    target = monitor.root / "probe-00000000.json"
    provenance = monitor.root / "initial-probe-import.json"
    if target.exists() and target.read_bytes() != raw:
        raise ValueError("a different initial quality baseline already exists")
    note = json.loads(provenance.read_bytes()) if provenance.exists() else None
    if note is not None and (
        note.get("initial_policy_snapshot_id") != probe.policy_snapshot_id
        or note.get("evidence_id") != probe.evidence_id
    ):
        raise ValueError("initial quality import provenance differs")
    if optimizer_step == 0:
        expected_snapshot = policy_snapshot_id
    else:
        if not target.is_file() or note is None:
            raise ValueError("resume can only reuse an already imported identical T0 baseline")
        expected_snapshot = note["initial_policy_snapshot_id"]
    decision = evaluate_quality(
        monitor.policy,
        baseline=probe,
        probes=(),
        policy_step=0,
        policy_snapshot_id=expected_snapshot,
    )
    if decision.status != "verified" or decision.action != "continue":
        raise ValueError(
            f"imported T0 does not pass the declared quality rules: {decision.to_value()}"
        )
    # Nothing is installed until all identity/count/T0 admission checks pass.
    # Exclusive creation never replaces a baseline or provenance from another run.
    if note is None:
        with provenance.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "source": str(source.resolve()),
                    "evidence_id": probe.evidence_id,
                    "initial_policy_snapshot_id": policy_snapshot_id,
                    "meaning": "original diagnostic T0 import; no collection or training update",
                    "decision": decision.to_value(),
                },
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            stream.write("\n")
    if not target.exists():
        with target.open("xb") as stream:
            stream.write(raw)
    return probe
