"""Frozen OOD source projections using the IID actor/scorer separation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from skillev.evaluation.corpus_search import INPUT_PROFILE
from skillev.evaluation.input_metric_contracts import OOD_BENCHMARKS, PublicTaskView
from skillev.evaluation.integrity_pipeline import FrozenPanel

from .integrity_sources import SourcePanel


def load_ood_panel(config: dict[str, Any]) -> SourcePanel:
    counts = config["evaluation_sample_counts"]
    entries = []
    targets: dict[str, dict[str, Any]] = {}
    interactive: dict[str, dict[str, Any]] = {}
    public_input_receipts = {}
    for benchmark, count in counts.items():
        if benchmark not in OOD_BENCHMARKS:
            raise ValueError("OOD benchmark is not enabled for this round")
        if type(count) is not int or count < 1:
            raise ValueError("OOD count must be positive")
        with Path(config["ood_sources"][benchmark]).open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) != count:
            raise ValueError("frozen OOD source differs from the declared sample count")
        for row in rows:
            task_id = row["task_id"]
            if task_id in targets or task_id in interactive:
                raise ValueError("duplicate OOD source identity")
            entry = PublicTaskView.from_record(task_id, benchmark, row["public"])
            if benchmark in config.get("corpus_retrieval", {}):
                entry = replace(entry, input_profile=INPUT_PROFILE)
            entries.append(entry)
            targets[task_id] = row.get("target", {})
            if benchmark == "musique" and "public_input_receipt" in row:
                public_input_receipts[task_id] = row["public_input_receipt"]["paragraphs"]
            if benchmark == "scienceworld":
                interactive[task_id] = row["interactive"]
    provenance = config["ood_provenance"]
    panel = FrozenPanel(
        tuple(entries),
        config.get("ood_panel_exposure", "frozen-result-blind-ood-infrastructure-panel"),
        json.dumps(provenance, sort_keys=True),
        tuple(counts.items()),
        OOD_BENCHMARKS,
    )
    panel.validate(canary=False)
    return SourcePanel(panel, targets, interactive, provenance, public_input_receipts)
