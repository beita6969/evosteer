"""Copy a committed recovery prefix; never roll back or rewrite the source run.

The directory basename stays the scientific experiment ID. A different parent
and an explicit branch declaration identify the new condition, including in
metrics. Pending rollouts, later commits and old quality scores are not imported.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from skillev.runtime import StepTransactionJournal, StepTransactionState
from skillev.training.checkpoint import FilesystemTrainingCheckpointStore
from skillev.training.inflight import durable_json


def branch_checkpoint(*, source_root: Path, snapshot: Path, target_root: Path) -> Path:
    source_root, snapshot, target_root = (p.resolve() for p in (source_root, snapshot, target_root))
    if source_root == target_root or source_root.name != target_root.name:
        raise ValueError("branch needs a new parent, retaining the source experiment name")
    metadata = FilesystemTrainingCheckpointStore(root=snapshot.parent).load_metadata(snapshot)
    step = metadata.optimizer_step
    if step < 1 or metadata.experiment_id != source_root.name:
        raise ValueError("branch requires a committed trained snapshot of this experiment")
    journal = StepTransactionJournal(source_root / "checkpoints" / "step-transactions")
    records = tuple(journal.load(i) for i in range(1, step + 1))
    if any(r.state is not StepTransactionState.COMMITTED for r in records):
        raise ValueError("branch prefix contains an unresolved transaction")
    event_ids = {e.event_id for r in records for e in r.source_events}
    if not event_ids:
        raise ValueError("branch prefix has no committed source events")
    # Exclusive target creation: failure leaves evidence to inspect, not an
    # automatic retry that overwrites the branch or samples substitute work.
    target_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    checkpoints = target_root / "checkpoints"
    checkpoints.mkdir()
    target = checkpoints / snapshot.name
    shutil.copytree(snapshot, target)
    destination = checkpoints / "step-transactions"
    destination.mkdir()
    for record in records:
        name = f"step-{record.optimizer_step:08d}.json"
        shutil.copy2(journal.directory / name, destination / name)
    with (
        (source_root / "events.jsonl").open("rb") as inp,
        (target_root / "events.jsonl").open("xb") as out,
    ):
        for line in inp:
            if not line.endswith(b"\n"):
                raise ValueError("branch prefix contains an incomplete source event")
            event = json.loads(line)
            if event["event_type"] == "training_step_committed":
                if event["payload"]["optimizer_step"] > step:
                    raise ValueError("required branch events not found before later commit")
            out.write(line)
            event_ids.discard(event["event_id"])
            if not event_ids:
                break
        if event_ids:
            raise ValueError("branch is missing original committed source events")
    for name in ("formal-config.json", "run-clock.json", "resolved-run-plan.json"):
        shutil.copy2(source_root / name, target_root / name)
    for path in source_root.glob("effective-condition-process-*.json"):
        shutil.copy2(path, target_root / path.name)
    durable_json(
        target_root / "branch-source.json",
        {
            "format": "committed-condition-branch@1",
            "source_root": str(source_root),
            "source_checkpoint": str(snapshot),
            "optimizer_step": step,
            "metrics_start_step": step + 1,
            "sampling_condition": metadata.identity.sampling_schedule_algorithm,
            "history": "original source prefix; later source commits retained only in parent run",
            "quality": "new-condition baseline must be measured at the restored policy step",
        },
    )
    return target
