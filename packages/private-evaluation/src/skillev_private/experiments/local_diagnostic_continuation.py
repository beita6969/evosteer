"""Explicit append-only diagnostic plan extension, using original committed evidence."""

from __future__ import annotations

import json
import sqlite3
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from skillev.application_config import ApplicationConfig
from skillev.application_continuation import PlanContinuation
from skillev.runtime import (
    BudgetReservation,
    BudgetSettlement,
    BudgetVector,
    FullRuntimeExecutionState,
    LedgerEntry,
    StepTransactionJournal,
    StepTransactionState,
)
from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan
from skillev.training.checkpoint import FilesystemTrainingCheckpointStore
from skillev_private.benchmarks.protocol_v13_training import Protocol13TrainingRecord

from .bayesian_improve_training import FormalTrainingBindings
from .bayesian_training_config import BayesianFormalConfig
from .protocol_v10_application_input import ProtocolV10ApplicationIdentity
from .training_data_condition import data_condition_scientific


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def completed_budget(root: Path, through: int) -> tuple[LedgerEntry, ...]:
    """Restore original reservations, including author calls, without re-emitting them."""
    charges: dict[str, dict[str, Any]] = {}
    for step in range(1, through + 1):
        directory = root / "inflight" / f"step-{step:08d}"
        batch = read(directory / "batch.json")
        for rollout in batch["rollouts"]:
            value = read(directory / f"trajectory-{rollout['position']:06d}.json")
            if value["artifact"]["record"]["trajectory_id"] != rollout["trajectory_id"]:
                raise ValueError("original artifact is not the committed planned trajectory")
            for entry in value["budget_entries"]:
                if (
                    entry["run_id"] != root.name
                    or entry["attempt_id"] != "diagnostic-seed0"
                    or entry["invocation_id"] != rollout["trajectory_id"]
                ):
                    raise ValueError("original call charge belongs to another rollout/attempt")
                rid = entry["reservation_id"]
                if rid in charges:
                    raise ValueError("original call charge occurs more than once")
                charges[rid] = entry
    # Native judge calls may have their own accounting scope: do not pretend
    # they were charged to the training ledger. Unknown requests still forbid resume.
    with sqlite3.connect((root / "requests.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        if db.execute("SELECT 1 FROM requests WHERE state!='COMPLETED' LIMIT 1").fetchone():
            raise ValueError("diagnostic has an unresolved original request; do not resample")
        authors = {}
        for raw_id, payload in db.execute("SELECT identity,payload FROM requests"):
            identity = json.loads(raw_id)
            if identity[:2] == ["author", root.name]:
                authors[identity[2]] = json.loads(zlib.decompress(payload))["sampling_params"][
                    "sampling_seed"
                ]
    pending = {}
    restored = []
    seen = set()
    with (root / "events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["run_id"] != root.name or event["attempt_id"] != "diagnostic-seed0":
                raise ValueError("original event belongs to another diagnostic attempt")
            payload = event["payload"]
            kind = event["event_type"]
            if kind == "training_step_committed" and payload["optimizer_step"] > through:
                raise ValueError("refuse to resume behind a later committed transaction")
            if kind == "budget_reserved":
                rid = payload["reservation_id"]
                if rid in seen:
                    raise ValueError("duplicate original reservation")
                seen.add(rid)
                charge = charges.get(rid)
                if charge is not None:
                    if charge["maximum"] != payload["maximum"]:
                        raise ValueError("artifact and original reservation disagree")
                    invocation = charge["invocation_id"]
                elif rid in authors:
                    invocation = f"phi-authoring:{authors[rid]}"
                else:
                    raise ValueError("original reservation has no persisted call provenance")
                pending[rid] = BudgetReservation(
                    rid,
                    root.name,
                    "diagnostic-seed0",
                    invocation,
                    BudgetVector.from_value(payload["maximum"]),
                )
            elif kind == "budget_settled":
                rid = payload["reservation_id"]
                if rid not in pending:
                    raise ValueError("settlement has no original reservation")
                if rid in charges and charges[rid]["actual"] != payload["actual"]:
                    raise ValueError("artifact and original settlement disagree")
                restored.append(
                    LedgerEntry.reserved(pending.pop(rid)).settled(
                        BudgetSettlement(rid, BudgetVector.from_value(payload["actual"]))
                    )
                )
    if pending or set(charges) - seen:
        raise ValueError("original diagnostic budget is incomplete")
    return tuple(restored)


@dataclass(frozen=True)
class DiagnosticAppend:
    source_application: ApplicationConfig
    source_plan: ExactAttemptRunPlan
    plan: ExactAttemptRunPlan
    snapshot: Any
    entries: tuple[LedgerEntry, ...]

    def identity(
        self,
        source: ProtocolV10ApplicationIdentity,
        candidate: ProtocolV10ApplicationIdentity,
        task_ids: tuple[str, ...],
    ) -> tuple[ProtocolV10ApplicationIdentity, PlanContinuation]:
        if source.runtime_snapshot_identity() != self.snapshot.identity:
            raise ValueError("reconstructed original declaration differs from Step3 checkpoint")
        target = replace(
            candidate,
            snapshot_identity=replace(
                source.snapshot_identity,
                run_plan_hash=self.plan.content_hash,
                public_identity_content_hash=candidate.snapshot_identity.public_identity_content_hash,
            ),
            initial_run_cursor=replace(
                self.snapshot.execution_state.run_cursor, run_plan_hash=self.plan.content_hash
            ),
        )
        transition = PlanContinuation(source, 3, task_ids[:84], task_ids)
        transition.require_target(target)
        return target, transition


def require_append(
    *,
    root: Path,
    resume: Path,
    config: BayesianFormalConfig,
    bindings: FormalTrainingBindings,
    selected: tuple[Protocol13TrainingRecord, ...],
) -> DiagnosticAppend:
    if not root.is_dir() or resume.resolve().parent != (root / "checkpoints").resolve():
        raise ValueError("append must restore this same diagnostic run's original checkpoint")
    old = read(root / "diagnostic-condition.json")
    if old["format"] != "skillev-single-local-gradient-diagnostic@1" or config.steps != 10:
        raise ValueError("only the explicit original three-step to ten-step extension is supported")
    if replace(config, steps=3).to_value() != old["formal"]:
        raise ValueError("append cannot change the diagnostic model, method, prompt or budgets")
    new_data = data_condition_scientific(bindings.data_condition, selected)["data_condition"]
    assert isinstance(new_data, dict)
    ordered = new_data["ordered_selected_sources"]
    assert isinstance(ordered, list)
    if ordered[:84] != old["data_condition"]["ordered_selected_sources"]:
        raise ValueError("new task schedule changed original source/population/occurrence order")
    previous_binding = read(root / "bindings-private.json")
    # Data declaration may extend the schedule, never change deployment or sampling.
    current = bindings
    for name in (
        "preparation",
        "dataset",
        "deployments",
        "evalplus_python",
        "evalplus_source_root",
        "endpoint",
        "base_model",
        "adapter_namespace",
        "serving_gpu_uuid",
    ):
        item = getattr(current, name)
        if (str(item) if isinstance(item, Path) else item) != previous_binding[name]:
            raise ValueError(f"append changed original {name}")
    if list(bindings.training_gpu_uuids) != previous_binding["training_gpu_uuids"]:
        raise ValueError("append changed its declared single gradient owner")
    prior = read(root / "resolved-run-plan.json")
    source_plan = ExactAttemptRunPlan.from_value(prior["run_plan"])
    if source_plan != ExactAttemptRunPlan(2, 1, 2):
        raise ValueError("append source is not the original diagnostic plan")
    snapshot = FilesystemTrainingCheckpointStore(root=root / "checkpoints").load_metadata(resume)
    state = snapshot.execution_state
    if snapshot.optimizer_step != 3 or not isinstance(state, FullRuntimeExecutionState):
        raise ValueError("append requires full committed Step3 state")
    if state.task_cursor.cursor != 84 or state.run_cursor.completed_training_steps != 3:
        raise ValueError("source cursor differs from the three complete B28 batches")
    state.run_cursor.require_plan(source_plan)
    journal = StepTransactionJournal((root / "checkpoints/step-transactions").resolve())
    if journal.pending() or any(
        journal.load(i).state is not StepTransactionState.COMMITTED for i in (1, 2, 3)
    ):
        raise ValueError("all original transactions must be complete before extending the plan")
    return DiagnosticAppend(
        ApplicationConfig.from_value(prior["application"]),
        source_plan,
        source_plan.append(phase_search_steps=6, closure_steps=1),
        snapshot,
        completed_budget(root, 3),
    )
