"""Read-only IID harness using the formal episode engine, not the IID direct-answer actor."""

from __future__ import annotations

import json
import math
import shutil
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skillev.contracts import normalize_json
from skillev.diagnostics.rollout_trace import JsonlRolloutTraceSink
from skillev.evolution import (
    BaseRolloutSessionFactory,
    RetrievingRolloutSessionFactory,
    TaskConditionedSkillRetriever,
)
from skillev.rollout import RolloutArtifact, RolloutGenerator, RolloutTokenizerProtocol
from skillev.rollout.readonly_collection import (
    EvaluationEpisodeOutcome,
    ReadOnlyPanelContinuation,
    ReadOnlyPanelResult,
    collect_readonly_panel,
    evaluation_isolation,
)
from skillev.runtime import (
    BudgetLedger,
    BudgetReservation,
    BudgetSettlement,
    BudgetVector,
    LiveAttemptEventLog,
    RuntimeEventEmitter,
    SkillLibrary,
    SkillLibraryState,
)
from skillev.runtime.budget_ledger import LedgerEntry
from skillev.training.inflight import durable_json
from skillev.training.rollout_workflow import RolloutWorkflowBinding, RolloutWorkflowResources
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig

from .iid_architecture import (
    IIDSnapshotSelection,
    decode_record,
    expanded_controls,
    require_iid_architecture_match,
)


def _read_private(path: Path) -> Any:
    return json.loads(path.read_text())


def continuation_prefix(source: Path) -> tuple[list[dict[str, Any]], int]:
    """Only complete-prefix + failed lazy preparation is eligible, never a crashed episode."""
    plan = _read_private(source / "episodes" / "plan-private.json")
    rows = [
        _read_private(source / "episodes" / f"episode-{i:06d}-private.json")
        for i in range(len(plan["tasks"]))
    ]
    start = next((i for i, row in enumerate(rows) if row["artifact"] is None), len(rows))
    if start == len(rows) or any(row["artifact"] is not None for row in rows[start:]):
        raise ValueError("continuation requires a failed unstarted suffix after a complete prefix")
    if any(
        row["position"] != i or row["task_id"] != plan["tasks"][i]["task_id"]
        for i, row in enumerate(rows)
    ):
        raise ValueError("saved episode positions differ from frozen panel")
    progress = _read_private(source / "episodes" / "rollout-progress.json")
    if progress["status"] != "failed":
        raise ValueError("source is not a published failed collection")
    if any(
        row["canonical_position"] >= start or row["stage"] != "artifact-ready"
        for row in progress["trajectories"]
    ):
        raise ValueError("a suffix episode started or has an unknown outcome")
    if {row["canonical_position"] for row in progress["trajectories"]} != set(range(start)):
        raise ValueError("source progress does not prove the complete original prefix")
    marker = source / "episodes" / f"chunk-{start:06d}.json"
    if marker.exists():
        if _read_private(marker).get("status") != "preparation-failed-before-episodes":
            raise ValueError("failure did not precede all episode execution")
    elif any(
        row.get("infrastructure_error") != "OfficialEnvironmentInfrastructureError"
        for row in rows[start:]
    ):
        # Compatibility only for the observed old hydration failure representation.
        raise ValueError("legacy suffix lacks the observed native hydration failure evidence")
    if any(
        (source / "episodes" / f"started-episode-{i:06d}.json").exists()
        for i in range(start, len(rows))
    ):
        raise ValueError("a suffix episode was started")
    return rows, start


def copy_continuation_journal(source: Path, destination: Path) -> None:
    """Preserve exact request identities in an evaluation-only branch; never authorize retries."""
    rows, start = continuation_prefix(source)
    completed_ids = {row["artifact"]["record"]["trajectory_id"] for row in rows[:start]}
    original = source.parent / f"{source.name}-evaluation-requests.sqlite3"
    if destination.exists():
        raise ValueError("continuation journal destination already exists")
    with sqlite3.connect(original.as_uri() + "?mode=ro", uri=True) as db:
        for identity, state in db.execute("SELECT identity,state FROM requests"):
            key = json.loads(identity)
            if state != "COMPLETED":
                raise ValueError("source has a pending/unknown model or judge request")
            # Current actor IDs start with episode; judge scopes put it second.
            if not isinstance(key, list) or not any(item in completed_ids for item in key[:2]):
                raise ValueError("request is not attributable to an original completed episode")
        for (episode,) in db.execute("SELECT episode FROM episode_routes"):
            if episode not in completed_ids:
                raise ValueError("an unstarted suffix already has a serving route")
        destination.touch(mode=0o600, exist_ok=False)
        with sqlite3.connect(destination) as target:
            db.backup(target)


def load_iid_continuation(
    source: Path,
    *,
    expanded: dict[str, Any],
    chunk_size: int,
    tokenizer: RolloutTokenizerProtocol,
    original_request: Path | None = None,
) -> tuple[ReadOnlyPanelContinuation, str, str]:
    if _read_private(source / "expanded-controls-private.json") != expanded:
        raise ValueError("continuation changed frozen panel/controls/arm/policy/library")
    rows, start = continuation_prefix(source)
    plan = _read_private(source / "episodes" / "plan-private.json")
    if start % chunk_size or plan.get("chunk_size", chunk_size) != chunk_size:
        raise ValueError("continuation changed original chunk coordinates")
    # Old collectors did not persist chunk_size: exact artifact-ready chunks in
    # progress alone cannot determine it; require the original run declaration.
    if "chunk_size" not in plan:
        if original_request is None:
            raise ValueError("legacy source requires its original execution request")
        declared = _read_private(original_request)
        if declared.get("chunk_size") != chunk_size or declared.get("arm") != expanded["arm"]:
            raise ValueError("continuation differs from original chunk/arm declaration")
    outcomes = tuple(
        EvaluationEpisodeOutcome(
            i, row["task_id"], RolloutArtifact.from_value(row["artifact"], tokenizer=tokenizer)
        )
        for i, row in enumerate(rows[:start])
    )
    ids = {row.artifact.record.trajectory_id for row in outcomes if row.artifact is not None}
    entries: list[LedgerEntry] = []
    ledger_path = source / "episodes" / "ledger-private.json"
    if ledger_path.exists():
        saved = _read_private(ledger_path)
        run_id, attempt_id = saved["run_id"], saved["attempt_id"]
        for row in saved["entries"]:
            if row["actual"] is None or row["invocation_id"] not in ids:
                raise ValueError("source has unsettled or unstarted episode budget")
            reservation = BudgetReservation(
                row["reservation_id"],
                row["run_id"],
                row["attempt_id"],
                row["invocation_id"],
                BudgetVector.from_value(row["maximum"]),
            )
            entries.append(
                LedgerEntry.reserved(reservation).settled(
                    BudgetSettlement(row["reservation_id"], BudgetVector.from_value(row["actual"]))
                )
            )
    else:
        # Reconstruct the original exact reserve/settle events, not synthetic charges.
        events = [
            _read
            for line in (source / "events-private.jsonl").read_text().splitlines()
            if (_read := json.loads(line))
        ]
        if not events:
            raise ValueError("original settled budget event evidence is absent")
        run_id, attempt_id = events[0]["run_id"], events[0]["attempt_id"]
        pending: dict[str, BudgetReservation] = {}
        settled_ids: set[str] = set()
        for event in events:
            if (event["run_id"], event["attempt_id"]) != (run_id, attempt_id):
                raise ValueError("source events mix execution identities")
            payload = event["payload"]
            if event["event_type"] == "budget_reserved":
                rid = payload["reservation_id"]
                owners = [tid for tid in ids if rid.startswith(tid + ":")]
                if len(owners) != 1 or rid in pending or rid in settled_ids:
                    raise ValueError("budget reservation is not unique to completed prefix")
                pending[rid] = BudgetReservation(
                    rid, run_id, attempt_id, owners[0], BudgetVector.from_value(payload["maximum"])
                )
            elif event["event_type"] == "budget_settled":
                rid = payload["reservation_id"]
                if rid not in pending:
                    raise ValueError("settlement has no original pending reservation")
                entries.append(
                    LedgerEntry.reserved(pending.pop(rid)).settled(
                        BudgetSettlement(rid, BudgetVector.from_value(payload["actual"]))
                    )
                )
                settled_ids.add(rid)
            elif event["event_type"] == "rollout_started" and payload["trajectory_id"] not in ids:
                raise ValueError("an unstarted suffix has original rollout-start evidence")
        if pending:
            raise ValueError("source budget contains unknown/unsettled requests")
    summary = _read_private(source / "summary.json")
    total = BudgetVector()
    for entry in entries:
        assert entry.settlement is not None
        total = total.add(entry.settlement.actual)
    if total != BudgetVector.from_value(summary["consumed_budget"]):
        raise ValueError("original exact budget events disagree with published settled summary")
    return (
        ReadOnlyPanelContinuation(
            source / "episodes", outcomes, tuple(entries), plan, summary["elapsed_seconds"]
        ),
        run_id,
        attempt_id,
    )


def iid_aggregate(
    result: ReadOnlyPanelResult,
    architecture: dict[str, Any],
) -> dict[str, Any]:
    """Native metric means use all planned IDs; infra missing values are not zero labels."""
    expected = architecture["panel"]["records"]
    if [(r.position, r.task_id) for r in result.outcomes] != [
        (i, r["public_task"]["task_id"]) for i, r in enumerate(expected)
    ]:
        raise ValueError("IID results must account for every frozen position exactly once")
    by_domain: dict[str, Any] = {}
    for domain in dict.fromkeys(case["source"]["benchmark"] for case in expected):
        outcomes = [
            row
            for row, case in zip(result.outcomes, expected, strict=True)
            if case["source"]["benchmark"] == domain
        ]
        rewards = [row.artifact.record.reward for row in outcomes if row.artifact is not None]
        metrics: dict[str, list[float]] = {}
        for reward in rewards:
            native = reward.native_payload.get("public_metrics", {})
            # HumanEval / ALFWorld have an intrinsically binary native metric.
            values = (
                native
                if isinstance(native, dict) and native
                else {reward.native_metric_name: reward.value}
            )
            for name, value in values.items():
                if (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                ):
                    metrics.setdefault(name, []).append(float(value))
        means = {
            name: math.fsum(values) / len(outcomes) if len(values) == len(outcomes) else None
            for name, values in metrics.items()
        }
        complete = len(rewards) == len(outcomes) and bool(outcomes)
        rule = architecture["acceptance_rules"][domain]
        measured = means.get(rule["metric"])
        accepted = (
            complete
            and measured is not None
            and (
                measured > rule["minimum"]
                if rule.get("comparison", "at-least") == "strictly-above"
                else measured >= rule["minimum"]
            )
        )
        by_domain[domain] = {
            "planned": len(outcomes),
            "completed": len(rewards),
            "infrastructure_failures_or_not_started": len(outcomes) - len(rewards),
            "success_count": sum(r.success for r in rewards) if complete else None,
            "success_rate": sum(r.success for r in rewards) / len(outcomes) if complete else None,
            "reward_mean": math.fsum(r.value for r in rewards) / len(outcomes)
            if complete
            else None,
            "native_metrics": means,
            "acceptance_rule": rule,
            "accepted": accepted,
        }
    return {
        "record_kind": "iid-evaluation",
        "architecture_id": architecture["architecture_id"],
        "panel_purpose": architecture["panel"].get("purpose"),
        "population_exposure": architecture["panel"].get("exposure"),
        "independent_unseen_final_test": False,
        "prior_project_evaluation_source_count": len(
            architecture["panel"].get("prior_project_evaluation_overlap", [])
        ),
        "denominator": "all-frozen-panel-ids",
        "domains": by_domain,
        "baseline_accepted": all(row["accepted"] for row in by_domain.values()),
        "elapsed_seconds": result.elapsed_seconds,
        "consumed_budget": result.consumed_budget.to_value(),
        **evaluation_isolation(),
    }


async def run_iid_episodes(
    *,
    architecture: dict[str, Any],
    selection: IIDSnapshotSelection,
    generator: RolloutGenerator,
    base_sessions: BaseRolloutSessionFactory,
    actual_controls: dict[str, Any],
    output: Path,
    chunk_size: int,
    resources: RolloutWorkflowResources | None = None,
    reference: dict[str, Any] | None = None,
    allow_library_change: bool = False,
    validate_execution: Callable[[], None],
    continuation_from: Path | None = None,
    continuation_original_request: Path | None = None,
) -> dict[str, Any]:
    """There is no trainer, state-restoration API or training evidence dependency.

    `actual_controls` is assembled by the live native session/service binding,
    rather than a loose override dictionary. A saved Step-0 reference is required
    for every trained-policy or evolved-library intervention.
    """
    expanded = expanded_controls(architecture, selection)
    actual = json.loads(json.dumps(actual_controls))
    if actual != expanded["controls"]:
        raise ValueError("live IID components differ from the frozen execution architecture")
    initial = SkillLibraryState.from_value(architecture["initial_library"])
    library = SkillLibraryState.from_value(expanded["library_snapshot"])
    if reference is not None:
        axes = require_iid_architecture_match(
            reference, expanded, allow_library_change=allow_library_change
        )
    elif selection.optimizer_step or (selection.arm != "skills-off" and library != initial):
        raise ValueError("trained or evolved IID evaluation requires its Step-0 controls")
    else:
        axes = ()
    if generator.snapshot() != selection.policy:
        raise ValueError("generator does not serve the selected immutable policy snapshot")
    formal = BayesianFormalConfig(**actual["formal"])
    rollout = formal.sampling_config
    tasks = tuple(decode_record(row).input for row in architecture["panel"]["records"])
    binding = RolloutWorkflowBinding.from_value(actual["workflow"])
    resources = resources or RolloutWorkflowResources(binding)
    if resources.binding != binding:
        raise ValueError("live request scheduling differs from the frozen architecture")
    continuation = None
    run_id, attempt_id = output.name, "evaluation-only"
    if continuation_from is not None:
        continuation, run_id, attempt_id = load_iid_continuation(
            continuation_from,
            expanded=expanded,
            chunk_size=chunk_size,
            tokenizer=generator.tokenizer,
            original_request=continuation_original_request,
        )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    durable_json(output / "expanded-controls-private.json", normalize_json(expanded))
    ledger = BudgetLedger(
        run_id=run_id,
        attempt_id=attempt_id,
        cap=rollout.per_rollout_maximum.scale(len(tasks)),
    )
    events = LiveAttemptEventLog(
        output / "events-private.jsonl", run_id=run_id, attempt_id=attempt_id
    )
    durable_json(
        output / "continuation-execution-private.json",
        {
            "chunk_size": chunk_size,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "source": None if continuation_from is None else str(continuation_from),
        },
    )
    if continuation_original_request is not None:
        shutil.copyfile(continuation_original_request, output / "source-execution-request.json")
    if continuation_from is not None:
        shutil.copyfile(
            continuation_from / "events-private.jsonl", output / "source-events-private.jsonl"
        )
        shutil.copyfile(continuation_from / "summary.json", output / "source-summary.json")
    # This private library wrapper is a retrieval dependency, not an application
    # library. Neither it nor the immutable state is passed to an evolution policy.
    sessions = RetrievingRolloutSessionFactory(
        base_factory=base_sessions,
        retriever=TaskConditionedSkillRetriever(library=SkillLibrary(library)),
    )
    async with JsonlRolloutTraceSink.create((output / "trace-private.jsonl").resolve()) as trace:
        result = await collect_readonly_panel(
            root=output / "episodes",
            tasks=tasks,
            generator=generator,
            sessions=sessions,
            library=library,
            rollout=rollout,
            assembler=rollout.context_assembler(maximum_h0_tokens=actual["maximum_h0_tokens"]),
            epsilon_min=formal.epsilon,
            condition_id=architecture["architecture_id"],
            sampling_schedule_id=actual["sampling_schedule"],
            ordered_sequence_id=architecture["architecture_id"],
            resources=resources,
            ledger=ledger,
            emitter=RuntimeEventEmitter(events, "iid-evaluation"),
            clock=lambda: datetime.now(UTC).isoformat(),
            chunk_size=chunk_size,
            trace_sink=trace,
            anchor_ordinal=actual["sampling_anchor"],
            continuation=continuation,
        )
    # A completed collector is not enough: fail before publishing an accepted
    # summary if the service/profile or selected published adapter drifted.
    try:
        validate_execution()
    except Exception as error:
        durable_json(
            output / "execution-failure.json",
            {"status": "execution-validation-failed", "error_type": type(error).__name__},
        )
        raise
    summary = {
        **iid_aggregate(result, architecture),
        "arm": selection.arm,
        "policy_step": selection.optimizer_step,
        "policy_snapshot_id": selection.policy.snapshot_id,
        "library_version": library.current_version,
        "snapshot_axes_changed": axes,
        "execution_validation": "live-components-unchanged",
    }
    durable_json(output / "summary.json", normalize_json(summary))
    return summary
