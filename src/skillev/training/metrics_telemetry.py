"""Read-only aggregate enrichment; source commits and timing remain immutable.

Only numeric aggregates and opaque operational coordinates reach the export.
Process-counter differences are interval measurements, not per-step server usage.
GPU hours are declared reservations, never GPU-utilization measurements.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from skillev.contracts import JsonValue, canonical_json
from skillev.contracts.skill_exposure import TWO_SKILL_CATALOG_EXPOSURE

from .metrics_contract import TrainingMetricsSnapshot, _object, _objects
from .skill_discovery_metrics import skill_discovery_facts

if TYPE_CHECKING:
    from .metrics_export import MetricsStore

TIMINGS = (
    "step_wall_seconds",
    "rollout_span_seconds",
    "gradient_span_seconds",
    "gradient_tail_seconds",
    "overlap_seconds",
    "durability_seconds",
)
COUNTERS = (
    "server_generated_tokens",
    "admitted_content_tokens",
    "admitted_stop_tokens",
    "discarded_suffix_tokens",
)


def number(value: JsonValue) -> float | None:
    return (
        float(value)
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        else None
    )


def performance_rows(path: Path | None) -> dict[tuple[int, str], dict[str, JsonValue]]:
    result: dict[tuple[int, str], dict[str, JsonValue]] = {}
    if path is None or not path.exists():
        return result
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            if row.get("committed") is not True:
                continue
            if type(row.get("optimizer_step")) is not int or not isinstance(
                row.get("batch_id"), str
            ):
                raise ValueError("performance must identify its committed step and batch")
            key = row["optimizer_step"], row["batch_id"]
            if key in result and result[key] != row:
                raise ValueError("conflicting committed performance records")
            result[key] = row
    return result


def skill_target_facts(records: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    """Only the explicitly requested two-skill condition has a compliance target.

    Count distinct executed IDs, not mentions, catalog exposure or repeated reads.
    Keep historical @1 telemetry unchanged and incomplete evidence unknown.
    """
    targeted = [
        record
        for record in records
        if _object(_object(record.get("initial_context", {})).get("meta", {})).get("skill_exposure")
        == TWO_SKILL_CATALOG_EXPOSURE
    ]
    if not targeted:
        return {}
    distinct: list[int] = []
    unavailable: list[bool] = []
    for record in targeted:
        calls = [edge.get("invoked_skill_ids") for edge in _objects(record["steps"])]
        if all(isinstance(v, list) and all(isinstance(s, str) for s in v) for v in calls):
            distinct.append(len({s for v in calls if isinstance(v, list) for s in v}))
        visible = _object(record["initial_context"]).get("retrieved_skill_ids")
        if isinstance(visible, list) and all(isinstance(s, str) for s in visible):
            unavailable.append(len(set(visible)) < 2)
    complete = len(distinct) == len(targeted)
    return {
        "skill_target_trajectory_count": len(targeted),
        "skill_target_unknown_trajectory_count": len(targeted) - len(distinct),
        "skill_target_met_trajectory_count": sum(n >= 2 for n in distinct) if complete else None,
        "skill_target_met_fraction": (
            sum(n >= 2 for n in distinct) / len(targeted) if complete else None
        ),
        "skill_distinct_per_trajectory_mean": sum(distinct) / len(targeted) if complete else None,
        "skill_catalog_below_target_trajectory_count": (
            sum(unavailable) if len(unavailable) == len(targeted) else None
        ),
    }


def step_facts(event: dict[str, JsonValue]) -> dict[str, JsonValue]:
    payload = _object(event["payload"])
    records = _objects(payload["records"])
    edges = [edge for record in records for edge in _objects(record["steps"])]
    invocations = [edge.get("invoked_skill_ids") for edge in edges]
    invocation_count = (
        sum(len(v) for v in invocations if isinstance(v, list))
        if all(isinstance(v, list) and all(isinstance(x, str) for x in v) for v in invocations)
        else None
    )
    posterior = payload.get("posterior_batch")
    updates = posterior.get("updates") if isinstance(posterior, dict) else None
    cells: list[str] | None = None
    if isinstance(updates, list):
        cells = sorted({canonical_json([_object(v)["skill_id"], _object(v)["z"]]) for v in updates})
    domains: dict[str, JsonValue] = {}
    for record in records:
        reward = _object(record["reward"])
        native = _object(reward["native_payload"])
        domain = str(native.get("benchmark_id", "unknown"))
        row = domains.setdefault(domain, {"success_count": 0, "trajectory_count": 0})
        assert isinstance(row, dict)
        row["success_count"] = cast(int, row["success_count"]) + int(reward["success"] is True)
        row["trajectory_count"] = cast(int, row["trajectory_count"]) + 1
    # The historical key holds effective training success. Keep that series,
    # but expose the unmodified native label separately under the new condition.
    if any(
        "format_content_review" in _object(_object(r["reward"])["native_payload"]) for r in records
    ):
        for domain, row in domains.items():
            assert isinstance(row, dict)
            row["native_success_count"] = 0
            for record in records:
                reward = _object(record["reward"])
                native = _object(reward["native_payload"])
                if native.get("benchmark_id", "unknown") != domain:
                    continue
                review = native.get("format_content_review")
                original = _object(review["native_result"]) if isinstance(review, dict) else reward
                row["native_success_count"] = cast(int, row["native_success_count"]) + int(
                    original["success"] is True
                )
    return {
        **skill_discovery_facts(records),
        "batch_id": payload["batch_id"],
        "skill_invocation_count": invocation_count,
        "posterior_update_event_count": len(updates) if isinstance(updates, list) else None,
        "cells": cast(JsonValue, cells),
        "native_success_counts": domains,
        **skill_target_facts(records),
    }


class CommittedTelemetry:
    def __init__(
        self,
        store: MetricsStore,
        *,
        run_id: str,
        condition_id: str,
        performance: Path | None = None,
        gpu_count: int | None = None,
        gpu_process_ids: tuple[str, ...] = (),
        committed_after: int = 0,
        allow_missing_performance: bool = False,
    ) -> None:
        if not store.enriched:
            raise ValueError("telemetry requires a new enriched output/store")
        if not run_id or not condition_id or committed_after < 0:
            raise ValueError("telemetry requires explicit run/condition/segment coordinates")
        if (gpu_count is None) != (not gpu_process_ids) or (
            gpu_count is not None and gpu_count < 1
        ):
            raise ValueError("GPU reservations require positive count and explicit process IDs")
        self.store, self.run_id, self.condition_id = store, run_id, condition_id
        self.performance, self.gpu_count, self.gpu_process_ids = (
            performance,
            gpu_count,
            gpu_process_ids,
        )
        self.committed_after, self.allow_missing = committed_after, allow_missing_performance
        db = store.connection
        db.execute("CREATE TABLE IF NOT EXISTS telemetry_contract (payload TEXT NOT NULL)")
        contract = canonical_json(
            {
                "format": "committed-telemetry@2",
                "run_id": run_id,
                "condition_id": condition_id,
                "performance": str(performance.resolve()) if performance else None,
                "gpu_count": gpu_count,
                "gpu_process_ids": sorted(gpu_process_ids),
                "committed_after": committed_after,
                "allow_missing_performance": allow_missing_performance,
            }
        )
        saved = db.execute("SELECT payload FROM telemetry_contract").fetchone()
        if saved is not None and saved[0] != contract:
            raise ValueError("telemetry declaration changed; use a new output/store")
        if saved is None:
            db.execute("INSERT INTO telemetry_contract VALUES (?)", (contract,))
        db.execute(
            "CREATE TABLE IF NOT EXISTS telemetry_facts "
            "(event_id TEXT PRIMARY KEY, step INTEGER, kind TEXT, payload TEXT)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS telemetry_pending "
            "(step INTEGER PRIMARY KEY, batch TEXT UNIQUE, payload TEXT, finalized TEXT)"
        )
        db.commit()

    def observe(self, event: dict[str, JsonValue]) -> None:
        if event.get("run_id") != self.run_id:
            raise ValueError("telemetry event belongs to another observed run")
        kind = event.get("event_type")
        if kind not in {
            "training_step_committed",
            "phase_detection_recorded",
            "evolution_cycle_committed",
            "evolution_no_op_committed",
        }:
            return
        payload = _object(event["payload"])
        step = payload["optimizer_step"]
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError("telemetry event needs a committed optimizer coordinate")
        if kind == "training_step_committed":
            facts = step_facts(event)
        elif kind == "phase_detection_recorded":
            facts = {
                "batch_id": payload.get("batch_id"),
                "reason": payload.get("reason"),
                "trigger_rule": payload.get("trigger_rule"),
            }
        elif kind == "evolution_cycle_committed":
            mutation = _object(payload["mutation"])
            facts = {
                "mutation_count": int(
                    payload["library_version_before"] != payload["library_version_after"]
                ),
                "mutation_actions": len(_objects(mutation["actions"])),
                "authoring_usage": payload.get("authoring_usage"),
                "new_skill_ids": [
                    _object(_object(doc).get("manifest", {})).get("skill_id")
                    for doc in _objects(mutation.get("new_documents", []))
                ]
                if isinstance(mutation.get("new_documents"), list)
                else None,
            }
        else:
            facts = {}
        raw = canonical_json(facts)
        db = self.store.connection
        existing = db.execute(
            "SELECT step,kind,payload FROM telemetry_facts WHERE event_id=?", (event["event_id"],)
        ).fetchone()
        if existing is not None and existing != (step, kind, raw):
            raise ValueError("telemetry source event changed")
        with db:
            db.execute(
                "INSERT OR IGNORE INTO telemetry_facts VALUES (?,?,?,?)",
                (event["event_id"], step, kind, raw),
            )
            if kind == "training_step_committed" and step > self.committed_after:
                snapshot = self.store.actions.enrich(
                    TrainingMetricsSnapshot.from_event(event, condition_id=self.condition_id), event
                )
                base = json.dumps(asdict(snapshot), sort_keys=True, allow_nan=False)
                old = db.execute(
                    "SELECT payload FROM telemetry_pending WHERE step=?", (step,)
                ).fetchone()
                if old is not None and old[0] != base:
                    raise ValueError("committed telemetry batch changed")
                db.execute(
                    "INSERT OR IGNORE INTO telemetry_pending VALUES (?,?,?,NULL)",
                    (step, snapshot.batch_id, base),
                )

    def facts(self, step: int, kind: str) -> list[dict[str, JsonValue]]:
        return [
            json.loads(row[0])
            for row in self.store.connection.execute(
                "SELECT payload FROM telemetry_facts WHERE step=? AND kind=? ORDER BY event_id",
                (step, kind),
            )
        ]

    def phase(self, step: int, batch: str | None = None) -> dict[str, JsonValue]:
        checks = self.facts(step, "phase_detection_recorded")
        if batch is not None and any(v.get("batch_id") not in (None, batch) for v in checks):
            raise ValueError("phase detection differs from committed batch")
        cycles = self.facts(step, "evolution_cycle_committed")
        noops = self.facts(step, "evolution_no_op_committed")
        reasons = [v.get("reason") for v in checks]
        known = bool(checks) and all(isinstance(v, str) and v for v in reasons)
        rules_known = known and all(
            v.get("reason") != "phase-detected" or v.get("trigger_rule") is not None for v in checks
        )
        triggered = sum(v == "phase-detected" for v in reasons) if known else None
        resolved = bool(cycles or noops) or triggered == 0
        result: dict[str, JsonValue] = {
            "phase_detection_record_count": len(checks) if checks else None,
            "phase_check_count": sum(v != "slot-does-not-allow-phase-detection" for v in reasons)
            if known
            else None,
            "phase_trigger_count": triggered,
            "cold_start_trigger_count": sum(
                v.get("trigger_rule") == "zero-coverage-generate@1" for v in checks
            )
            if rules_known
            else None,
            "natural_phase_trigger_count": sum(
                v.get("trigger_rule") == "residual-and-entropy" for v in checks
            )
            if rules_known
            else None,
            "phase_no_op_count": len(noops) if resolved else None,
            "library_mutation_count": sum(int(str(v["mutation_count"])) for v in cycles)
            if resolved
            else None,
            "library_mutation_action_count": sum(int(str(v["mutation_actions"])) for v in cycles)
            if resolved
            else None,
        }
        for field in ("input_tokens", "output_tokens", "model_calls"):
            values = [
                number(_object(v["authoring_usage"]).get(field))
                for v in cycles
                if isinstance(v.get("authoring_usage"), dict)
            ]
            result[f"authoring_{field}"] = (
                sum(v for v in values if v is not None)
                if cycles and len(values) == len(cycles) and all(v is not None for v in values)
                else None
            )
        return result

    def build(
        self, snapshot: TrainingMetricsSnapshot, rows: dict[tuple[int, str], dict[str, JsonValue]]
    ) -> TrainingMetricsSnapshot:
        step = snapshot.optimizer_step
        facts = self.facts(step, "training_step_committed")
        if len(facts) != 1:
            raise ValueError("one authoritative commit required per telemetry step")
        fact = facts[0]
        metrics = {
            k: v
            for k, v in fact.items()
            if k not in ("cells", "batch_id", "skill_visible_ids", "skill_invoked_ids")
        }
        prior_cycles = [
            cycle for i in range(1, step) for cycle in self.facts(i, "evolution_cycle_committed")
        ]
        history_known = all(
            self.phase(i)["library_mutation_count"] is not None for i in range(1, step)
        ) and all(isinstance(cycle.get("new_skill_ids"), list) for cycle in prior_cycles)
        evolved = {
            skill
            for cycle in prior_cycles
            for skill in cast(list[JsonValue], cycle.get("new_skill_ids") or [])
            if isinstance(skill, str)
        }
        for label, field in (("visible", "skill_visible_ids"), ("invoked", "skill_invoked_ids")):
            ids = fact.get(field)
            metrics[f"evolved_skill_{label}_count"] = (
                len(evolved.intersection(cast(list[str], ids)))
                if history_known and isinstance(ids, list)
                else None
            )
        cells = fact.get("cells")
        metrics["posterior_cells_touched_count"] = len(cells) if isinstance(cells, list) else None
        metrics.update(self.phase(step, snapshot.batch_id))
        for label, start in (("cumulative", 1), ("segment_cumulative", self.committed_after + 1)):
            history = [self.facts(i, "training_step_committed") for i in range(start, step + 1)]
            complete = all(len(v) == 1 and isinstance(v[0].get("cells"), list) for v in history)
            metrics[f"posterior_update_event_count_{label}"] = (
                sum(int(str(v[0]["posterior_update_event_count"])) for v in history)
                if complete
                else None
            )
            metrics[f"posterior_cells_touched_{label}"] = (
                len({str(cell) for v in history for cell in cast(list[JsonValue], v[0]["cells"])})
                if complete
                else None
            )
        row = rows.get((step, snapshot.batch_id), {})
        rollout = row.get("rollout")
        rollout = rollout if isinstance(rollout, dict) else {}
        if rollout.get("batch_id") not in (None, snapshot.batch_id):
            raise ValueError("rollout performance belongs to another batch")
        for field in TIMINGS:
            metrics[field] = number(row.get(field))
        metrics["logical_input_tokens"] = number(rollout.get("prompt_tokens"))
        metrics["logical_output_tokens"] = number(rollout.get("completion_tokens"))
        metrics["logical_model_calls"] = number(rollout.get("model_calls"))
        process = row.get("process_instance_id")
        metrics["performance_process_instance_id"] = process
        metrics["measured_server_input_tokens"] = None
        # Sidecar rows alone never establish a source commit or a GPU charge.
        history_rows = []
        for key, value in sorted(rows.items()):
            source = self.facts(key[0], "training_step_committed")
            if key[0] <= step and len(source) == 1 and source[0]["batch_id"] == key[1]:
                history_rows.append((key, value))
        previous = [
            v
            for key, v in history_rows
            if key[0] < step and process is not None and v.get("process_instance_id") == process
        ]
        physical = row.get("physical_generation_usage_cumulative")
        physical = physical if isinstance(physical, dict) else {}
        before = previous[-1].get("physical_generation_usage_cumulative") if previous else None
        before = before if isinstance(before, dict) else {}
        for name in COUNTERS:
            current, old = number(physical.get(name)), number(before.get(name))
            metrics[name + "_process_cumulative"] = current
            metrics[name + "_delta"] = (
                current - old
                if current is not None and old is not None and current >= old
                else None
            )
        declared = isinstance(process, str) and process in self.gpu_process_ids
        count = self.gpu_count if declared else None
        wall, elapsed = (
            number(row.get("step_wall_seconds")),
            number(row.get("process_elapsed_seconds")),
        )
        metrics["reserved_gpu_count"] = count
        metrics["reserved_gpu_hours_step"] = (
            count * wall / 3600 if count and wall is not None else None
        )
        metrics["reserved_gpu_hours_process_elapsed"] = (
            count * elapsed / 3600 if count and elapsed is not None else None
        )
        eligible = [
            v
            for key, v in history_rows
            if key[0] > self.committed_after
            and v.get("process_instance_id") in self.gpu_process_ids
        ]
        walls = [number(v.get("step_wall_seconds")) for v in eligible]
        metrics["reserved_gpu_hours_observed_cumulative"] = (
            self.gpu_count * sum(v for v in walls if v is not None) / 3600
            if self.gpu_count and walls and all(v is not None for v in walls)
            else None
        )
        latest: dict[str, float] = {}
        for value in eligible:
            duration = number(value.get("process_elapsed_seconds"))
            if duration is not None:
                latest[str(value["process_instance_id"])] = duration
        metrics["reserved_gpu_hours_processes_observed_cumulative"] = (
            self.gpu_count * sum(latest.values()) / 3600 if self.gpu_count and latest else None
        )
        metrics["gpu_processes_observed_count"] = len(latest) if self.gpu_count and latest else None
        metrics["scope"] = {
            "posterior_cumulative": (
                "complete source-event prefix from step 1 across original conditions; "
                "null for missing evidence"
            ),
            "posterior_segment_cumulative": (
                f"source commits strictly after step {self.committed_after}"
            ),
            "tokens": (
                "settled logical budget ledger entries during sealed batch collection; "
                "no text retokenization or assumed R/action/remote-grader split"
            ),
            "server_deltas": (
                "between observed same-process samples; first/reset unknown; "
                "may include nonstep work, not a step token charge"
            ),
            "gpu": "declared-reservation-not-utilization; explicitly declared process IDs only",
            "gpu_step": "reserved roles times committed wall; observed segment, not outside-step",
            "gpu_process": (
                "reserved roles times process elapsed including preparation/quality/waiting; "
                "latest once per observed declared process; never run_elapsed"
            ),
            "outside_step_cost": (
                "not separately measured; process and step scopes overlap, do not add them"
            ),
        }
        return replace(snapshot, telemetry=metrics)

    def flush(self, rows: dict[tuple[int, str], dict[str, JsonValue]]) -> int:
        db = self.store.connection
        count = 0
        pending = db.execute(
            "SELECT p.step,p.batch,p.payload,p.finalized FROM telemetry_pending p "
            "WHERE NOT EXISTS (SELECT 1 FROM metrics m WHERE m.run=? AND m.batch=p.batch) "
            "ORDER BY p.step",
            (self.run_id,),
        ).fetchall()
        for step, batch, raw, finalized in pending:
            if finalized is None:
                if (
                    self.performance is not None
                    and not self.allow_missing
                    and (step, batch) not in rows
                ):
                    break
                enriched = self.build(TrainingMetricsSnapshot(**json.loads(raw)), rows)
                finalized = json.dumps(asdict(enriched), sort_keys=True, allow_nan=False)
                with db:
                    db.execute(
                        "UPDATE telemetry_pending SET finalized=? WHERE step=?", (finalized, step)
                    )
            count += self.store.add(TrainingMetricsSnapshot(**json.loads(finalized)))
        return count
