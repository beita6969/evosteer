"""Recover the observed pre-dispatch HealthBench import failure without resampling.

This is not a general retry facility. Static owner responses must be completely
recorded, and every failed score must have stopped at the missing blobfile import
before a judge client was constructed. Original artifacts remain untouched.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from skillev.contracts import JsonValue
from skillev.runtime.request_journal import DurableRequestJournal, UnknownRequestOutcomeError
from skillev.training.inflight import durable_json


class RecordedOwnerJournal(DurableRequestJournal):
    """A started episode may consume existing responses, never dispatch another."""

    def __init__(self, path: Path, recorded_episodes: frozenset[str]) -> None:
        super().__init__(path)
        self.recorded_episodes = recorded_episodes

    def request(
        self,
        *,
        identity: tuple[str, ...],
        endpoint: str,
        payload: dict[str, JsonValue],
        send: Callable[[], tuple[int, JsonValue]],
    ) -> tuple[int, JsonValue]:
        def no_new_response() -> tuple[int, JsonValue]:
            raise UnknownRequestOutcomeError("started owner episode has no recorded response")

        return super().request(
            identity=identity,
            endpoint=endpoint,
            payload=payload,
            send=no_new_response if identity[0] in self.recorded_episodes else send,
        )


def prepare_dependency_recovery(
    source: Path,
    destination: Path,
    *,
    expanded: dict[str, Any],
    chunk_size: int,
) -> RecordedOwnerJournal:
    """Copy a fully settled response journal after the observed import-only failure."""
    source, destination = source.resolve(), destination.resolve()
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
        or destination.exists()
    ):
        raise ValueError("dependency recovery requires a separate new output directory")

    def read(path: Path) -> Any:
        return json.loads(path.read_text())

    if read(source / "expanded-controls-private.json") != expanded:
        raise ValueError("dependency recovery cannot change frozen inputs or controls")
    episodes = source / "episodes"
    plan = read(episodes / "plan-private.json")
    progress = read(episodes / "rollout-progress.json")
    if plan["chunk_size"] != chunk_size or progress["status"] != "failed":
        raise ValueError("source must be a settled failed run with the same chunk size")
    rows = progress["trajectories"]
    if not rows or any(r["stage"] not in {"artifact-ready", "failed"} for r in rows):
        raise ValueError("source still has an active or unknown owner episode")
    positions = {r["canonical_position"] for r in rows}
    recorded = frozenset(r["trajectory_id"] for r in rows)
    markers = [read(p) for p in episodes.glob("started-episode-*.json")]
    if {m["position"] for m in markers} != positions or {
        m["trajectory_id"] for m in markers
    } != recorded:
        raise ValueError("started episode records disagree")

    def benchmark(position: int) -> str:
        return str(expanded["panel"]["records"][position]["source"]["benchmark"])

    if any(benchmark(r["canonical_position"]) == "alfworld" for r in rows):
        raise ValueError("dependency recovery does not replay environment actions")
    failed = {r["canonical_position"] for r in rows if r["stage"] == "failed"}
    failures = [read(p) for p in episodes.glob("failure-episode-*-private.json")]
    if not failed or {f["position"] for f in failures} != failed:
        raise ValueError("source failure coverage is incomplete")
    for failure in failures:
        if (
            benchmark(failure["position"]) != "healthbench"
            or failure["error_type"] != "TerminalEvaluatorError"
            or "_load_official" not in failure["traceback"]
            or "ModuleNotFoundError: No module named 'blobfile'" not in failure["traceback"]
        ):
            raise ValueError("failure is not the observed pre-dispatch dependency failure")
    original = source.parent / f"{source.name}-evaluation-requests.sqlite3"
    criteria = original.with_name(original.stem + "-healthbench-criteria")
    ledgers = [read(p) for p in criteria.glob("*.json")]
    if {row["binding"]["task_id"] for row in ledgers} != {
        failure["task_id"] for failure in failures
    } or any(
        row["status"] != "incomplete-grading"
        or row.get("error_type") != "ModuleNotFoundError"
        or row["requests"]
        or row["result"] is not None
        for row in ledgers
    ):
        raise ValueError("a judge request may already have been sent; recovery is not allowed")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    journal = destination.parent / f"{destination.name}-evaluation-requests.sqlite3"
    if journal.exists():
        raise ValueError("recovery journal must be new")
    with sqlite3.connect(original.as_uri() + "?mode=ro", uri=True) as db:
        requests = list(db.execute("SELECT identity,state,status FROM requests"))
        if not requests or any(
            json.loads(identity)[0] not in recorded or state != "COMPLETED" or status != 200
            for identity, state, status in requests
        ):
            raise ValueError("owner requests are not all durably completed")
        journal.touch(mode=0o600, exist_ok=False)
        with sqlite3.connect(journal) as target:
            db.backup(target)
    durable_json(
        destination.parent / f"{destination.name}-dependency-recovery.json",
        {
            "source": str(source),
            "recovery": "blobfile-import-before-judge-dispatch@1",
            "recorded_owner_episodes": sorted(recorded),
            "recorded_owner_responses": len(requests),
            "prior_judge_requests": 0,
            "resample_started_episodes": False,
            "new_generations": "originally-unstarted-episodes-only",
            "cost_accounting": "original-generations-count-once-plus-new-calls",
        },
    )
    return RecordedOwnerJournal(journal, recorded)
