"""Opt-in new-run event axes; one SDK writer and read-only controller forwarding."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime

from skillev.training.controller_status import controller_observation
from skillev.training.metric_events import EVENT_CONTRACT


def define_axes(run):
    for axis in (
        "train/optimizer_step",
        "quality/policy_step",
        "telemetry/optimizer_step",
        "heartbeat/unix",
    ):
        run.define_metric(axis)
    for prefix, axis in (
        ("train", "train/optimizer_step"),
        ("quality", "quality/policy_step"),
        ("telemetry", "telemetry/optimizer_step"),
        ("heartbeat", "heartbeat/unix"),
    ):
        run.define_metric(prefix + "/*", step_metric=axis, step_sync=False)
    for prefix in ("timing", "tokens", "skills", "posterior", "phase", "evolution", "resources"):
        run.define_metric(prefix + "/*", step_metric="train/optimizer_step", step_sync=False)


def declare_source(path, *, run_id, condition_id):
    if not run_id or not condition_id:
        raise ValueError(
            "new event contract needs the actual source run and condition before quality0"
        )
    value = {"run_id": run_id, "condition_id": condition_id}
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("new run cannot inherit another source/condition identity")
    else:
        with path.open("x") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())


def quality_points(summary):
    groups = {}
    for key, value in summary.items():
        _, coordinate, name = key.split("/", 2)
        step = int(coordinate.removeprefix("step"))
        groups.setdefault(step, {})["quality/" + name] = value
    return [
        {"record_kind": "quality_evaluation", "quality/policy_step": step, **groups[step]}
        for step in sorted(groups)
    ]


class QualityPublisher:
    def __init__(self, wandb, path, state, timeout):
        self.wandb, self.path, self.timeout = wandb, path, timeout
        self.pending = state / "quality-pending.json"
        self.known = self.remote()
        if self.pending.exists():
            point = json.loads(self.pending.read_text())
            self.await_visible(point)
            self.pending.unlink()

    def remote(self):
        result = {}
        for row in self.wandb.Api().run(self.path).scan_history(use_cache=False):
            if row.get("record_kind") != "quality_evaluation":
                continue
            step = row["quality/policy_step"]
            point = {k: v for k, v in row.items() if k == "record_kind" or k.startswith("quality/")}
            if step in result and result[step] != point:
                raise RuntimeError("conflicting quality history at one policy snapshot")
            result[step] = point
        return result

    def await_visible(self, point):
        deadline = time.monotonic() + self.timeout
        while True:
            row = self.remote().get(point["quality/policy_step"])
            if row is not None:
                if any(row.get(k) != v for k, v in point.items()):
                    raise RuntimeError("completed quality evidence differs from remote history")
                self.known[point["quality/policy_step"]] = point
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("quality ACK unavailable; pending retained, not resent")
            time.sleep(min(5, max(0, deadline - time.monotonic())))

    def publish(self, run, points):
        for point in points:
            step = point["quality/policy_step"]
            if step in self.known:
                if any(self.known[step].get(k) != v for k, v in point.items()):
                    raise RuntimeError("local and remote quality evidence differ")
                continue
            with self.pending.open("w") as stream:
                json.dump(point, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            run.log(point, commit=True)
            self.await_visible(point)
            self.pending.unlink()


def heartbeat(
    run,
    *,
    known,
    controller,
    source_run_id,
    stale_after,
    progress_after,
    last_sent,
    monotonic_now,
    last_uploaded_at=None,
):
    if last_sent is not None and monotonic_now - last_sent < 60:
        return last_sent
    value = None
    if controller is not None:
        try:
            value = json.loads(controller.read_text())
        except FileNotFoundError:
            pass
    observation = controller_observation(
        value,
        now=datetime.now(UTC),
        heartbeat_stale_after=stale_after,
        progress_stale_after=progress_after,
        expected_run_id=source_run_id,
    )
    last_ack = max(known, default=0)
    last_commit = observation.get("last_committed_step")
    point = {
        "record_kind": "uploader_heartbeat",
        "heartbeat/unix": time.time(),
        "heartbeat/last_acknowledged_optimizer_step": last_ack,
        "heartbeat/last_uploaded_at": last_uploaded_at,
        "heartbeat/upload_lag_steps": max(0, last_commit - last_ack)
        if type(last_commit) is int
        else None,
        "heartbeat/scope": "uploader-not-training-process",
        "heartbeat/event_contract": EVENT_CONTRACT,
    }
    for name in (
        "state",
        "declared_state",
        "freshness",
        "heartbeat_age_seconds",
        "phase_progress_age_seconds",
        "last_committed_step",
        "source_commit_id",
        "committed_at",
        "evidence_status",
    ):
        point["heartbeat/controller_" + name] = observation.get(name)
    run.summary.update(point)
    run.log(point, commit=True)
    return monotonic_now
