"""Publish committed batch aggregates with a local login; never upload trajectories.

Input is produced by `python -m skillev.training.metrics_export`. Only the
allowlisted scalar fields below leave the machine. An uncertain prior upload
must become visible in remote history before this process can continue: it is
not silently sent again. Use a new run ID for a new scientific condition.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import time
from pathlib import Path

import training_wandb_events
import training_wandb_telemetry

from skillev.training.metric_events import (
    EVENT_CONTRACT,
    NATIVE_FIELDS,
    committed_native_fields,
    supplemental_point,
    training_point,
)

METRICS = (
    "trajectory_count",
    "success_count",
    "batch_success_fraction",
    "reward_sum",
    "reward_mean",
    "ttb_loss",
    "raw_delta_sq_mean",
    "horizon_mean",
    "horizon_max",
    "action_count",
    "action_content_tokens",
    "parsed_count",
    "structural_valid_count",
    "parse_error_count",
    "schema_invalid_count",
    "first_turn_valid_count",
    "action_structure_valid_fraction",
    "first_turn_structure_valid_fraction",
    "admitted_count",
    "executed_count",
    "assessed_action_count",
    "execution_returned_success_count",
    "accepted_submission_count",
    "environment_terminal_count",
    "unique_source_question_count",
    "unknown_source_trajectory_count",
    "positive_delta_count",
    "negative_delta_count",
    "log_z_mean",
    "sum_forward_mean",
    "sum_backward_mean",
)

FORMAT_REVIEW_METRICS = (
    "native_reward_mean",
    "native_success_fraction",
    "format_review_count",
    "format_review_approved_count",
    "format_review_reward_gain",
    "format_review_api_input_tokens",
    "format_review_api_output_tokens",
)


def load_points(
    path: Path, *, source_state: Path | None = None, separated: bool = False
) -> list[dict]:
    points = {}
    identities = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            if row.get("format") not in {
                "skillev-training-metrics@1",
                "skillev-training-metrics@2",
            }:
                raise ValueError("W&B input is not a committed metrics snapshot")
            identities.add((row["run_id"], row["condition_id"]))
            step = row["optimizer_step"]
            if type(step) is not int or step < 1 or row["sampled_policy_step"] != step - 1:
                raise ValueError("W&B policy/update coordinates disagree")
            point = {
                "optimizer_step": step,
                "sampled_policy_step": row["sampled_policy_step"],
                "source_commit_id": row["commit_id"],
            }
            names = (
                METRICS
                + tuple(name for name in FORMAT_REVIEW_METRICS if name in row["metrics"])
                + (
                    (
                        "terminal_evidence_record_count",
                        "valid_terminal_record_count",
                        "valid_terminal_record_fraction",
                    )
                    if separated
                    else ()
                )
            )
            for name in names:
                value = row["metrics"].get(name)
                if value is not None and (
                    type(value) not in (int, float) or not math.isfinite(value)
                ):
                    raise ValueError("W&B metrics must be finite scalars or null")
                point["train/" + name] = value
            if step in points and points[step] != point:
                raise ValueError("conflicting committed metric point")
            points[step] = point
    if len(identities) > 1:
        raise ValueError("one W&B run cannot combine different source runs or conditions")
    if identities and source_state is not None:
        source_run, condition = next(iter(identities))
        source_identity = {"run_id": source_run, "condition_id": condition}
        if source_state.exists():
            if json.loads(source_state.read_text()) != source_identity:
                raise ValueError("uploader source run or condition changed between polls")
        else:
            temporary = source_state.with_suffix(".tmp")
            with temporary.open("w") as out:
                json.dump(source_identity, out)
                out.flush()
                os.fsync(out.fileno())
            temporary.replace(source_state)
    return [points[step] for step in sorted(points)]


QUALITY_METRICS = {
    "success_fraction": "panel/success_fraction",
    "reward": "panel/reward_mean",
    "first_turn_valid_fraction": "panel/first_turn_structure_valid_fraction",
    "action_structure_valid_fraction": "panel/action_structure_valid_fraction",
    "admitted_fraction": "panel/admitted_fraction",
    "trajectory_count": "panel/trajectory_count",
    "action_count": "panel/action_count",
}
for _domain in (
    "hotpotqa",
    "triviaqa",
    "aime-2026",
    "healthbench",
    "alfworld",
    "mbpp-plus",
    "humaneval",
):
    for _metric in ("success_fraction", "reward_mean"):
        QUALITY_METRICS[f"{_domain}_{_metric}"] = f"{_domain}/{_metric}"


def load_quality_summaries(root, *, source_state, committed_steps, include_zero=False):
    """Keep completed aggregate probes in the SAME SDK writer as training.

    Separate API summary writes were actually lost when the live SDK next
    published its cached summary. Never turn a quality probe into an Adam step.
    The mirror supplies only completed probe/decision files and source identity.
    """
    if root is None or not source_state.exists() or not (root / "source.json").exists():
        return {}
    source = json.loads(source_state.read_text())
    manifest = json.loads((root / "source.json").read_text())
    if any(manifest.get(k) != source[k] for k in ("run_id", "condition_id")):
        raise ValueError("quality source run or condition differs from training")
    scope = manifest.get("reference_scope", "unspecified")
    if scope not in (
        "owner-selected-composite",
        "fixed-held-out",
        "independent-iid",
        "unspecified",
    ):
        raise ValueError("unknown quality reference scope")
    result = {}
    for path in sorted(root.glob("probe-*.json")):
        probe = json.loads(path.read_text())
        step = probe["policy_step"]
        if type(step) is not int or step < 0:
            raise ValueError("quality policy step must be a nonnegative integer")
        if (step == 0 and not include_zero) or (step != 0 and step not in committed_steps):
            continue
        if probe["condition_id"] != source["condition_id"]:
            raise ValueError("quality probe belongs to another condition")
        decisions = root / f"decisions-{step:08d}.json"
        try:
            saved = json.loads(decisions.read_text())
        except FileNotFoundError:
            # The live mirror can replace this file between discovery and read.
            # Defer only this incomplete pair; never suppress malformed evidence.
            continue
        if not saved:
            continue
        decision = saved[-1]
        if (
            decision["policy_step"] != step
            or decision["policy_snapshot_id"] != probe["policy_snapshot_id"]
            or probe["evidence_id"] not in decision["evidence_ids"]
        ):
            raise ValueError("quality decision differs from its completed probe")
        prefix = f"quality/step{step}/"
        if decision["status"] not in ("verified", "warning", "regressed", "metrics-missing"):
            raise ValueError("quality probe has no completed decision")
        if decision["action"] not in ("continue", "pause-after-commit"):
            raise ValueError("quality action is not a declared decision")
        metrics = dict(QUALITY_METRICS)
        if include_zero:
            for domain, fields in NATIVE_FIELDS.items():
                for name, native in fields.items():
                    metrics[f"{domain}_{name}"] = f"{domain}/{native}"
        for name, metric in metrics.items():
            value = probe["metrics"].get(metric)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("quality metrics must be finite scalars or null")
            result[prefix + name] = value
        for name in ("status", "action", "triggered_rules", "metrics_unavailable"):
            result[prefix + name] = decision[name]
        if include_zero:
            result[prefix + "policy_snapshot_id"] = probe["policy_snapshot_id"]
            result[prefix + "condition_id"] = probe["condition_id"]
        result[prefix + "evidence_id"] = probe["evidence_id"]
        result[prefix + "t0_scope"] = scope
    return result


class IncompleteRemoteHistoryError(RuntimeError):
    """The live summary advanced before its queryable history index."""


def remote_points(wandb, path):
    run = wandb.Api().run(path)
    points, rows = {}, {}
    identical_rows = 0
    # A resumed SDK run has actually returned the same full history row twice,
    # including _step and _timestamp. Do not mistake it for a new commit or
    # resend it. Read full rows so matching IDs cannot conceal changed metrics.
    # W&B 0.30 cached a one-row history for this live run even though its
    # server summary was at step 6. Fresh server history is required for ACKs.
    for point in run.scan_history(use_cache=False):
        if "optimizer_step" not in point or "source_commit_id" not in point:
            continue
        step, commit = int(point["optimizer_step"]), point["source_commit_id"]
        if step in points:
            if point != rows[step]:
                raise RuntimeError("remote history contains conflicting optimizer-step rows")
            identical_rows += 1
            continue
        points[step], rows[step] = commit, dict(point)
    expected = getattr(run, "summary", {}).get("optimizer_step", 0)
    if type(expected) is not int or expected < 0:
        raise RuntimeError("remote optimizer cursor is invalid")
    first = getattr(run, "config", {}).get("first_optimizer_step", 1)
    if type(first) is not int or first < 1:
        raise RuntimeError("remote condition-segment start is invalid")
    last = max(expected, max(points, default=0))
    if set(points) != set(range(first, last + 1)):
        raise IncompleteRemoteHistoryError(
            "remote history incomplete; no existing step may be resent"
        )
    if identical_rows:
        print(json.dumps({"remote_identical_history_rows": identical_rows}), flush=True)
    return points


def await_remote_points(wandb, path, seconds):
    # A real step-7 ACK observed the new summary before all history rows were
    # queryable. Poll reads only; never repeat run.log() to repair visibility.
    deadline = time.monotonic() + seconds
    while True:
        try:
            return remote_points(wandb, path)
        except IncompleteRemoteHistoryError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(5, remaining))


def await_visible(wandb, path, step, commit, seconds):
    deadline = time.monotonic() + seconds
    while True:
        remote = await_remote_points(wandb, path, max(0, deadline - time.monotonic()))
        if step in remote:
            if remote[step] != commit:
                raise RuntimeError("remote step belongs to another commit")
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "upload acknowledgment unavailable; pending point retained, not resent"
            )
        time.sleep(5)


def publish_uploader_heartbeat(run, *, known, last_sent, monotonic_now):
    """Keep the metrics-sync job visible while a long batch has no new point.

    System telemetry is intentionally disabled for privacy. Use the existing
    summary writer, never a synthetic history row or a training-liveness claim.
    """
    if last_sent is not None and monotonic_now - last_sent < 60:
        return last_sent
    run.summary.update(
        {
            "telemetry/uploader_heartbeat_unix": time.time(),
            "telemetry/last_acknowledged_optimizer_step": max(known, default=0),
            "telemetry/heartbeat_scope": "uploader-not-training-process",
        }
    )
    return monotonic_now


def sync(args):
    import wandb

    path = f"{args.entity}/{args.project}/{args.run_id}"
    pending = args.state / "pending.json"
    identity = args.state / "identity.json"
    resuming = identity.exists() or not args.new
    separated = getattr(args, "event_contract", "legacy") == EVENT_CONTRACT
    first = getattr(args, "first_optimizer_step", 1)
    if separated:
        training_wandb_events.declare_source(
            args.state / "source.json", run_id=args.source_run_id, condition_id=args.condition_id
        )
    target = {"entity": args.entity, "project": args.project, "run_id": args.run_id}
    if separated:
        target["event_contract"] = EVENT_CONTRACT
    if first != 1:
        target["first_optimizer_step"] = first
    if identity.exists():
        if json.loads(identity.read_text()) != target:
            raise ValueError("uploader state directory belongs to another W&B run")
        known = await_remote_points(wandb, path, args.ack_timeout)
    elif args.new:
        known = {}
    else:
        known = await_remote_points(wandb, path, args.ack_timeout)
    if separated and resuming:
        remote_config = wandb.Api().run(path).config
        if any(
            remote_config.get(k) != v
            for k, v in {
                "event_contract": EVENT_CONTRACT,
                "source_run_id": args.source_run_id,
                "condition_id": args.condition_id,
            }.items()
        ):
            raise ValueError(
                "separated metrics require the same new-run source/event contract; "
                "never migrate old history"
            )
    # Resolve a crash after log() without generating duplicate points.
    if pending.exists():
        point = json.loads(pending.read_text())
        await_visible(
            wandb, path, point["optimizer_step"], point["source_commit_id"], args.ack_timeout
        )
        known[point["optimizer_step"]] = point["source_commit_id"]
        pending.unlink()
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        resume="must" if identity.exists() or not args.new else "never",
        job_type="committed-training-metrics-sync",
        dir=str(args.state),
        config={
            "raw_data_uploaded": False,
            "first_optimizer_step": first,
            "metrics_format": "skillev-training-metrics@1",
            **(
                {
                    "event_contract": EVENT_CONTRACT,
                    "source_run_id": args.source_run_id,
                    "condition_id": args.condition_id,
                }
                if separated
                else {}
            ),
            "job_finish_is_not_training_completion": True,
        },
        settings=wandb.Settings(
            disable_code=True,
            disable_git=True,
            save_code=False,
            console="off",
            x_disable_stats=True,
            x_disable_meta=True,
            x_disable_machine_info=True,
            x_disable_viewer=True,
        ),
    )
    identity.write_text(json.dumps(target))
    if separated:
        training_wandb_events.define_axes(run)
    else:
        run.define_metric("optimizer_step")
        run.define_metric("*", step_metric="optimizer_step")
        run.define_metric("telemetry/optimizer_step")
    telemetry = None
    quality_publisher = None
    if separated and (args.state / "quality-pending.json").exists():
        quality_publisher = training_wandb_events.QualityPublisher(
            wandb, path, args.state, args.ack_timeout
        )
    last_quality = {}
    last_heartbeat = None
    last_uploaded_at = None
    upload_observation = args.state / "last-uploaded.json"
    if separated and upload_observation.exists():
        last_uploaded_at = json.loads(upload_observation.read_text())["observed_at"]
    combined = set()
    if separated and resuming:
        combined = {
            r["source_commit_id"]
            for r in wandb.Api().run(path).scan_history(use_cache=False)
            if r.get("record_kind") == "committed_training_step" and "telemetry_combined" in r
        }

    def update_quality(points):
        nonlocal last_quality, quality_publisher
        quality = load_quality_summaries(
            args.quality_root,
            source_state=args.state / "source.json",
            committed_steps={point["optimizer_step"] for point in points} & known.keys(),
            include_zero=separated,
        )
        if quality and quality != last_quality:
            run.summary.update(quality)
            if separated:
                if quality_publisher is None:
                    quality_publisher = training_wandb_events.QualityPublisher(
                        wandb, path, args.state, args.ack_timeout
                    )
                quality_publisher.publish(run, training_wandb_events.quality_points(quality))
            last_quality = quality

    try:
        while True:
            points = load_points(
                args.metrics, source_state=args.state / "source.json", separated=separated
            )
            supplemental = training_wandb_telemetry.load_points(args.metrics)
            if separated:
                by_step = {p["telemetry/optimizer_step"]: p for p in supplemental}
                native = committed_native_fields(
                    getattr(args, "events", None),
                    run_id=args.source_run_id,
                    expected_steps={p["source_commit_id"]: p["optimizer_step"] for p in points},
                )
                points = [
                    training_point(
                        p, by_step.get(p["optimizer_step"]), native.get(p["source_commit_id"])
                    )
                    for p in points
                ]
                if getattr(args, "events", None) is not None:
                    # Native source mirror must reach the same complete commit.
                    # Do not freeze null, ACK it, then lose the late native data.
                    ready = []
                    for p in points:
                        if p["source_commit_id"] not in native:
                            break
                        ready.append(p)
                    points = ready
                for p in points:
                    if "train/action_execution_valid_fraction" in p:
                        p["train/execution_returned_success_fraction"] = p[
                            "train/action_execution_valid_fraction"
                        ]
            update_quality(points)
            for point in points:
                if separated and any(
                    p["telemetry/source_commit_id"] == point["source_commit_id"]
                    for p in supplemental
                ):
                    point["telemetry_combined"] = True
                step, commit = point["optimizer_step"], point["source_commit_id"]
                if step < first:
                    raise ValueError("source metrics precede this condition segment")
                if step in known:
                    if known[step] != commit:
                        raise RuntimeError("local and W&B committed points differ")
                    continue
                if known and step <= max(known):
                    raise RuntimeError(
                        "remote history has a gap; do not silently discard an old step"
                    )
                if step != max(known, default=first - 1) + 1:
                    raise ValueError("local condition metrics omit a committed step")
                with pending.open("w") as out:
                    json.dump(point, out, allow_nan=False)
                    out.flush()
                    os.fsync(out.fileno())
                # SDK history is append-only; optimizer_step is the scientific x axis.
                # Supplemental old-step telemetry must not make future training
                # points collide with an already used SDK internal history index.
                run.log(point, commit=True)
                await_visible(wandb, path, step, commit, args.ack_timeout)
                known[step] = commit
                if separated:
                    last_uploaded_at = training_wandb_events.datetime.now(
                        training_wandb_events.UTC
                    ).isoformat()
                    upload_observation.write_text(
                        json.dumps(
                            {
                                "observed_at": last_uploaded_at,
                                "optimizer_step": step,
                                "source_commit_id": commit,
                            }
                        )
                    )
                if separated and point.get("telemetry_combined"):
                    combined.add(commit)
                pending.unlink()
                print(json.dumps({"uploaded_optimizer_step": step, "url": run.url}), flush=True)
            update_quality(points)
            if separated:
                supplemental = [
                    supplemental_point({**p, **native.get(p["telemetry/source_commit_id"], {})})
                    for p in supplemental
                    if p["telemetry/source_commit_id"] not in combined
                ]
            if supplemental:
                if telemetry is None:
                    telemetry = training_wandb_telemetry.Publisher(
                        wandb, path, args.state, args.ack_timeout
                    )
                telemetry.publish(run, supplemental, known)
            if not args.follow:
                break
            if separated:
                last_heartbeat = training_wandb_events.heartbeat(
                    run,
                    known=known,
                    controller=args.controller_status,
                    source_run_id=args.source_run_id,
                    stale_after=args.controller_stale_after,
                    progress_after=args.controller_progress_stale_after,
                    last_sent=last_heartbeat,
                    monotonic_now=time.monotonic(),
                    last_uploaded_at=last_uploaded_at,
                )
            else:
                last_heartbeat = publish_uploader_heartbeat(
                    run, known=known, last_sent=last_heartbeat, monotonic_now=time.monotonic()
                )
            time.sleep(5)
    finally:
        run.summary["sync_job_is_not_training_completion"] = True
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, help="Mirrored completed aggregate probes")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--first-optimizer-step", type=int, default=1)
    parser.add_argument("--event-contract", choices=("legacy", EVENT_CONTRACT), default="legacy")
    parser.add_argument("--source-run-id")
    parser.add_argument("--condition-id")
    parser.add_argument(
        "--events", type=Path, help="Private authoritative commits for native aggregate allowlist"
    )
    parser.add_argument("--controller-status", type=Path)
    parser.add_argument("--controller-stale-after", type=float)
    parser.add_argument("--controller-progress-stale-after", type=float)
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--ack-timeout", type=float, default=120.0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("metrics sync is CPU-only; set CUDA_VISIBLE_DEVICES to empty")
    if not math.isfinite(args.ack_timeout) or args.ack_timeout <= 0:
        parser.error("ack timeout must be positive")
    if args.first_optimizer_step < 1:
        parser.error("first optimizer step must be positive")
    if args.event_contract == EVENT_CONTRACT and args.follow:
        if any(
            v is None or not math.isfinite(v) or v <= 0
            for v in (args.controller_stale_after, args.controller_progress_stale_after)
        ):
            parser.error("declare positive controller heartbeat/progress freshness intervals")
    args.state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (args.state / "sync.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sync(args)


if __name__ == "__main__":
    main()
