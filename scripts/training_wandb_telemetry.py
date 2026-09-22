"""Aggregate-only supplemental history, keyed to an existing training commit.

Older W&B rows cannot be edited in place. A separate metric axis allows filling
missing telemetry without emitting a second optimizer-commit row. SDK history
indices remain append-only; scientific coordinates remain explicit.
"""

from __future__ import annotations

import json
import math
import os
import time

DOMAINS = ("hotpotqa", "triviaqa", "aime-2026", "healthbench", "alfworld", "mbpp-plus", "humaneval")
DEFINITIONS = {
    "binary_success": "effective terminal labels; format reviews retain separate native labels",
    "format_review": "Step81 opt-in training content review, never relabelled as native pass",
    "action_admission": "admitted / all sampled actions; incomplete evidence is null",
    "execution_validity": "successful execution returns / executed calls; not task success",
    "domains": "per-domain batch aggregates, four rollouts of one source; not IID accuracy",
    "skill_invocations": "executed credited skill IDs, not retrieval or prompt exposure",
    "skill_target": (
        "HISTORICAL withdrawn catalog-then-read@2: two distinct executed skill IDs; "
        "a prompt target, not forced execution; skill reads remain actions/turns; "
        "older conditions or incomplete evidence remain null"
    ),
    "posterior": "committed update events/cells; invocation events are not independent samples",
    "mutation": "actual installed library mutation, distinct from detection, trigger and no-op",
    "gpu_hours": "declared GPU reservation accounting, not device utilization or FLOPs",
    "step_gpu_hours": "declared GPU count times transaction wall; excludes quality gaps",
    "process_gpu_hours": "declared GPU count times process elapsed; includes waiting/quality",
    "tokens": "settled collection budget ledger; no assumed actor/grader split",
    "server_deltas": "same-process counter intervals; may include quality work; first unknown",
    "gpu_scopes": "step and process scopes overlap and must not be added together",
    "unknown": "missing evidence remains null, never zero",
    "history": "supplemental rows reference original commits, not extra optimizer steps",
}
FIELDS = {
    "skill_invocation_count": "skills/invocation_count",
    "skill_visible_id_count": "skills/visible_id_count",
    "skill_body_read_count": "skills/body_read_count",
    "skill_body_read_failed_count": "skills/body_read_failed_count",
    "skill_body_read_unknown_count": "skills/body_read_unknown_count",
    "skill_read_followed_by_action_count": "skills/read_followed_by_action_count",
    "evolved_skill_visible_count": "skills/evolved_visible_count",
    "evolved_skill_invoked_count": "skills/evolved_invoked_count",
    "cold_start_trigger_count": "phase/cold_start_trigger_count",
    "natural_phase_trigger_count": "phase/natural_trigger_count",
    "skill_target_trajectory_count": "skills/target_trajectory_count",
    "skill_target_unknown_trajectory_count": "skills/target_unknown_trajectory_count",
    "skill_target_met_trajectory_count": "skills/two_distinct_met_trajectory_count",
    "skill_target_met_fraction": "skills/two_distinct_met_fraction",
    "skill_distinct_per_trajectory_mean": "skills/distinct_per_trajectory_mean",
    "skill_catalog_below_target_trajectory_count": "skills/catalog_below_two_trajectory_count",
    "posterior_update_event_count": "posterior/update_event_count",
    "posterior_cells_touched_count": "posterior/cells_touched_count",
    "posterior_update_event_count_cumulative": "posterior/update_event_count_cumulative",
    "posterior_cells_touched_cumulative": "posterior/cells_touched_cumulative",
    "posterior_update_event_count_segment_cumulative": (
        "posterior/update_event_count_segment_cumulative"
    ),
    "posterior_cells_touched_segment_cumulative": "posterior/cells_touched_segment_cumulative",
    "phase_detection_record_count": "phase/detection_record_count",
    "phase_check_count": "phase/check_count",
    "phase_trigger_count": "phase/trigger_count",
    "phase_no_op_count": "phase/no_op_count",
    "library_mutation_count": "evolution/library_mutation_count",
    "library_mutation_action_count": "evolution/mutation_action_count",
    "logical_input_tokens": "tokens/logical_input_tokens",
    "logical_output_tokens": "tokens/logical_output_tokens",
    "logical_model_calls": "tokens/logical_model_calls",
    "authoring_input_tokens": "tokens/authoring_input_tokens",
    "authoring_output_tokens": "tokens/authoring_output_tokens",
    "authoring_model_calls": "tokens/authoring_model_calls",
    "step_wall_seconds": "timing/step_wall_seconds",
    "rollout_span_seconds": "timing/rollout_span_seconds",
    "gradient_span_seconds": "timing/gradient_span_seconds",
    "gradient_tail_seconds": "timing/gradient_tail_seconds",
    "overlap_seconds": "timing/overlap_seconds",
    "durability_seconds": "timing/durability_seconds",
    "reserved_gpu_hours_step": "resources/reserved_gpu_hours_step",
    "reserved_gpu_hours_observed_cumulative": "resources/reserved_gpu_hours_observed_cumulative",
    "reserved_gpu_hours_process_elapsed": "resources/reserved_gpu_hours_process_elapsed",
    "reserved_gpu_hours_processes_observed_cumulative": (
        "resources/reserved_gpu_hours_processes_observed_cumulative"
    ),
    "reserved_gpu_count": "resources/reserved_gpu_count",
}

for _counter in (
    "server_generated_tokens",
    "admitted_content_tokens",
    "admitted_stop_tokens",
    "discarded_suffix_tokens",
):
    for _scope in ("delta", "process_cumulative"):
        FIELDS[f"{_counter}_{_scope}"] = f"tokens/{_counter}_{_scope}"


def _scalar(value):
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
        raise ValueError("supplemental metrics must be finite numbers or null")
    return value


def _fraction(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return _scalar(numerator) / _scalar(denominator)


def load_points(path):
    points = {}
    for line in path.read_text().splitlines(keepends=True):
        if not line.endswith("\n"):
            break
        row = json.loads(line)
        if row.get("format") != "skillev-training-metrics@2":
            continue
        step = row["optimizer_step"]
        if type(step) is not int or step < 1:
            raise ValueError("supplemental metrics need a committed positive step")
        point = {
            "telemetry/optimizer_step": step,
            "telemetry/source_commit_id": row["commit_id"],
            "telemetry/format": "committed-training-telemetry@1",
        }
        m = row["metrics"]
        point.update(
            {
                "train/binary_success_rate": _scalar(m.get("batch_success_fraction")),
                "train/action_admitted_fraction": _fraction(
                    m.get("admitted_count"), m.get("action_count")
                ),
                "train/action_execution_valid_fraction": _fraction(
                    m.get("execution_returned_success_count"), m.get("executed_count")
                ),
            }
        )
        for domain in DOMAINS:
            values = row.get("native_metrics_by_domain", {}).get(domain, {})
            for name in ("reward_mean", "success_fraction", "trajectory_count"):
                point[f"train/domain/{domain}/{name}"] = _scalar(values.get(name))
            for name in (
                "native_reward_mean",
                "native_success_fraction",
                "format_review_count",
                "format_review_approved_count",
                "format_review_reward_gain",
            ):
                if name in values:
                    point[f"train/domain/{domain}/{name}"] = _scalar(values[name])
            counts = row.get("telemetry", {}).get("native_success_counts", {}).get(domain, {})
            point[f"train/domain/{domain}/success_count"] = _scalar(counts.get("success_count"))
            if "native_success_count" in counts:
                point[f"train/domain/{domain}/native_success_count"] = _scalar(
                    counts["native_success_count"]
                )
        for source, target in FIELDS.items():
            point[target] = _scalar(row.get("telemetry", {}).get(source))
        if step in points and points[step] != point:
            raise ValueError("conflicting supplemental metric values")
        points[step] = point
    return [points[step] for step in sorted(points)]


def remote_rows(wandb, path):
    rows, raw = {}, {}
    for row in wandb.Api().run(path).scan_history(use_cache=False):
        if "telemetry/source_commit_id" not in row:
            continue
        step = row["telemetry/optimizer_step"]
        values = {k: v for k, v in row.items() if not k.startswith("_")}
        if step in raw and raw[step] != row:
            raise RuntimeError("conflicting supplemental remote rows")
        raw[step] = row
        rows[step] = values
    return rows


def require_same(expected, actual):
    # Missing/null metrics are deliberately unknown. No raw text is compared or uploaded.
    if any(actual.get(key) != value for key, value in expected.items()):
        raise RuntimeError("local and remote supplemental evidence differ")


def await_visible(wandb, path, point, seconds):
    deadline = time.monotonic() + seconds
    while True:
        row = remote_rows(wandb, path).get(point["telemetry/optimizer_step"])
        if row is not None:
            require_same(point, row)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("supplemental ACK unavailable; pending row retained, not resent")
        time.sleep(min(5, max(0, deadline - time.monotonic())))


class Publisher:
    def __init__(self, wandb, path, state, timeout):
        self.wandb, self.path, self.timeout = wandb, path, timeout
        self.pending = state / "telemetry-pending.json"
        self.known = remote_rows(wandb, path)
        if self.pending.exists():
            point = json.loads(self.pending.read_text())
            await_visible(wandb, path, point, timeout)
            self.known[point["telemetry/optimizer_step"]] = point
            self.pending.unlink()

    def publish(self, run, points, committed):
        run.summary["telemetry/metric_definitions"] = DEFINITIONS
        for point in points:
            step, commit = point["telemetry/optimizer_step"], point["telemetry/source_commit_id"]
            if step not in committed:
                continue
            if committed[step] != commit:
                raise ValueError("telemetry does not refer to the acknowledged training commit")
            if step in self.known:
                require_same(point, self.known[step])
                continue
            for name in point:
                if name not in (
                    "telemetry/optimizer_step",
                    "telemetry/source_commit_id",
                    "telemetry/format",
                    "record_kind",
                ):
                    run.define_metric(name, step_metric="telemetry/optimizer_step", step_sync=False)
            with self.pending.open("w") as out:
                json.dump(point, out, allow_nan=False)
                out.flush()
                os.fsync(out.fileno())
            run.log(point, commit=True)
            await_visible(self.wandb, self.path, point, self.timeout)
            self.known[step] = point
            self.pending.unlink()
            print(json.dumps({"uploaded_telemetry_optimizer_step": step}), flush=True)
