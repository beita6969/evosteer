"""Read-only batch arithmetic by actual source/trajectory, not causal attribution."""

import math
from collections import defaultdict

from skillev.contracts import JsonValue

from .step_math import PreparedTTBStep


def ttb_update_diagnostics(prepared: PreparedTTBStep) -> dict[str, JsonValue]:
    rows: list[JsonValue] = []
    by_domain: dict[str, list[float]] = defaultdict(list)
    total = len(prepared.batch.artifacts)
    residuals = {r.trajectory_id: r for r in prepared.residuals}
    for artifact in prepared.batch.artifacts:
        record = artifact.record
        r = residuals[record.trajectory_id]
        source = record.reward.native_payload.get("training_evidence_source")
        source = source if isinstance(source, dict) else {}
        domain = str(
            source.get("benchmark_id", record.initial_context.meta.get("benchmark_id", "unknown"))
        )
        loss = (r.delta / r.horizon) ** 2
        by_domain[domain].append(loss)
        rows.append(
            {
                "trajectory_id": record.trajectory_id,
                "source_question_id": source.get("source_question_id"),
                "domain": domain,
                "delta": r.delta,
                "delta_squared": r.delta**2,
                "normalized_loss": loss,
                "horizon": r.horizon,
                "action_token_counts": [step.action_token_count for step in record.steps],
                "binary_success": record.reward.success,
                "observation_statuses": [step.observation_status for step in record.steps],
                "log_z": r.log_z,
                "sum_forward": r.sum_forward,
                "sum_backward": r.sum_backward,
                "forward_delta_coefficient": 2 * r.delta / (total * r.horizon**2),
                "token_coefficients": [
                    2 * r.delta / (total * r.horizon**2 * step.action_token_count)
                    for step in record.steps
                ],
            }
        )
    return {
        "format": "read-only-ttb-decomposition@1",
        "interpretation": "loss contribution is not gradient contribution or causal task benefit",
        "sampling_policy_snapshot_id": prepared.snapshot_before.snapshot_id,
        "sampled_policy_step": prepared.batch.optimizer_step - 1,
        "optimizer_step": prepared.batch.optimizer_step,
        "trajectory_count": total,
        "trajectories": rows,
        "domains": {
            domain: {
                "trajectory_count": len(values),
                "mean_normalized_loss": math.fsum(values) / len(values),
                "batch_loss_contribution": math.fsum(values) / total,
            }
            for domain, values in sorted(by_domain.items())
        },
    }
