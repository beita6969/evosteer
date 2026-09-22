"""CPU-only gate for frozen training-sequence terminal-evaluator infrastructure."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.contracts import JsonValue, stable_hash
from skillev.rollout import (
    NoSubmissionReason,
    NoTerminalSubmission,
    RolloutTask,
    RolloutTermination,
    TerminalEvaluationRequest,
)

from ..benchmarks.catalog import PrivateBenchmarkCatalog
from ..benchmarks.terminal_admission import (
    TerminalEvaluatorRouteAdmission,
    admit_terminal_evaluator_routes,
)


@dataclass(frozen=True, slots=True)
class TerminalEvaluatorGateReport:
    """Answer-free gate result safe for a public infrastructure report."""

    route_admission: TerminalEvaluatorRouteAdmission
    first_task_id_hash: str
    first_benchmark_id: str
    first_environment_id: str
    first_reward_value: float
    first_reward_success: bool
    first_verifier_version: str
    no_submission_replay_count: int
    verifier_version_counts: tuple[tuple[str, int], ...]

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "first_benchmark_id": self.first_benchmark_id,
            "first_environment_id": self.first_environment_id,
            "first_reward_success": self.first_reward_success,
            "first_reward_value": self.first_reward_value,
            "first_task_id_hash": self.first_task_id_hash,
            "first_verifier_version": self.first_verifier_version,
            "no_submission_replay_count": self.no_submission_replay_count,
            "route_admission": self.route_admission.to_value(),
            "verifier_version_counts": dict(self.verifier_version_counts),
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


async def run_terminal_evaluator_gate(
    catalog: PrivateBenchmarkCatalog,
    tasks: tuple[RolloutTask, ...],
) -> TerminalEvaluatorGateReport:
    """Admit all routes, then replay the exact first horizon boundary without a model."""

    admission = admit_terminal_evaluator_routes(catalog, tasks)
    first = tasks[0]
    context = first.public_context
    benchmark_id = context.get("benchmark_id") if isinstance(context, dict) else None
    if type(benchmark_id) is not str or not benchmark_id:
        raise ValueError("first training task has no benchmark identity")
    session = catalog.route((first,)).create(first)
    reward = await session.evaluator.evaluate(
        TerminalEvaluationRequest(
            trajectory_id="terminal-evaluator-gate-first-task",
            task_id=first.task_id,
            termination=RolloutTermination.HORIZON_EXHAUSTED,
            evaluation_input=NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED),
            public_transcript_hash=stable_hash(
                {
                    "environment_id": first.environment_id,
                    "termination": RolloutTermination.HORIZON_EXHAUSTED.value,
                }
            ),
        )
    )
    if reward.value != 0.0 or reward.success:
        raise ValueError("first-task no-submission outcome must be an exact zero failure")
    if reward.environment_id != first.environment_id:
        raise ValueError("first-task terminal reward has another environment identity")
    return TerminalEvaluatorGateReport(
        route_admission=admission,
        first_task_id_hash=stable_hash(first.task_id),
        first_benchmark_id=benchmark_id,
        first_environment_id=first.environment_id,
        first_reward_value=reward.value,
        first_reward_success=reward.success,
        first_verifier_version=reward.verifier_version,
        no_submission_replay_count=1,
        verifier_version_counts=((reward.verifier_version, 1),),
    )


__all__ = ["TerminalEvaluatorGateReport", "run_terminal_evaluator_gate"]
