"""Read-only evidence maturity, distinct from cold-start eligibility and efficacy.

Repeated reads remain real posterior events. They do not multiply independent
sources here; this diagnostic is never consulted by the evolution policy.
"""

from dataclasses import asdict, dataclass
from typing import cast

from skillev.contracts import JsonValue

from .evidence_context import TrajectoryEvidenceContext
from .posterior_state import PosteriorEventProvenance


@dataclass(frozen=True)
class CoverageDiagnosticPolicy:
    minimum_read_trajectories: int = 16
    minimum_read_source_questions: int = 8
    minimum_read_batches: int = 3

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 1 for n in asdict(self).values()):
            raise ValueError("coverage diagnostic counts must be positive")


def skill_coverage_report(
    provenance: PosteriorEventProvenance,
    policy: CoverageDiagnosticPolicy | None = None,
) -> dict[str, JsonValue]:
    policy = policy or CoverageDiagnosticPolicy()
    scopes: dict[
        tuple[str, str, str | None, tuple[str, ...] | None],
        list[tuple[str, TrajectoryEvidenceContext]],
    ] = {}
    for batch in provenance.batches:
        for context in batch.trajectory_contexts:
            key = (
                batch.library_version,
                context.task_family,
                context.context_id,
                context.available_tools,
            )
            scopes.setdefault(key, []).append((batch.posterior.batch_id, context))
    rows: list[JsonValue] = []
    for (library, family, context_id, tools), observations in scopes.items():
        reads = [
            (b, c)
            for b, c in observations
            if any(link.body_returned is True for link in c.invocation_links or ())
        ]
        known = all(
            c.invocation_links is not None
            and all(link.body_returned is not None for link in c.invocation_links)
            for _, c in observations
        )
        sources = {c.canonical_source_key for _, c in reads if c.canonical_source_key is not None}
        batches = {b for b, _ in reads}
        broad = (
            len(reads) >= policy.minimum_read_trajectories
            and len(sources) >= policy.minimum_read_source_questions
            and len(batches) >= policy.minimum_read_batches
        )
        maturity = (
            "unknown-historical-evidence"
            if not known
            else "never-read"
            if not reads
            else "broader-evidence-benefit-unestablished"
            if broad
            else "sparse-read-evidence"
        )
        links = [
            link
            for _, c in reads
            for link in c.invocation_links or ()
            if link.body_returned is True
        ]
        direct = [c for _, c in observations if c.invocation_links == ()]

        def successes(contexts: list[TrajectoryEvidenceContext]) -> int | None:
            return (
                sum(c.terminal_success is True for c in contexts)
                if all(c.terminal_success is not None for c in contexts)
                else None
            )

        rows.append(
            {
                "library_version": library,
                "task_family": family,
                "context_id": context_id,
                "available_tools": list(tools) if tools is not None else None,
                "trajectory_count": len(observations),
                "read_trajectory_count": len(reads),
                "read_source_question_count": len(sources),
                "read_batch_count": len(batches),
                "source_identity_unknown_reads": sum(
                    c.canonical_source_key is None for _, c in reads
                ),
                "body_read_event_count": len(links),
                "repeat_same_version_read_count": sum(
                    link.repeat_same_version is True for link in links
                )
                if all(link.repeat_same_version is not None for link in links)
                else None,
                "read_followed_by_task_action_count": sum(
                    bool(link.following_task_action_steps) for link in links
                )
                if all(link.following_task_action_steps is not None for link in links)
                else None,
                "read_trajectory_successes": successes([c for _, c in reads]),
                "no_read_trajectory_count": len(direct),
                "no_read_trajectory_successes": successes(direct),
                "evidence_maturity": maturity,
                "verified_effective_use": None,
                "efficacy_status": "requires-independent-training-side-utility-comparison",
            }
        )
    return {
        "format": "skill-coverage-maturity@1",
        "diagnostic_policy": cast(dict[str, JsonValue], asdict(policy)),
        "scopes": rows,
        "interpretation": (
            "read/no-read outcomes are selected observational groups, not a causal contrast"
        ),
        "posterior_or_evolution_changes": False,
        "stable_effective_state": "never inferred from call count, posterior mass, or one success",
    }
