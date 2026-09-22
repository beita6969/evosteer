from dataclasses import replace
from types import SimpleNamespace

from skillev.training.evidence_context import TrajectoryEvidenceContext
from skillev.training.invocation_evidence import InvocationExecutionLink
from skillev.training.skill_coverage import CoverageDiagnosticPolicy, skill_coverage_report


def context(trajectory, source, *, reads=0, success=False):
    links = tuple(
        InvocationExecutionLink(
            i + 1,
            "procedure",
            True,
            "success",
            (reads + 1,),
            success,
            body_returned=True,
            returned_skill_version="v1",
            returned_library_version="library",
            body_visible_execution_steps=(reads + 1,),
            read_ordinal=i + 1,
            repeat_same_version=i > 0,
            following_task_action_steps=(reads + 1,),
        )
        for i in range(reads)
    )
    return TrajectoryEvidenceContext(
        trajectory,
        "synthetic/task",
        "synthetic",
        "training-development",
        source,
        ("procedure",),
        ("procedure",),
        ("procedure",),
        ("procedure",),
        reads + 1,
        reads,
        invocation_links=links,
        context_id="synthetic-context",
        available_tools=(),
        terminal_success=success,
        terminal_reward=float(success),
    )


def report(contexts, policy=None):
    provenance = SimpleNamespace(
        batches=(
            SimpleNamespace(
                library_version="library",
                posterior=SimpleNamespace(batch_id="batch"),
                trajectory_contexts=contexts,
            ),
        )
    )
    return skill_coverage_report(provenance, policy)["scopes"][0]


def test_many_repeated_reads_of_one_source_are_still_sparse():
    row = report(
        [
            context("t1", "same", reads=20),
            context("t2", "same", reads=20),
            context("t3", "easy", success=True),
        ]
    )
    assert row["body_read_event_count"] == 40
    assert row["read_source_question_count"] == 1
    assert row["read_trajectory_count"] == 2
    assert row["repeat_same_version_read_count"] == 38
    assert row["evidence_maturity"] == "sparse-read-evidence"
    assert row["read_trajectory_successes"] == 0
    assert row["no_read_trajectory_successes"] == 1
    assert row["verified_effective_use"] is None


def test_broad_success_evidence_does_not_automatically_qualify_skill_utility():
    row = report(
        [context("t1", "s1", reads=1, success=True), context("t2", "s2", reads=1, success=True)],
        CoverageDiagnosticPolicy(2, 2, 1),
    )
    assert row["evidence_maturity"] == "broader-evidence-benefit-unestablished"
    assert row["verified_effective_use"] is None


def test_never_called_and_historical_unknown_are_distinct_and_restore_preserves_facts():
    original = context("t1", "s1")
    assert TrajectoryEvidenceContext.from_value(original.to_value()) == original
    assert report([original])["evidence_maturity"] == "never-read"
    unknown = replace(original, invocation_links=None, terminal_success=None, terminal_reward=None)
    assert report([unknown])["evidence_maturity"] == "unknown-historical-evidence"
