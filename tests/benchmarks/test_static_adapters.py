from __future__ import annotations

import asyncio

import pytest
from skillev_private.benchmarks import (
    PrivateStaticBenchmarkCase,
    PrivateStaticBenchmarkEvaluator,
    PrivateStaticBenchmarkSessionFactory,
    PrivateStaticTarget,
    StaticScoringRule,
)

from skillev.benchmarks import (
    BenchmarkPublicItem,
    CompletionBenchmarkEnvironment,
    OrderedBenchmarkTaskProvider,
)
from skillev.contracts import canonical_json, stable_hash
from skillev.rollout import (
    NoSubmissionReason,
    NoTerminalSubmission,
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
)
from skillev.runtime import ActionKind, EnvironmentMethodFailedError, StructuredAction

PRIVATE_CANARY = "PRIVATE-ANSWER-CANARY-NEVER-PUBLIC"


def _public_item(*, task_id: str = "synthetic-static-001") -> BenchmarkPublicItem:
    return BenchmarkPublicItem(
        benchmark_id="synthetic-static",
        dataset_revision="fixture@1",
        split="dev",
        task_id=task_id,
        task_family="synthetic-static/short-answer",
        query="Return the public fixture response.",
        public_context={"answer_format": "short-text"},
    )


def _request(item: BenchmarkPublicItem, answer: str) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="trajectory-001",
        task_id=item.task_id,
        termination=RolloutTermination.COMPLETED,
        evaluation_input=SubmittedTerminalValue({"answer": answer}),
        public_transcript_hash=stable_hash({"public": "transcript"}),
    )


def _no_submission_request(item: BenchmarkPublicItem) -> TerminalEvaluationRequest:
    return TerminalEvaluationRequest(
        trajectory_id="trajectory-no-submission",
        task_id=item.task_id,
        termination=RolloutTermination.HORIZON_EXHAUSTED,
        evaluation_input=NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED),
        public_transcript_hash=stable_hash({"public": "no submission"}),
    )


def test_no_submission_is_deterministic_zero_reward() -> None:
    item = _public_item()
    evaluator = PrivateStaticBenchmarkEvaluator(
        PrivateStaticBenchmarkCase(
            item,
            PrivateStaticTarget(item.task_id, StaticScoringRule.EXACT_TEXT, ("private",)),
        )
    )

    reward = asyncio.run(evaluator.evaluate(_no_submission_request(item)))

    assert reward.value == 0.0
    assert reward.success is False
    assert reward.native_payload["no_submission_reason"] == "horizon-exhausted"


def test_public_projection_and_provider_never_expose_private_target() -> None:
    item = _public_item()
    target = PrivateStaticTarget(
        item.task_id,
        StaticScoringRule.EXACT_TEXT,
        (PRIVATE_CANARY,),
    )
    case = PrivateStaticBenchmarkCase(item, target)
    provider = OrderedBenchmarkTaskProvider((item,))

    task = provider.next_task()

    assert task == case.public.to_rollout_task()
    assert PRIVATE_CANARY not in canonical_json(task.to_value())
    with pytest.raises(RuntimeError):
        provider.next_task()


def test_completion_environment_records_only_explicit_skill_invocation() -> None:
    environment = CompletionBenchmarkEnvironment(_public_item())
    skill_action = StructuredAction(
        kind=ActionKind.SKILL,
        name="apply-skill",
        arguments={},
        resource_id="skill-runtime",
        skill_id="skill-alpha",
    )

    observation = asyncio.run(environment.execute(skill_action, step_index=1))

    assert observation.invoked_skill_ids == ("skill-alpha",)
    assert observation.budget_usage.tool_calls == 1
    assert environment.validate_completion({"answer": "public prediction"})
    assert not environment.validate_completion({"answer": ""})
    assert not hasattr(environment, "checkpoint")
    assert not hasattr(environment, "restore")


def test_completion_environment_rejects_unimplemented_tool_semantics() -> None:
    environment = CompletionBenchmarkEnvironment(_public_item())
    tool_action = StructuredAction(
        kind=ActionKind.TOOL,
        name="search",
        arguments={"query": "public"},
        resource_id="search",
    )
    with pytest.raises(EnvironmentMethodFailedError) as captured:
        asyncio.run(environment.execute(tool_action, step_index=1))
    assert captured.value.budget_usage.tool_calls == 1

    skill_action = StructuredAction(
        kind=ActionKind.SKILL,
        name="apply-skill",
        arguments={},
        resource_id="skill-runtime",
        skill_id="skill-alpha",
    )
    # BoundedAgent may consume intervening parse/schema-invalid turns without
    # calling this environment. A later admitted action must still be executable.
    observation = asyncio.run(environment.execute(skill_action, step_index=3))
    assert observation.invoked_skill_ids == ("skill-alpha",)

    with pytest.raises(RuntimeError):
        asyncio.run(environment.execute(skill_action, step_index=2))


@pytest.mark.parametrize(
    ("rule", "accepted", "prediction", "expected"),
    [
        (StaticScoringRule.EXACT_TEXT, ("The Public Answer",), "public answer", 1.0),
        (StaticScoringRule.OPTION, ("B",), "b", 1.0),
        (StaticScoringRule.INTEGER, ("0042",), "42", 1.0),
        (StaticScoringRule.INTEGER, ("42",), "42.0", 0.0),
        (StaticScoringRule.TOKEN_F1, ("alpha beta gamma",), "alpha gamma", 0.8),
    ],
)
def test_private_static_scoring_rules(
    rule: StaticScoringRule,
    accepted: tuple[str, ...],
    prediction: str,
    expected: float,
) -> None:
    item = _public_item()
    evaluator = PrivateStaticBenchmarkEvaluator(
        PrivateStaticBenchmarkCase(
            item,
            PrivateStaticTarget(item.task_id, rule, accepted),
        )
    )
    reward = asyncio.run(evaluator.evaluate(_request(item, prediction)))
    assert reward.value == pytest.approx(expected)
    assert reward.success is (expected == 1.0)
    if rule is StaticScoringRule.TOKEN_F1:
        assert reward.native_payload["public_metrics"] == {"exact-match": 0.0}
    else:
        assert reward.native_payload["public_metrics"] == {}
    assert accepted[0] not in canonical_json(reward.native_payload)


@pytest.mark.parametrize(
    ("benchmark_id", "task_family", "rule", "expected_metric"),
    [
        ("medqa", "medqa/us-four-option", StaticScoringRule.OPTION, "accuracy"),
        ("aime-2026", "aime-2026/integer-answer", StaticScoringRule.INTEGER, "accuracy"),
        ("nq-open", "nq-open/open-domain", StaticScoringRule.EXACT_TEXT, "exact-match"),
    ],
)
def test_formal_static_benchmarks_emit_the_protocol_metric_name(
    benchmark_id: str,
    task_family: str,
    rule: StaticScoringRule,
    expected_metric: str,
) -> None:
    item = BenchmarkPublicItem(
        benchmark_id=benchmark_id,
        dataset_revision="fixture@1",
        split="dev",
        task_id=f"{benchmark_id}/fixture-001",
        task_family=task_family,
        query="Return A.",
        public_context={"answer_format": "short-text"},
    )
    evaluator = PrivateStaticBenchmarkEvaluator(
        PrivateStaticBenchmarkCase(
            item,
            PrivateStaticTarget(item.task_id, rule, ("A",)),
        )
    )

    reward = asyncio.run(evaluator.evaluate(_request(item, "A")))

    assert reward.native_metric_name == expected_metric


def test_private_session_factory_pairs_exact_public_projection() -> None:
    item = _public_item()
    case = PrivateStaticBenchmarkCase(
        item,
        PrivateStaticTarget(item.task_id, StaticScoringRule.EXACT_TEXT, (PRIVATE_CANARY,)),
    )
    factory = PrivateStaticBenchmarkSessionFactory((case,))

    bundle = factory.create(item.to_rollout_task())
    reward = asyncio.run(bundle.evaluator.evaluate(_request(item, PRIVATE_CANARY)))

    assert reward.value == 1.0
    assert bundle.environment.environment_id == item.environment_id
    assert PRIVATE_CANARY not in canonical_json(item.to_rollout_task().to_value())
