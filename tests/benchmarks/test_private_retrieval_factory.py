from __future__ import annotations

import asyncio

from skillev_private.benchmarks import (
    PrivateRetrievalBenchmarkSessionFactory,
    PrivateStaticBenchmarkCase,
    PrivateStaticTarget,
    StaticScoringRule,
)

from skillev.benchmarks import (
    BenchmarkPublicItem,
    DocumentPassage,
    OrderedRetrievalTaskProvider,
    QABenchmark,
    RetrievalIndex,
    build_retrieval_index,
)
from skillev.contracts import canonical_json, stable_hash
from skillev.rollout import (
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
)
from skillev.runtime import ActionKind, StructuredAction


def test_private_target_stays_outside_retrieval_task_and_observations(tmp_path) -> None:
    private_answer = "PRIVATE-RETRIEVAL-ANSWER-CANARY"
    path = tmp_path / "public.sqlite"
    manifest = build_retrieval_index(
        path,
        (
            DocumentPassage(
                passage_id="public-passage",
                document_id="public-document",
                title="Public title",
                text="Public passage text used only for retrieval.",
            ),
        ),
        corpus_name="fixture-public",
        corpus_version="fixture@1",
    )
    item = BenchmarkPublicItem(
        benchmark_id=QABenchmark.HOTPOT_QA.value,
        dataset_revision="fixture@1",
        split="dev",
        task_id="hotpot-fixture-001",
        task_family="hotpotqa/bridge",
        query="Which public passage should be read?",
        public_context={"answer_format": "short-text"},
    )
    case = PrivateStaticBenchmarkCase(
        item,
        PrivateStaticTarget(item.task_id, StaticScoringRule.TOKEN_F1, (private_answer,)),
    )

    with RetrievalIndex.open(path) as index:
        provider = OrderedRetrievalTaskProvider((item,), manifest)
        task = provider.next_task()
        bundle = PrivateRetrievalBenchmarkSessionFactory(
            (case,),
            index,
            QABenchmark.HOTPOT_QA,
        ).create(task)
        observation = asyncio.run(
            bundle.environment.execute(
                StructuredAction(
                    kind=ActionKind.TOOL,
                    name="search",
                    arguments={"limit": 1, "query": "public passage"},
                    resource_id="qa-retrieval",
                ),
                step_index=1,
            )
        )
        reward = asyncio.run(
            bundle.evaluator.evaluate(
                TerminalEvaluationRequest(
                    trajectory_id="trajectory-fixture",
                    task_id=item.task_id,
                    termination=RolloutTermination.COMPLETED,
                    evaluation_input=SubmittedTerminalValue({"answer": private_answer}),
                    public_transcript_hash=stable_hash({"public": "transcript"}),
                )
            )
        )

    public_wire = canonical_json({"observation": observation.to_value(), "task": task.to_value()})
    assert private_answer not in public_wire
    assert reward.value == 1.0
    assert reward.environment_id == task.environment_id
