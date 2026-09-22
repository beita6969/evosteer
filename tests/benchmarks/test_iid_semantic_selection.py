from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from skillev_private.benchmarks.catalog import (
    PrivateBenchmarkCatalog,
    PrivateBenchmarkWorkload,
)
from skillev_private.benchmarks.semantic_selection import (
    IID_EPISODES_PER_BENCHMARK,
    SemanticCandidate,
    SemanticReviewDecision,
    SemanticReviewItem,
    build_iid_semantic_selection,
    build_semantic_review_packets,
    coverage_ranked_candidates,
    load_iid_semantic_selection,
    load_semantic_review_decisions,
    load_semantic_review_packets,
    materialize_iid_training_catalog,
    materialize_iid_training_subcatalog,
    publish_iid_semantic_selection,
    publish_semantic_review_decisions,
    publish_semantic_review_packets,
    semantic_candidate_for_task,
)
from skillev_private.benchmarks.terminal_admission import admit_terminal_evaluator_routes

from skillev.contracts import SuccessRule, TerminalReward, stable_hash
from skillev.experiments import BENCHMARK_SPECS, Benchmark, TrainingUse
from skillev.rollout import (
    NoSubmissionReason,
    NoTerminalSubmission,
    RolloutSessionBundle,
    RolloutTask,
    RolloutTermination,
    TerminalEvaluationRequest,
)


@dataclass(frozen=True, slots=True)
class _Evaluator:
    source_task_id: str

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.source_task_id:
            raise AssertionError("episode alias did not restore the source task identity")
        return TerminalReward(
            value=0.0,
            success=False,
            success_rule=SuccessRule.R_EQUALS_ONE,
            success_threshold=None,
            native_metric_name="semantic-selection-fixture",
            native_payload={},
            environment_id="semantic-selection-fixture",
            verifier_version="semantic-selection-fixture@1",
        )


@dataclass(frozen=True, slots=True)
class _Factory:
    tasks: tuple[RolloutTask, ...]

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        if task not in self.tasks:
            raise ValueError("source task is outside the fixture factory")
        return RolloutSessionBundle(
            environment=object(),  # type: ignore[arg-type]
            evaluator=_Evaluator(task.task_id),
            retrieved_skills=(),
        )


def _iid_benchmarks() -> tuple[Benchmark, ...]:
    return tuple(
        spec.benchmark for spec in BENCHMARK_SPECS if spec.training_use is TrainingUse.TRAINING_MIX
    )


def _candidate(benchmark: Benchmark, index: int) -> SemanticCandidate:
    task_id = f"{benchmark.value}/source/{index:04d}"
    task_family = f"{benchmark.value}/fixture"
    public_task = RolloutTask(
        task_id=task_id,
        environment_id=f"fixture:{benchmark.value}",
        task_family=task_family,
        context_id=f"fixture:{task_id}",
        query="Public synthetic semantic-review task.",
        available_tools=(),
        public_context={"benchmark_id": benchmark.value},
    )
    return SemanticCandidate(
        task_id=task_id,
        public_task_hash=stable_hash(public_task.to_value()),
        task_family=task_family,
        coverage_tags=(f"kind-{index % 3}",),
        length_bucket=("short", "medium", "long")[index % 3],
        difficulty_band=("easy", "medium", "hard")[index % 3],
    )


def _task(candidate: SemanticCandidate, benchmark: Benchmark) -> RolloutTask:
    return RolloutTask(
        task_id=candidate.task_id,
        environment_id=f"fixture:{benchmark.value}",
        task_family=candidate.task_family,
        context_id=f"fixture:{candidate.task_id}",
        query="Public synthetic semantic-review task.",
        available_tools=(),
        public_context={"benchmark_id": benchmark.value},
    )


def test_semantic_review_materializes_fixed_domain_blocks_and_unique_repeats(
    tmp_path: Path,
) -> None:
    candidates_by_benchmark: dict[Benchmark, tuple[SemanticCandidate, ...]] = {}
    reviews: list[SemanticReviewDecision] = []
    workloads: list[PrivateBenchmarkWorkload] = []
    for ordinal, benchmark in enumerate(_iid_benchmarks()):
        population = 7 if ordinal == 0 else IID_EPISODES_PER_BENCHMARK + 3
        candidates = tuple(_candidate(benchmark, index) for index in range(population))
        candidates_by_benchmark[benchmark] = candidates
        ranked = coverage_ranked_candidates(benchmark, candidates)
        selected_ids = {
            item.task_id
            for item in (ranked if population < IID_EPISODES_PER_BENCHMARK else ranked[:512])
        }
        reviews.extend(
            SemanticReviewDecision(
                task_id=item.task_id,
                public_task_hash=item.public_task_hash,
                accepted=item.task_id in selected_ids,
                reason_codes=(
                    "representative" if item.task_id in selected_ids else "coverage-redundant",
                ),
            )
            for item in ranked
        )
        source_tasks = tuple(_task(item, benchmark) for item in candidates)
        workloads.append(
            PrivateBenchmarkWorkload(
                benchmark,
                source_tasks,
                _Factory(source_tasks),
            )
        )

    selection = build_iid_semantic_selection(
        candidates_by_benchmark=candidates_by_benchmark,
        reviews=tuple(reviews),
    )
    assert len(selection.episodes) == 4608
    assert len({item.episode_id for item in selection.episodes}) == 4608
    first_block = selection.episodes[:512]
    assert len({item.source_task_id for item in first_block}) == 7
    assert max(item.repeat_ordinal for item in first_block) > 0

    source_catalog = PrivateBenchmarkCatalog(tuple(workloads))
    packet_path = tmp_path / "iid-semantic-review-packets.jsonl"
    packets = build_semantic_review_packets(source_catalog)
    publish_semantic_review_packets(packets, packet_path)
    assert load_semantic_review_packets(packet_path) == packets
    review_path = tmp_path / "iid-semantic-review-decisions.jsonl"
    publish_semantic_review_decisions(tuple(reviews), review_path)
    assert load_semantic_review_decisions(review_path) == tuple(reviews)

    selection_path = tmp_path / "iid-semantic-selection.json"
    publish_iid_semantic_selection(selection, selection_path)
    assert load_iid_semantic_selection(selection_path) == selection

    catalog = materialize_iid_training_catalog(source_catalog, selection)
    assert tuple(item.benchmark for item in catalog.workloads) == _iid_benchmarks()
    assert all(len(item.tasks) == 512 for item in catalog.workloads)
    aliased = catalog.workloads[0].tasks[-1]
    routed = catalog.workloads[0].session_factory.create(aliased)
    reward = asyncio.run(
        routed.evaluator.evaluate(
            TerminalEvaluationRequest(
                trajectory_id="semantic-selection-alias-test",
                task_id=aliased.task_id,
                termination=RolloutTermination.HORIZON_EXHAUSTED,
                evaluation_input=NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED),
                public_transcript_hash=stable_hash("semantic-selection-alias-test"),
            )
        )
    )
    assert reward.value == 0.0
    admission = admit_terminal_evaluator_routes(catalog, (aliased,))
    assert admission.task_count == 1

    frozen_ids = tuple(item.episode_id for item in selection.episodes[:3])
    subcatalog = materialize_iid_training_subcatalog(
        source_catalog,
        selection,
        ordered_episode_ids=frozen_ids,
    )
    assert tuple(item.benchmark for item in subcatalog.workloads) == (_iid_benchmarks()[0],)
    assert tuple(task.task_id for task in subcatalog.workloads[0].tasks) == frozen_ids
    with pytest.raises(ValueError):
        materialize_iid_training_subcatalog(
            source_catalog,
            selection,
            ordered_episode_ids=("not-a-reviewed-episode",),
        )


def test_semantic_review_jsonl_round_trip_preserves_unicode_line_separator(
    tmp_path: Path,
) -> None:
    benchmark = BENCHMARK_SPECS[0].benchmark
    task = _task(_candidate(benchmark, 0), benchmark)
    task = replace(task, query="Public task with a Unicode\u2028line separator.")
    candidate = semantic_candidate_for_task(task)
    packet = SemanticReviewItem(BENCHMARK_SPECS[0].benchmark, candidate, task)
    packet_path = tmp_path / "packets.jsonl"
    publish_semantic_review_packets((packet,), packet_path)

    decision = SemanticReviewDecision(
        task_id=task.task_id,
        public_task_hash=candidate.public_task_hash,
        accepted=True,
        reason_codes=("clear-public-task",),
    )
    decision_path = tmp_path / "decisions.jsonl"
    publish_semantic_review_decisions((decision,), decision_path)

    assert load_semantic_review_packets(packet_path) == (packet,)
    assert load_semantic_review_decisions(decision_path) == (decision,)


def test_small_population_requires_every_source_task_to_be_reviewed() -> None:
    candidates_by_benchmark: dict[Benchmark, tuple[SemanticCandidate, ...]] = {}
    reviews: list[SemanticReviewDecision] = []
    for ordinal, benchmark in enumerate(_iid_benchmarks()):
        population = 3 if ordinal == 0 else 512
        candidates = tuple(_candidate(benchmark, index) for index in range(population))
        candidates_by_benchmark[benchmark] = candidates
        ranked = coverage_ranked_candidates(benchmark, candidates)
        visited = ranked[:-1] if ordinal == 0 else ranked
        reviews.extend(
            SemanticReviewDecision(
                task_id=item.task_id,
                public_task_hash=item.public_task_hash,
                accepted=True,
                reason_codes=("representative",),
            )
            for item in visited
        )

    with pytest.raises(ValueError):
        build_iid_semantic_selection(
            candidates_by_benchmark=candidates_by_benchmark,
            reviews=tuple(reviews),
        )
