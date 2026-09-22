"""Public-only semantic review and fixed 512-episode IID materialization."""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path

from skillev.contracts import JsonValue, TerminalReward, canonical_json, stable_hash
from skillev.experiments import (
    ACTIVE_BENCHMARKS,
    BENCHMARK_SPECS,
    FIXED_SEED,
    Benchmark,
    TrainingUse,
)
from skillev.rollout import (
    RolloutSessionBundle,
    RolloutTask,
    TerminalEvaluationRequest,
    TerminalEvaluator,
    TerminalEvaluatorError,
)

from .catalog import PrivateBenchmarkCatalog, PrivateBenchmarkWorkload, PrivateSessionFactory

IID_EPISODES_PER_BENCHMARK = 512
SEMANTIC_SELECTION_ALGORITHM = "coverage-strata-then-codex-review@1"
SEMANTIC_SELECTION_FORMAT = "skillev-private-iid-semantic-selection@1"
SEMANTIC_REVIEW_PACKET_FORMAT = "skillev-private-iid-semantic-review-packet@1"


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _iid_benchmarks() -> tuple[Benchmark, ...]:
    return tuple(
        spec.benchmark for spec in BENCHMARK_SPECS if spec.training_use is TrainingUse.TRAINING_MIX
    )


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    task_id: str
    public_task_hash: str
    task_family: str
    coverage_tags: tuple[str, ...]
    length_bucket: str
    difficulty_band: str

    def __post_init__(self) -> None:
        for field in (
            "task_id",
            "public_task_hash",
            "task_family",
            "length_bucket",
            "difficulty_band",
        ):
            _text(getattr(self, field), field=field)
        if not self.coverage_tags or any(
            type(item) is not str or not item.strip() for item in self.coverage_tags
        ):
            raise ValueError("coverage_tags must be non-empty text")
        if tuple(sorted(set(self.coverage_tags))) != self.coverage_tags:
            raise ValueError("coverage_tags must be sorted and unique")

    @property
    def stratum(self) -> tuple[str, ...]:
        return (
            self.task_family,
            *self.coverage_tags,
            f"difficulty:{self.difficulty_band}",
            f"length:{self.length_bucket}",
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "coverage_tags": list(self.coverage_tags),
            "difficulty_band": self.difficulty_band,
            "length_bucket": self.length_bucket,
            "public_task_hash": self.public_task_hash,
            "task_family": self.task_family,
            "task_id": self.task_id,
        }

    @classmethod
    def from_value(cls, value: object) -> SemanticCandidate:
        if not isinstance(value, dict) or set(value) != {
            "coverage_tags",
            "difficulty_band",
            "length_bucket",
            "public_task_hash",
            "task_family",
            "task_id",
        }:
            raise ValueError("semantic candidate has incompatible fields")
        tags = value["coverage_tags"]
        if not isinstance(tags, list) or any(type(item) is not str for item in tags):
            raise TypeError("semantic candidate coverage_tags must be a text array")
        return cls(
            task_id=_text(value["task_id"], field="task_id"),
            public_task_hash=_text(value["public_task_hash"], field="public_task_hash"),
            task_family=_text(value["task_family"], field="task_family"),
            coverage_tags=tuple(tags),
            length_bucket=_text(value["length_bucket"], field="length_bucket"),
            difficulty_band=_text(value["difficulty_band"], field="difficulty_band"),
        )


@dataclass(frozen=True, slots=True)
class SemanticReviewItem:
    benchmark: Benchmark
    candidate: SemanticCandidate
    public_task: RolloutTask
    format: str = SEMANTIC_REVIEW_PACKET_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark) or not isinstance(
            self.candidate, SemanticCandidate
        ):
            raise TypeError("semantic review item requires closed benchmark metadata")
        if not isinstance(self.public_task, RolloutTask):
            raise TypeError("semantic review item requires a public rollout task")
        if self.candidate.task_id != self.public_task.task_id:
            raise ValueError("semantic candidate and public task identities differ")
        if self.candidate.public_task_hash != stable_hash(self.public_task.to_value()):
            raise ValueError("semantic candidate hash differs from public task bytes")
        context = self.public_task.public_context
        if not isinstance(context, dict) or context.get("benchmark_id") != self.benchmark.value:
            raise ValueError("semantic review item has an incompatible benchmark identity")
        if self.format != SEMANTIC_REVIEW_PACKET_FORMAT:
            raise ValueError("unsupported semantic review packet format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "candidate": self.candidate.to_value(),
            "format": self.format,
            "public_task": self.public_task.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> SemanticReviewItem:
        if not isinstance(value, dict) or set(value) != {
            "benchmark",
            "candidate",
            "format",
            "public_task",
        }:
            raise ValueError("semantic review item has incompatible fields")
        return cls(
            benchmark=Benchmark(_text(value["benchmark"], field="benchmark")),
            candidate=SemanticCandidate.from_value(value["candidate"]),
            public_task=RolloutTask.from_value(value["public_task"]),
            format=_text(value["format"], field="format"),
        )


@dataclass(frozen=True, slots=True)
class SemanticReviewDecision:
    task_id: str
    public_task_hash: str
    accepted: bool
    reason_codes: tuple[str, ...]
    reviewer: str = "codex-primary-agent"

    def __post_init__(self) -> None:
        _text(self.task_id, field="task_id")
        _text(self.public_task_hash, field="public_task_hash")
        _text(self.reviewer, field="reviewer")
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be a boolean")
        if not self.reason_codes or any(
            type(item) is not str or not item.strip() for item in self.reason_codes
        ):
            raise ValueError("reason_codes must be non-empty text")
        if tuple(sorted(set(self.reason_codes))) != self.reason_codes:
            raise ValueError("reason_codes must be sorted and unique")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "accepted": self.accepted,
            "public_task_hash": self.public_task_hash,
            "reason_codes": list(self.reason_codes),
            "reviewer": self.reviewer,
            "task_id": self.task_id,
        }

    @classmethod
    def from_value(cls, value: object) -> SemanticReviewDecision:
        if not isinstance(value, dict) or set(value) != {
            "accepted",
            "public_task_hash",
            "reason_codes",
            "reviewer",
            "task_id",
        }:
            raise ValueError("semantic review decision has incompatible fields")
        reasons = value["reason_codes"]
        if not isinstance(reasons, list) or any(type(item) is not str for item in reasons):
            raise TypeError("semantic review reason_codes must be a text array")
        return cls(
            task_id=_text(value["task_id"], field="task_id"),
            public_task_hash=_text(value["public_task_hash"], field="public_task_hash"),
            accepted=value["accepted"],
            reason_codes=tuple(reasons),
            reviewer=_text(value["reviewer"], field="reviewer"),
        )


@dataclass(frozen=True, slots=True)
class IIDTrainingEpisode:
    benchmark: Benchmark
    episode_id: str
    source_task_id: str
    repeat_ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark):
            raise TypeError("episode benchmark must be closed")
        _text(self.episode_id, field="episode_id")
        _text(self.source_task_id, field="source_task_id")
        if type(self.repeat_ordinal) is not int or self.repeat_ordinal < 0:
            raise ValueError("repeat_ordinal must be non-negative")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "episode_id": self.episode_id,
            "repeat_ordinal": self.repeat_ordinal,
            "source_task_id": self.source_task_id,
        }

    @classmethod
    def from_value(cls, value: object) -> IIDTrainingEpisode:
        if not isinstance(value, dict) or set(value) != {
            "benchmark",
            "episode_id",
            "repeat_ordinal",
            "source_task_id",
        }:
            raise ValueError("IID training episode has incompatible fields")
        return cls(
            benchmark=Benchmark(_text(value["benchmark"], field="benchmark")),
            episode_id=_text(value["episode_id"], field="episode_id"),
            source_task_id=_text(value["source_task_id"], field="source_task_id"),
            repeat_ordinal=value["repeat_ordinal"],
        )


@dataclass(frozen=True, slots=True)
class IIDSemanticSelection:
    candidates_hash: str
    reviews: tuple[SemanticReviewDecision, ...]
    episodes: tuple[IIDTrainingEpisode, ...]
    algorithm: str = SEMANTIC_SELECTION_ALGORITHM
    fixed_seed: int = FIXED_SEED
    format: str = SEMANTIC_SELECTION_FORMAT

    def __post_init__(self) -> None:
        _text(self.candidates_hash, field="candidates_hash")
        if self.algorithm != SEMANTIC_SELECTION_ALGORITHM or self.fixed_seed != FIXED_SEED:
            raise ValueError("IID selection algorithm or seed differs from protocol")
        if self.format != SEMANTIC_SELECTION_FORMAT:
            raise ValueError("unsupported IID semantic selection format")
        expected = tuple(b for b in _iid_benchmarks() for _ in range(IID_EPISODES_PER_BENCHMARK))
        if tuple(item.benchmark for item in self.episodes) != expected:
            raise ValueError("IID semantic selection is not in frozen domain-block order")
        if len({item.episode_id for item in self.episodes}) != len(self.episodes):
            raise ValueError("IID semantic selection repeats an episode identity")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "algorithm": self.algorithm,
            "candidates_hash": self.candidates_hash,
            "episodes": [item.to_value() for item in self.episodes],
            "fixed_seed": self.fixed_seed,
            "format": self.format,
            "reviews": [item.to_value() for item in self.reviews],
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    @classmethod
    def from_value(cls, value: object) -> IIDSemanticSelection:
        if not isinstance(value, dict) or set(value) != {
            "algorithm",
            "candidates_hash",
            "episodes",
            "fixed_seed",
            "format",
            "reviews",
        }:
            raise ValueError("IID semantic selection has incompatible fields")
        reviews, episodes = value["reviews"], value["episodes"]
        if not isinstance(reviews, list) or not isinstance(episodes, list):
            raise TypeError("IID semantic selection reviews and episodes must be arrays")
        return cls(
            candidates_hash=_text(value["candidates_hash"], field="candidates_hash"),
            reviews=tuple(SemanticReviewDecision.from_value(item) for item in reviews),
            episodes=tuple(IIDTrainingEpisode.from_value(item) for item in episodes),
            algorithm=_text(value["algorithm"], field="algorithm"),
            fixed_seed=value["fixed_seed"],
            format=_text(value["format"], field="format"),
        )


def coverage_ranked_candidates(
    benchmark: Benchmark, candidates: tuple[SemanticCandidate, ...]
) -> tuple[SemanticCandidate, ...]:
    if not candidates or len({item.task_id for item in candidates}) != len(candidates):
        raise ValueError("semantic candidates must be non-empty and unique")
    strata: dict[tuple[str, ...], list[SemanticCandidate]] = defaultdict(list)
    for candidate in candidates:
        strata[candidate.stratum].append(candidate)
    queues = {
        key: deque(
            sorted(
                values,
                key=lambda item: (
                    stable_hash(
                        {
                            "algorithm": SEMANTIC_SELECTION_ALGORITHM,
                            "benchmark": benchmark.value,
                            "seed": FIXED_SEED,
                            "task_id": item.task_id,
                        }
                    ),
                    item.task_id,
                ),
            )
        )
        for key, values in strata.items()
    }
    ordered: list[SemanticCandidate] = []
    while queues:
        for key in sorted(queues):
            ordered.append(queues[key].popleft())
            if not queues[key]:
                del queues[key]
    return tuple(ordered)


def _public_difficulty(task: RolloutTask) -> str:
    if isinstance(task.public_context, dict):
        for field in ("difficulty", "level", "qsubtype", "task_type"):
            value = task.public_context.get(field)
            if type(value) is str and value.strip():
                return f"{field}:{value.strip().lower()}"
    return "unspecified"


def semantic_candidate_for_task(task: RolloutTask) -> SemanticCandidate:
    context = task.public_context
    if not isinstance(context, dict) or type(context.get("benchmark_id")) is not str:
        raise ValueError("semantic candidate is missing benchmark_id")
    tags = {
        f"context:{task.context_id}",
        f"tools:{','.join(task.available_tools) if task.available_tools else 'none'}",
    }
    for field in ("domain", "subdomain", "type", "qtype", "category"):
        value = context.get(field)
        if type(value) is str and value.strip():
            tags.add(f"{field}:{value.strip().lower()}")
    size = len(canonical_json(task.to_value()).encode())
    length_bucket = "short" if size <= 2_000 else "medium" if size <= 8_000 else "long"
    return SemanticCandidate(
        task.task_id,
        stable_hash(task.to_value()),
        task.task_family,
        tuple(sorted(tags)),
        length_bucket,
        _public_difficulty(task),
    )


def build_semantic_review_packets(
    catalog: PrivateBenchmarkCatalog,
) -> tuple[SemanticReviewItem, ...]:
    packets: list[SemanticReviewItem] = []
    for benchmark in _iid_benchmarks():
        workload = catalog.workload(benchmark)
        by_id = {task.task_id: task for task in workload.tasks}
        candidates = tuple(semantic_candidate_for_task(task) for task in workload.tasks)
        packets.extend(
            SemanticReviewItem(benchmark, candidate, by_id[candidate.task_id])
            for candidate in coverage_ranked_candidates(benchmark, candidates)
        )
    return tuple(packets)


def _episode_id(benchmark: Benchmark, source_task_id: str, repeat_ordinal: int) -> str:
    digest = stable_hash(
        {
            "benchmark": benchmark.value,
            "repeat_ordinal": repeat_ordinal,
            "source_task_id": source_task_id,
        }
    ).removeprefix("sha256:")
    return f"{benchmark.value}/episode/{digest}"


def build_iid_semantic_selection(
    *,
    candidates_by_benchmark: dict[Benchmark, tuple[SemanticCandidate, ...]],
    reviews: tuple[SemanticReviewDecision, ...],
) -> IIDSemanticSelection:
    review_by_id = {item.task_id: item for item in reviews}
    if len(review_by_id) != len(reviews) or set(candidates_by_benchmark) != set(_iid_benchmarks()):
        raise ValueError("semantic review inputs differ from IID protocol")
    candidate_values: list[dict[str, JsonValue]] = []
    episodes: list[IIDTrainingEpisode] = []
    visited: set[str] = set()
    for benchmark in _iid_benchmarks():
        ranked = coverage_ranked_candidates(benchmark, candidates_by_benchmark[benchmark])
        candidate_values.extend(item.to_value() for item in ranked)
        accepted: list[str] = []
        for candidate in ranked:
            review = review_by_id.get(candidate.task_id)
            if review is None:
                raise ValueError("every IID source task requires an individual semantic review")
            visited.add(review.task_id)
            if review.public_task_hash != candidate.public_task_hash:
                raise ValueError("semantic review belongs to different public task bytes")
            if review.accepted:
                accepted.append(candidate.task_id)
        if not accepted:
            raise ValueError("semantic review rejected an entire IID benchmark")
        if len(ranked) >= IID_EPISODES_PER_BENCHMARK and len(accepted) != 512:
            raise ValueError("large IID population must have exactly 512 accepted reviews")
        source_ids = accepted[:IID_EPISODES_PER_BENCHMARK]
        for index in range(len(source_ids), IID_EPISODES_PER_BENCHMARK):
            source_ids.append(accepted[index % len(accepted)])
        occurrence: dict[str, int] = defaultdict(int)
        for source_task_id in source_ids:
            ordinal = occurrence[source_task_id]
            occurrence[source_task_id] += 1
            episodes.append(
                IIDTrainingEpisode(
                    benchmark,
                    _episode_id(benchmark, source_task_id, ordinal),
                    source_task_id,
                    ordinal,
                )
            )
    if visited != set(review_by_id):
        raise ValueError("semantic review ledger contains decisions outside the source populations")
    return IIDSemanticSelection(stable_hash(candidate_values), reviews, tuple(episodes))


@dataclass(frozen=True, slots=True)
class _EpisodeAliasEvaluator:
    source_evaluator: TerminalEvaluator
    episode_task_id: str
    source_task_id: str

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.episode_task_id:
            raise TerminalEvaluatorError("terminal request reached another training episode")
        return await self.source_evaluator.evaluate(replace(request, task_id=self.source_task_id))


@dataclass(frozen=True, slots=True)
class _EpisodeAliasFactory:
    source_factory: PrivateSessionFactory
    routes: tuple[tuple[str, RolloutTask], ...]

    def _source_task(self, task: RolloutTask) -> RolloutTask:
        matches = tuple(source for episode_id, source in self.routes if episode_id == task.task_id)
        if len(matches) != 1:
            raise ValueError("training episode has no unique source route")
        return matches[0]

    def terminal_admission_route(
        self, task: RolloutTask
    ) -> tuple[PrivateSessionFactory, RolloutTask]:
        return self.source_factory, self._source_task(task)

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        source_task = self._source_task(task)
        source_bundle = self.source_factory.create(source_task)
        if not isinstance(source_bundle, RolloutSessionBundle):
            raise TypeError("episode source factory returned an incompatible session bundle")
        return replace(
            source_bundle,
            evaluator=_EpisodeAliasEvaluator(
                source_evaluator=source_bundle.evaluator,
                episode_task_id=task.task_id,
                source_task_id=source_task.task_id,
            ),
        )


def materialize_iid_training_catalog(
    source_catalog: PrivateBenchmarkCatalog, selection: IIDSemanticSelection
) -> PrivateBenchmarkCatalog:
    workloads: list[PrivateBenchmarkWorkload] = []
    for benchmark in _iid_benchmarks():
        source = source_catalog.workload(benchmark)
        by_id = {task.task_id: task for task in source.tasks}
        selected = tuple(item for item in selection.episodes if item.benchmark is benchmark)
        routes: list[tuple[str, RolloutTask]] = []
        tasks: list[RolloutTask] = []
        for episode in selected:
            try:
                source_task = by_id[episode.source_task_id]
            except KeyError as error:
                raise ValueError("selected IID source task is absent from catalog") from error
            routes.append((episode.episode_id, source_task))
            tasks.append(replace(source_task, task_id=episode.episode_id))
        workloads.append(
            PrivateBenchmarkWorkload(
                benchmark, tuple(tasks), _EpisodeAliasFactory(source.session_factory, tuple(routes))
            )
        )
    return PrivateBenchmarkCatalog(tuple(workloads))


def materialize_iid_training_subcatalog(
    source_catalog: PrivateBenchmarkCatalog,
    selection: IIDSemanticSelection,
    *,
    ordered_episode_ids: tuple[str, ...],
) -> PrivateBenchmarkCatalog:
    """Materialize exactly the benchmark routes used by one frozen B2 schedule.

    The episode order remains owned by ``PrivateFrozenTaskSequence``.  This
    helper only avoids opening unrelated process deployments before training;
    every requested ID must still be an exact member of the reviewed IID
    selection and every source route must exist in ``source_catalog``.
    """

    if not ordered_episode_ids or len(set(ordered_episode_ids)) != len(ordered_episode_ids):
        raise ValueError("training subcatalog requires unique frozen episode IDs")
    episode_by_id = {episode.episode_id: episode for episode in selection.episodes}
    if not set(ordered_episode_ids) <= set(episode_by_id):
        raise ValueError("training schedule contains an episode outside the IID selection")
    required = {episode_by_id[episode_id].benchmark for episode_id in ordered_episode_ids}
    workloads: list[PrivateBenchmarkWorkload] = []
    for benchmark in ACTIVE_BENCHMARKS:
        if benchmark not in required:
            continue
        source = source_catalog.workload(benchmark)
        by_id = {task.task_id: task for task in source.tasks}
        selected = tuple(
            episode_by_id[episode_id]
            for episode_id in ordered_episode_ids
            if episode_by_id[episode_id].benchmark is benchmark
        )
        routes: list[tuple[str, RolloutTask]] = []
        tasks: list[RolloutTask] = []
        for episode in selected:
            try:
                source_task = by_id[episode.source_task_id]
            except KeyError as error:
                raise ValueError("selected IID source task is absent from catalog") from error
            routes.append((episode.episode_id, source_task))
            tasks.append(replace(source_task, task_id=episode.episode_id))
        workloads.append(
            PrivateBenchmarkWorkload(
                benchmark,
                tuple(tasks),
                _EpisodeAliasFactory(source.session_factory, tuple(routes)),
            )
        )
    return PrivateBenchmarkCatalog(tuple(workloads))


def _publish_jsonl_once(values: tuple[dict[str, JsonValue], ...], path: Path) -> None:
    payload = "".join(canonical_json(item) + "\n" for item in values).encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def publish_semantic_review_packets(packets: tuple[SemanticReviewItem, ...], path: Path) -> None:
    if not packets:
        raise ValueError("semantic review packet queue cannot be empty")
    _publish_jsonl_once(tuple(item.to_value() for item in packets), path)


def load_semantic_review_packets(path: Path) -> tuple[SemanticReviewItem, ...]:
    with path.open(encoding="utf-8", newline="\n") as stream:
        packets = tuple(
            SemanticReviewItem.from_value(json.loads(line)) for line in stream if line.strip()
        )
    if not packets or len({item.candidate.task_id for item in packets}) != len(packets):
        raise ValueError("semantic review packet queue must be non-empty and unique")
    return packets


def publish_semantic_review_decisions(
    reviews: tuple[SemanticReviewDecision, ...], path: Path
) -> None:
    if not reviews or len({item.task_id for item in reviews}) != len(reviews):
        raise ValueError("semantic review decisions must be non-empty and unique")
    _publish_jsonl_once(tuple(item.to_value() for item in reviews), path)


def load_semantic_review_decisions(path: Path) -> tuple[SemanticReviewDecision, ...]:
    with path.open(encoding="utf-8", newline="\n") as stream:
        reviews = tuple(
            SemanticReviewDecision.from_value(json.loads(line)) for line in stream if line.strip()
        )
    if not reviews or len({item.task_id for item in reviews}) != len(reviews):
        raise ValueError("semantic review decisions must be non-empty and unique")
    return reviews


def publish_iid_semantic_selection(selection: IIDSemanticSelection, path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(selection.to_value()) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_iid_semantic_selection(path: Path) -> IIDSemanticSelection:
    return IIDSemanticSelection.from_value(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "IID_EPISODES_PER_BENCHMARK",
    "SEMANTIC_REVIEW_PACKET_FORMAT",
    "SEMANTIC_SELECTION_ALGORITHM",
    "SEMANTIC_SELECTION_FORMAT",
    "IIDSemanticSelection",
    "IIDTrainingEpisode",
    "SemanticCandidate",
    "SemanticReviewDecision",
    "SemanticReviewItem",
    "build_iid_semantic_selection",
    "build_semantic_review_packets",
    "coverage_ranked_candidates",
    "load_iid_semantic_selection",
    "load_semantic_review_decisions",
    "load_semantic_review_packets",
    "materialize_iid_training_catalog",
    "materialize_iid_training_subcatalog",
    "publish_iid_semantic_selection",
    "publish_semantic_review_decisions",
    "publish_semantic_review_packets",
    "semantic_candidate_for_task",
]
