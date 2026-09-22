"""Private, answer-isolated Protocol 13 formal-training dataset.

The canonical artifact is one mixed JSONL stream with a uniform ``input`` / ``output``
envelope.  ``input`` is the complete model-visible :class:`~skillev.rollout.RolloutTask`;
``output`` is a trusted-evaluator payload and must never be passed to the policy.  A second
JSONL file containing only the inputs is emitted so runtime code has a safe-by-construction
loading path.

This legacy acquisition format contains eight domains and 250 questions per domain.
New training uses protocol_v13_seven_training and never executes WebShop.
The historical file remains readable without rewriting its labels or source IDs.  Every
optimizer step receives one question from every domain; runtime expands each question
into four independently seeded trajectories, for an effective batch of 32.  Source tasks
are sampled in deterministic shuffled cycles.  A source population smaller than 250 is
therefore repeated, but every occurrence receives a distinct episode identity.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from skillev.contracts import JsonValue, normalize_json
from skillev.evaluation.current_iid.protocol13.catalog import (
    ACTIVE_PROTOCOL13_BENCHMARKS,
    CHECKPOINT_COUNT,
    CHECKPOINT_EVERY_STEPS,
    EFFECTIVE_BATCH_SIZE,
    QUESTIONS_PER_DOMAIN,
    QUESTIONS_PER_STEP,
    TRAINING_QUESTION_COUNT,
    TRAINING_STEPS,
    TRAINING_TRAJECTORY_COUNT,
    TRAJECTORIES_PER_QUESTION,
    Protocol13Benchmark,
)
from skillev.rollout import RolloutTask

TRAINING_RECORD_FORMAT = "skillev-private-protocol13-training-record@2"
TRAINING_SUMMARY_FORMAT = "skillev-private-protocol13-training-summary@2"
TRAINING_SELECTION_ALGORITHM = "dataset-balanced-step-major-shuffled-cycles-seed-0@2"
TRAINING_DATASET_FILE = "training.jsonl"
MODEL_INPUT_FILE = "model-inputs.jsonl"
SUMMARY_FILE = "summary.json"

_EVALUATORS: Mapping[Protocol13Benchmark, str] = {
    Protocol13Benchmark.HOTPOT_QA: "hotpotqa-official-em-f1",
    Protocol13Benchmark.TRIVIA_QA: "triviaqa-official-alias-em-f1",
    Protocol13Benchmark.AIME_2026: "integer-exact",
    Protocol13Benchmark.HEALTHBENCH: "simple-evals-rubric",
    Protocol13Benchmark.WEB_SHOP: "webshop-native-reward",
    Protocol13Benchmark.ALF_WORLD: "alfworld-success",
    Protocol13Benchmark.MBPP_PLUS: "evalplus-base-plus",
    Protocol13Benchmark.HUMAN_EVAL: "humaneval-native",
}


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text without NUL")
    return value


def _integer(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _object(value: object, *, field: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise TypeError(f"{field} must be a JSON object")
    return normalized


@dataclass(frozen=True, slots=True)
class Protocol13TrainingSourceCase:
    """One unique private source task before the 250-question lane is built."""

    benchmark: Protocol13Benchmark
    population_id: str
    source_version: str
    source_id: str
    task: RolloutTask
    evaluator_kind: str
    evaluator_target: dict[str, JsonValue]

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Protocol13Benchmark):
            raise TypeError("training source benchmark must belong to Protocol 13")
        for field in ("population_id", "source_version", "source_id", "evaluator_kind"):
            _text(getattr(self, field), field=field)
        if self.evaluator_kind != _EVALUATORS[self.benchmark]:
            raise ValueError("training source evaluator differs from its benchmark")
        if not isinstance(self.task, RolloutTask):
            raise TypeError("training source task must be a RolloutTask")
        if self.task.task_id != self.source_id:
            raise ValueError("training source task ID differs from its private source ID")
        if not self.task.task_family.startswith(f"{self.benchmark.value}/"):
            raise ValueError("training source task family differs from its benchmark")
        context = self.task.public_context
        if not isinstance(context, dict):
            raise ValueError("training source public context must be an object")
        if (
            context.get("benchmark_id") != self.benchmark.value
            or context.get("dataset_revision") != self.source_version
            or context.get("split") != "training"
        ):
            raise ValueError("training source public identity is inconsistent")
        object.__setattr__(
            self,
            "evaluator_target",
            _object(self.evaluator_target, field="evaluator target"),
        )


@dataclass(frozen=True, slots=True)
class Protocol13TrainingEpisode:
    benchmark: Protocol13Benchmark
    population_id: str
    episode_id: str
    source_id: str
    repeat_ordinal: int
    block_position: int
    optimizer_step: int
    global_position: int

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Protocol13Benchmark):
            raise TypeError("training episode benchmark must belong to Protocol 13")
        for field in ("population_id", "episode_id", "source_id"):
            _text(getattr(self, field), field=field)
        for field in (
            "repeat_ordinal",
            "block_position",
            "optimizer_step",
            "global_position",
        ):
            _integer(getattr(self, field), field=field)
        if self.block_position >= QUESTIONS_PER_DOMAIN:
            raise ValueError("training episode lies outside its domain block")
        if not 1 <= self.optimizer_step <= TRAINING_STEPS:
            raise ValueError("training episode has an invalid optimizer step")
        if self.block_position != self.optimizer_step - 1:
            raise ValueError("training episode domain position differs from its optimizer step")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "block_position": self.block_position,
            "episode_id": self.episode_id,
            "global_position": self.global_position,
            "optimizer_step": self.optimizer_step,
            "population_id": self.population_id,
            "repeat_ordinal": self.repeat_ordinal,
            "source_id": self.source_id,
        }

    @classmethod
    def from_value(cls, value: object) -> Protocol13TrainingEpisode:
        data = _object(value, field="training episode")
        fields = {
            "benchmark",
            "block_position",
            "episode_id",
            "global_position",
            "optimizer_step",
            "population_id",
            "repeat_ordinal",
            "source_id",
        }
        if set(data) != fields:
            raise ValueError("training episode has incompatible fields")
        return cls(
            benchmark=Protocol13Benchmark(_text(data["benchmark"], field="benchmark")),
            population_id=_text(data["population_id"], field="population_id"),
            episode_id=_text(data["episode_id"], field="episode_id"),
            source_id=_text(data["source_id"], field="source_id"),
            repeat_ordinal=_integer(data["repeat_ordinal"], field="repeat_ordinal"),
            block_position=_integer(data["block_position"], field="block_position"),
            optimizer_step=_integer(data["optimizer_step"], field="optimizer_step"),
            global_position=_integer(data["global_position"], field="global_position"),
        )


@dataclass(frozen=True, slots=True)
class Protocol13TrainingOutput:
    """Verifier-only half of a unified record."""

    evaluator_kind: str
    target: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _text(self.evaluator_kind, field="evaluator kind")
        object.__setattr__(self, "target", _object(self.target, field="evaluator target"))

    def to_value(self) -> dict[str, JsonValue]:
        return {"evaluator_kind": self.evaluator_kind, "target": self.target}

    @classmethod
    def from_value(cls, value: object) -> Protocol13TrainingOutput:
        data = _object(value, field="training output")
        if set(data) != {"evaluator_kind", "target"}:
            raise ValueError("training output has incompatible fields")
        return cls(
            evaluator_kind=_text(data["evaluator_kind"], field="evaluator kind"),
            target=_object(data["target"], field="evaluator target"),
        )


@dataclass(frozen=True, slots=True)
class Protocol13TrainingRecord:
    episode: Protocol13TrainingEpisode
    input: RolloutTask
    output: Protocol13TrainingOutput
    format: str = TRAINING_RECORD_FORMAT

    def __post_init__(self) -> None:
        if self.format != TRAINING_RECORD_FORMAT:
            raise ValueError("unsupported Protocol 13 training record format")
        if not isinstance(self.episode, Protocol13TrainingEpisode):
            raise TypeError("training record requires an episode")
        if not isinstance(self.input, RolloutTask):
            raise TypeError("training record input must be a RolloutTask")
        if not isinstance(self.output, Protocol13TrainingOutput):
            raise TypeError("training record output must be verifier-only output")
        if self.input.task_id != self.episode.episode_id:
            raise ValueError("training record input differs from its episode ID")
        if self.output.evaluator_kind != _EVALUATORS[self.episode.benchmark]:
            raise ValueError("training record evaluator differs from its benchmark")
        context = self.input.public_context
        if not isinstance(context, dict) or context.get("benchmark_id") != (
            self.episode.benchmark.value
        ):
            raise ValueError("training record input belongs to another benchmark")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "episode": self.episode.to_value(),
            "format": self.format,
            "input": self.input.to_value(),
            "output": self.output.to_value(),
        }

    def model_input_value(self) -> dict[str, JsonValue]:
        """Return the only record projection that policy code may consume."""

        return {
            "episode_id": self.episode.episode_id,
            "format": "skillev-protocol13-model-input@2",
            "input": self.input.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> Protocol13TrainingRecord:
        data = _object(value, field="training record")
        if set(data) != {"episode", "format", "input", "output"}:
            raise ValueError("training record has incompatible fields")
        return cls(
            episode=Protocol13TrainingEpisode.from_value(data["episode"]),
            input=RolloutTask.from_value(data["input"]),
            output=Protocol13TrainingOutput.from_value(data["output"]),
            format=_text(data["format"], field="training record format"),
        )


def _cycle_order(
    cases: Sequence[Protocol13TrainingSourceCase],
    *,
    benchmark_index: int,
    cycle: int,
) -> list[Protocol13TrainingSourceCase]:
    ordered = sorted(cases, key=lambda case: case.source_id)
    seed = benchmark_index * 1_000_003 + cycle * 10_007
    random.Random(seed).shuffle(ordered)  # noqa: S311 - frozen scientific schedule
    return ordered


def build_protocol13_training_records(
    sources: Mapping[Protocol13Benchmark, Sequence[Protocol13TrainingSourceCase]],
) -> tuple[Protocol13TrainingRecord, ...]:
    """Build the exact 2,000-question, step-major balanced training stream."""

    if set(sources) != set(ACTIVE_PROTOCOL13_BENCHMARKS):
        raise ValueError("training sources must cover the exact Protocol 13 IID catalog")
    selected_by_benchmark: dict[
        Protocol13Benchmark, tuple[tuple[Protocol13TrainingSourceCase, int], ...]
    ] = {}
    for benchmark_index, benchmark in enumerate(ACTIVE_PROTOCOL13_BENCHMARKS):
        cases = tuple(sources[benchmark])
        if not cases or any(case.benchmark is not benchmark for case in cases):
            raise ValueError("training source domain is empty or mislabeled")
        ids = tuple(case.source_id for case in cases)
        if len(ids) != len(set(ids)):
            raise ValueError("training source population repeats a source ID")
        population_ids = {case.population_id for case in cases}
        versions = {case.source_version for case in cases}
        if len(population_ids) != 1 or len(versions) != 1:
            raise ValueError("training source domain has multiple population identities")
        selected: list[Protocol13TrainingSourceCase] = []
        cycle = 0
        while len(selected) < QUESTIONS_PER_DOMAIN:
            selected.extend(_cycle_order(cases, benchmark_index=benchmark_index, cycle=cycle))
            cycle += 1
        occurrences: Counter[str] = Counter()
        lane: list[tuple[Protocol13TrainingSourceCase, int]] = []
        for case in selected[:QUESTIONS_PER_DOMAIN]:
            repeat_ordinal = occurrences[case.source_id]
            occurrences[case.source_id] += 1
            lane.append((case, repeat_ordinal))
        selected_by_benchmark[benchmark] = tuple(lane)

    provisional: list[tuple[Protocol13TrainingEpisode, Protocol13TrainingSourceCase]] = []
    for block_position in range(QUESTIONS_PER_DOMAIN):
        step_benchmarks = list(ACTIVE_PROTOCOL13_BENCHMARKS)
        random.Random(block_position).shuffle(  # noqa: S311 - frozen scientific schedule
            step_benchmarks
        )
        for benchmark in step_benchmarks:
            case, repeat_ordinal = selected_by_benchmark[benchmark][block_position]
            episode = Protocol13TrainingEpisode(
                benchmark=benchmark,
                population_id=case.population_id,
                episode_id=f"protocol13/{benchmark.value}/training/{block_position:04d}",
                source_id=case.source_id,
                repeat_ordinal=repeat_ordinal,
                block_position=block_position,
                optimizer_step=block_position + 1,
                global_position=len(provisional),
            )
            provisional.append((episode, case))
    records = tuple(
        Protocol13TrainingRecord(
            episode=episode,
            input=replace(case.task, task_id=episode.episode_id),
            output=Protocol13TrainingOutput(case.evaluator_kind, case.evaluator_target),
        )
        for episode, case in provisional
    )
    validate_protocol13_training_records(records)
    return records


def expand_protocol13_training_trajectories(
    records: Sequence[Protocol13TrainingRecord],
) -> tuple[Protocol13TrainingRecord, ...]:
    """Expand complete balanced question steps into four rollout bindings each.

    The persisted dataset remains one record per question.  This runtime-only
    projection gives every rollout a unique task identity so generation seeds,
    environments, evaluators, and trajectory receipts cannot alias each other.
    Each effective batch is ordered as four passes over the same eight-question
    step, preserving one question per IID domain and four trajectories per
    question.
    """

    if not records or len(records) % QUESTIONS_PER_STEP:
        raise ValueError("trajectory expansion requires complete eight-question steps")
    expanded: list[Protocol13TrainingRecord] = []
    for offset in range(0, len(records), QUESTIONS_PER_STEP):
        step_records = tuple(records[offset : offset + QUESTIONS_PER_STEP])
        steps = {record.episode.optimizer_step for record in step_records}
        benchmarks = {record.episode.benchmark for record in step_records}
        if len(steps) != 1 or benchmarks != set(ACTIVE_PROTOCOL13_BENCHMARKS):
            raise ValueError("trajectory expansion received an unbalanced question step")
        for trajectory_ordinal in range(TRAJECTORIES_PER_QUESTION):
            for record in step_records:
                task_id = f"{record.episode.episode_id}/rollout-{trajectory_ordinal:02d}"
                expanded.append(
                    replace(
                        record,
                        episode=replace(record.episode, episode_id=task_id),
                        input=replace(record.input, task_id=task_id),
                    )
                )
    expected = len(records) * TRAJECTORIES_PER_QUESTION
    if len(expanded) != expected or len({record.input.task_id for record in expanded}) != expected:
        raise AssertionError("trajectory expansion did not produce unique rollout bindings")
    return tuple(expanded)


def validate_protocol13_training_records(
    records: Sequence[Protocol13TrainingRecord],
) -> None:
    if len(records) != TRAINING_QUESTION_COUNT:
        raise ValueError("Protocol 13 training dataset must contain 2,000 questions")
    if tuple(record.episode.global_position for record in records) != tuple(
        range(TRAINING_QUESTION_COUNT)
    ):
        raise ValueError("training global positions must be contiguous")
    episode_ids = tuple(record.episode.episode_id for record in records)
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("training episode IDs must be unique")
    for benchmark in ACTIVE_PROTOCOL13_BENCHMARKS:
        domain = tuple(record for record in records if record.episode.benchmark is benchmark)
        if len(domain) != QUESTIONS_PER_DOMAIN:
            raise ValueError("every Protocol 13 benchmark must contribute 250 questions")
        positions = tuple(sorted(record.episode.block_position for record in domain))
        if positions != tuple(range(QUESTIONS_PER_DOMAIN)):
            raise ValueError("training domain positions must be contiguous")
        by_source: dict[str, list[int]] = {}
        for record in domain:
            by_source.setdefault(record.episode.source_id, []).append(record.episode.repeat_ordinal)
        if any(sorted(values) != list(range(len(values))) for values in by_source.values()):
            raise ValueError("training repeat ordinals are not contiguous")
    for step_index in range(TRAINING_STEPS):
        start = step_index * QUESTIONS_PER_STEP
        step = tuple(records[start : start + QUESTIONS_PER_STEP])
        if (
            len(step) != QUESTIONS_PER_STEP
            or {record.episode.benchmark for record in step} != set(ACTIVE_PROTOCOL13_BENCHMARKS)
            or {record.episode.optimizer_step for record in step} != {step_index + 1}
        ):
            raise ValueError("every training step must contain one question from every domain")


def training_summary(
    records: Sequence[Protocol13TrainingRecord],
) -> dict[str, JsonValue]:
    validate_protocol13_training_records(records)
    domains: list[JsonValue] = []
    for benchmark in ACTIVE_PROTOCOL13_BENCHMARKS:
        selected = tuple(record for record in records if record.episode.benchmark is benchmark)
        source_counts = Counter(record.episode.source_id for record in selected)
        contexts = {
            cast(dict[str, JsonValue], record.input.public_context)["dataset_revision"]
            for record in selected
        }
        population_ids = {record.episode.population_id for record in selected}
        if len(contexts) != 1 or len(population_ids) != 1:
            raise ValueError("training domain identities differ while summarizing")
        domains.append(
            {
                "benchmark": benchmark.value,
                "evaluator_kind": _EVALUATORS[benchmark],
                "max_source_occurrences": max(source_counts.values()),
                "population_id": next(iter(population_ids)),
                "question_count": len(selected),
                "repeated_question_count": sum(count - 1 for count in source_counts.values()),
                "source_version": cast(str, next(iter(contexts))),
                "unique_source_count": len(source_counts),
            }
        )
    return {
        "checkpoint_count": CHECKPOINT_COUNT,
        "checkpoint_every_steps": CHECKPOINT_EVERY_STEPS,
        "domain_count": len(ACTIVE_PROTOCOL13_BENCHMARKS),
        "domains": domains,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "format": TRAINING_SUMMARY_FORMAT,
        "model_input_file": MODEL_INPUT_FILE,
        "question_count": len(records),
        "questions_per_domain": QUESTIONS_PER_DOMAIN,
        "questions_per_step": QUESTIONS_PER_STEP,
        "selection_seed": 0,
        "selection_algorithm": TRAINING_SELECTION_ALGORITHM,
        "training_steps": TRAINING_STEPS,
        "training_dataset_file": TRAINING_DATASET_FILE,
        "trajectories_per_question": TRAJECTORIES_PER_QUESTION,
        "trajectory_count": TRAINING_TRAJECTORY_COUNT,
    }


def _write_jsonl(path: Path, values: Iterable[dict[str, JsonValue]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(
                json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())


def write_protocol13_training_dataset(
    records: Sequence[Protocol13TrainingRecord],
    output_dir: Path,
) -> dict[str, JsonValue]:
    """Publish an owner-only private artifact without overwriting an earlier run."""

    validate_protocol13_training_records(records)
    if not output_dir.is_absolute() or output_dir.exists():
        raise ValueError("training output directory must be a fresh absolute path")
    staging = output_dir.with_name(f".{output_dir.name}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True, mode=0o700)
    try:
        _write_jsonl(staging / TRAINING_DATASET_FILE, (record.to_value() for record in records))
        _write_jsonl(staging / MODEL_INPUT_FILE, (record.model_input_value() for record in records))
        summary = training_summary(records)
        summary_path = staging / SUMMARY_FILE
        descriptor = os.open(summary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, output_dir)
        return summary
    except BaseException:
        for name in (TRAINING_DATASET_FILE, MODEL_INPUT_FILE, SUMMARY_FILE):
            (staging / name).unlink(missing_ok=True)
        staging.rmdir()
        raise


def load_protocol13_training_records(path: Path) -> tuple[Protocol13TrainingRecord, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="\n") as stream:
        records = tuple(
            Protocol13TrainingRecord.from_value(json.loads(line)) for line in stream if line.strip()
        )
    validate_protocol13_training_records(records)
    return records


def validate_protocol13_training_dataset(output_dir: Path) -> dict[str, JsonValue]:
    """Re-read all three files and prove the model-only projection is exact."""

    records = load_protocol13_training_records(output_dir / TRAINING_DATASET_FILE)
    with (output_dir / MODEL_INPUT_FILE).open(encoding="utf-8", newline="\n") as stream:
        model_inputs = tuple(json.loads(line) for line in stream if line.strip())
    expected_inputs = tuple(record.model_input_value() for record in records)
    if model_inputs != expected_inputs:
        raise ValueError("model input projection differs from the unified dataset")
    summary = _object(
        json.loads((output_dir / SUMMARY_FILE).read_text(encoding="utf-8")),
        field="training summary",
    )
    expected_summary = training_summary(records)
    if summary != expected_summary:
        raise ValueError("training summary differs from the dataset")
    return expected_summary


__all__ = [
    "MODEL_INPUT_FILE",
    "SUMMARY_FILE",
    "TRAINING_DATASET_FILE",
    "TRAINING_RECORD_FORMAT",
    "TRAINING_SELECTION_ALGORITHM",
    "TRAINING_SUMMARY_FORMAT",
    "Protocol13TrainingEpisode",
    "Protocol13TrainingOutput",
    "Protocol13TrainingRecord",
    "Protocol13TrainingSourceCase",
    "build_protocol13_training_records",
    "expand_protocol13_training_trajectories",
    "load_protocol13_training_records",
    "training_summary",
    "validate_protocol13_training_dataset",
    "validate_protocol13_training_records",
    "write_protocol13_training_dataset",
]
