from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest
from skillev_private.benchmarks.protocol_v13_training import (
    MODEL_INPUT_FILE,
    SUMMARY_FILE,
    TRAINING_DATASET_FILE,
    Protocol13TrainingSourceCase,
    build_protocol13_training_records,
    expand_protocol13_training_trajectories,
    load_protocol13_training_records,
    training_summary,
    validate_protocol13_training_dataset,
    write_protocol13_training_dataset,
)
from skillev_private.benchmarks.protocol_v13_training_sources import (
    NONFINITE_FLOAT_FORMAT,
    restore_nonfinite_evaluator_values,
)

from skillev.evaluation.current_iid.protocol13.catalog import (
    ACTIVE_PROTOCOL13_BENCHMARKS,
    EFFECTIVE_BATCH_SIZE,
    QUESTIONS_PER_DOMAIN,
    QUESTIONS_PER_STEP,
    TRAINING_QUESTION_COUNT,
    TRAJECTORIES_PER_QUESTION,
    Protocol13Benchmark,
)
from skillev.rollout import RolloutTask

_EVALUATORS = {
    Protocol13Benchmark.HOTPOT_QA: "hotpotqa-official-em-f1",
    Protocol13Benchmark.TRIVIA_QA: "triviaqa-official-alias-em-f1",
    Protocol13Benchmark.AIME_2026: "integer-exact",
    Protocol13Benchmark.HEALTHBENCH: "simple-evals-rubric",
    Protocol13Benchmark.WEB_SHOP: "webshop-native-reward",
    Protocol13Benchmark.ALF_WORLD: "alfworld-success",
    Protocol13Benchmark.MBPP_PLUS: "evalplus-base-plus",
    Protocol13Benchmark.HUMAN_EVAL: "humaneval-native",
}


def _case(benchmark: Protocol13Benchmark, index: int) -> Protocol13TrainingSourceCase:
    source_id = f"{benchmark.value}-source-{index:04d}"
    version = f"{benchmark.value}-revision"
    return Protocol13TrainingSourceCase(
        benchmark=benchmark,
        population_id=f"{benchmark.value}-train-v13",
        source_version=version,
        source_id=source_id,
        task=RolloutTask(
            task_id=source_id,
            environment_id=f"benchmark:{benchmark.value}@{version}",
            task_family=f"{benchmark.value}/test-family",
            context_id=f"{benchmark.value}:training",
            query=f"public query {index}",
            available_tools=(),
            public_context={
                "benchmark_id": benchmark.value,
                "dataset_revision": version,
                "payload": {"public": index},
                "split": "training",
            },
        ),
        evaluator_kind=_EVALUATORS[benchmark],
        evaluator_target={"private_target": f"target-{index}"},
    )


def _sources(
    default_count: int = 3,
) -> dict[Protocol13Benchmark, tuple[Protocol13TrainingSourceCase, ...]]:
    return {
        benchmark: tuple(_case(benchmark, index) for index in range(default_count))
        for benchmark in ACTIVE_PROTOCOL13_BENCHMARKS
    }


def test_exact_eight_schedule_is_deterministic_and_repeats_only_source_identity() -> None:
    sources = _sources()
    first = build_protocol13_training_records(sources)
    second = build_protocol13_training_records(sources)

    assert first == second
    assert len(first) == TRAINING_QUESTION_COUNT
    assert len({record.episode.episode_id for record in first}) == TRAINING_QUESTION_COUNT
    assert {tuple(record.to_value()) for record in first} == {
        ("episode", "format", "input", "output")
    }
    for step_index in range(QUESTIONS_PER_DOMAIN):
        step = first[step_index * QUESTIONS_PER_STEP : (step_index + 1) * QUESTIONS_PER_STEP]
        assert {record.episode.benchmark for record in step} == set(ACTIVE_PROTOCOL13_BENCHMARKS)
        assert {record.episode.optimizer_step for record in step} == {step_index + 1}
    for benchmark in ACTIVE_PROTOCOL13_BENCHMARKS:
        domain = [record for record in first if record.episode.benchmark is benchmark]
        assert len(domain) == QUESTIONS_PER_DOMAIN
        assert len({record.episode.source_id for record in domain}) == 3
        for source_id in {record.episode.source_id for record in domain}:
            ordinals = sorted(
                record.episode.repeat_ordinal
                for record in domain
                if record.episode.source_id == source_id
            )
            assert ordinals == list(range(len(ordinals)))


def test_population_larger_than_domain_block_is_not_repeated() -> None:
    sources = _sources(default_count=251)
    records = build_protocol13_training_records(sources)

    for benchmark in ACTIVE_PROTOCOL13_BENCHMARKS:
        domain = [record for record in records if record.episode.benchmark is benchmark]
        assert len({record.episode.source_id for record in domain}) == QUESTIONS_PER_DOMAIN
        assert {record.episode.repeat_ordinal for record in domain} == {0}


def test_runtime_expansion_gives_every_question_four_unique_rollouts() -> None:
    questions = build_protocol13_training_records(_sources())[:QUESTIONS_PER_STEP]

    trajectories = expand_protocol13_training_trajectories(questions)

    assert len(trajectories) == EFFECTIVE_BATCH_SIZE
    assert len({record.input.task_id for record in trajectories}) == EFFECTIVE_BATCH_SIZE
    assert all(record.input.task_id == record.episode.episode_id for record in trajectories)
    counts = Counter(record.episode.source_id for record in trajectories)
    assert set(counts.values()) == {TRAJECTORIES_PER_QUESTION}
    for offset in range(0, EFFECTIVE_BATCH_SIZE, QUESTIONS_PER_STEP):
        assert {
            record.episode.benchmark
            for record in trajectories[offset : offset + QUESTIONS_PER_STEP]
        } == set(ACTIVE_PROTOCOL13_BENCHMARKS)


def test_writer_emits_owner_only_unified_and_model_safe_streams(tmp_path: Path) -> None:
    records = build_protocol13_training_records(_sources(default_count=2))
    output = tmp_path / "protocol13-training"

    summary = write_protocol13_training_dataset(records, output)

    assert summary == training_summary(records)
    assert validate_protocol13_training_dataset(output) == summary
    assert load_protocol13_training_records(output / TRAINING_DATASET_FILE) == records
    assert (output / SUMMARY_FILE).stat().st_mode & 0o777 == 0o600
    assert (output / TRAINING_DATASET_FILE).stat().st_mode & 0o777 == 0o600
    assert (output / MODEL_INPUT_FILE).stat().st_mode & 0o777 == 0o600
    with (output / MODEL_INPUT_FILE).open(encoding="utf-8") as stream:
        values = [json.loads(line) for line in stream]
    assert len(values) == TRAINING_QUESTION_COUNT
    assert {tuple(value) for value in values} == {("episode_id", "format", "input")}
    assert all("output" not in value and "target" not in value for value in values)
    counts = Counter(value["input"]["public_context"]["benchmark_id"] for value in values)
    assert set(counts.values()) == {QUESTIONS_PER_DOMAIN}

    with pytest.raises(ValueError, match="fresh absolute path"):
        write_protocol13_training_dataset(records, output)


def test_duplicate_source_identity_is_rejected() -> None:
    sources = _sources()
    hotpot = sources[Protocol13Benchmark.HOTPOT_QA]
    sources[Protocol13Benchmark.HOTPOT_QA] = (hotpot[0], hotpot[0])

    with pytest.raises(ValueError, match="repeats a source ID"):
        build_protocol13_training_records(sources)


def test_evalplus_nonfinite_values_have_a_valid_json_carrier() -> None:
    encoded = {
        "nan": {"format": NONFINITE_FLOAT_FORMAT, "value": "nan"},
        "positive": {"format": NONFINITE_FLOAT_FORMAT, "value": "+inf"},
        "negative": {"format": NONFINITE_FLOAT_FORMAT, "value": "-inf"},
    }

    restored = restore_nonfinite_evaluator_values(json.loads(json.dumps(encoded)))

    assert isinstance(restored, dict)
    assert math.isnan(restored["nan"])
    assert restored["positive"] == float("inf")
    assert restored["negative"] == float("-inf")
