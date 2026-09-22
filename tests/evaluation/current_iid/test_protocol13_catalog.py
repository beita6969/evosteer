from skillev.evaluation.current_iid.protocol13.catalog import (
    ACTIVE_PROTOCOL13_BENCHMARKS,
    CHECKPOINT_COUNT,
    CHECKPOINT_EVERY_STEPS,
    EFFECTIVE_BATCH_SIZE,
    FINAL_RECORD_COUNT,
    QUESTIONS_PER_DOMAIN,
    QUESTIONS_PER_STEP,
    TRAINING_QUESTION_COUNT,
    TRAINING_STEPS,
    TRAINING_TRAJECTORY_COUNT,
    TRAJECTORIES_PER_QUESTION,
    Protocol13Benchmark,
    expected_final_count,
)


def test_protocol13_is_exactly_the_authoritative_eight() -> None:
    assert tuple(Protocol13Benchmark) == ACTIVE_PROTOCOL13_BENCHMARKS
    assert "spreadsheetbench" not in {item.value for item in Protocol13Benchmark}
    assert "appworld" not in {item.value for item in Protocol13Benchmark}
    assert "swe-bench" not in {item.value for item in Protocol13Benchmark}
    assert QUESTIONS_PER_DOMAIN == 250
    assert QUESTIONS_PER_STEP == 8
    assert TRAJECTORIES_PER_QUESTION == 4
    assert EFFECTIVE_BATCH_SIZE == 32
    assert TRAINING_QUESTION_COUNT == 2_000
    assert TRAINING_TRAJECTORY_COUNT == 8_000
    assert TRAINING_STEPS == 250
    assert CHECKPOINT_EVERY_STEPS == 10
    assert CHECKPOINT_COUNT == 25
    assert FINAL_RECORD_COUNT == 926
    assert sum(expected_final_count(item) for item in ACTIVE_PROTOCOL13_BENCHMARKS) == 926
