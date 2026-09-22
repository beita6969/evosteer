"""Immutable exact-eight catalog for Protocol 13.

Protocol 12 remains the historical exact-nine identity.  This module deliberately
uses a separate enum so a diagnostic benchmark cannot be parsed and admitted late.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class Protocol13Benchmark(StrEnum):
    HOTPOT_QA = "hotpotqa"
    TRIVIA_QA = "triviaqa"
    AIME_2026 = "aime-2026"
    HEALTHBENCH = "healthbench"
    WEB_SHOP = "webshop"
    ALF_WORLD = "alfworld"
    MBPP_PLUS = "mbpp-plus"
    HUMAN_EVAL = "humaneval"


ACTIVE_PROTOCOL13_BENCHMARKS: Final = (
    Protocol13Benchmark.HOTPOT_QA,
    Protocol13Benchmark.TRIVIA_QA,
    Protocol13Benchmark.AIME_2026,
    Protocol13Benchmark.HEALTHBENCH,
    Protocol13Benchmark.WEB_SHOP,
    Protocol13Benchmark.ALF_WORLD,
    Protocol13Benchmark.MBPP_PLUS,
    Protocol13Benchmark.HUMAN_EVAL,
)

QUESTIONS_PER_DOMAIN: Final = 250
QUESTIONS_PER_STEP: Final = len(ACTIVE_PROTOCOL13_BENCHMARKS)
TRAJECTORIES_PER_QUESTION: Final = 4
EFFECTIVE_BATCH_SIZE: Final = QUESTIONS_PER_STEP * TRAJECTORIES_PER_QUESTION
TRAINING_STEPS: Final = QUESTIONS_PER_DOMAIN
TRAINING_QUESTION_COUNT: Final = QUESTIONS_PER_DOMAIN * QUESTIONS_PER_STEP
TRAINING_TRAJECTORY_COUNT: Final = TRAINING_STEPS * EFFECTIVE_BATCH_SIZE
CHECKPOINT_EVERY_STEPS: Final = 10
CHECKPOINT_COUNT: Final = TRAINING_STEPS // CHECKPOINT_EVERY_STEPS
FINAL_RECORD_COUNT: Final = 30 + 128 * (len(ACTIVE_PROTOCOL13_BENCHMARKS) - 1)


def expected_final_count(benchmark: Protocol13Benchmark) -> int:
    return 30 if benchmark is Protocol13Benchmark.AIME_2026 else 128


if TRAINING_QUESTION_COUNT != 2_000:
    raise AssertionError("Protocol 13 training population must contain 2,000 questions")
if TRAINING_TRAJECTORY_COUNT != 8_000:
    raise AssertionError("Protocol 13 must generate 8,000 training trajectories")
if TRAINING_STEPS != 250:
    raise AssertionError("Protocol 13 must contain 250 optimizer steps")
if CHECKPOINT_COUNT != 25:
    raise AssertionError("Protocol 13 must save 25 cadence checkpoints")
if FINAL_RECORD_COUNT != 926:
    raise AssertionError("Protocol 13 final panel must contain 926 records")


__all__ = [
    "ACTIVE_PROTOCOL13_BENCHMARKS",
    "CHECKPOINT_COUNT",
    "CHECKPOINT_EVERY_STEPS",
    "EFFECTIVE_BATCH_SIZE",
    "FINAL_RECORD_COUNT",
    "QUESTIONS_PER_DOMAIN",
    "QUESTIONS_PER_STEP",
    "TRAINING_QUESTION_COUNT",
    "TRAINING_STEPS",
    "TRAINING_TRAJECTORY_COUNT",
    "TRAJECTORIES_PER_QUESTION",
    "Protocol13Benchmark",
    "expected_final_count",
]
