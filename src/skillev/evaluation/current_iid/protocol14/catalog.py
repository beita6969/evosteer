"""Immutable corrected exact-eight IID catalog for Protocol 14."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class Protocol14Benchmark(StrEnum):
    HOTPOT_QA = "hotpotqa"
    TRIVIA_QA = "triviaqa"
    AIME_2026 = "aime-2026"
    HEALTHBENCH = "healthbench"
    WEB_SHOP = "webshop"
    ALF_WORLD = "alfworld"
    MBPP_PLUS = "mbpp-plus"
    HUMAN_EVAL = "humaneval"


ACTIVE_PROTOCOL14_BENCHMARKS: Final = tuple(Protocol14Benchmark)
FINAL_RECORD_COUNT: Final = 30 + 128 * 7


def expected_final_count(benchmark: Protocol14Benchmark) -> int:
    """Return the frozen panel size; AIME 2026 is the documented all-30 exception."""

    return 30 if benchmark is Protocol14Benchmark.AIME_2026 else 128


if FINAL_RECORD_COUNT != 926:
    raise AssertionError("Protocol 14 final panel must contain 926 records")
if sum(expected_final_count(item) for item in ACTIVE_PROTOCOL14_BENCHMARKS) != 926:
    raise AssertionError("Protocol 14 catalog count does not close")


__all__ = [
    "ACTIVE_PROTOCOL14_BENCHMARKS",
    "FINAL_RECORD_COUNT",
    "Protocol14Benchmark",
    "expected_final_count",
]
