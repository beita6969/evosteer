"""Historical Protocol 12 catalog, not the current project's seven-IID catalog.

Names containing CURRENT below retain their original Protocol 12 identity and
counts for historical readers. New Step-0 runs use input_metric_contracts and
must not execute this old population or relabel it as the current seven domains.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class CurrentIIDBenchmark(StrEnum):
    HOTPOT_QA = "hotpotqa"
    TRIVIA_QA = "triviaqa"
    AIME_2026 = "aime-2026"
    HEALTHBENCH = "healthbench"
    WEBSHOP = "webshop"
    ALFWORLD = "alfworld"
    SPREADSHEETBENCH = "spreadsheetbench"
    MBPP_PLUS = "mbpp-plus"
    HUMAN_EVAL = "humaneval"

    # Historical/diagnostic identities are intentionally not active.
    APPWORLD_DIAGNOSTIC = "appworld"
    SWE_BENCH_HISTORICAL = "swe-bench"
    MBPP_PLUS_HARD_LEGACY = "mbpp-plus-hard"


ACTIVE_CURRENT_IID_BENCHMARKS: Final = (
    CurrentIIDBenchmark.HOTPOT_QA,
    CurrentIIDBenchmark.TRIVIA_QA,
    CurrentIIDBenchmark.AIME_2026,
    CurrentIIDBenchmark.HEALTHBENCH,
    CurrentIIDBenchmark.WEBSHOP,
    CurrentIIDBenchmark.ALFWORLD,
    CurrentIIDBenchmark.SPREADSHEETBENCH,
    CurrentIIDBenchmark.MBPP_PLUS,
    CurrentIIDBenchmark.HUMAN_EVAL,
)

CURRENT_IID_EPISODES_PER_DOMAIN: Final = 512
CURRENT_IID_BATCH_SIZE: Final = 16
CURRENT_IID_TRAINING_RECORD_COUNT: Final = 4_608
CURRENT_IID_OPTIMIZER_STEPS: Final = 288
CURRENT_IID_FINAL_RECORD_COUNT: Final = 1_054


def expected_final_count(benchmark: CurrentIIDBenchmark) -> int:
    if benchmark not in ACTIVE_CURRENT_IID_BENCHMARKS:
        raise ValueError(f"{benchmark.value} is not an active current-IID benchmark")
    return 30 if benchmark is CurrentIIDBenchmark.AIME_2026 else 128


__all__ = [
    "ACTIVE_CURRENT_IID_BENCHMARKS",
    "CURRENT_IID_BATCH_SIZE",
    "CURRENT_IID_EPISODES_PER_DOMAIN",
    "CURRENT_IID_FINAL_RECORD_COUNT",
    "CURRENT_IID_OPTIMIZER_STEPS",
    "CURRENT_IID_TRAINING_RECORD_COUNT",
    "CurrentIIDBenchmark",
    "expected_final_count",
]
