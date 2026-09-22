from __future__ import annotations

import asyncio
import json
from pathlib import Path

from skillev_private.direct_reference.journal import PrivateGenerationJournal

from skillev.evaluation.direct_baseline import DirectBenchmark
from skillev.evaluation.direct_baseline.runner import RawGenerationRecord


def _record(task_id: str, request_attempt: int) -> RawGenerationRecord:
    return RawGenerationRecord(
        task_id=task_id,
        benchmark=DirectBenchmark.HOTPOT_QA,
        raw_text="public candidate response",
        reasoning_text=None,
        finish_reason="stop",
        prompt_tokens=4,
        completion_tokens=3,
        response_model="qwen35-direct-base",
        response_id=None,
        service_instance_id="service-0",
        panel_index=0,
        service_slot=0,
        prompt_profile_id="prompt@1",
        parser_profile_id="parser@1",
        decoding_profile_id="decoding@1",
        population_id="population@1",
        run_seed=0,
        request_attempt=request_attempt,
        infrastructure_error=None,
    )


def test_private_generation_journal_can_durably_append_a_resumed_attempt(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "generations.jsonl").resolve()

    async def write() -> None:
        async with PrivateGenerationJournal(path) as journal:
            await journal.append(_record("task-1", 1))
        async with PrivateGenerationJournal(path, append_existing=True) as journal:
            await journal.append(_record("task-2", 1))

    asyncio.run(write())

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(row["task_id"], row["request_attempt"]) for row in rows] == [
        ("task-1", 1),
        ("task-2", 1),
    ]
