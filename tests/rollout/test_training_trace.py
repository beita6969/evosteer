import asyncio
import json
from pathlib import Path

import pytest

from skillev.diagnostics.rollout_trace import (
    JsonlRolloutTraceSink,
    RolloutTraceEvent,
    RolloutTraceStage,
    reject_private_keys,
)


def test_private_trace_writer_flushes_lifecycle(tmp_path: Path) -> None:
    path = (tmp_path / "trace.jsonl").resolve()

    async def run() -> None:
        async with JsonlRolloutTraceSink.create(path) as sink:
            await sink.record(
                RolloutTraceEvent(
                    "trajectory-1",
                    "task-1",
                    1,
                    RolloutTraceStage.ACTION_RESULT,
                    {"raw_text": "public action"},
                )
            )

    asyncio.run(run())
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["stage"] == "action-result"
    assert row["public_payload"] == {"raw_text": "public action"}


def test_trace_renderer_rejects_private_evaluator_keys_recursively() -> None:
    reject_private_keys({"public": [{"value": 1}]})
    with pytest.raises(ValueError):
        reject_private_keys({"nested": {"rubrics": ["secret"]}})
