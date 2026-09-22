"""Official numeric test inputs survive the evaluation-only JSON boundary."""

import gzip
import json
import math

import pytest
from skillev_private.benchmarks.protocol_v13_mbpp_worker import _restore
from skillev_private.evaluation.iid_episode_sources import IIDSourceIdentity, evaluation_records
from skillev_private.evaluation.integrity_sources import SourcePanel, _json_rows

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.integrity_pipeline import FrozenPanel


@pytest.mark.parametrize("compressed", [False, True])
def test_iid_nonfinite_native_inputs_round_trip_without_reaching_public_task(tmp_path, compressed):
    target = {
        "base_input": [[1.5, float("inf"), float("-inf")]],
        "plus_input": [[float("nan"), {"nested": float("inf")}]],
    }
    path = tmp_path / ("synthetic.jsonl.gz" if compressed else "synthetic.jsonl")
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"private_target": target}) + "\n")
    loaded = _json_rows(path)[0]
    entry = PublicTaskView.from_record("mbpp-plus/synthetic", "mbpp-plus", {"prompt": "def f(x):"})
    identity = IIDSourceIdentity(
        entry.task_id, "mbpp-plus", "synthetic", "synthetic-panel", "synthetic-v1", "evaluation"
    )
    source = SourcePanel(
        FrozenPanel((entry,), "synthetic", "synthetic", (("mbpp-plus", 1),)),
        {entry.task_id: loaded},
        {},
        {},
    )
    (record,) = evaluation_records(source, (identity,), max_turns=25, seed=0)
    wire = json.loads(json.dumps(record.output.target, allow_nan=False))
    restored = _restore(wire)
    assert restored["base_input"] == target["base_input"]
    assert math.isnan(restored["plus_input"][0][0])
    assert restored["plus_input"][0][1]["nested"] == float("inf")
    assert record.input.query == "def f(x):"
    assert "private_target" not in record.input.public_context
