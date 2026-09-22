"""Declared scoring resources, independent lanes and verbatim public carriers."""

import io
import json
import multiprocessing
import sys
import time
import types

import pytest
from skillev_private.benchmarks import protocol_v13_mbpp_worker
from skillev_private.evaluation.integrity_mbpp import (
    MBPPScorerProfile,
    mbpp_failure_kind,
    resolve_mbpp_profile,
)
from skillev_private.evaluation.integrity_sources import mbpp_public_prompt


def test_native_profile_reads_the_selected_source_defaults_and_outer_budget(tmp_path):
    source = tmp_path / "evalplus"
    source.mkdir()
    (source / "config.py").write_text(
        "DEFAULT_MIN_TIME_LIMIT = 4.0\nDEFAULT_GT_TIME_LIMIT_FACTOR = 4.0\n"
    )
    settings = {"source_root": str(tmp_path), "timeout_seconds": 900}
    assert resolve_mbpp_profile(settings).min_time_limit == 4
    assert resolve_mbpp_profile(settings).result_transport == "single-writer-raw@1"
    assert (
        resolve_mbpp_profile({**settings, "profile": {"profile_id": "old"}}).result_transport
        == "synchronized"
    )
    with pytest.raises(ValueError):
        resolve_mbpp_profile({**settings, "profile": {"min_time_limit": 0.1}})
    custom = resolve_mbpp_profile(
        {
            **settings,
            "profile": {
                "profile_id": "custom-short",
                "condition": "custom-time-limits",
                "min_time_limit": 0.1,
                "gt_time_limit_factor": 2.0,
            },
        }
    )
    assert custom.condition == "custom-time-limits"
    with pytest.raises(ValueError):
        resolve_mbpp_profile({**settings, "timeout_seconds": 10})


def _writer_stopped_during_result_transfer(values, progress, signal):
    values[0] = True
    progress.value = 1
    signal.send(True)
    time.sleep(60)


def test_parent_can_read_native_results_after_writer_is_terminated():
    native = types.SimpleNamespace()
    protocol_v13_mbpp_worker._configure_result_transport(native, "single-writer-raw@1")
    values, progress = native.Array("b", [False, False]), native.Value("i", 0)
    ctx = multiprocessing.get_context("fork")
    reader, writer = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_writer_stopped_during_result_transfer, args=(values, progress, writer)
    )
    process.start()
    try:
        assert reader.poll(5)
        assert reader.recv()
        process.terminate()
        process.join(5)
        assert not process.is_alive()
        assert values[: progress.value] == [1]
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        reader.close()
        writer.close()


@pytest.mark.parametrize("first", ["fail", "timeout"])
def test_fixed_source_pass_constant_and_plus_lane_are_preserved_after_base_failure(
    monkeypatch, first
):
    modules = {
        name: types.ModuleType(name)
        for name in (
            "evalplus",
            "evalplus.data",
            "evalplus.data.mbpp",
            "evalplus.eval",
            "evalplus.eval._special_oracle",
            "evalplus.gen",
            "evalplus.gen.util",
        )
    }
    calls = []
    verdicts = iter([(first, [False]), ("pass", [True])])

    def check(*args, **kwargs):
        calls.append((args[2], kwargs))
        return next(verdicts)

    modules["evalplus.eval"].PASS = "pass"
    modules["evalplus.eval"].untrusted_check = check
    modules["evalplus.data.mbpp"].mbpp_deserialize_inputs = lambda _, value: value
    modules["evalplus.eval._special_oracle"].MBPP_OUTPUT_NOT_NONE_TASKS = set()
    modules["evalplus.gen.util"].trusted_exec = lambda *a, **kw: ([2], [0.01])
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    request = {
        "format": "skillev-mbpp-request@2",
        "operation": "evaluate-mbpp-plus",
        "task_id": "synthetic",
        "source_task_id": "synthetic",
        "prompt": "",
        "submission": "def f(x): return x + 1",
        "private_target": {
            "assertion": [],
            "atol": 0,
            "base_input": [[1]],
            "plus_input": [[2]],
            "canonical_solution": "def f(x): return x + 1",
            "contract": [],
            "entry_point": "f",
        },
        "scorer_profile": MBPPScorerProfile().to_value(),
    }
    output = io.StringIO()
    monkeypatch.setattr(
        sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(json.dumps(request).encode()))
    )
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setenv("EVALPLUS_MAX_MEMORY_BYTES", "1")
    monkeypatch.setenv("EVALPLUS_TIMEOUT_PER_TASK", "invalid-inherited-setting")
    protocol_v13_mbpp_worker.main()
    result = json.loads(output.getvalue())
    assert not result["base_passed"]
    assert result["plus_passed"]
    assert result["status"] == "failed"
    assert result["lanes"]["base"]["native_status"] == first
    assert result["lanes"]["plus"]["native_status"] == "pass"
    assert len(calls) == 2
    assert all(
        row[1]["min_time_limit"] == 4 and row[1]["gt_time_limit_factor"] == 4 for row in calls
    )
    assert mbpp_failure_kind(result) == (
        "base-timeout" if first == "timeout" else "base-test-failure"
    )


def test_public_carrier_preserves_interface_and_examples_but_never_copies_private_fields():
    prompt = 'def f(x):\n    """Add one. Example: f(1) == 2."""\n'
    target = {"prompt": prompt, "canonical_solution": "PRIVATE SOLUTION", "plus_input": ["PRIVATE"]}
    assert mbpp_public_prompt({"prompt": prompt}, target) == prompt
    with pytest.raises(ValueError):
        mbpp_public_prompt(prompt.replace("Example: f(1) == 2.", ""), target)
    with pytest.raises(ValueError):
        mbpp_public_prompt(prompt + "Precomputed advice.", target)
