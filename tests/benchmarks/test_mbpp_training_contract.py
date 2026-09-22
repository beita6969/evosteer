"""One real native scorer condition for training labels and IID verdicts."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from skillev_private.benchmarks import protocol_v13_mbpp_worker
from skillev_private.benchmarks.mbpp_scoring import (
    MBPPScorerProfile,
    MBPPScoringInfrastructureError,
    decode_mbpp_verdict,
    mbpp_request,
)
from skillev_private.benchmarks.protocol_v10_workers import PrivateJSONWorker
from skillev_private.benchmarks.protocol_v13_training_sessions import _MBPPPlusEvaluator

from skillev.contracts import canonical_json
from skillev.rollout import TerminalEvaluatorError
from skillev.runtime import RuntimeSnapshotIdentity
from tests.benchmarks.test_protocol_v13_training_sessions import _mbpp_record, _request
from tests.runtime.test_runtime_snapshot_v4 import _snapshot


def _payload(profile: MBPPScorerProfile, submission: str) -> dict[str, object]:
    return mbpp_request(
        profile=profile,
        private_target={
            "assertion": [],
            "atol": 0,
            "base_input": [[1]],
            "canonical_solution": "def solve(value): return value + 1\n",
            "contract": [],
            "entry_point": "solve",
            "plus_input": [[2]],
        },
        prompt="",
        source_task_id="Mbpp/99999",
        submission=submission,
        task_id="synthetic-mbpp-contract",
    )


def _worker(tmp_path: Path, request: dict[str, object]) -> dict[str, object]:
    completed = subprocess.run(  # noqa: S603 -- repository worker, synthetic stdin only
        [sys.executable, str(Path(protocol_v13_mbpp_worker.__file__).resolve())],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
        cwd=tmp_path,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    return json.loads(completed.stdout)


def test_explicit_time_conditions_change_native_verdict_without_implicit_fallback(tmp_path):
    native = MBPPScorerProfile()
    short = replace(
        native,
        profile_id="synthetic-short",
        condition="custom-time-limits",
        min_time_limit=0.01,
        gt_time_limit_factor=1.0,
    )
    program = "import time\ndef solve(value):\n    time.sleep(0.1)\n    return value + 1\n"
    assert decode_mbpp_verdict(_worker(tmp_path, _payload(native, program)), native)["base_passed"]
    result = decode_mbpp_verdict(_worker(tmp_path, _payload(short, program)), short)
    assert not result["base_passed"]
    assert not result["plus_passed"]
    with pytest.raises(MBPPScoringInfrastructureError):
        decode_mbpp_verdict(result, native)


def test_no_profile_is_an_infrastructure_error_not_a_legacy_training_label(tmp_path):
    request = _payload(MBPPScorerProfile(), "def solve(value): return value + 1\n")
    del request["scorer_profile"]
    result = _worker(tmp_path, request)
    assert result["infrastructure_error"] is True
    assert "base_passed" not in result


def test_native_process_deadline_is_a_verdict_not_a_worker_infrastructure_failure(tmp_path):
    profile = replace(
        MBPPScorerProfile(),
        profile_id="synthetic-module-timeout",
        condition="custom-time-limits",
        min_time_limit=0.01,
        gt_time_limit_factor=1.0,
    )
    result = decode_mbpp_verdict(_worker(tmp_path, _payload(profile, "while True: pass")), profile)
    assert not result["base_passed"]
    assert not result["plus_passed"]
    assert result["lanes"]["base"]["native_status"] == "timeout"
    assert result["lanes"]["plus"]["native_status"] == "timeout"


def test_real_training_decoder_preserves_native_profile_and_unknown_results(tmp_path):
    record = _mbpp_record()
    record = replace(
        record,
        episode=replace(record.episode, source_id="Mbpp/99999"),
        output=replace(
            record.output,
            target={
                "assertion": [],
                "atol": 0,
                "base_input": [[]],
                "canonical_solution": "def solve(): return 1\n",
                "contract": [],
                "entry_point": "solve",
                "plus_input": [[]],
            },
        ),
    )
    worker = PrivateJSONWorker(
        command=(sys.executable, str(Path(protocol_v13_mbpp_worker.__file__).resolve())),
        working_directory=tmp_path,
        timeout_seconds=30,
    )
    profile = MBPPScorerProfile()
    evaluator = _MBPPPlusEvaluator(record, record.input, worker, profile)
    reward = asyncio.run(
        evaluator.evaluate(_request(record.input.task_id, "def solve(): return 1\n"))
    )
    assert reward.value == 1
    assert reward.success
    native = decode_mbpp_verdict(reward.native_payload["native_verdict"], profile)
    assert native["base_passed"]
    assert native["plus_passed"]
    assert reward.native_payload["scorer_profile"] == profile.to_value()
    broken = replace(
        record,
        output=replace(
            record.output,
            target={
                **record.output.target,
                "canonical_solution": "raise RuntimeError('synthetic reference failure')",
            },
        ),
    )
    with pytest.raises(TerminalEvaluatorError):
        asyncio.run(
            _MBPPPlusEvaluator(broken, broken.input, worker, profile).evaluate(
                _request(broken.input.task_id, "def solve(): return 1\n")
            )
        )


def test_snapshot_identity_carries_scoring_condition_and_preserves_legacy_unknown():
    base = _snapshot().identity
    native = replace(
        base,
        terminal_evaluation_conditions_json=canonical_json(
            {"mbpp-plus": MBPPScorerProfile().to_value()}
        ),
    )
    custom = replace(
        native,
        terminal_evaluation_conditions_json=canonical_json(
            {
                "mbpp-plus": replace(
                    MBPPScorerProfile(),
                    profile_id="short",
                    condition="custom-time-limits",
                    min_time_limit=0.1,
                    gt_time_limit_factor=2.0,
                ).to_value()
            }
        ),
    )
    assert RuntimeSnapshotIdentity.from_value(native.to_value()) == native
    assert custom != native
    legacy = base.to_value()
    legacy.pop("terminal_evaluation_conditions")
    legacy.pop("task_feature_mapping_version")
    legacy["format"] = "skillev-runtime-snapshot-identity@6"
    restored = RuntimeSnapshotIdentity.from_value(legacy)
    assert restored.terminal_evaluation_conditions_json is None
    assert restored.to_value() == legacy
    assert restored != native


def test_trusted_source_coordinates_do_not_change_reward_or_actor_task():
    from skillev_private.benchmarks.protocol_v13_training_sessions import _SourceBoundEvaluator

    from tests.benchmarks.test_protocol_v13_training_sessions import _mbpp_verdict

    class Worker:
        async def request(self, value):
            return _mbpp_verdict(True, True)

    record = _mbpp_record()
    task_before = record.input.to_value()
    evaluator = _SourceBoundEvaluator(
        _MBPPPlusEvaluator(record, record.input, Worker(), MBPPScorerProfile()), record
    )
    reward = asyncio.run(
        evaluator.evaluate(_request(record.input.task_id, "def solve(): return 1"))
    )
    assert reward.success
    assert reward.value == 1
    assert (
        reward.native_payload["training_evidence_source"]["source_question_id"]
        == record.episode.source_id
    )
    assert record.input.to_value() == task_before
    assert (
        reward.native_payload["training_evidence_source"]["occurrence_id"]
        == record.episode.episode_id
    )
    assert "training_evidence_source" not in repr(task_before)


def test_training_default_is_the_declared_native_iid_profile():
    import evalplus
    import yaml
    from skillev_private.benchmarks.mbpp_scoring import resolve_mbpp_profile

    raw = yaml.safe_load(Path("configs/evaluation/mbpp_evalplus_native.yaml").read_text())
    source_root = Path(evalplus.__file__).resolve().parent.parent
    iid = resolve_mbpp_profile({"source_root": str(source_root), "profile": raw})
    assert iid == MBPPScorerProfile()
