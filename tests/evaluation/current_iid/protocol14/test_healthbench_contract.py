from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from skillev.evaluation.current_iid.protocol14.catalog import Protocol14Benchmark
from skillev.evaluation.current_iid.protocol14.config import load_execution_contracts_v4
from skillev.evaluation.current_iid.protocol14.contracts import (
    ExecutionLane,
    FormalEligibility,
)
from skillev.evaluation.healthbench_official import load_qwen_judge_profile
from skillev.experiments.protocol_v14 import load_protocol_v14

ROOT = Path(__file__).parents[4]


def _load_script(module_name: str, filename: str) -> ModuleType:
    spec = spec_from_file_location(module_name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load test script {filename}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


receipt_builder = _load_script(
    "_protocol14_test_receipt_builder", "build_qwen35_protocol14_receipt.py"
)
health_runner = _load_script(
    "_protocol14_test_health_runner", "run_qwen35_protocol14_healthbench.py"
)


def test_healthbench_actor_and_judge_are_fully_bound() -> None:
    protocol = load_protocol_v14(
        ROOT / "configs/evaluation/protocol_v14.yaml",
        ROOT / "configs/evaluation/protocol_v14_sources.yaml",
    )
    execution = load_execution_contracts_v4(
        ROOT / "configs/evaluation/protocol_v14_conditions.yaml", protocol=protocol
    )[Protocol14Benchmark.HEALTHBENCH]
    judge = load_qwen_judge_profile(
        ROOT / "configs/evaluation/protocol_v14_healthbench_grader.yaml"
    )

    assert execution.expected_count == 128
    assert execution.selection_rule == "frozen-random-0-sample-128"
    assert execution.lane is ExecutionLane.DIAGNOSTIC
    assert execution.formal_eligibility is FormalEligibility.DIAGNOSTIC_ONLY
    assert execution.thinking_mode.value == "disabled"
    assert execution.decoding.temperature == 0.5
    assert execution.decoding.top_p == 1.0
    assert execution.decoding.top_k == 0
    assert execution.decoding.max_new_tokens == 2048
    assert execution.aggregation.seeds == (0,)
    assert execution.aggregation.run_count == 1
    assert execution.aggregation.reducer == "identity"
    assert execution.aggregation.dispersion is None
    assert execution.grader_profile == judge.profile_id
    assert judge.backend == "qwen-sglang"
    assert judge.model_route == execution.actor_route
    assert judge.call_mode == "per-rubric"
    assert judge.system_message == "You are a helpful assistant."
    assert judge.max_tokens == 2048
    assert judge.temperature == 0.5
    assert judge.top_p is None
    assert judge.top_k is None
    assert judge.seed is None
    assert not judge.enable_thinking
    assert judge.semantic_repair_attempts == 1
    assert not judge.official_gpt_comparable
    assert not receipt_builder._required_for_formal_gate(execution)


def test_healthbench_grader_builds_qwen_sglang_transports(monkeypatch) -> None:
    judge = load_qwen_judge_profile(
        ROOT / "configs/evaluation/protocol_v14_healthbench_grader.yaml"
    )
    client = object()
    checked: list[tuple[object, str]] = []
    monkeypatch.setattr(
        health_runner,
        "_candidate_client",
        lambda endpoint, timeout_seconds: client,
    )
    monkeypatch.setattr(
        health_runner,
        "_require_candidate_route",
        lambda value, route: checked.append((value, route)),
    )

    grader = health_runner._qwen_grader(
        endpoint_bases=["http://judge-1/v1", "http://judge-2/v1"],
        profile=judge,
        sampler_response_type=object,
    )

    assert checked == [(client, judge.model_route), (client, judge.model_route)]
    assert len(grader.transports) == 2
    assert all(transport.model == judge.model_route for transport in grader.transports)
    assert all(transport.temperature == 0.5 for transport in grader.transports)
    assert all(transport.max_tokens == 2048 for transport in grader.transports)
    assert all(transport.top_p is None for transport in grader.transports)
    assert all(transport.top_k is None for transport in grader.transports)
    assert all(transport.seed is None for transport in grader.transports)
    assert all(not transport.enable_thinking for transport in grader.transports)


def test_healthbench_route_check_accepts_requested_base_on_adapter_capable_server() -> None:
    client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=(
                    SimpleNamespace(id="qwen35-direct-base"),
                    SimpleNamespace(id="trained-forward-adapter"),
                )
            )
        )
    )

    health_runner._require_candidate_route(client, "qwen35-direct-base")
    with pytest.raises(RuntimeError):
        health_runner._require_candidate_route(client, "missing-route")
