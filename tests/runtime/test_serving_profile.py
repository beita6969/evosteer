from __future__ import annotations

import copy

import pytest
from skillev_private.experiments.bayesian_serving import register_serving

from skillev.runtime.service_topology import InferenceService, ServiceTopology
from skillev.runtime.serving_profile import (
    require_same_profile,
    require_training_service,
    serving_profile,
)
from skillev.runtime.sglang_gateway import SGLangGateway
from tests.runtime.test_sglang_pool import pool_fixture


def info():
    return {
        "model_path": "/models/qwen",
        "tokenizer_path": "/models/qwen",
        "served_model_name": "qwen",
        "dtype": "auto",
        "context_length": 67584,
        "enable_lora": True,
        "enable_deterministic_inference": True,
        "sampling_backend": "pytorch",
        "version": "test-serving-version",
        "page_size": 64,
        "chunked_prefill_size": 4096,
    }


def test_flat_and_nested_actual_settings_without_fabricated_measurements():
    raw = info()
    flat = serving_profile(raw)
    assert serving_profile({"server_args": raw, "version": raw["version"]}) == flat
    assert flat["revision"] is None
    with pytest.raises(ValueError):
        serving_profile({"server_args": {}})
    # A telemetry toggle does not change token execution.
    require_same_profile(flat, serving_profile({**raw, "enable_request_time_stats_logging": True}))
    with pytest.raises(ValueError):
        require_same_profile(flat, serving_profile({**raw, "page_size": 32}))


@pytest.mark.parametrize(
    "change",
    [
        {"model_path": "/models/other"},
        {"tokenizer_path": "/tokenizers/other"},
        {"enable_lora": False},
        {"context_length": 32000},
        {"sampling_backend": "other"},
        {"enable_deterministic_inference": False},
    ],
)
def test_actual_service_must_match_declared_model_and_execution(change):
    with pytest.raises(ValueError):
        require_training_service(
            serving_profile({**info(), **change}),
            model_path="/models/qwen",
            tokenizer_path="/models/qwen",
            base_model="qwen",
            minimum_context=67584,
            actor=True,
        )


def test_registration_restore_and_publication_recheck(tmp_path, monkeypatch):
    pool, controls = pool_fixture()
    live = [info(), info()]
    monkeypatch.setattr(
        SGLangGateway,
        "read_serving_profile",
        lambda self: serving_profile(live[pool.members.index(self)]),
    )
    topology = ServiceTopology(
        (
            InferenceService("a", "http://localhost:8000", "GPU-a"),
            InferenceService("b", "http://localhost:8001", "GPU-b"),
        ),
        ("a", "b"),
        ("a",),
        ("a",),
        ("GPU-c", "GPU-d"),
    )
    args = {
        "root": tmp_path,
        "model_path": "/models/qwen",
        "tokenizer_path": "/models/qwen",
        "minimum_context": 67584,
    }
    original = register_serving(pool, topology, **args)
    assert register_serving(pool, topology, **args) == original
    # A restarted replica with another prefill profile is caught before load.
    live[1]["chunked_prefill_size"] = 2048
    before = copy.deepcopy(controls[1].calls)
    with pytest.raises(ValueError):
        pool.prepare_supervisor_adapter(adapter_path="/adapters/v1", adapter_revision="v1")
    assert controls[1].calls == before
    with pytest.raises(ValueError):
        register_serving(pool, topology, **args)
    live[0]["chunked_prefill_size"] = 2048
    # Even when all replicas agree, resume cannot silently change old settings.
    with pytest.raises(ValueError):
        register_serving(pool, topology, **args)
