"""Private collector pool wiring with synthetic service observations only."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from skillev_private.experiments import development_collection as dev
from skillev_private.experiments.readonly_replicas import ReadonlyReplicaGenerator

from skillev.training.rollout_workflow import RolloutWorkflowBinding
from tests.training.test_development_collection import request_fixture, write
from tests.training.test_readonly_replicas import ENDPOINTS


def test_selection_binds_ordered_pool_and_defaults_remain_unchanged(tmp_path):
    request, _, _ = request_fixture(tmp_path)
    _, original = dev.selection(request)
    assert "actor_replica_pool" not in original
    request["actor_endpoints"] = list(ENDPOINTS)
    _, pooled = dev.selection(request)
    assert pooled["actor_replica_pool"]["actor_endpoints"] == list(ENDPOINTS)
    request["comparison_reference"] = write(tmp_path / "reference.json", pooled)
    request["actor_endpoints"] = list(reversed(ENDPOINTS))
    with pytest.raises(ValueError):
        dev.selection(request)


@pytest.mark.parametrize(
    "drift",
    [
        None,
        "before",
        "after",
        "before-capacity",
        "after-capacity",
        "before-missing",
        "admission",
        "before-requests",
        "after-requests",
        "before-requests-missing",
    ],
)
def test_all_services_checked_original_collector_and_transports_closed(
    tmp_path, monkeypatch, make_training_harness, drift
):
    request, record, policy = request_fixture(tmp_path)
    harness = make_training_harness()
    request.update(
        endpoint=ENDPOINTS[0],
        actor_endpoints=list(ENDPOINTS),
        mbpp_interpreter="synthetic-python",
        evalplus_source_root="synthetic-source",
        mbpp_profile="synthetic-profile",
        development_practice="training-development-skill-practice@1",
    )
    admission = drift == "admission" or (drift is not None and "requests" in drift)
    if admission:
        request["replica_actor_requests"] = 32
    for key in ("backbone", "formal_config", "deployments"):
        request[key] = write(tmp_path / key, {})
    profile = {"dtype": "bfloat16", "quantization": None, "served_model_name": "synthetic-base"}
    request["serving_profile"] = write(tmp_path / "profile.json", profile)
    backbone = SimpleNamespace(
        tokenizer_id=policy.tokenizer_id,
        base_model_path="synthetic-base",
        tokenizer_path=None,
        to_value=dict,
    )
    formal = SimpleNamespace(
        require_backbone=lambda _: None,
        sampling_config=replace(harness.config.rollout, base_seed=0),
        performance_profile="unused",
        to_value=dict,
        max_input_tokens=2048,
        maximum_reasoning_tokens=128,
        maximum_action_tokens=1024,
        healthbench_judge="synthetic-judge",
        domains=("hotpotqa",),
        hotpot_deliberation=False,
        task_budget=None,
        static_task_budget=None,
        domain_task_budgets={},
        application_config=lambda _: SimpleNamespace(trainer=harness.config),
    )
    monkeypatch.setattr(dev, "load_fresh_config", lambda _: formal)
    monkeypatch.setattr(
        dev, "QwenMultimodalBackboneConfig", SimpleNamespace(from_value=lambda _: backbone)
    )
    tokenizer = SimpleNamespace(tokenizer_id=policy.tokenizer_id)
    monkeypatch.setattr(
        dev, "QwenTokenizerAdapter", SimpleNamespace(from_config=lambda _: tokenizer)
    )
    inspections = []

    def observe(endpoint):
        stage = "before" if len(inspections) < len(ENDPOINTS) else "after"
        inspections.append((stage, endpoint))
        actual = {
            **profile,
            "max_req_input_len": 4096 + ENDPOINTS.index(endpoint),
            "max_total_num_tokens": 8192 + ENDPOINTS.index(endpoint),
            "max_running_requests": 32 + ENDPOINTS.index(endpoint),
        }
        if endpoint == ENDPOINTS[-1] and drift == stage + "-requests":
            actual["max_running_requests"] = 31
        if endpoint == ENDPOINTS[-1] and drift == stage + "-requests-missing":
            actual.pop("max_running_requests")
        if endpoint == ENDPOINTS[-1] and drift == stage + "-capacity":
            actual["max_total_num_tokens"] = 2500  # input fits, input+output does not
        if endpoint == ENDPOINTS[-1] and drift == stage + "-missing":
            actual.pop("max_req_input_len")
        if stage == drift and endpoint == ENDPOINTS[-1]:
            actual["dtype"] = "float16"
        return {"server_info": actual, "model_info": {"model_path": backbone.base_model_path}}

    monkeypatch.setattr(dev, "observe_service", observe)
    monkeypatch.setattr(dev, "serving_profile", lambda row: {key: row[key] for key in profile})
    monkeypatch.setattr(dev, "require_training_service", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dev,
        "TrainingPerformanceConfig",
        SimpleNamespace(load=lambda _: SimpleNamespace(workflow=lambda: RolloutWorkflowBinding())),
    )
    monkeypatch.setattr(dev, "resolve_mbpp_profile", lambda _: object())
    monkeypatch.setattr(dev, "native_scorer_contracts", lambda *args, **kwargs: {"synthetic": True})
    calls = []

    async def sessions(rows, **kwargs):
        assert rows == (record,)
        calls.append("hydrate")
        return (), object()

    monkeypatch.setattr(dev, "build_protocol13_training_sessions", sessions)
    real = dev.ExternalSGLangRolloutGenerator
    members = []

    def member(**kwargs):
        assert kwargs["gateway"] is None
        value = real(**kwargs)
        members.append(value)
        return value

    monkeypatch.setattr(dev, "ExternalSGLangRolloutGenerator", member)

    async def collect(**kwargs):
        calls.append("collect")
        resources = kwargs["workflow_resources"]
        if admission:
            assert all(resources.model_limiter(e).limit == 32 for e in ENDPOINTS)
            assert resources.model_limiter(ENDPOINTS[0]) is not resources.model_limiter(
                ENDPOINTS[1]
            )
        else:
            assert all(resources.model_limiter(e) is resources.model_requests for e in ENDPOINTS)
        assert "replica_capacity_before" not in kwargs["condition"].scientific
        assert len(kwargs["condition"].execution["replica_capacity_before"]) == 3
        assert isinstance(kwargs["generator"], ReadonlyReplicaGenerator)
        assert kwargs["sampling_schedule_id"] == request["comparison_id"]
        assert kwargs["ordered_task_sequence_id"] == request["comparison_id"]
        assert isinstance(kwargs["base_sessions"], dev.PracticeSessionFactory)
        assert kwargs["tasks"] == (record.input,)
        assert kwargs["execution_controls"]["actor_replica_pool"]["actor_endpoints"] == list(
            ENDPOINTS
        )

    monkeypatch.setattr(dev, "collect_training_condition", collect)
    output = tmp_path / "output"
    if drift not in (None, "admission"):
        with pytest.raises(ValueError):
            asyncio.run(dev.run(request, output))
    else:
        result = asyncio.run(dev.run(request, output))
        assert result["training_updates"] == 0
    if drift and drift.startswith("before"):
        assert calls == []
        assert members == []
        assert inspections == [("before", endpoint) for endpoint in ENDPOINTS]
    else:
        assert calls == ["hydrate", "collect"]
        assert inspections == [
            (stage, endpoint) for stage in ("before", "after") for endpoint in ENDPOINTS
        ]
        assert all(m._executor._shutdown for m in members)
        controls = json.loads((output / "controls-private.json").read_text())
        assert list(controls["replica_service_profiles"]) == list(ENDPOINTS)
        assert controls["development_practice"]["profile"] == request["development_practice"]

    before = json.loads((output / "replica-capacities-before-private.json").read_text())
    if drift == "before-missing":
        assert before[ENDPOINTS[-1]]["max_req_input_len"] is None
    if drift == "after-capacity":
        after = json.loads((output / "replica-capacities-after-private.json").read_text())
        assert after[ENDPOINTS[-1]]["max_total_num_tokens"] == 2500
    if drift in (None, "admission"):
        after_controls = json.loads((output / "controls-after-private.json").read_text())
        assert after_controls["replica_capacity_after"] == before
        assert len({row["max_total_num_tokens"] for row in before.values()}) == 3
        if admission:
            assert len({row["max_running_requests"] for row in before.values()}) == 3
        else:
            assert all("max_running_requests" not in row for row in before.values())
