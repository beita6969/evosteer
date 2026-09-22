from __future__ import annotations

from dataclasses import replace

import pytest

from skillev.runtime.service_topology import InferenceService, ServiceTopology
from skillev.runtime.sglang_gateway import SGLangGateway, SGLangGatewayConfig, SGLangGatewayError
from skillev.runtime.sglang_pool import SGLangActorPool


class Control:
    def __init__(self):
        self.loaded = set()
        self.fail = None
        self.calls = []

    def request(self, *, method, url, payload, **kwargs):
        path = url.rsplit("/", 1)[-1]
        self.calls.append((path, payload))
        if self.fail == path:
            self.fail = None
            return 500, {}
        if path == "load_lora_adapter":
            self.loaded.add(payload["lora_name"])
        if path == "unload_lora_adapter":
            self.loaded.discard(payload["lora_name"])
        if path == "models":
            return 200, {"data": [{"id": v} for v in ["qwen", *sorted(self.loaded)]]}
        if path == "completions":
            return 200, {"choices": [{}]} if payload["model"] in self.loaded else {}
        return 200, {}


def pool_fixture():
    controls = (Control(), Control())
    members = tuple(
        SGLangGateway(
            SGLangGatewayConfig(f"http://localhost:{8000 + i}", "qwen", "actor", control_retries=0),
            _control_transport=c,
        )
        for i, c in enumerate(controls)
    )
    return SGLangActorPool(members), controls


def publish(pool, version):
    swap = pool.prepare_supervisor_adapter(
        adapter_path=f"/adapters/{version}", adapter_revision=version
    )
    return pool.commit_supervisor_adapter(swap)


def test_all_members_ready_before_balanced_sticky_episode_admission():
    pool, _ = pool_fixture()
    with pytest.raises(SGLangGatewayError):
        pool.acquire_episode("early")
    prepared = pool.prepare_supervisor_adapter(adapter_path="/adapters/v1", adapter_revision="v1")
    with pytest.raises(SGLangGatewayError):
        pool.acquire_episode("not-committed")
    pool.commit_supervisor_adapter(prepared)
    routes = [pool.acquire_episode(str(i)) for i in range(4)]
    assert routes == [pool.members[0], pool.members[1], pool.members[0], pool.members[1]]
    assert [v.supervisor_inflight for v in pool.members] == [2, 2]
    with pytest.raises(SGLangGatewayError):
        publish(pool, "v2")
    for i in range(4):
        pool.release_episode(str(i))
    publish(pool, "v2")
    assert pool.acquire_episode("next").adapter_generation.adapter_revision == "v2"
    pool.release_episode("next")


def test_partial_prepare_does_not_release_candidate_or_resample():
    pool, controls = pool_fixture()
    publish(pool, "v1")
    controls[1].fail = "load_lora_adapter"
    with pytest.raises(SGLangGatewayError):
        publish(pool, "v2")
    assert all(v.adapter_generation.adapter_revision == "v1" for v in pool.members)
    assert all(len(v.loaded) == 1 for v in controls)
    with pytest.raises(SGLangGatewayError):
        pool.acquire_episode("blocked")
    # Explicit control recovery, never a /generate retry or base fallback.
    pool.restore_supervisor_adapter(adapter_path="/adapters/v1", adapter_revision="v1")
    pool.acquire_episode("restored")
    pool.release_episode("restored")


def test_partial_commit_stays_closed_until_new_controller_restores_all_members():
    pool, controls = pool_fixture()
    publish(pool, "v1")
    swap = pool.prepare_supervisor_adapter(adapter_path="/adapters/v2", adapter_revision="v2")
    controls[1].fail = "unload_lora_adapter"
    with pytest.raises(SGLangGatewayError):
        pool.commit_supervisor_adapter(swap)
    with pytest.raises(SGLangGatewayError):
        pool.acquire_episode("blocked")
    restored = SGLangActorPool(
        tuple(
            SGLangGateway(v.config, _control_transport=c)
            for v, c in zip(pool.members, controls, strict=True)
        )
    )
    restored.restore_supervisor_adapter(adapter_path="/adapters/v2", adapter_revision="v2")
    assert restored.acquire_episode("resumed").adapter_generation.adapter_revision == "v2"
    restored.release_episode("resumed")


def test_duplicate_episode_and_wrong_transition_rejected():
    pool, _ = pool_fixture()
    publish(pool, "v1")
    pool.acquire_episode("one")
    with pytest.raises(ValueError):
        pool.acquire_episode("one")
    pool.release_episode("one")
    swap = pool.prepare_supervisor_adapter(adapter_path="/adapters/v2", adapter_revision="v2")
    with pytest.raises(ValueError):
        pool.commit_supervisor_adapter(replace(swap))
    pool.rollback_supervisor_adapter(swap)
    assert pool.acquire_episode("next").adapter_generation.adapter_revision == "v1"
    pool.release_episode("next")


def topology():
    return ServiceTopology(
        (
            InferenceService("a", "http://localhost:8000", "GPU-a"),
            InferenceService("b", "http://localhost:8001", "GPU-b"),
            InferenceService("frozen", "http://localhost:8002", "GPU-c"),
        ),
        ("a", "b"),
        ("frozen",),
        ("frozen",),
        ("GPU-d", "GPU-e"),
    )


def test_explicit_five_card_topology_roundtrips_without_double_allocating_frozen_roles():
    config = topology()
    assert ServiceTopology.from_value(config.to_value()) == config
    config.require_device_mapping("GPU-d,GPU-e", 2)
    assert config.members("judge") == config.members("author")
    with pytest.raises(ValueError):
        config.require_device_mapping("GPU-e,GPU-d", 2)


@pytest.mark.parametrize(
    "changes",
    [
        {"gradient_workers": ("GPU-a", "GPU-e")},
        {"actor_pool": ("a", "a")},
        {"judge_pool": ("absent",)},
        {"author_pool": ()},
    ],
)
def test_incompatible_role_placements_fail(changes):
    with pytest.raises(ValueError):
        replace(topology(), **changes)


def test_native_requests_keep_episode_route_and_restore_original_tokens(tmp_path):
    import asyncio

    from skillev.rollout import ExternalSGLangRolloutConfig, ExternalSGLangRolloutGenerator
    from skillev.runtime.request_journal import DurableRequestJournal
    from tests.rollout.test_external_sglang_generator import (
        _request,
        _snapshot,
        _Tokenizer,
        _Transport,
    )

    pool, _ = pool_fixture()
    publish(pool, "v1")
    pool.bind_policy_snapshot(_snapshot().snapshot_id)
    transport = _Transport(
        {
            "output_ids": [10, 11],
            "text": "AB",
            "meta_info": {
                "completion_tokens": 2,
                "prompt_tokens": 3,
                "finish_reason": {"type": "length"},
            },
        }
    )

    def generator():
        return ExternalSGLangRolloutGenerator(
            config=ExternalSGLangRolloutConfig(pool.config.api_root),
            tokenizer=_Tokenizer(),
            gateway=pool,
            snapshot_provider=_snapshot,
            transport=transport,
            request_journal=DurableRequestJournal(tmp_path / "requests.sqlite3"),
        )

    first = generator()
    for episode in ("a", "b"):
        first.begin_episode(episode, _snapshot().snapshot_id)
    for episode in ("b", "a", "b"):
        req = replace(_request(), episode_id=episode, turn_index=1)
        assert asyncio.run(first.generate(req)).content_token_ids == (10, 11)
    assert len(transport.calls) == 2
    assert transport.calls[0][1] == "http://localhost:8001/generate"
    assert transport.calls[1][1] == "http://localhost:8000/generate"
    assert first.physical_usage["server_generated_tokens"] == 4
    for episode in ("a", "b"):
        first.end_episode(episode)
    first.close()
    # Restart with opposite arrival order: saved routes win, no fresh send.
    second = generator()
    for episode in ("b", "a"):
        second.begin_episode(episode, _snapshot().snapshot_id)
        req = replace(_request(), episode_id=episode, turn_index=1)
        assert asyncio.run(second.generate(req)).content_token_ids == (10, 11)
        second.end_episode(episode)
    assert len(transport.calls) == 2
    assert second.physical_usage["server_generated_tokens"] == 0
    second.close()


def test_generic_provider_cannot_bypass_pool_episode_barrier():
    from skillev.runtime.sglang_gateway import SGLangRole

    pool, _ = pool_fixture()
    publish(pool, "v1")
    with pytest.raises(SGLangGatewayError):
        pool._begin_request(SGLangRole.SUPERVISOR)
    pool.bind_policy_snapshot("policy1")
    with pytest.raises(SGLangGatewayError):
        pool.acquire_episode("stale", "policy0")
    assert pool.acquire_episode("valid", "policy1") is pool.members[0]
    pool.release_episode("valid")


def test_new_episode_placement_uses_work_not_only_session_count():
    pool, _ = pool_fixture()
    publish(pool, "v1")
    assert pool.acquire_episode("long", estimated_work=20) is pool.members[0]
    assert pool.acquire_episode("short", estimated_work=2) is pool.members[1]
    assert pool.acquire_episode("next", estimated_work=2) is pool.members[1]
    pool.update_episode_work("long", 1)
    assert pool.acquire_episode("later") is pool.members[0]
    # Existing long session never moves when its estimate changes.
    assert pool._sessions["long"] is pool.members[0]
    for episode in ("long", "short", "next", "later"):
        pool.release_episode(episode)
