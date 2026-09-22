"""Real journal/external generator with synthetic responses, never model HTTP."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from skillev_private.experiments.readonly_replicas import (
    ReadonlyReplicaGenerator,
    replica_declaration,
)

from skillev.diagnostics.rollout_progress import RolloutProgress, bind_progress
from skillev.rollout import ExternalSGLangRolloutConfig, ExternalSGLangRolloutGenerator
from skillev.runtime.request_journal import DurableRequestJournal, UnknownRequestOutcomeError
from tests.rollout.test_external_sglang_generator import _request, _snapshot, _Tokenizer, _Transport
from tests.training.test_development_collection import replace_policy

ENDPOINTS = tuple(f"http://localhost:{port}" for port in (19001, 19002, 19003))


def pool(path, *, broken=None):
    policy = replace_policy(_snapshot())
    journal = DurableRequestJournal(path)
    transports = [
        _Transport(
            {
                "meta_info": {
                    "completion_tokens": 2,
                    "prompt_tokens": 3,
                    "finish_reason": {"type": "length"},
                },
                "output_ids": [10, 11],
                "text": "AB",
            }
        )
        for _ in ENDPOINTS
    ]
    if broken is not None:

        class Failed:
            def __init__(self):
                self.calls = []

            def request(self, **kwargs):
                self.calls.append(kwargs)
                raise TimeoutError("synthetic unknown dispatch")

        transports[broken] = Failed()
    members = tuple(
        ExternalSGLangRolloutGenerator(
            config=ExternalSGLangRolloutConfig(endpoint),
            tokenizer=_Tokenizer(),
            gateway=None,
            snapshot_provider=lambda: policy,
            transport=transport,
            request_journal=journal,
        )
        for endpoint, transport in zip(ENDPOINTS, transports, strict=True)
    )
    return ReadonlyReplicaGenerator(members, journal), transports, policy


def request(policy, episode, turn=1):
    return replace(
        _request(),
        expected_policy_snapshot_id=policy.snapshot_id,
        episode_id=episode,
        turn_index=turn,
        library_version="synthetic-library",
        seed=147,
        action_boundary_version="native-model-stop@1",
    )


def test_same_position_across_arms_sticky_and_original_sampling_request(tmp_path):
    routes = []
    for arm in ("off", "on"):
        generator, transports, policy = pool(tmp_path / f"{arm}.sqlite3")
        episode = f"{arm}/original-trajectory"
        try:
            with bind_progress(RolloutProgress(canonical_position=4)):
                generator.begin_episode(episode, policy.snapshot_id)
            first = request(policy, episode)
            with bind_progress(RolloutProgress(canonical_position=8)):
                # Turn progress cannot reroute an already bound episode.
                result = asyncio.run(generator.generate(first))
                asyncio.run(generator.generate(request(policy, episode, 2)))
                routes.append(generator.execution_endpoint(first))
            generator.end_episode(episode)
            assert result.content_token_ids == (10, 11)
            assert [len(t.calls) for t in transports] == [0, 2, 0]
            payload = transports[1].calls[0][2]
            assert payload["input_ids"] == list(first.input_ids)
            assert payload["sampling_params"]["sampling_seed"] == first.seed
            assert "lora_path" not in payload
            assert generator.journal.episode_route(episode, policy.snapshot_id) == ENDPOINTS[1]
            assert generator.physical_usage["server_generated_tokens"] == 4
        finally:
            generator.close()
    assert routes == [ENDPOINTS[1], ENDPOINTS[1]]


def test_new_process_recovers_same_exact_result_route_and_rejects_reassignment(tmp_path):
    path = tmp_path / "requests.sqlite3"
    generator, transports, policy = pool(path)
    original = request(policy, "same-episode")
    with bind_progress(RolloutProgress(canonical_position=2)):
        generator.begin_episode(original.episode_id, policy.snapshot_id)
    result = asyncio.run(generator.generate(original))
    generator.end_episode(original.episode_id)
    generator.close()
    resumed, transports, _ = pool(path)
    try:
        with bind_progress(RolloutProgress(canonical_position=2)):
            resumed.begin_episode(original.episode_id, policy.snapshot_id)
        assert asyncio.run(resumed.generate(original)) == result
        assert all(not t.calls for t in transports)
        resumed.end_episode(original.episode_id)
        with bind_progress(RolloutProgress(canonical_position=0)), pytest.raises(ValueError):
            resumed.begin_episode(original.episode_id, policy.snapshot_id)
        with pytest.raises(ValueError):
            resumed.journal.episode_route(original.episode_id, "different-policy")
    finally:
        resumed.close()


def test_dispatch_failure_never_changes_replica_or_retries_unknown(tmp_path):
    path = tmp_path / "requests.sqlite3"
    generator, transports, policy = pool(path, broken=1)
    original = request(policy, "unknown")
    with bind_progress(RolloutProgress(canonical_position=1)):
        generator.begin_episode(original.episode_id, policy.snapshot_id)
    try:
        with pytest.raises(TimeoutError):
            asyncio.run(generator.generate(original))
        assert [len(t.calls) for t in transports] == [0, 1, 0]
    finally:
        generator.end_episode(original.episode_id)
        generator.close()
    resumed, transports, _ = pool(path)
    try:
        with bind_progress(RolloutProgress(canonical_position=1)):
            resumed.begin_episode(original.episode_id, policy.snapshot_id)
        with pytest.raises(UnknownRequestOutcomeError):
            asyncio.run(resumed.generate(original))
        assert all(not t.calls for t in transports)
        resumed.end_episode(original.episode_id)
    finally:
        resumed.close()


def test_explicit_pool_validation_and_close_every_replica_even_on_error(tmp_path):
    assert replica_declaration({}) == {}
    for endpoints in ([], "http://localhost:3", [ENDPOINTS[0], ENDPOINTS[0] + "/v1"]):
        with pytest.raises(ValueError):
            replica_declaration({"actor_endpoints": endpoints})
    generator, _, policy = pool(tmp_path / "requests.sqlite3")
    try:
        with pytest.raises(ValueError):
            generator.begin_episode("no-position", policy.snapshot_id)
        original = generator.members[0].gateway
        generator.members[0].gateway = object()
        with pytest.raises(ValueError):
            ReadonlyReplicaGenerator(generator.members, generator.journal)
        generator.members[0].gateway = original
    finally:
        generator.close()
    closed = []

    def close(index):
        closed.append(index)
        if index == 1:
            raise ValueError("synthetic cleanup")

    generator.members = tuple(SimpleNamespace(close=lambda index=i: close(index)) for i in range(3))
    with pytest.raises(ValueError):
        generator.close()
    assert sorted(closed) == [0, 1, 2]
