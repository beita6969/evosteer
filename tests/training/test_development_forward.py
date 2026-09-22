"""Synthetic metadata/control/response fixtures, not real Qwen warmup evidence."""

import asyncio
import json
from dataclasses import replace

import pytest
from skillev_private.experiments import development_collection as dev
from skillev_private.experiments import development_forward as forward
from skillev_private.experiments.protocol_v13_training_debug import _PREPARATION_FORMAT
from skillev_private.experiments.warmup_initialization import WARMUP_PREPARATION_FORMAT

from skillev.contracts import canonical_json
from skillev.policy import PrivateInitialCheckpointBinding, QwenMultimodalBackboneConfig
from skillev.policy.checkpoint import POLICY_STATE_FILE, PolicyCheckpointState
from skillev.policy.versions import TrainableVersions
from skillev.rollout import (
    ExternalSGLangRolloutConfig,
    ExternalSGLangRolloutGenerator,
    PolicySnapshot,
)
from skillev.runtime.request_journal import DurableRequestJournal, UnknownRequestOutcomeError
from skillev.runtime.sglang_gateway import SGLangGateway, SGLangGatewayError
from tests.rollout.test_external_sglang_generator import _request, _snapshot, _Tokenizer, _Transport
from tests.training.test_development_collection import request_fixture, write


def initialization_inputs(tmp_path, request, backbone, config, kind, sampling):
    config = QwenMultimodalBackboneConfig.from_value(config.to_value())
    warmup = kind == "skill-use-warmup"
    versions = TrainableVersions.from_backbone(backbone)
    version = "warmup-initialization/synthetic@0" if warmup else versions.forward
    state = PolicyCheckpointState(
        backbone_id=backbone.backbone_id,
        forward_version=version,
        backward_version=versions.backward,
        z_version=backbone.z_version,
        optimizer_step=0,
        trainable_state=backbone.trainable_state_identity,
    )
    checkpoint = tmp_path / "metadata-only-synthetic-policy"
    checkpoint.mkdir()
    (checkpoint / POLICY_STATE_FILE).write_text(canonical_json(state.to_value()))
    library = json.loads(open(request["initial_library"]).read())
    declaration = {
        "format": WARMUP_PREPARATION_FORMAT if warmup else _PREPARATION_FORMAT,
        "backbone": config.to_value(),
        "initial_checkpoint": PrivateInitialCheckpointBinding(
            str(checkpoint), state.trainable_state
        ).to_value(),
    }
    if warmup:
        declaration["initialization"] = {
            "kind": "skill-use-warmup@1",
            "warmup_updates": 2,
            "original_step_zero": False,
            "candidate": {"library": library, "sampling": sampling},
        }
    policy = PolicySnapshot.create(
        backbone_id=state.backbone_id,
        forward_adapter_version=state.forward_version,
        tokenizer_id=config.tokenizer_id,
        backend_id="sglang-native-exact-token",
        initial_trainable_state_hash=state.trainable_state.content_hash,
    )
    request.pop("step0_policy")
    request["forward_initialization"] = {
        "format": forward.FORMAT,
        "kind": kind,
        "preparation": write(tmp_path / "preparation.json", declaration),
        "policy": write(tmp_path / "forward-policy.json", policy.to_value()),
        "published_forward_adapter": "synthetic-published-forward",
    }
    return policy, config


@pytest.mark.parametrize("kind", ["fresh-forward", "skill-use-warmup"])
def test_saved_metadata_pins_policy_and_start_kind_before_calls(
    tmp_path, training_backbone, training_backbone_config, kind
):
    request, _, _ = request_fixture(tmp_path)
    policy, config = initialization_inputs(
        tmp_path, request, training_backbone, training_backbone_config, kind, {"synthetic": 7}
    )
    records, frozen = dev.selection(request)
    actual, declared = forward.resolve_forward(request)
    assert actual == policy
    assert declared["original_step_zero"] is (kind == "fresh-forward")
    assert not declared["model_loaded_by_collector"]
    forward.require_forward_candidate(
        declared,
        backbone=config.to_value(),
        sampling={"synthetic": 7},
        library=frozen["initial_library"],
    )
    assert dev.aggregate(tmp_path, frozen, elapsed_seconds=0)["forward_initialization"] == declared
    for change in ("kind", "policy", "pool"):
        altered = {**request, "forward_initialization": dict(request["forward_initialization"])}
        if change == "kind":
            altered["forward_initialization"]["kind"] = (
                "fresh-forward" if kind == "skill-use-warmup" else "skill-use-warmup"
            )
        elif change == "policy":
            changed = PolicySnapshot.create(
                backbone_id=policy.backbone_id,
                forward_adapter_version="other-version",
                tokenizer_id=policy.tokenizer_id,
                backend_id=policy.backend_id,
                initial_trainable_state_hash=policy.initial_trainable_state_hash,
            )
            altered["forward_initialization"]["policy"] = write(
                tmp_path / "bad-policy.json", changed.to_value()
            )
        else:
            altered["actor_endpoints"] = ["http://localhost:9"]
        with pytest.raises(ValueError):
            dev.selection(altered)
    if kind == "skill-use-warmup":
        for field in ("sampling", "library"):
            kwargs = {
                "backbone": config.to_value(),
                "sampling": {"synthetic": 7},
                "library": frozen["initial_library"],
            }
            kwargs[field] = {"changed": True}
            with pytest.raises(ValueError):
                forward.require_forward_candidate(declared, **kwargs)
    assert dev.selection(request)[0] == records


@pytest.mark.parametrize("failure", [None, "missing-adapter", "unknown-dispatch"])
def test_bound_generator_uses_real_lora_payload_and_exact_journal_without_fallback(
    tmp_path, monkeypatch, failure
):
    controls = []

    class Control:
        def request(self, **kwargs):
            controls.append(kwargs)
            assert kwargs["method"] == "GET"
            if kwargs["url"].endswith("/health"):
                return 200, {}
            return 200, {
                "data": [
                    {"id": x}
                    for x in (["base"] if failure == "missing-adapter" else ["base", "actual-F"])
                ]
            }

    def gateway(config):
        assert config.control_retries == 0
        return SGLangGateway(config, _control_transport=Control())

    monkeypatch.setattr(forward, "SGLangGateway", gateway)
    policy = _snapshot()
    declaration = {"published_forward_adapter": "actual-F"}
    if failure == "missing-adapter":
        with pytest.raises(SGLangGatewayError):
            forward.bind_forward_gateway(
                declaration, endpoint="http://localhost:9", base_model="base", policy=policy
            )
        assert not list(tmp_path.iterdir())
        return
    bound = forward.bind_forward_gateway(
        declaration, endpoint="http://localhost:9", base_model="base", policy=policy
    )
    payload = {
        "meta_info": {
            "completion_tokens": 2,
            "prompt_tokens": 3,
            "finish_reason": {"type": "length"},
        },
        "output_ids": [10, 11],
        "text": "AB",
    }
    transport = _Transport(payload)
    if failure == "unknown-dispatch":

        class FailedTransport:
            def __init__(self):
                self.calls = []

            def request(self, **kwargs):
                self.calls.append(kwargs)
                raise TimeoutError("synthetic unknown dispatch")

        transport = FailedTransport()
    original = replace(
        _request(),
        action_boundary_version="native-model-stop@1",
        episode_id="synthetic-readonly",
        turn_index=1,
        library_version="synthetic-library",
    )
    journal = DurableRequestJournal(tmp_path / "evaluation.sqlite3")
    for recovered in (False, True):
        generator = ExternalSGLangRolloutGenerator(
            config=ExternalSGLangRolloutConfig("http://localhost:9"),
            tokenizer=_Tokenizer(),
            gateway=bound,
            snapshot_provider=lambda: forward.fixed_snapshot(bound, policy),
            transport=transport,
            request_journal=journal,
        )
        try:
            if failure:
                with pytest.raises(UnknownRequestOutcomeError if recovered else TimeoutError):
                    asyncio.run(generator.generate(original))
            else:
                result = asyncio.run(generator.generate(original))
                assert result.content_token_ids == (10, 11)
        finally:
            generator.close()
    assert len(transport.calls) == 1
    if failure is None:
        posted = transport.calls[0][2]
        assert posted["lora_path"] == "actual-F"
        assert posted["input_ids"] == list(original.input_ids)
        assert posted["sampling_params"]["sampling_seed"] == original.seed
    bound._generation = replace(bound.adapter_generation, adapter_revision="changed-F")
    with pytest.raises(ValueError):
        forward.fixed_snapshot(bound, policy)
    assert len(controls) == 2  # no publication or control retries
