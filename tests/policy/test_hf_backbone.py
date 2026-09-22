from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from transformers import (
    PreTrainedTokenizerFast,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)

import skillev.policy.hf_backbone as hf_backbone_module
from skillev.contracts import canonical_json
from skillev.policy import (
    AUTHORING_JSON_ROOT_BOUNDARY_VERSION,
    AdapterRole,
    AuthoringGenerationRequest,
    PolicyGenerationRequest,
    QwenBackboneConfig,
    QwenPolicyBackbone,
    TrainableStateIdentity,
    qwen_tokenizer_artifact_identity,
)
from skillev.policy.checkpoint import inspect_policy_checkpoint
from skillev.rollout import LocalPolicyGenerator

PREFIX_IDS = (1, 4, 5)
ACTION_IDS = (6, 7)
QUERY_IDS = (1, 5)


def _groups(backbone: QwenPolicyBackbone) -> dict[str, tuple[nn.Parameter, ...]]:
    groups = backbone.parameter_groups()
    return {
        "forward": groups.forward,
        "backward": groups.backward,
        "z": groups.z_head,
    }


def _role_parameters(
    backbone: QwenPolicyBackbone,
    role: AdapterRole,
) -> tuple[nn.Parameter, ...]:
    groups = backbone.parameter_groups()
    return groups.forward if role is AdapterRole.FORWARD_POLICY else groups.backward


def _clear_gradients(backbone: QwenPolicyBackbone) -> None:
    for parameters in _groups(backbone).values():
        for parameter in parameters:
            parameter.grad = None


def _base_parameters(backbone: QwenPolicyBackbone) -> tuple[nn.Parameter, ...]:
    trainable = {
        id(parameter) for parameters in _groups(backbone).values() for parameter in parameters
    }
    return tuple(
        parameter for parameter in backbone._model.parameters() if id(parameter) not in trainable
    )


def _manual_token_logprobs(
    backbone: QwenPolicyBackbone,
    role: AdapterRole,
) -> torch.Tensor:
    full = PREFIX_IDS + ACTION_IDS
    backbone._activate_adapter(role)
    inputs = torch.tensor([full], dtype=torch.long)
    outputs = backbone._model(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits[
        0,
        len(PREFIX_IDS) - 1 : len(PREFIX_IDS) + len(ACTION_IDS) - 1,
        :,
    ]
    targets = torch.tensor(ACTION_IDS, dtype=torch.long).unsqueeze(1)
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, targets).squeeze(1)


@pytest.fixture(scope="session")
def tiny_qwen35_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model_path = tmp_path_factory.mktemp("tiny-qwen35-selective-logits")
    vocabulary = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        **{f"token-{index}": index + 4 for index in range(28)},
    }
    tokenizer_backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} "
        "{{ message['content'] }}{% endfor %}"
        "{% if add_generation_prompt %} assistant{% endif %}"
    )
    tokenizer.save_pretrained(model_path)
    config = Qwen3_5TextConfig(
        vocab_size=len(vocabulary),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng():
        torch.manual_seed(20260810)
        Qwen3_5ForCausalLM(config).save_pretrained(model_path)
    return model_path


@pytest.fixture
def qwen35_selective_config(tiny_qwen35_model_path: Path) -> QwenBackboneConfig:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tiny_qwen35_model_path)
    return QwenBackboneConfig(
        base_model_path=str(tiny_qwen35_model_path),
        revision="local-pinned",
        tokenizer_id="tiny-qwen35-tokenizer@1",
        tokenizer_content_hash=qwen_tokenizer_artifact_identity(
            tokenizer=tokenizer,
            tokenizer_id="tiny-qwen35-tokenizer@1",
            revision="local-pinned",
        ).content_hash,
        hidden_size=16,
        eos_token_ids=(2,),
        torch_dtype="float32",
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_target_modules=("q_proj", "v_proj"),
        z_hidden_width=8,
        device="cpu",
    )


@pytest.fixture
def qwen35_selective_backbone(
    qwen35_selective_config: QwenBackboneConfig,
) -> QwenPolicyBackbone:
    return QwenPolicyBackbone(qwen35_selective_config)


def test_backbone_loads_tokenizer_from_distinct_private_root(
    tmp_path: Path,
    tiny_qwen35_model_path: Path,
    qwen35_selective_config: QwenBackboneConfig,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    for filename in ("config.json", "model.safetensors"):
        shutil.copy2(tiny_qwen35_model_path / filename, model_path / filename)

    backbone = QwenPolicyBackbone(
        replace(
            qwen35_selective_config,
            base_model_path=str(model_path),
            tokenizer_path=str(tiny_qwen35_model_path),
        )
    )

    assert backbone.tokenizer.encode("token-1")


def _replace_adapter(
    backbone: QwenPolicyBackbone,
    role: AdapterRole,
    *,
    scale: float,
) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(_role_parameters(backbone, role), start=1):
            parameter.fill_(scale * index)


@pytest.mark.parametrize("role", tuple(AdapterRole))
def test_score_matches_manual_decoder_fence(
    backbone: QwenPolicyBackbone,
    role: AdapterRole,
) -> None:
    actual = backbone.score(PREFIX_IDS, ACTION_IDS, role)
    expected = _manual_token_logprobs(backbone, role)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual.shape == (len(ACTION_IDS),)
    assert actual.dtype is torch.float32
    assert actual.requires_grad


@pytest.mark.parametrize("role", tuple(AdapterRole))
def test_qwen35_selective_action_logits_match_full_logits_and_gradients(
    qwen35_selective_backbone: QwenPolicyBackbone,
    role: AdapterRole,
) -> None:
    backbone = qwen35_selective_backbone
    _clear_gradients(backbone)
    expected = _manual_token_logprobs(backbone, role)
    expected.sum().backward()
    expected_gradients = tuple(
        parameter.grad.detach().clone() for parameter in _role_parameters(backbone, role)
    )

    _clear_gradients(backbone)
    actual = backbone.score(PREFIX_IDS, ACTION_IDS, role)
    actual.sum().backward()
    actual_gradients = tuple(
        parameter.grad.detach().clone() for parameter in _role_parameters(backbone, role)
    )

    torch.testing.assert_close(actual, expected.detach(), rtol=1e-6, atol=1e-6)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=1e-6,
            atol=1e-7,
        )


def test_qwen35_episode_cache_matches_stateless_generation_and_reuses_prefix(
    qwen35_selective_backbone: QwenPolicyBackbone,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = qwen35_selective_backbone
    long_prefix = (1, *((4, 5, 6, 7) * 18))
    first_request = PolicyGenerationRequest(
        input_ids=long_prefix,
        max_new_tokens=3,
        seed=20260810,
        decoding_snapshot_id="cache-equivalence@1",
    )
    stateless_first = backbone.generate_policy(first_request)
    second_input = (*long_prefix, *stateless_first.content_token_ids, 8, 9)
    second_request = PolicyGenerationRequest(
        input_ids=second_input,
        max_new_tokens=3,
        seed=20260811,
        decoding_snapshot_id="cache-equivalence@1",
    )
    original_sampler = backbone._sample_policy_token
    stateless_logits: list[torch.Tensor] = []

    def capture_stateless(
        logits: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        stateless_logits.append(logits.detach().clone())
        return original_sampler(logits, generator=generator)

    monkeypatch.setattr(backbone, "_sample_policy_token", capture_stateless)
    stateless_second = backbone.generate_policy(second_request)
    monkeypatch.setattr(backbone, "_sample_policy_token", original_sampler)

    backbone.begin_policy_episode("episode-cache-equivalence")
    cached_first = backbone.generate_policy(first_request)
    cached_logits: list[torch.Tensor] = []

    def capture_cached(
        logits: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        cached_logits.append(logits.detach().clone())
        return original_sampler(logits, generator=generator)

    monkeypatch.setattr(backbone, "_sample_policy_token", capture_cached)
    cached_second = backbone.generate_policy(second_request)
    reused = backbone._last_policy_reused_token_count
    prefill = backbone._last_policy_prefill_token_count
    backbone.end_policy_episode("episode-cache-equivalence")

    assert cached_first == stateless_first
    assert cached_second == stateless_second
    assert len(cached_logits) == len(stateless_logits)
    for actual, expected in zip(cached_logits, stateless_logits, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert reused > 0
    assert reused + prefill == len(second_input)


def test_qwen35_episode_cache_crops_a_changed_bpe_boundary_exactly(
    qwen35_selective_backbone: QwenPolicyBackbone,
) -> None:
    backbone = qwen35_selective_backbone
    first_input = (1, *((4, 5, 6, 7) * 18))
    first_request = PolicyGenerationRequest(
        input_ids=first_input,
        max_new_tokens=2,
        seed=20260810,
        decoding_snapshot_id="cache-prefix@1",
    )
    boundary_changed_request = PolicyGenerationRequest(
        input_ids=(*first_input[:20], *((11, 12) * 28)),
        max_new_tokens=2,
        seed=20260811,
        decoding_snapshot_id="cache-prefix@1",
    )
    expected = backbone.generate_policy(boundary_changed_request)

    backbone.begin_policy_episode("episode-prefix-contract")
    backbone.generate_policy(first_request)
    actual = backbone.generate_policy(boundary_changed_request)

    assert actual == expected
    assert backbone._last_policy_reused_token_count == len(first_input) - 64


def test_qwen35_episode_cache_rejects_a_disjoint_prompt_without_fallback(
    qwen35_selective_backbone: QwenPolicyBackbone,
) -> None:
    backbone = qwen35_selective_backbone
    first_input = (1, *((4, 5, 6, 7) * 18))
    request = PolicyGenerationRequest(
        input_ids=first_input,
        max_new_tokens=2,
        seed=20260810,
        decoding_snapshot_id="cache-prefix@1",
    )
    backbone.begin_policy_episode("episode-disjoint-prefix")
    backbone.generate_policy(request)

    with pytest.raises(RuntimeError, match="canonical prompt prefix"):
        backbone.generate_policy(
            PolicyGenerationRequest(
                input_ids=(9, 10, 11, 12),
                max_new_tokens=2,
                seed=20260811,
                decoding_snapshot_id="cache-prefix@1",
            )
        )


@pytest.mark.parametrize("checkpoint_min_tokens", [1, len(PREFIX_IDS) + len(ACTION_IDS) + 1])
def test_qwen35_nonreentrant_checkpointed_scoring_preserves_three_way_gradients(
    qwen35_selective_config: QwenBackboneConfig,
    checkpoint_min_tokens: int,
) -> None:
    from skillev.policy.scoring_execution import TeacherForcingConfig
    from skillev.training.performance_config import TrainingPerformanceConfig

    reference = QwenPolicyBackbone(qwen35_selective_config)
    backbone = QwenPolicyBackbone(
        replace(qwen35_selective_config, teacher_forced_gradient_checkpointing=True),
        performance=TrainingPerformanceConfig(
            teacher_forcing=TeacherForcingConfig(checkpoint_min_tokens=checkpoint_min_tokens)
        ),
    )
    with torch.no_grad():
        for expected, actual in zip(
            (
                *reference.parameter_groups().forward,
                *reference.parameter_groups().backward,
                *reference.parameter_groups().z_head,
            ),
            (
                *backbone.parameter_groups().forward,
                *backbone.parameter_groups().backward,
                *backbone.parameter_groups().z_head,
            ),
            strict=True,
        ):
            actual.copy_(expected)

    for role in AdapterRole:
        _clear_gradients(reference)
        _clear_gradients(backbone)
        expected_score = reference.score(PREFIX_IDS, ACTION_IDS, role).mean()
        actual_score = backbone.score(PREFIX_IDS, ACTION_IDS, role).mean()
        expected_score.backward()
        actual_score.backward()
        assert backbone.drain_scoring_metrics()[0]["checkpointed"] is (
            len(PREFIX_IDS) + len(ACTION_IDS) >= checkpoint_min_tokens
        )
        torch.testing.assert_close(actual_score, expected_score, rtol=1e-6, atol=1e-7)
        for actual, expected in zip(
            _role_parameters(backbone, role),
            _role_parameters(reference, role),
            strict=True,
        ):
            assert actual.grad is not None
            assert expected.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)

    _clear_gradients(reference)
    _clear_gradients(backbone)
    expected_z = reference.z_value(QUERY_IDS)
    actual_z = backbone.z_value(QUERY_IDS)
    torch.testing.assert_close(actual_z, expected_z, rtol=1e-6, atol=1e-7)
    expected_z.backward()
    actual_z.backward()
    for actual, expected in zip(
        backbone.parameter_groups().z_head,
        reference.parameter_groups().z_head,
        strict=True,
    ):
        assert actual.grad is not None
        assert expected.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)
    assert backbone._model.is_gradient_checkpointing


def test_long_checkpointed_scoring_offloads_saved_tensors_without_changing_gradients(
    qwen35_selective_config: QwenBackboneConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class RecordingContext:
        def __enter__(self) -> None:
            events.append("enter")

        def __exit__(self, *exc_info: object) -> None:
            events.append(("exit", exc_info[0]))

    from skillev.policy.scoring_execution import TeacherForcingConfig
    from skillev.training.performance_config import TrainingPerformanceConfig

    backbone = QwenPolicyBackbone(
        replace(qwen35_selective_config, teacher_forced_gradient_checkpointing=True),
        performance=TrainingPerformanceConfig(
            teacher_forcing=TeacherForcingConfig(
                offload_min_tokens=len(PREFIX_IDS) + len(ACTION_IDS)
            )
        ),
    )
    monkeypatch.setattr(backbone._teacher_forcing.offload, "context", RecordingContext)

    score = backbone.score(PREFIX_IDS, ACTION_IDS, AdapterRole.FORWARD_POLICY).mean()
    score.backward()

    assert events == ["enter", ("exit", None)]
    assert backbone.drain_scoring_metrics()[0]["offloaded"] is True
    assert all(parameter.grad is not None for parameter in backbone.parameter_groups().forward)


def test_qwen35_cuda_checkpointing_requires_pinned_fla_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def chunk_kernel() -> None:
        pass

    chunk_kernel.__module__ = "fla.ops.gated_delta_rule.chunk"
    delta_type = type(
        "Qwen3_5GatedDeltaNet",
        (torch.nn.Module,),
        {
            "__init__": lambda self: (
                torch.nn.Module.__init__(self),
                setattr(self, "chunk_gated_delta_rule", chunk_kernel),
            )[-1]
        },
    )
    model = torch.nn.Sequential(delta_type())
    monkeypatch.setattr(
        hf_backbone_module.importlib.metadata,
        "version",
        lambda package: (
            hf_backbone_module.QWEN35_GATED_DELTA_KERNEL_VERSION
            if package == hf_backbone_module.QWEN35_GATED_DELTA_KERNEL_PACKAGE
            else pytest.fail(f"unexpected package: {package}")
        ),
    )

    hf_backbone_module.require_qwen35_gated_delta_kernel(model)

    model[0].chunk_gated_delta_rule.__module__ = "transformers.integrations.hub_kernels"
    with pytest.raises(RuntimeError, match="did not select"):
        hf_backbone_module.require_qwen35_gated_delta_kernel(model)


def test_score_is_deterministic_and_role_gradients_are_isolated(
    backbone: QwenPolicyBackbone,
) -> None:
    _clear_gradients(backbone)
    first = backbone.score(PREFIX_IDS, ACTION_IDS, AdapterRole.FORWARD_POLICY)
    second = backbone.score(PREFIX_IDS, ACTION_IDS, AdapterRole.FORWARD_POLICY)
    assert torch.equal(first, second)

    first.sum().backward()
    assert all(
        parameter.grad is not None
        for parameter in _role_parameters(backbone, AdapterRole.FORWARD_POLICY)
    )
    assert all(
        parameter.grad is None
        for parameter in _role_parameters(backbone, AdapterRole.BACKWARD_POLICY)
    )
    assert all(parameter.grad is None for parameter in backbone.parameter_groups().z_head)
    assert all(parameter.grad is None for parameter in _base_parameters(backbone))


def test_parameter_groups_are_fixed_nonempty_and_disjoint(
    backbone: QwenPolicyBackbone,
) -> None:
    first = backbone.parameter_groups()
    second = backbone.parameter_groups()
    identities = [
        {id(parameter) for parameter in first.forward},
        {id(parameter) for parameter in first.backward},
        {id(parameter) for parameter in first.z_head},
    ]

    assert first == second
    assert all(identities)
    assert identities[0].isdisjoint(identities[1])
    assert identities[0].isdisjoint(identities[2])
    assert identities[1].isdisjoint(identities[2])


def test_backbone_id_is_static_while_versions_track_policy_updates(
    backbone: QwenPolicyBackbone,
) -> None:
    identity = backbone.backbone_id
    versions_before = {role: backbone.adapter_version(role) for role in AdapterRole}
    _replace_adapter(backbone, AdapterRole.FORWARD_POLICY, scale=0.03)

    assert backbone.backbone_id == identity
    backbone.mark_policy_update(7)
    assert backbone.backbone_id == identity
    for role, before in versions_before.items():
        assert backbone.adapter_version(role) == f"{before.rpartition('@')[0]}@7"
    assert backbone.z_version.endswith("@7")


def test_z_uses_query_only_frozen_features_and_gradients_reach_only_z(
    backbone: QwenPolicyBackbone,
) -> None:
    _clear_gradients(backbone)
    value_before = backbone.z_value(QUERY_IDS)
    _replace_adapter(backbone, AdapterRole.FORWARD_POLICY, scale=0.03)
    _replace_adapter(backbone, AdapterRole.BACKWARD_POLICY, scale=-0.02)
    value_after = backbone.z_value(QUERY_IDS)

    assert torch.equal(value_before.detach(), value_after.detach())
    value_after.backward()
    assert all(parameter.grad is not None for parameter in backbone.parameter_groups().z_head)
    assert all(
        parameter.grad is None
        for role in AdapterRole
        for parameter in _role_parameters(backbone, role)
    )
    assert all(parameter.grad is None for parameter in _base_parameters(backbone))


def test_reset_z_is_seeded_deterministic_and_has_no_unseeded_entry(
    backbone: QwenPolicyBackbone,
) -> None:
    first_version = backbone.reset_z(20260725)
    first = {
        name: tensor.detach().clone() for name, tensor in backbone._z_head.state_dict().items()
    }
    second_version = backbone.reset_z(20260725)
    second = backbone._z_head.state_dict()

    assert first_version == second_version == f"z-reset-{20260725:016x}@0"
    assert all(torch.equal(first[name], tensor) for name, tensor in second.items())

    backbone.reset_z(20260726)
    assert any(
        not torch.equal(first[name], tensor)
        for name, tensor in backbone._z_head.state_dict().items()
    )
    with pytest.raises(TypeError):
        backbone.reset_z()  # type: ignore[call-arg]


def test_generation_is_locally_seeded_and_raw_policy_sampling_is_reproducible(
    backbone: QwenPolicyBackbone,
) -> None:
    request = PolicyGenerationRequest(
        input_ids=PREFIX_IDS,
        max_new_tokens=4,
        seed=17,
        decoding_snapshot_id="decoding@3",
    )
    rng_before = torch.random.get_rng_state().clone()

    first = backbone.generate_policy(request)
    second = backbone.generate_policy(request)

    assert first == second
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_base_generation_is_adapter_free(
    backbone: QwenPolicyBackbone,
) -> None:
    request = AuthoringGenerationRequest(
        input_ids=PREFIX_IDS,
        max_new_tokens=2,
        temperature=0.8,
        top_p=0.9,
        seed=19,
        template_version="authoring@3",
        completion_boundary_version=AUTHORING_JSON_ROOT_BOUNDARY_VERSION,
    )
    before = backbone.generate_base(request)
    _replace_adapter(backbone, AdapterRole.FORWARD_POLICY, scale=0.05)
    _replace_adapter(backbone, AdapterRole.BACKWARD_POLICY, scale=-0.05)
    after = backbone.generate_base(request)

    assert before == after


def test_base_generation_stops_at_the_first_complete_json_root(
    backbone: QwenPolicyBackbone,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BoundaryTokenizer:
        tokenizer_id = "boundary-tokenizer@1"

        @staticmethod
        def decode(token_ids: tuple[int, ...]) -> str:
            return "".join({10: "{", 11: "}", 12: "x"}[item] for item in token_ids)

    sampled: list[int] = []
    sequence = iter((10, 11, 12))

    def sample(*_: object, **__: object) -> torch.Tensor:
        token = next(sequence)
        sampled.append(token)
        return torch.tensor([token], dtype=torch.long)

    monkeypatch.setattr(backbone, "_tokenizer", _BoundaryTokenizer())
    monkeypatch.setattr(backbone, "_sample_authoring_token", sample)

    result = backbone.generate_base(
        AuthoringGenerationRequest(
            input_ids=PREFIX_IDS,
            max_new_tokens=3,
            temperature=0.8,
            top_p=0.9,
            seed=19,
            template_version="authoring@3",
            completion_boundary_version=AUTHORING_JSON_ROOT_BOUNDARY_VERSION,
        )
    )

    assert result.content_token_ids == (10, 11)
    assert result.stop_token_ids == ()
    assert result.finish_reason == "json-root"
    assert sampled == [10, 11]


def _assert_parameters_equal(
    expected: Iterable[nn.Parameter],
    actual: Iterable[nn.Parameter],
) -> None:
    for left, right in zip(expected, actual, strict=True):
        assert torch.equal(left.detach(), right.detach())


def test_checkpoint_has_one_v3_manifest_and_round_trips_exact_state(
    backbone: QwenPolicyBackbone,
    backbone_config: QwenBackboneConfig,
    tmp_path: Path,
) -> None:
    _replace_adapter(backbone, AdapterRole.FORWARD_POLICY, scale=0.03)
    _replace_adapter(backbone, AdapterRole.BACKWARD_POLICY, scale=-0.02)
    backbone.reset_z(23)
    backbone.mark_policy_update(5)
    checkpoint = tmp_path / "policy"
    backbone.save_checkpoint(str(checkpoint))

    inspection = inspect_policy_checkpoint(checkpoint)
    assert inspection.state.trainable_state == backbone.trainable_state_identity
    assert inspection.forward_tensor_names
    assert inspection.backward_tensor_names
    assert inspection.z_tensor_names

    assert (checkpoint / "policy_state.json").is_file()
    assert (checkpoint / "forward_adapter" / "adapter_model.safetensors").is_file()
    assert (checkpoint / "backward_adapter" / "adapter_model.safetensors").is_file()
    assert (checkpoint / "forward_adapter" / "adapter_config.json").is_file()
    assert (checkpoint / "backward_adapter" / "adapter_config.json").is_file()
    assert (checkpoint / "z_head.pt").is_file()
    assert not any("fingerprint" in path.name for path in checkpoint.rglob("*"))

    restored = QwenPolicyBackbone(backbone_config)
    restored.load_checkpoint(str(checkpoint))
    _assert_parameters_equal(
        backbone.parameter_groups().forward,
        restored.parameter_groups().forward,
    )
    _assert_parameters_equal(
        backbone.parameter_groups().backward,
        restored.parameter_groups().backward,
    )
    _assert_parameters_equal(
        backbone.parameter_groups().z_head,
        restored.parameter_groups().z_head,
    )
    assert {role: restored.adapter_version(role) for role in AdapterRole} == {
        role: backbone.adapter_version(role) for role in AdapterRole
    }
    assert restored.z_version == backbone.z_version


def test_checkpoint_and_initial_binding_reject_changed_trainable_bytes(
    backbone: QwenPolicyBackbone,
    backbone_config: QwenBackboneConfig,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "policy"
    backbone.save_checkpoint(str(checkpoint))
    original = backbone.trainable_state_identity

    mismatched = TrainableStateIdentity.create(
        backbone_deployment_hash=original.backbone_deployment_hash,
        forward_adapter_hash="sha256:" + "f" * 64,
        backward_adapter_hash=original.backward_adapter_hash,
        z_head_hash=original.z_head_hash,
    )
    with pytest.raises(ValueError):
        backbone.bind_initial_trainable_state(mismatched)

    adapter_file = checkpoint / "forward_adapter" / "adapter_model.safetensors"
    tensors = load_file(str(adapter_file))
    name = next(iter(tensors))
    tensors[name] = tensors[name] + 1
    save_file(tensors, str(adapter_file))

    restored = QwenPolicyBackbone(backbone_config)
    with pytest.raises(ValueError):
        restored.load_checkpoint(str(checkpoint))
    with pytest.raises(ValueError):
        inspect_policy_checkpoint(checkpoint)


def test_cpu_checkpoint_inspection_rejects_missing_and_invalid_tensor_files(
    backbone: QwenPolicyBackbone,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    backbone.save_checkpoint(str(missing))
    (missing / "z_head.pt").rename(missing / "z_head.missing")
    with pytest.raises(FileNotFoundError):
        inspect_policy_checkpoint(missing)

    invalid = tmp_path / "invalid"
    backbone.save_checkpoint(str(invalid))
    torch.save({"not-a-tensor": "invalid"}, invalid / "z_head.pt")
    with pytest.raises(ValueError):
        inspect_policy_checkpoint(invalid)


def test_cpu_checkpoint_inspection_rejects_metadata_tensor_mismatch(
    backbone: QwenPolicyBackbone,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "mismatched"
    backbone.save_checkpoint(str(checkpoint))
    metadata_path = checkpoint / "policy_state.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["trainable_state"]["z_head_hash"] = "sha256:" + "0" * 64
    metadata_path.write_text(canonical_json(metadata), encoding="utf-8")

    with pytest.raises(ValueError):
        inspect_policy_checkpoint(checkpoint)


def test_local_generator_snapshot_changes_only_after_version_advance(
    backbone: QwenPolicyBackbone,
) -> None:
    generator = LocalPolicyGenerator(backbone)
    before = generator.snapshot()
    _replace_adapter(backbone, AdapterRole.FORWARD_POLICY, scale=0.03)

    assert generator.snapshot() == before
    backbone.mark_policy_update(1)
    assert generator.snapshot() != before


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("role", list(AdapterRole))
@pytest.mark.parametrize("checkpointed", [False, True])
def test_qwen_hybrid_edge_microbatch_preserves_padding_scores_and_gradients(
    qwen35_selective_config, size, role, checkpointed
):
    from skillev.policy.scoring_execution import TeacherForcingConfig
    from skillev.training.performance_config import TrainingPerformanceConfig

    backbone = QwenPolicyBackbone(
        replace(qwen35_selective_config, teacher_forced_gradient_checkpointing=checkpointed),
        performance=TrainingPerformanceConfig(
            teacher_forcing=TeacherForcingConfig(microbatch_size=size)
        ),
    )
    edges = (
        ((1, 4), (6, 7, 8)),
        ((1, 4, 5, 6, 7), (8,)),
        ((1, 5, 4, 6), (7, 8)),
        ((1,), (4, 5, 6, 7)),
    )[:size]
    baseline = tuple(backbone.score(prefix, action, role) for prefix, action in edges)
    torch.stack([score.mean() for score in baseline]).sum().backward()
    expected = [p.grad.detach().clone() for p in _role_parameters(backbone, role)]
    _clear_gradients(backbone)
    batched = backbone.score_edge_microbatch(edges, role)
    for actual, reference in zip(batched, baseline, strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.stack([score.mean() for score in batched]).sum().backward()
    for parameter, reference in zip(_role_parameters(backbone, role), expected, strict=True):
        torch.testing.assert_close(parameter.grad, reference, rtol=1e-5, atol=1e-6)


def test_multimodal_outer_config_gets_padding_from_text_config(qwen35_selective_config):
    from transformers import Qwen3_5Config

    from skillev.policy.scoring_execution import TeacherForcingConfig
    from skillev.policy.teacher_forcing import TeacherForcingExecutor

    backbone = QwenPolicyBackbone(qwen35_selective_config)
    backbone._activate_adapter(AdapterRole.FORWARD_POLICY)

    class MultimodalEnvelope:
        config = Qwen3_5Config(text_config=backbone._model.config.to_dict())

        def train(self, mode):
            backbone._model.train(mode)

        def eval(self):
            backbone._model.eval()

        def __call__(self, **kwargs):
            return backbone._model(**kwargs)

    expected = backbone.score(PREFIX_IDS, ACTION_IDS, AdapterRole.FORWARD_POLICY)
    result = TeacherForcingExecutor(TeacherForcingConfig()).score(
        model=MultimodalEnvelope(),
        device=torch.device("cpu"),
        edges=((PREFIX_IDS, ACTION_IDS),),
        role=AdapterRole.FORWARD_POLICY,
        checkpoint_enabled=False,
    )
    torch.testing.assert_close(result[0], expected, rtol=0, atol=0)


def test_scoring_interval_reuses_adapter_and_keeps_mode_through_backward(
    qwen35_selective_config, monkeypatch
):
    config = replace(qwen35_selective_config, teacher_forced_gradient_checkpointing=True)
    backbone = QwenPolicyBackbone(config)
    reference = QwenPolicyBackbone(config)
    with torch.no_grad():
        for group in ("forward", "backward", "z_head"):
            for actual, expected in zip(
                getattr(backbone.parameter_groups(), group),
                getattr(reference.parameter_groups(), group),
                strict=True,
            ):
                actual.copy_(expected)
    scores = []
    for _ in range(3):
        value = reference.score(PREFIX_IDS, ACTION_IDS, AdapterRole.FORWARD_POLICY)
        scores.append(value.detach())
        value.mean().backward()
    activations = []
    activate = backbone._activate_adapter

    def tracked(role):
        activations.append(role)
        return activate(role)

    monkeypatch.setattr(backbone, "_activate_adapter", tracked)
    with backbone.scoring_session(AdapterRole.FORWARD_POLICY):
        for expected in scores:
            value = backbone.score(PREFIX_IDS, ACTION_IDS, AdapterRole.FORWARD_POLICY)
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            assert backbone._model.training
            value.mean().backward()
            assert backbone._model.training
        with pytest.raises(RuntimeError):
            backbone.z_value(QUERY_IDS)
    assert activations == [AdapterRole.FORWARD_POLICY]
    assert not backbone._model.training
    for actual, expected in zip(
        backbone.parameter_groups().forward, reference.parameter_groups().forward, strict=True
    ):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    with pytest.raises(ValueError), backbone.scoring_session(AdapterRole.BACKWARD_POLICY):
        raise ValueError("injected scoring failure")
    assert not backbone._model.training
    assert backbone._scoring_role is None
