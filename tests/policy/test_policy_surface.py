from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import Any
from unittest.mock import patch

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

import skillev.policy.interface as policy_interface
from skillev.policy import (
    AUTHORING_JSON_ROOT_BOUNDARY_VERSION,
    ROLLOUT_SOURCE_MESSAGES_BEGIN,
    ROLLOUT_SOURCE_MESSAGES_END,
    AdapterRole,
    AuthoringGenerationRequest,
    GenerationResult,
    PolicyBackbone,
    PolicyGenerationRequest,
    QwenBackboneConfig,
    QwenPolicyBackbone,
    QwenTokenizerAdapter,
    qwen_tokenizer_artifact_identity,
)


def _fast_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        WordLevel(
            vocab={
                "[UNK]": 0,
                "[BOS]": 1,
                "[EOS]": 2,
                "[PAD]": 3,
                "hello": 4,
                "world": 5,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} "
        "{{ message['content'] }}{% endfor %}"
        "{% if add_generation_prompt %} assistant{% endif %}"
    )
    return tokenizer


def _tokenizer_content_hash(tokenizer: PreTrainedTokenizerFast) -> str:
    return qwen_tokenizer_artifact_identity(
        tokenizer=tokenizer,
        tokenizer_id="tokenizer@pinned",
        revision="pinned",
    ).content_hash


def _config(**changes: Any) -> QwenBackboneConfig:
    values: dict[str, Any] = {
        "base_model_path": "local-model",
        "revision": "pinned",
        "tokenizer_id": "qwen-tokenizer@1",
        "tokenizer_content_hash": "sha256:" + "0" * 64,
        "hidden_size": 16,
        "device": "cpu",
        "torch_dtype": "float32",
        "lora_rank": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.1,
        "lora_target_modules": ("q_proj",),
        "z_hidden_width": 8,
        "eos_token_ids": (2,),
    }
    values.update(changes)
    return QwenBackboneConfig(**values)


def _policy_request(**changes: Any) -> PolicyGenerationRequest:
    values: dict[str, Any] = {
        "input_ids": (1, 2),
        "max_new_tokens": 3,
        "seed": 17,
        "decoding_snapshot_id": "decoding@3",
    }
    values.update(changes)
    return PolicyGenerationRequest(**values)


def _authoring_request(**changes: Any) -> AuthoringGenerationRequest:
    values: dict[str, Any] = {
        "input_ids": (1, 2),
        "max_new_tokens": 3,
        "temperature": 0.8,
        "top_p": 0.9,
        "seed": 17,
        "template_version": "authoring@3",
        "completion_boundary_version": AUTHORING_JSON_ROOT_BOUNDARY_VERSION,
    }
    values.update(changes)
    return AuthoringGenerationRequest(**values)


def test_policy_surface_contains_only_deterministic_reset_and_parameter_groups() -> None:
    members = vars(policy_interface)

    assert "PreparedZReset" not in members
    assert "prepare_z_reset" not in vars(PolicyBackbone)
    assert "apply_z_reset" not in vars(PolicyBackbone)
    assert "recover_z_reset" not in vars(PolicyBackbone)
    assert "trainable_parameters" not in vars(PolicyBackbone)
    assert tuple(inspect.signature(PolicyBackbone.reset_z).parameters) == (
        "self",
        "seed",
    )
    assert tuple(inspect.signature(PolicyBackbone.parameter_groups).parameters) == ("self",)


@pytest.mark.parametrize(
    "record",
    [
        _config(),
        _policy_request(),
        _authoring_request(),
        GenerationResult((4,), (), "length"),
    ],
)
def test_public_values_are_frozen_and_slotted(record: object) -> None:
    assert is_dataclass(record)
    assert "__dict__" not in type(record).__slots__
    with pytest.raises(FrozenInstanceError):
        setattr(record, fields(record)[0].name, object())


@pytest.mark.parametrize(
    "changes",
    [
        {"base_model_path": ""},
        {"revision": ""},
        {"tokenizer_id": ""},
        {"hidden_size": 0},
        {"device": "auto"},
        {"torch_dtype": "float16"},
        {"attention_implementation": "auto"},
        {"lora_rank": 0},
        {"lora_alpha": 0},
        {"lora_dropout": 1.0},
        {"lora_target_modules": ()},
        {"lora_target_modules": ("q_proj", "q_proj")},
        {"z_hidden_width": 0},
        {"eos_token_ids": ()},
        {"eos_token_ids": (2, 2)},
    ],
)
def test_qwen_config_rejects_unpinned_or_out_of_domain_values(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _config(**changes)


def test_teacher_forced_checkpointing_is_explicit_and_requires_zero_dropout() -> None:
    config = _config(
        lora_dropout=0.0,
        teacher_forced_gradient_checkpointing=True,
    )

    assert QwenBackboneConfig.from_value(config.to_value()) == config
    with pytest.raises(ValueError):
        _config(teacher_forced_gradient_checkpointing=True)
    with pytest.raises(TypeError):
        _config(teacher_forced_gradient_checkpointing=1)


@pytest.mark.parametrize("backbone_type", [PolicyBackbone, QwenPolicyBackbone])
def test_generate_score_and_z_signatures_are_token_id_only(
    backbone_type: type[object],
) -> None:
    assert tuple(inspect.signature(backbone_type.generate_policy).parameters) == (
        "self",
        "request",
    )
    assert tuple(inspect.signature(backbone_type.generate_base).parameters) == (
        "self",
        "request",
    )
    assert tuple(inspect.signature(backbone_type.score).parameters) == (
        "self",
        "prefix_ids",
        "action_ids",
        "role",
    )
    assert tuple(inspect.signature(backbone_type.z_value).parameters) == (
        "self",
        "query_ids",
    )


def test_qwen_tokenizer_requires_explicit_fast_backend_and_identity() -> None:
    raw = _fast_tokenizer()
    adapter = QwenTokenizerAdapter(
        tokenizer=raw,
        tokenizer_id="tokenizer@pinned",
        revision="pinned",
        expected_content_hash=_tokenizer_content_hash(raw),
    )

    assert adapter.tokenizer_id == "tokenizer@pinned"
    assert adapter.encode("hello world") == [4, 5]
    with pytest.raises(TypeError):
        QwenTokenizerAdapter(  # type: ignore[arg-type]
            tokenizer=object(),
            tokenizer_id="tokenizer@pinned",
            revision="pinned",
            expected_content_hash=_tokenizer_content_hash(raw),
        )
    with pytest.raises(ValueError):
        QwenTokenizerAdapter(
            tokenizer=raw,
            tokenizer_id="",
            revision="pinned",
            expected_content_hash=_tokenizer_content_hash(raw),
        )


def test_qwen_tokenizer_uses_one_explicit_nonthinking_authoring_template() -> None:
    raw = _fast_tokenizer()
    adapter = QwenTokenizerAdapter(
        tokenizer=raw,
        tokenizer_id="tokenizer@pinned",
        revision="pinned",
        expected_content_hash=_tokenizer_content_hash(raw),
    )

    with patch.object(raw, "apply_chat_template", wraps=raw.apply_chat_template) as render:
        encoded = adapter.encode_authoring_prompt("hello world")

    assert encoded
    render.assert_called_once()
    messages = render.call_args.args[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"] == "hello world"
    assert render.call_args.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "return_dict": False,
    }


def test_qwen_tokenizer_uses_controller_system_nonthinking_rollout_template() -> None:
    raw = _fast_tokenizer()
    adapter = QwenTokenizerAdapter(
        tokenizer=raw,
        tokenizer_id="tokenizer@pinned",
        revision="pinned",
        expected_content_hash=_tokenizer_content_hash(raw),
    )

    with patch.object(raw, "apply_chat_template", wraps=raw.apply_chat_template) as render:
        encoded = adapter.encode_rollout_prompt("hello world")

    assert encoded
    render.assert_called_once()
    messages = render.call_args.args[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"] == "hello world"
    assert render.call_args.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "return_dict": False,
    }


def test_qwen_tokenizer_exposes_evaluation_thinking_rollout_template() -> None:
    raw = _fast_tokenizer()
    adapter = QwenTokenizerAdapter(
        tokenizer=raw,
        tokenizer_id="tokenizer@pinned",
        revision="pinned",
        expected_content_hash=_tokenizer_content_hash(raw),
    )

    with patch.object(raw, "apply_chat_template", wraps=raw.apply_chat_template) as render:
        encoded = adapter.encode_rollout_prompt_with_thinking("solve this problem")

    assert encoded
    render.assert_called_once()
    messages = render.call_args.args[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"] == "solve this problem"
    assert render.call_args.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": True,
        "return_dict": False,
    }


def test_qwen_tokenizer_exposes_non_json_step_zero_terminal_template() -> None:
    raw = _fast_tokenizer()
    adapter = QwenTokenizerAdapter(
        tokenizer=raw,
        tokenizer_id="tokenizer@pinned",
        revision="pinned",
        expected_content_hash=_tokenizer_content_hash(raw),
    )

    with patch.object(raw, "apply_chat_template", wraps=raw.apply_chat_template) as render:
        encoded = adapter.encode_step_zero_terminal_prompt("Terminal payload: answer")

    assert encoded
    messages = render.call_args.args[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "terminal response renderer" in messages[0]["content"]
    assert "structured-action-json" not in messages[0]["content"]
    assert render.call_args.kwargs["enable_thinking"] is False


def test_qwen_rollout_template_preserves_source_messages_with_one_system_turn() -> None:
    raw = _fast_tokenizer()
    adapter = QwenTokenizerAdapter(
        tokenizer=raw,
        tokenizer_id="tokenizer@pinned",
        revision="pinned",
        expected_content_hash=_tokenizer_content_hash(raw),
    )
    envelope = (
        "controller-before\n"
        + ROLLOUT_SOURCE_MESSAGES_BEGIN
        + '[{"content":"system context","role":"system"},'
        + '{"content":"patient question","role":"user"}]'
        + ROLLOUT_SOURCE_MESSAGES_END
        + "controller-after"
    )

    with patch.object(raw, "apply_chat_template", wraps=raw.apply_chat_template) as render:
        encoded = adapter.encode_rollout_prompt(envelope)

    assert encoded
    messages = render.call_args.args[0]
    assert [message["role"] for message in messages] == ["system", "user", "user"]
    assert messages == [
        {
            "role": "system",
            "content": (
                policy_interface.ROLLOUT_CONTROLLER_SYSTEM_MESSAGE
                + "\n\nSource system instructions:\nsystem context"
            ),
        },
        {"role": "user", "content": "patient question"},
        {"role": "user", "content": "controller-before\ncontroller-after"},
    ]


def test_qwen_tokenizer_exact_decode_keeps_special_tokens() -> None:
    raw = _fast_tokenizer()
    adapter = QwenTokenizerAdapter(
        tokenizer=raw,
        tokenizer_id="tokenizer@pinned",
        revision="pinned",
        expected_content_hash=_tokenizer_content_hash(raw),
    )

    with patch.object(raw, "decode", wraps=raw.decode) as decode:
        text = adapter.decode((1, 4, 2))

    assert text == "[BOS] hello [EOS]"
    decode.assert_called_once_with(
        [1, 4, 2],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def test_adapter_role_is_closed() -> None:
    assert tuple(AdapterRole) == (
        AdapterRole.FORWARD_POLICY,
        AdapterRole.BACKWARD_POLICY,
    )
