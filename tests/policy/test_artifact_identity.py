"""Behavioral coverage for frozen-base and executable-tokenizer identities."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from transformers import PreTrainedTokenizerFast

from skillev.policy import (
    BaseModelArtifactIdentity,
    QwenPolicyBackbone,
    hf_backbone,
    qwen_backend_class,
    qwen_dtype_conversion_policy,
    qwen_tokenizer_artifact_identity,
)


def _write_unindexed_model(root: Path) -> None:
    (root / "config.json").write_text('{"architectures":["Qwen"]}', encoding="utf-8")
    (root / "generation_config.json").write_text('{"eos_token_id":1}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"first frozen model payload")


def _fast_tokenizer(*, byte_level: bool) -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        WordLevel(
            vocab={"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "[PAD]": 3, "hello": 4},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False) if byte_level else Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    return tokenizer


def test_base_model_identity_commits_to_every_weight_byte_without_private_root(
    tmp_path: Path,
) -> None:
    _write_unindexed_model(tmp_path)
    identity = BaseModelArtifactIdentity.from_directory(
        directory=tmp_path,
        backend_class="transformers.AutoModelForCausalLM",
        upstream_revision="commit-a",
        dtype_conversion_policy="from_pretrained:bfloat16;module_to:bfloat16",
    )

    assert identity.weight_index is None
    assert tuple(item.relative_path for item in identity.weight_shards) == ("model.safetensors",)
    assert str(tmp_path) not in json.dumps(identity.to_value())
    assert BaseModelArtifactIdentity.from_value(identity.to_value()) == identity
    assert (
        BaseModelArtifactIdentity.create(
            backend_class=identity.backend_class,
            upstream_revision=identity.upstream_revision,
            dtype_conversion_policy=identity.dtype_conversion_policy,
            model_config=identity.model_config,
            generation_config=identity.generation_config,
            weight_index=identity.weight_index,
            weight_shards=identity.weight_shards,
        )
        == identity
    )

    (tmp_path / "model.safetensors").write_bytes(b"changed frozen model payload")
    changed = BaseModelArtifactIdentity.from_directory(
        directory=tmp_path,
        backend_class="transformers.AutoModelForCausalLM",
        upstream_revision="commit-a",
        dtype_conversion_policy="from_pretrained:bfloat16;module_to:bfloat16",
    )
    assert changed.content_hash != identity.content_hash


def test_base_model_identity_uses_indexed_shard_manifest(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "generation_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 6},
                "weight_map": {
                    "layer.a": "model-00002-of-00002.safetensors",
                    "layer.b": "model-00001-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    identity = BaseModelArtifactIdentity.from_directory(
        directory=tmp_path,
        backend_class="transformers.AutoModelForMultimodalLM",
        upstream_revision="commit-b",
        dtype_conversion_policy="from_pretrained:float32;module_to:float32",
    )

    assert identity.weight_index is not None
    assert tuple(item.relative_path for item in identity.weight_shards) == (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )


def test_base_model_identity_records_absent_optional_generation_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"official model without generation config")

    identity = BaseModelArtifactIdentity.from_directory(
        directory=tmp_path,
        backend_class="transformers.AutoModelForMultimodalLM",
        upstream_revision="commit-without-generation-config",
        dtype_conversion_policy="from_pretrained:bfloat16;module_to:bfloat16",
    )

    assert identity.generation_config is None
    assert identity.to_value()["generation_config"] is None
    assert BaseModelArtifactIdentity.from_value(identity.to_value()) == identity


def test_qwen_tokenizer_identity_commits_to_full_backend_serialization() -> None:
    whitespace = _fast_tokenizer(byte_level=False)
    byte_level = _fast_tokenizer(byte_level=True)

    first = qwen_tokenizer_artifact_identity(
        tokenizer=whitespace,
        tokenizer_id="tiny-tokenizer",
        revision="commit-a",
    )
    second = qwen_tokenizer_artifact_identity(
        tokenizer=byte_level,
        tokenizer_id="tiny-tokenizer",
        revision="commit-a",
    )

    assert first.content_hash != second.content_hash
    assert type(first).from_value(first.to_value()) == first


def test_qwen_loader_rejects_a_base_manifest_that_is_not_the_loaded_files(
    backbone_config,
    tiny_model_path: Path,
) -> None:
    actual = BaseModelArtifactIdentity.from_directory(
        directory=tiny_model_path,
        backend_class=qwen_backend_class(backbone_config),
        upstream_revision=backbone_config.revision,
        dtype_conversion_policy=qwen_dtype_conversion_policy(backbone_config),
    )
    accepted = replace(backbone_config, base_model_artifact=actual)
    backbone = QwenPolicyBackbone(accepted)
    assert backbone.backbone_id

    different_revision = BaseModelArtifactIdentity.from_directory(
        directory=tiny_model_path,
        backend_class=qwen_backend_class(backbone_config),
        upstream_revision="another-commit",
        dtype_conversion_policy=qwen_dtype_conversion_policy(backbone_config),
    )
    with pytest.raises(ValueError):
        QwenPolicyBackbone(replace(backbone_config, base_model_artifact=different_revision))


def test_qwen_loader_rejects_artifact_changed_during_model_load(
    backbone_config,
    tiny_model_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "mutable-tiny-model"
    shutil.copytree(tiny_model_path, model_path)
    config = replace(backbone_config, base_model_path=str(model_path))
    pinned = BaseModelArtifactIdentity.from_directory(
        directory=model_path,
        backend_class=qwen_backend_class(config),
        upstream_revision=config.revision,
        dtype_conversion_policy=qwen_dtype_conversion_policy(config),
    )
    original_loader = hf_backbone.AutoModelForCausalLM.from_pretrained

    def mutate_after_load(*args: object, **kwargs: object):
        model = original_loader(*args, **kwargs)
        (model_path / "generation_config.json").write_text(
            '{"eos_token_id": 1, "pad_token_id": 1}',
            encoding="utf-8",
        )
        return model

    monkeypatch.setattr(
        hf_backbone.AutoModelForCausalLM,
        "from_pretrained",
        mutate_after_load,
    )

    with pytest.raises(ValueError):
        QwenPolicyBackbone(replace(config, base_model_artifact=pinned))
