"""Offline tiny-backbone fixtures for Protocol-v3 trainer tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from skillev.policy import (
    QwenBackboneConfig,
    QwenPolicyBackbone,
    qwen_tokenizer_artifact_identity,
)
from tests.training.fakes import TrainingHarness, build_training_harness, tiny_tokenizer_corpus

TrainingBackboneFactory = Callable[[], QwenPolicyBackbone]
TrainingHarnessFactory = Callable[..., TrainingHarness]


@pytest.fixture(scope="session")
def training_tiny_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model_path = tmp_path_factory.mktemp("tiny-v3-training-backbone")
    backend = Tokenizer(BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    backend.decoder = ByteLevelDecoder()
    backend.train_from_iterator(
        tiny_tokenizer_corpus(),
        BpeTrainer(
            vocab_size=400,
            min_frequency=1,
            special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
            initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        model_max_length=4096,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} "
        "{{ message['content'] }}{% endfor %}"
        "{% if add_generation_prompt %} assistant{% endif %}"
    )
    tokenizer.save_pretrained(model_path)
    config = GPT2Config(
        vocab_size=len(tokenizer.get_vocab()),
        n_positions=4096,
        n_ctx=4096,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    with torch.random.fork_rng():
        torch.manual_seed(20260725)
        GPT2LMHeadModel(config).save_pretrained(model_path)
    return model_path


@pytest.fixture
def training_backbone_config(
    training_tiny_model_path: Path,
) -> QwenBackboneConfig:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(training_tiny_model_path)
    return QwenBackboneConfig(
        base_model_path=str(training_tiny_model_path),
        revision="local-pinned",
        tokenizer_id="tiny-v3-training-tokenizer",
        tokenizer_content_hash=qwen_tokenizer_artifact_identity(
            tokenizer=tokenizer,
            tokenizer_id="tiny-v3-training-tokenizer",
            revision="local-pinned",
        ).content_hash,
        hidden_size=16,
        eos_token_ids=(int(GPT2Config.from_pretrained(training_tiny_model_path).eos_token_id),),
        torch_dtype="float32",
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.25,
        lora_target_modules=("c_attn",),
        z_hidden_width=8,
        device="cpu",
    )


@pytest.fixture
def make_training_backbone(
    training_backbone_config: QwenBackboneConfig,
) -> TrainingBackboneFactory:
    return lambda: QwenPolicyBackbone(training_backbone_config)


@pytest.fixture
def training_backbone(
    make_training_backbone: TrainingBackboneFactory,
) -> QwenPolicyBackbone:
    return make_training_backbone()


@pytest.fixture
def make_training_harness(
    tmp_path: Path,
    training_backbone: QwenPolicyBackbone,
) -> TrainingHarnessFactory:
    count = 0

    def factory(
        *,
        backbone: QwenPolicyBackbone | None = None,
        **kwargs: object,
    ) -> TrainingHarness:
        nonlocal count
        count += 1
        return build_training_harness(
            tmp_path / f"harness-{count:03d}",
            backbone=training_backbone if backbone is None else backbone,
            **kwargs,
        )

    return factory
