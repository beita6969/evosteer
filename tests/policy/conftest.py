from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from skillev.policy import (
    QwenBackboneConfig,
    QwenPolicyBackbone,
    qwen_tokenizer_artifact_identity,
)


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a complete random causal LM and fast tokenizer without network I/O."""

    model_path = tmp_path_factory.mktemp("tiny-policy-backbone")
    vocabulary = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        **{f"token-{index}": index + 4 for index in range(28)},
    }
    tokenizer_backend = Tokenizer(
        WordLevel(vocab=vocabulary, unk_token="[UNK]"),
    )
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

    config = GPT2Config(
        vocab_size=len(vocabulary),
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=vocabulary["[BOS]"],
        eos_token_id=vocabulary["[EOS]"],
        pad_token_id=vocabulary["[PAD]"],
        use_cache=True,
    )
    # Keep creation reproducible without changing the test process's global RNG.
    with torch.random.fork_rng():
        torch.manual_seed(20260719)
        GPT2LMHeadModel(config).save_pretrained(model_path)
    return model_path


@pytest.fixture
def backbone_config(tiny_model_path: Path) -> QwenBackboneConfig:
    """Use the production Qwen class with an intentionally nonzero dropout."""

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tiny_model_path)
    return QwenBackboneConfig(
        base_model_path=str(tiny_model_path),
        revision="local-pinned",
        tokenizer_id="tiny-qwen-tokenizer@1",
        tokenizer_content_hash=qwen_tokenizer_artifact_identity(
            tokenizer=tokenizer,
            tokenizer_id="tiny-qwen-tokenizer@1",
            revision="local-pinned",
        ).content_hash,
        hidden_size=16,
        eos_token_ids=(int(GPT2Config.from_pretrained(tiny_model_path).eos_token_id),),
        torch_dtype="float32",
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.35,
        lora_target_modules=("c_attn",),
        z_hidden_width=8,
        device="cpu",
    )


@pytest.fixture
def tiny_backbone(backbone_config: QwenBackboneConfig) -> QwenPolicyBackbone:
    return QwenPolicyBackbone(backbone_config)


@pytest.fixture
def backbone(tiny_backbone: QwenPolicyBackbone) -> QwenPolicyBackbone:
    """Short alias used by the focused behavioral tests in this directory."""

    return tiny_backbone
