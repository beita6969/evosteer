"""Explicit fast-tokenizer adapter for the pinned Qwen backend."""

from __future__ import annotations

import importlib.metadata
import math
from collections.abc import Callable, Mapping
from typing import Any, cast

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from skillev.contracts import JsonValue, stable_hash
from skillev.contracts.identity import validate_sha256

from .config import QwenBackboneConfig
from .interface import (
    ROLLOUT_TERMINAL_SYSTEM_MESSAGE,
    STEP_ZERO_REASONING_SYSTEM_MESSAGE,
    rollout_chat_messages,
)
from .phase_context import phase_chat_messages
from .tokenizer_identity import (
    PUBLIC_TOKENIZER_IDENTITY_FORMAT,
    PublicTokenizerIdentity,
    PublicTokenizerKind,
    TokenizerArtifactIdentity,
)

AUTHORING_SYSTEM_MESSAGE = (
    "You are a strict one-shot skill-authoring transducer. Transform the supplied source "
    "material according to the action-specific contract rather than copying a source "
    "document. Return only the final JSON object requested by the user. Do not expose "
    "analysis, thinking, markdown, or commentary."
)


class QwenTokenizerAdapter:
    """The sole tokenizer surface used by rollout, scoring, and authoring."""

    __slots__ = (
        "_artifact_identity",
        "_content_hash",
        "_revision",
        "_tokenizer",
        "_tokenizer_id",
    )

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        tokenizer_id: str,
        revision: str,
        expected_content_hash: str,
    ) -> None:
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise TypeError("Qwen backend requires PreTrainedTokenizerFast")
        if not isinstance(tokenizer_id, str) or not tokenizer_id.strip():
            raise ValueError("tokenizer_id must be non-empty text")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("tokenizer revision must be non-empty text")
        validate_sha256(expected_content_hash)
        if not tokenizer.chat_template:
            raise ValueError("Qwen tokenizer requires a chat_template")
        self._tokenizer = tokenizer
        self._tokenizer_id = tokenizer_id
        self._revision = revision
        self._artifact_identity = qwen_tokenizer_artifact_identity(
            tokenizer=tokenizer,
            tokenizer_id=tokenizer_id,
            revision=revision,
        )
        self._content_hash = self._artifact_identity.content_hash
        if self._content_hash != expected_content_hash:
            raise ValueError("tokenizer content differs from pinned deployment config")

    @classmethod
    def from_config(cls, config: QwenBackboneConfig) -> QwenTokenizerAdapter:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path or config.base_model_path,
            revision=config.revision,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise TypeError("Qwen backend requires PreTrainedTokenizerFast")
        return cls(
            tokenizer=tokenizer,
            tokenizer_id=config.tokenizer_id,
            revision=config.revision,
            expected_content_hash=config.tokenizer_content_hash,
        )

    @property
    def tokenizer_id(self) -> str:
        return self._tokenizer_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def content_hash(self) -> str:
        return self._content_hash

    @property
    def artifact_identity(self) -> TokenizerArtifactIdentity:
        """Return the complete executable tokenizer identity for formal use."""

        return self._artifact_identity

    @property
    def public_identity(self) -> PublicTokenizerIdentity:
        return PublicTokenizerIdentity(
            kind=PublicTokenizerKind.QWEN,
            tokenizer_id=self.tokenizer_id,
            revision=self.revision,
            content_hash=self.content_hash,
        )

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return cast(
            list[int],
            self._tokenizer.encode(
                text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            ),
        )

    def encode_rollout_prompt(self, text: str) -> list[int]:
        """Render one answer-free rollout prefix as a non-thinking chat turn."""

        return self._encode_rollout_prompt(text, enable_thinking=False)

    def encode_rollout_prompt_with_thinking(self, text: str) -> list[int]:
        """Render Qwen native thinking for a declared reasoning condition.

        Training c_t may opt in. Structured action sampling and F/B teacher
        forcing still share :meth:`encode_rollout_prompt`; native thinking is
        not inserted as an unscored prefix inside an action. IID owners also
        use this entry when their explicit benchmark condition requires it.
        """

        return self._encode_rollout_prompt(text, enable_thinking=True)

    def encode_step_zero_reasoning_prompt(
        self,
        text: str,
        *,
        enable_thinking: bool,
    ) -> list[int]:
        """Render seeded Step-0 reasoning without the JSON action system contract."""

        if not isinstance(text, str) or not text:
            raise ValueError("Step-0 reasoning prompt must be non-empty text")
        return cast(
            list[int],
            self._tokenizer.apply_chat_template(
                rollout_chat_messages(
                    text,
                    system_message=STEP_ZERO_REASONING_SYSTEM_MESSAGE,
                ),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                return_dict=False,
            ),
        )

    def encode_integrity_messages(
        self,
        messages: tuple[dict[str, str], ...],
        *,
        enable_thinking: bool,
    ) -> list[int]:
        """Evaluation-only public messages, without the legacy phase/skill system prompt.

        Training's raw-softmax prefixes and existing legacy entry points are unchanged.
        """
        return cast(
            list[int],
            self._tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                return_dict=False,
            ),
        )

    def encode_step_zero_terminal_prompt(self, text: str) -> list[int]:
        """Render a typed Step-0 terminal without the JSON-action controller contract."""

        if not isinstance(text, str) or not text:
            raise ValueError("terminal prompt must be non-empty text")
        return cast(
            list[int],
            self._tokenizer.apply_chat_template(
                rollout_chat_messages(
                    text,
                    system_message=ROLLOUT_TERMINAL_SYSTEM_MESSAGE,
                ),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            ),
        )

    def _encode_rollout_prompt(self, text: str, *, enable_thinking: bool) -> list[int]:
        if not isinstance(text, str) or not text:
            raise ValueError("rollout prompt must be non-empty text")
        messages, tools = phase_chat_messages(text)
        return cast(
            list[int],
            self._tokenizer.apply_chat_template(
                messages,
                **cast(dict[str, Any], {"tools": tools} if tools is not None else {}),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                return_dict=False,
            ),
        )

    def encode_authoring_prompt(self, text: str) -> list[int]:
        if not isinstance(text, str) or not text:
            raise ValueError("authoring prompt must be non-empty text")
        return cast(
            list[int],
            self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": AUTHORING_SYSTEM_MESSAGE},
                    {"role": "user", "content": text},
                ],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            ),
        )

    def decode(self, token_ids: tuple[int, ...]) -> str:
        if not isinstance(token_ids, tuple) or any(
            type(token_id) is not int or token_id < 0 for token_id in token_ids
        ):
            raise ValueError("token_ids must be a tuple of non-negative integers")
        return cast(
            str,
            self._tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        )

    def new_decode_stream(self) -> Callable[[int], str | None]:
        """Request-local decoding; preserves special tokens and partial UTF-8."""
        return incremental_token_decoder(self._tokenizer)


def incremental_token_decoder(tokenizer: Any) -> Callable[[int], str | None]:
    """Shared client/scheduler decoder with tokenizer dependencies in policy."""
    from tokenizers.decoders import DecodeStream  # type: ignore[import-untyped]

    stream = DecodeStream(skip_special_tokens=False)
    return lambda token_id: cast(str | None, stream.step(tokenizer.backend_tokenizer, token_id))


def qwen_tokenizer_artifact_identity(
    *,
    tokenizer: PreTrainedTokenizerFast,
    tokenizer_id: str,
    revision: str,
) -> TokenizerArtifactIdentity:
    """Hash every executable fast-tokenizer component used by Qwen.

    The complete Rust ``tokenizers`` serialization commits to the model,
    normalizer, pre-tokenizer, post-processor, decoder, and added-token
    behavior.  The surrounding Transformers configuration and package
    versions remain separate components because they also affect the public
    tokenizer surface.
    """

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError("Qwen backend requires PreTrainedTokenizerFast")
    if type(tokenizer_id) is not str or not tokenizer_id:
        raise ValueError("tokenizer_id must be non-empty text")
    if type(revision) is not str or not revision:
        raise ValueError("revision must be non-empty text")
    chat_template = tokenizer.chat_template
    if type(chat_template) is not str or not chat_template:
        raise ValueError("Qwen tokenizer requires a chat_template")
    config = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "clean_up_tokenization_spaces": tokenizer.clean_up_tokenization_spaces,
        "model_max_length": tokenizer.model_max_length,
        "padding_side": tokenizer.padding_side,
        "split_special_tokens": tokenizer.split_special_tokens,
        "truncation_side": tokenizer.truncation_side,
    }
    added_tokens = [
        {
            "id": token_id,
            "token": _tokenizer_value(token),
        }
        for token_id, token in sorted(tokenizer.added_tokens_decoder.items())
    ]
    return TokenizerArtifactIdentity.create(
        kind=PublicTokenizerKind.QWEN,
        tokenizer_id=tokenizer_id,
        revision=revision,
        backend_serialization_hash=stable_hash(
            {"backend_serialization": tokenizer.backend_tokenizer.to_str()}
        ),
        tokenizer_config_hash=stable_hash(config),
        chat_template_hash=stable_hash({"chat_template": chat_template}),
        special_tokens_hash=stable_hash(
            {"special_tokens_map": _tokenizer_value(tokenizer.special_tokens_map)}
        ),
        added_tokens_hash=stable_hash({"added_tokens": added_tokens}),
        transformers_version=importlib.metadata.version("transformers"),
        tokenizers_version=importlib.metadata.version("tokenizers"),
    )


def _tokenizer_value(value: object) -> JsonValue:
    """Translate known fast-tokenizer configuration values without paths.

    Unknown objects are rejected rather than stringified so a new executable
    tokenizer option cannot silently escape the formal identity.
    """

    if value is None or type(value) is str or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("tokenizer configuration contains a non-finite float")
        return value
    if isinstance(value, list | tuple):
        return [_tokenizer_value(item) for item in value]
    if isinstance(value, Mapping):
        if all(type(key) is str for key in value):
            return {key: _tokenizer_value(item) for key, item in sorted(value.items())}
        if all(type(key) is int for key in value):
            return {
                "integer_key_mapping": [
                    {"key": key, "value": _tokenizer_value(item)}
                    for key, item in sorted(value.items())
                ]
            }
        raise TypeError("tokenizer configuration mapping keys must be uniformly text or integer")
    attributes = (
        "content",
        "lstrip",
        "normalized",
        "rstrip",
        "single_word",
        "special",
    )
    if all(hasattr(value, attribute) for attribute in attributes):
        token = cast(Any, value)
        content = token.content
        flags = {
            "lstrip": token.lstrip,
            "normalized": token.normalized,
            "rstrip": token.rstrip,
            "single_word": token.single_word,
            "special": token.special,
        }
        if type(content) is not str or any(type(flag) is not bool for flag in flags.values()):
            raise TypeError("tokenizer added-token fields have unsupported types")
        return {"added_token": {"content": content, **flags}}
    raise TypeError("tokenizer configuration contains an unsupported executable value")


__all__ = [
    "AUTHORING_SYSTEM_MESSAGE",
    "PUBLIC_TOKENIZER_IDENTITY_FORMAT",
    "PublicTokenizerIdentity",
    "PublicTokenizerKind",
    "QwenTokenizerAdapter",
    "TokenizerArtifactIdentity",
    "qwen_tokenizer_artifact_identity",
]
