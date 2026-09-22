"""Dependency-light, token-ID-only interface to the policy model layer.

The v2 method deliberately exposes two different generation request types.
TTB rollout sampling is the raw categorical policy used by :meth:`score`;
frozen-base skill authoring has its own independently configured sampler.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import takewhile
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from skillev.contracts import JsonValue, TokenizerProtocol, canonical_json

from .json_root_boundary import AUTHORING_JSON_ROOT_BOUNDARY_VERSION
from .phase_context import OBSERVED_PUBLIC_HISTORY as OBSERVED_PUBLIC_HISTORY
from .phase_context import TYPED_PUBLIC_HISTORY as TYPED_PUBLIC_HISTORY
from .phase_context import PhaseContextSpec as PhaseContextSpec
from .tokenizer_identity import PublicTokenizerIdentity
from .trainable_state import TrainableStateIdentity
from .versions import TrainableVersions

if TYPE_CHECKING:
    import torch

_FINISH_REASONS: Final = frozenset({"json-root", "length", "stop"})
ROLLOUT_PROMPT_ENCODER_VERSION: Final = "qwen-chat-nonthinking@5"
THINKING_ROLLOUT_PROMPT_ENCODER_VERSION: Final = "qwen-native-reasoning-structured-action@1"
ROLLOUT_SOURCE_MESSAGES_BEGIN: Final = "<skillev-source-messages>\n"
ROLLOUT_SOURCE_MESSAGES_END: Final = "\n</skillev-source-messages>\n"
ROLLOUT_CONTROLLER_SYSTEM_MESSAGE: Final = (
    "Solve the task using the supplied information and available actions. The final "
    "prompt suffix indicates whether to reason about the task or send an action. "
    "Choose your own approach; the action interfaces are described in Available Actions."
)
STEP_ZERO_REASONING_SYSTEM_MESSAGE: Final = (
    "Reason about the task using the supplied information and any helpful approach. "
    "This is a reasoning phase; the executable action or terminal response follows separately."
)
ROLLOUT_TERMINAL_SYSTEM_MESSAGE: Final = (
    "You are the terminal response renderer for a benchmark task. Use the supplied task, "
    "source messages, retrieved guidance, and private reasoning, then obey the final "
    "Terminal payload contract in the last user message. Return only that payload. Do not "
    "emit JSON, markdown fences, headings, analysis, or commentary unless the benchmark-native "
    "payload itself explicitly requires them."
)


def rollout_chat_messages(
    text: str,
    *,
    system_message: str = ROLLOUT_CONTROLLER_SYSTEM_MESSAGE,
) -> list[dict[str, str]]:
    """Project one canonical rollout prefix into role-preserving chat messages."""

    if not isinstance(text, str) or not text:
        raise ValueError("rollout prompt must be non-empty text")
    if not isinstance(system_message, str) or not system_message.strip():
        raise ValueError("rollout system message must be non-empty text")
    begin = text.find(ROLLOUT_SOURCE_MESSAGES_BEGIN)
    if begin < 0:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text},
        ]
    payload_start = begin + len(ROLLOUT_SOURCE_MESSAGES_BEGIN)
    end = text.find(ROLLOUT_SOURCE_MESSAGES_END, payload_start)
    if end < 0 or text.find(ROLLOUT_SOURCE_MESSAGES_BEGIN, payload_start) >= 0:
        raise ValueError("rollout source-message envelope is malformed")
    raw_messages = text[payload_start:end]
    try:
        value = json.loads(raw_messages)
    except json.JSONDecodeError as error:
        raise ValueError("rollout source-message envelope is invalid JSON") from error
    if not isinstance(value, list) or canonical_json(value) != raw_messages:
        raise ValueError("rollout source-message envelope is not canonical")
    source_messages: list[dict[str, str]] = []
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"content", "role"}
            or item.get("role") not in {"system", "user", "assistant"}
            or type(item.get("content")) is not str
            or not item["content"].strip()
        ):
            raise ValueError("rollout source-message entry is incompatible")
        source_messages.append({"role": item["role"], "content": item["content"]})

    # Qwen's native template admits one leading system message.  Protocol-native
    # benchmark prompts also commonly carry a leading system instruction, so
    # preserving that instruction as a second system turn makes an otherwise
    # valid rollout impossible to encode.  Merge only the contiguous leading
    # system block into the controller message; a later system turn is still an
    # incompatible conversation rather than something we silently reorder.
    leading_system: list[str] = []
    first_non_system = 0
    while (
        first_non_system < len(source_messages)
        and source_messages[first_non_system]["role"] == "system"
    ):
        leading_system.append(source_messages[first_non_system]["content"])
        first_non_system += 1
    remaining = source_messages[first_non_system:]
    if any(message["role"] == "system" for message in remaining):
        raise ValueError("rollout source system messages must form one leading block")
    system_content = system_message
    if leading_system:
        system_content += "\n\nSource system instructions:\n" + "\n\n".join(leading_system)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
        *remaining,
    ]
    controller = text[:begin] + text[end + len(ROLLOUT_SOURCE_MESSAGES_END) :]
    messages.append({"role": "user", "content": controller})
    return messages


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite number")
    return normalized


def _validate_token_ids(
    value: object,
    *,
    field: str,
    non_empty: bool,
) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{field} must be a tuple of token ids")
    if non_empty and not value:
        raise ValueError(f"{field} must not be empty")
    if any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise ValueError(f"{field} must contain non-negative integer token ids")


class AdapterRole(StrEnum):
    """Closed identity set for the two independently parameterized policies."""

    FORWARD_POLICY = "forward-policy"
    BACKWARD_POLICY = "backward-policy"


@runtime_checkable
class RolloutPromptTokenizerProtocol(Protocol):
    """Tokenizer extension for one model-ready rollout condition."""

    def encode_rollout_prompt(self, text: str) -> list[int]: ...


def encode_rollout_prompt(tokenizer: TokenizerProtocol, text: str) -> list[int]:
    """Encode the exact rollout prefix used by both sampling and scoring.

    Production Qwen tokenizers render the prefix through the pinned chat
    template.  The raw ``encode`` fallback preserves dependency-light test and
    legacy tokenizer behavior without making Qwen silently bypass its template.
    """

    if not isinstance(text, str) or not text:
        raise ValueError("rollout prompt must be non-empty text")
    if isinstance(tokenizer, RolloutPromptTokenizerProtocol):
        encoded = tokenizer.encode_rollout_prompt(text)
    else:
        encoded = tokenizer.encode(text)
    if not isinstance(encoded, list) or not encoded:
        raise ValueError("rollout prompt must encode to at least one token")
    if any(type(token_id) is not int or token_id < 0 for token_id in encoded):
        raise ValueError("rollout prompt encoding must contain non-negative token ids")
    return encoded


@runtime_checkable
class ThinkingRolloutTokenizerProtocol(Protocol):
    def encode_rollout_prompt_with_thinking(self, text: str) -> list[int]: ...


def encode_reasoning_prompt(
    tokenizer: TokenizerProtocol, text: str, *, native_thinking: bool
) -> list[int]:
    """Native thinking belongs to c_t, not an unscored second action rationale.

    The later structured action is conditioned on that exact realized reasoning;
    its sampling and F/B teacher forcing continue to use encode_rollout_prompt.
    No extra model pass, token allowance or discarded action-token prefix.
    """
    if not native_thinking:
        return encode_rollout_prompt(tokenizer, text)
    if not isinstance(tokenizer, ThinkingRolloutTokenizerProtocol):
        raise TypeError("native reasoning requires a thinking-capable tokenizer")
    encoded = tokenizer.encode_rollout_prompt_with_thinking(text)
    if not encoded or any(type(t) is not int or t < 0 for t in encoded):
        raise ValueError("native reasoning prompt must contain valid token IDs")
    return encoded


INPUT_WINDOW_VERSION = "h0-head-tail-recent-tokens@1"
INPUT_WINDOW_META_KEY = "model_input_window"


@dataclass(frozen=True, slots=True)
class ModelInputWindow:
    """Keep H0 and recent tokens; exceptionally large H0 keeps its head and tail.

    On overflow, at least half the allowance is reserved for the current suffix
    and recent history. H0 receives up to the other half. If H0 itself does not
    fit that share, retain its first and last tokens (task/system and interfaces).
    Unused H0 capacity goes to recent history. All slices are original token IDs.
    """

    max_tokens: int
    format: str = INPUT_WINDOW_VERSION

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or self.max_tokens < 4:
            raise ValueError("model input window must allow at least four tokens")
        if self.format != INPUT_WINDOW_VERSION:
            raise ValueError("unsupported model input window")

    def to_value(self) -> dict[str, JsonValue]:
        return {"format": self.format, "max_tokens": self.max_tokens}

    @classmethod
    def from_value(cls, value: object) -> ModelInputWindow:
        if not isinstance(value, Mapping):
            raise ValueError("model input window must be an object")
        maximum, version = value.get("max_tokens"), value.get("format")
        if type(maximum) is not int or not isinstance(version, str):
            raise ValueError("invalid model input window fields")
        return cls(maximum, version)

    @classmethod
    def from_meta(cls, meta: Mapping[str, JsonValue]) -> ModelInputWindow | None:
        value = meta.get(INPUT_WINDOW_META_KEY)
        return None if value is None else cls.from_value(value)

    def retained_ranges(
        self, ids: list[int], initial_ids: list[int]
    ) -> tuple[tuple[int, int], ...]:
        """Original contiguous token ranges; also used by read-only input evidence."""
        if len(ids) <= self.max_tokens:
            return ((0, len(ids)),)
        common = sum(
            1
            for _ in takewhile(lambda pair: pair[0] == pair[1], zip(ids, initial_ids, strict=False))
        )
        recent_minimum = self.max_tokens // 2
        initial_end = min(common, len(ids) - recent_minimum)
        initial_count = min(initial_end, self.max_tokens - recent_minimum)
        head = (initial_count + 1) // 2
        tail = initial_count - head
        recent = self.max_tokens - initial_count
        ranges: list[tuple[int, int]] = []
        for start, end in (
            (0, head),
            (initial_end - tail, initial_end),
            (len(ids) - recent, len(ids)),
        ):
            if start == end:
                continue
            if ranges and ranges[-1][1] == start:
                ranges[-1] = (ranges[-1][0], end)
            else:
                ranges.append((start, end))
        return tuple(ranges)

    def apply(self, ids: list[int], initial_ids: list[int]) -> tuple[int, ...]:
        return tuple(
            token
            for start, end in self.retained_ranges(ids, initial_ids)
            for token in ids[start:end]
        )


@dataclass(frozen=True, slots=True)
class EncodedPolicyPrompt:
    ids: tuple[int, ...]
    original_tokens: int
    retained_segment_lengths: tuple[int, ...] | None = None

    @property
    def removed_tokens(self) -> int:
        return self.original_tokens - len(self.ids)


def encode_policy_prompt(
    tokenizer: TokenizerProtocol,
    text: str,
    *,
    initial_text: str,
    window: ModelInputWindow | None,
    native_thinking: bool = False,
) -> EncodedPolicyPrompt:
    ids = encode_reasoning_prompt(tokenizer, text, native_thinking=native_thinking)
    if window is None or len(ids) <= window.max_tokens:
        return EncodedPolicyPrompt(tuple(ids), len(ids))
    # Encode H0 only on overflow; both chat modes have the same leading context.
    initial_ids = encode_reasoning_prompt(tokenizer, initial_text, native_thinking=native_thinking)
    ranges = window.retained_ranges(ids, initial_ids)
    return EncodedPolicyPrompt(
        tuple(token for start, end in ranges for token in ids[start:end]),
        len(ids),
        tuple(end - start for start, end in ranges),
    )


class PolicyScoringMemoryError(RuntimeError):
    """A terminal device allocation failure during teacher-forced scoring."""

    def __init__(
        self,
        *,
        prefix_token_count: int,
        action_token_count: int,
        role: AdapterRole,
    ) -> None:
        super().__init__("teacher-forced policy scoring exhausted device memory")
        self.prefix_token_count = prefix_token_count
        self.action_token_count = action_token_count
        self.role = role


def _validate_generation_common(
    *,
    input_ids: tuple[int, ...],
    max_new_tokens: int,
    seed: int,
) -> None:
    _validate_token_ids(input_ids, field="input_ids", non_empty=True)
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")


@dataclass(frozen=True, slots=True)
class PolicyGenerationRequest:
    """One raw-softmax TTB rollout request, expressed only as token IDs.

    Temperature and nucleus filtering are intentionally absent.  The forward
    rollout distribution is exactly ``softmax(raw_logits)`` and is therefore
    the same distribution whose token log probabilities are returned by
    :meth:`PolicyBackbone.score`.
    """

    input_ids: tuple[int, ...]
    max_new_tokens: int
    seed: int
    decoding_snapshot_id: str

    def __post_init__(self) -> None:
        _validate_generation_common(
            input_ids=self.input_ids,
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,
        )
        if not isinstance(self.decoding_snapshot_id, str) or not self.decoding_snapshot_id.strip():
            raise ValueError("decoding_snapshot_id must be non-empty text")


@dataclass(frozen=True, slots=True)
class AuthoringGenerationRequest:
    """One frozen-base skill-authoring request.

    Authoring is outside the TTB trajectory distribution, so its sampling
    controls remain explicit and cannot be passed to policy generation.
    """

    input_ids: tuple[int, ...]
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    template_version: str
    completion_boundary_version: str

    def __post_init__(self) -> None:
        _validate_generation_common(
            input_ids=self.input_ids,
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,
        )
        temperature = _finite_float(self.temperature, field="temperature")
        if temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        top_p = _finite_float(self.top_p, field="top_p")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be finite and in (0, 1]")
        if not isinstance(self.template_version, str) or not self.template_version.strip():
            raise ValueError("template_version must be non-empty text")
        if self.completion_boundary_version != AUTHORING_JSON_ROOT_BOUNDARY_VERSION:
            raise ValueError("authoring completion boundary version is unsupported")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """New content token IDs and a disjoint explicit stop suffix."""

    content_token_ids: tuple[int, ...]
    stop_token_ids: tuple[int, ...]
    finish_reason: str

    def __post_init__(self) -> None:
        _validate_token_ids(
            self.content_token_ids,
            field="content_token_ids",
            non_empty=False,
        )
        _validate_token_ids(
            self.stop_token_ids,
            field="stop_token_ids",
            non_empty=False,
        )
        if not isinstance(self.finish_reason, str) or self.finish_reason not in _FINISH_REASONS:
            raise ValueError("finish_reason is unsupported")
        if self.finish_reason in {"json-root", "length"} and self.stop_token_ids:
            raise ValueError("non-stop generation cannot carry stop_token_ids")
        if self.finish_reason == "stop" and not self.stop_token_ids:
            raise ValueError("stop-finished generation must carry stop_token_ids")


class AuthoringTokenizerProtocol(
    TokenizerProtocol,
    RolloutPromptTokenizerProtocol,
    Protocol,
):
    """Exact tokenizer surface required by rollout and frozen-base authoring."""

    @property
    def public_identity(self) -> PublicTokenizerIdentity: ...

    def decode(self, token_ids: tuple[int, ...]) -> str: ...

    def encode_rollout_prompt(self, text: str) -> list[int]: ...

    def encode_authoring_prompt(self, text: str) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class PolicyParameterGroups:
    """The three disjoint trainable components owned by one policy backbone."""

    forward: tuple[torch.nn.Parameter, ...]
    backward: tuple[torch.nn.Parameter, ...]
    z_head: tuple[torch.nn.Parameter, ...]


class PolicyBackbone(Protocol):
    """The complete model-layer switchboard exposed to all higher layers."""

    @property
    def backbone_id(self) -> str:
        """Return a stable identity for the base, configuration, and checkpoint."""

        ...

    @property
    def tokenizer(self) -> AuthoringTokenizerProtocol:
        """Return the sole production tokenizer outlet, with stable identity."""

        ...

    def adapter_version(self, role: AdapterRole) -> str:
        """Return the current ``<checkpoint_id>@<step>`` adapter version."""

        ...

    @property
    def z_version(self) -> str:
        """Return the current ``<checkpoint_id>@<step>`` Z-head version."""

        ...

    @property
    def trainable_state_identity(self) -> TrainableStateIdentity:
        """Return the exact bytes currently loaded in adapters and Z."""

        ...

    @property
    def initial_trainable_state_hash(self) -> str:
        """Return the content hash fixed before the first formal rollout."""

        ...

    def bind_initial_trainable_state(self, expected: TrainableStateIdentity) -> None:
        """Bind a freshly loaded formal initial checkpoint to this backbone."""

        ...

    def mark_policy_update(self, optimizer_step: int) -> None:
        """Advance all trainable-component version suffixes to one step.

        The checkpoint/content lineage prefix is preserved.  Phase 5 calls
        this immediately after each optimizer update so the next rollout
        snapshot cannot be mistaken for the policy that produced the batch.
        """

        ...

    def generate_policy(self, request: PolicyGenerationRequest) -> GenerationResult:
        """Sample from ``softmax(raw_logits)`` with the forward adapter."""

        ...

    def synchronize_trainable_versions(self, versions: TrainableVersions) -> None:
        """Install metadata after checkpoint/tensor synchronization, not an optimizer update."""

        ...

    def begin_policy_episode(self, episode_id: str) -> None:
        """Start one explicit rollout cache scope under the current policy."""

        ...

    def end_policy_episode(self, episode_id: str) -> None:
        """Discard the exact cache scope after the rollout terminates."""

        ...

    def generate_base(self, request: AuthoringGenerationRequest) -> GenerationResult:
        """Sample authoring tokens from the adapter-free frozen base."""

        ...

    def score(
        self,
        prefix_ids: tuple[int, ...],
        action_ids: tuple[int, ...],
        role: AdapterRole,
    ) -> torch.Tensor:
        """Return one raw-softmax natural-log probability per action token.

        For the forward role this is exactly the categorical distribution used
        by :meth:`generate_policy`.  This method neither sums nor
        length-normalizes the result.
        """

        ...

    def z_value(self, query_ids: tuple[int, ...]) -> torch.Tensor:
        """Return a scalar log-Z tensor whose gradients reach only the Z head."""

        ...

    def reset_z(self, seed: int) -> str:
        """Deterministically reinitialize only Z and return its new version.

        Optimizer-state clearing belongs to the Phase 5 training layer, not to
        this model-layer operation.
        """

        ...

    def parameter_groups(self) -> PolicyParameterGroups:
        """Return fixed non-overlapping parameter tuples."""

        ...

    def named_trainable_parameters(self) -> dict[str, torch.nn.Parameter]:
        """Stable model-layer names for optimizer checkpoint association."""

        ...

    def save_checkpoint(self, directory: str) -> None:
        """Persist both adapters, the Z head, and their version metadata."""

        ...

    def load_checkpoint(self, directory: str) -> None:
        """Restore model-layer trainables and versions without copying the base."""

        ...
