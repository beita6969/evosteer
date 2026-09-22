"""Dependency-light identities and requests for on-policy rollout collection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from skillev.contracts import (
    JsonValue,
    ScientificSamplingCoordinate,
    normalize_json,
    stable_hash,
)
from skillev.policy.interface import (
    ROLLOUT_PROMPT_ENCODER_VERSION,
    THINKING_ROLLOUT_PROMPT_ENCODER_VERSION,
)
from skillev.runtime.full_skill_context import FullRetrievedSkillContext

from .action_root_boundary import ACTION_JSON_ROOT_BOUNDARY_VERSION
from .action_surface import (
    ActionSurface,
    ModelVisibleMessage,
    RolloutBudgetProfile,
)
from .prompt_profiles import InitialContextProfile

_UINT64_LIMIT = 2**64


def _require_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _require_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite number")
    return normalized


def _require_object(
    value: object,
    *,
    label: str,
    expected: set[str],
) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    if set(normalized) != expected:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


def _require_wire_float(value: object, *, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{field} must be a finite float")
    return value


def _require_wire_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    return value


def _require_text_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a JSON array")
    values = tuple(_require_text(item, field=field) for item in value)
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{field} must be sorted and unique")
    return values


def _require_json_value(value: JsonValue, *, field: str) -> JsonValue:
    if normalize_json(value) != value:
        raise TypeError(f"{field} must be normalized JSON")
    return value


class GenerationPhase(StrEnum):
    """The two generation calls made for every rollout step."""

    REASONING = "reasoning"
    ACTION = "action"


class RolloutTermination(StrEnum):
    """Terminal outcomes that still form a valid scientific trajectory."""

    COMPLETED = "completed"
    HORIZON_EXHAUSTED = "horizon-exhausted"
    NO_VALID_COMPLETE_ACTION = "no-valid-complete-action"


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """Immutable identity of the forward policy used by one trajectory."""

    snapshot_id: str
    backbone_id: str
    forward_adapter_version: str
    tokenizer_id: str
    backend_id: str
    initial_trainable_state_hash: str

    @staticmethod
    def derive_id(
        *,
        backbone_id: str,
        forward_adapter_version: str,
        tokenizer_id: str,
        backend_id: str,
        initial_trainable_state_hash: str,
    ) -> str:
        """Derive the snapshot commitment from every policy identity component."""

        return stable_hash(
            {
                "backbone_id": backbone_id,
                "backend_id": backend_id,
                "forward_adapter_version": forward_adapter_version,
                "initial_trainable_state_hash": initial_trainable_state_hash,
                "tokenizer_id": tokenizer_id,
            }
        )

    @classmethod
    def create(
        cls,
        *,
        backbone_id: str,
        forward_adapter_version: str,
        tokenizer_id: str,
        backend_id: str,
        initial_trainable_state_hash: str,
    ) -> PolicySnapshot:
        """Create a snapshot whose ID cannot be selected independently."""

        return cls(
            snapshot_id=cls.derive_id(
                backbone_id=backbone_id,
                forward_adapter_version=forward_adapter_version,
                tokenizer_id=tokenizer_id,
                backend_id=backend_id,
                initial_trainable_state_hash=initial_trainable_state_hash,
            ),
            backbone_id=backbone_id,
            forward_adapter_version=forward_adapter_version,
            tokenizer_id=tokenizer_id,
            backend_id=backend_id,
            initial_trainable_state_hash=initial_trainable_state_hash,
        )

    def __post_init__(self) -> None:
        for field, value in (
            ("snapshot_id", self.snapshot_id),
            ("backbone_id", self.backbone_id),
            ("forward_adapter_version", self.forward_adapter_version),
            ("tokenizer_id", self.tokenizer_id),
            ("backend_id", self.backend_id),
            ("initial_trainable_state_hash", self.initial_trainable_state_hash),
        ):
            _require_text(value, field=field)
        expected = self.derive_id(
            backbone_id=self.backbone_id,
            forward_adapter_version=self.forward_adapter_version,
            tokenizer_id=self.tokenizer_id,
            backend_id=self.backend_id,
            initial_trainable_state_hash=self.initial_trainable_state_hash,
        )
        if self.snapshot_id != expected:
            raise ValueError("snapshot_id does not match the policy identity components")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "backbone_id": self.backbone_id,
            "backend_id": self.backend_id,
            "forward_adapter_version": self.forward_adapter_version,
            "initial_trainable_state_hash": self.initial_trainable_state_hash,
            "snapshot_id": self.snapshot_id,
            "tokenizer_id": self.tokenizer_id,
        }

    @classmethod
    def from_value(cls, value: object) -> PolicySnapshot:
        data = _require_object(
            value,
            label="policy snapshot",
            expected={
                "backbone_id",
                "backend_id",
                "forward_adapter_version",
                "initial_trainable_state_hash",
                "snapshot_id",
                "tokenizer_id",
            },
        )
        return cls(
            snapshot_id=_require_text(data["snapshot_id"], field="snapshot_id"),
            backbone_id=_require_text(data["backbone_id"], field="backbone_id"),
            forward_adapter_version=_require_text(
                data["forward_adapter_version"],
                field="forward_adapter_version",
            ),
            tokenizer_id=_require_text(data["tokenizer_id"], field="tokenizer_id"),
            backend_id=_require_text(data["backend_id"], field="backend_id"),
            initial_trainable_state_hash=_require_text(
                data["initial_trainable_state_hash"],
                field="initial_trainable_state_hash",
            ),
        )


@dataclass(frozen=True, slots=True)
class DecodingSnapshot:
    """Raw-softmax policy execution parameters pinned for one trajectory."""

    snapshot_id: str
    max_reasoning_tokens: int
    max_action_tokens: int
    base_seed: int
    prompt_encoder_version: str = ROLLOUT_PROMPT_ENCODER_VERSION
    action_boundary_version: str = ACTION_JSON_ROOT_BOUNDARY_VERSION
    format_version: str = "skillev-policy-decoding@3"

    @staticmethod
    def derive_id(
        *,
        max_reasoning_tokens: int,
        max_action_tokens: int,
        base_seed: int,
        prompt_encoder_version: str = ROLLOUT_PROMPT_ENCODER_VERSION,
        action_boundary_version: str = ACTION_JSON_ROOT_BOUNDARY_VERSION,
        format_version: str = "skillev-policy-decoding@3",
    ) -> str:
        payload: dict[str, JsonValue] = {
            "base_seed": base_seed,
            "format_version": format_version,
            "max_action_tokens": max_action_tokens,
            "max_reasoning_tokens": max_reasoning_tokens,
        }
        if format_version != "skillev-policy-decoding@2":
            payload.update(
                {
                    "action_boundary_version": action_boundary_version,
                    "prompt_encoder_version": prompt_encoder_version,
                }
            )
        return stable_hash(payload)

    @classmethod
    def create(
        cls,
        *,
        max_reasoning_tokens: int,
        max_action_tokens: int,
        base_seed: int,
        prompt_encoder_version: str = ROLLOUT_PROMPT_ENCODER_VERSION,
        action_boundary_version: str = ACTION_JSON_ROOT_BOUNDARY_VERSION,
        format_version: str = "skillev-policy-decoding@3",
    ) -> DecodingSnapshot:
        return cls(
            snapshot_id=cls.derive_id(
                max_reasoning_tokens=max_reasoning_tokens,
                max_action_tokens=max_action_tokens,
                base_seed=base_seed,
                prompt_encoder_version=prompt_encoder_version,
                action_boundary_version=action_boundary_version,
                format_version=format_version,
            ),
            max_reasoning_tokens=max_reasoning_tokens,
            max_action_tokens=max_action_tokens,
            base_seed=base_seed,
            prompt_encoder_version=prompt_encoder_version,
            action_boundary_version=action_boundary_version,
            format_version=format_version,
        )

    def __post_init__(self) -> None:
        _require_text(self.snapshot_id, field="snapshot_id")
        if type(self.max_reasoning_tokens) is not int or self.max_reasoning_tokens <= 0:
            raise ValueError("max_reasoning_tokens must be positive")
        if type(self.max_action_tokens) is not int or self.max_action_tokens <= 0:
            raise ValueError("max_action_tokens must be positive")
        if type(self.base_seed) is not int or not 0 <= self.base_seed < _UINT64_LIMIT:
            raise ValueError("base_seed must be an unsigned 64-bit integer")
        _require_text(self.prompt_encoder_version, field="prompt_encoder_version")
        _require_text(self.action_boundary_version, field="action_boundary_version")
        if self.format_version not in {
            "skillev-policy-decoding@2",
            "skillev-policy-decoding@3",
        }:
            raise ValueError("unsupported policy decoding format")
        if self.format_version == "skillev-policy-decoding@3" and (
            self.prompt_encoder_version
            not in {ROLLOUT_PROMPT_ENCODER_VERSION, THINKING_ROLLOUT_PROMPT_ENCODER_VERSION}
            or self.action_boundary_version
            not in {ACTION_JSON_ROOT_BOUNDARY_VERSION, "native-model-stop@1"}
        ):
            raise ValueError("policy decoding runtime versions are unsupported")
        expected = self.derive_id(
            max_reasoning_tokens=self.max_reasoning_tokens,
            max_action_tokens=self.max_action_tokens,
            base_seed=self.base_seed,
            prompt_encoder_version=self.prompt_encoder_version,
            action_boundary_version=self.action_boundary_version,
            format_version=self.format_version,
        )
        if self.snapshot_id != expected:
            raise ValueError("snapshot_id does not match the decoding parameters")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "base_seed": self.base_seed,
            "action_boundary_version": self.action_boundary_version,
            "format_version": self.format_version,
            "max_action_tokens": self.max_action_tokens,
            "max_reasoning_tokens": self.max_reasoning_tokens,
            "prompt_encoder_version": self.prompt_encoder_version,
            "snapshot_id": self.snapshot_id,
        }

    @classmethod
    def from_value(cls, value: object) -> DecodingSnapshot:
        if not isinstance(value, dict):
            raise ValueError("decoding snapshot must be a JSON object")
        normalized = normalize_json(value)
        if not isinstance(normalized, dict):
            raise ValueError("decoding snapshot must be a JSON object")
        legacy = {
            "base_seed",
            "format_version",
            "max_action_tokens",
            "max_reasoning_tokens",
            "snapshot_id",
        }
        current = legacy | {"action_boundary_version", "prompt_encoder_version"}
        if frozenset(normalized) not in {frozenset(legacy), frozenset(current)}:
            raise ValueError("decoding snapshot has an incompatible field set")
        data = normalized
        legacy_wire = set(data) == legacy
        return cls(
            snapshot_id=_require_text(data["snapshot_id"], field="snapshot_id"),
            max_reasoning_tokens=_require_wire_int(
                data["max_reasoning_tokens"],
                field="max_reasoning_tokens",
            ),
            max_action_tokens=_require_wire_int(
                data["max_action_tokens"],
                field="max_action_tokens",
            ),
            base_seed=_require_wire_int(data["base_seed"], field="base_seed"),
            prompt_encoder_version=(
                "legacy-raw-prompt@1"
                if legacy_wire
                else _require_text(data["prompt_encoder_version"], field="prompt_encoder_version")
            ),
            action_boundary_version=(
                "none"
                if legacy_wire
                else _require_text(data["action_boundary_version"], field="action_boundary_version")
            ),
            format_version=_require_text(data["format_version"], field="format_version"),
        )


def derive_generation_seed(
    *,
    base_seed: int,
    coordinate: ScientificSamplingCoordinate,
    step_index: int,
    phase: GenerationPhase,
) -> int:
    """Derive a deterministic seed from scientific coordinates only."""

    if type(base_seed) is not int or not 0 <= base_seed < _UINT64_LIMIT:
        raise ValueError("base_seed must be an unsigned 64-bit integer")
    if not isinstance(coordinate, ScientificSamplingCoordinate):
        raise TypeError("coordinate must be ScientificSamplingCoordinate")
    if type(step_index) is not int or step_index < 1:
        raise ValueError("step_index must be positive")
    if not isinstance(phase, GenerationPhase):
        raise ValueError("phase must be a GenerationPhase")
    digest = stable_hash(
        {
            "base_seed": base_seed,
            "coordinate": coordinate.to_value(),
            "phase": phase.value,
            "step_index": step_index,
        }
    ).removeprefix("sha256:")
    return int(digest[:16], 16)


@dataclass(frozen=True, slots=True)
class RolloutTask:
    """Answer-free public task projection visible to the policy."""

    task_id: str
    environment_id: str
    task_family: str
    context_id: str
    query: str
    available_tools: tuple[str, ...]
    public_context: JsonValue
    action_surface: ActionSurface | None = None
    budget_profile: RolloutBudgetProfile | None = None
    model_visible_messages: tuple[ModelVisibleMessage, ...] = ()

    def __post_init__(self) -> None:
        for field, value in (
            ("task_id", self.task_id),
            ("environment_id", self.environment_id),
            ("task_family", self.task_family),
            ("context_id", self.context_id),
            ("query", self.query),
        ):
            normalized = normalize_json(value)
            object.__setattr__(self, field, _require_text(normalized, field=field))
        if not isinstance(self.available_tools, tuple) or any(
            type(tool) is not str or not tool.strip() for tool in self.available_tools
        ):
            raise ValueError("available_tools must be a tuple of non-empty text")
        if tuple(sorted(set(self.available_tools))) != self.available_tools:
            raise ValueError("available_tools must be sorted and unique")
        object.__setattr__(self, "public_context", normalize_json(self.public_context))
        if self.action_surface is not None and not isinstance(self.action_surface, ActionSurface):
            raise TypeError("action_surface must be an ActionSurface")
        if self.budget_profile is not None and not isinstance(
            self.budget_profile, RolloutBudgetProfile
        ):
            raise TypeError("budget_profile must be a RolloutBudgetProfile")
        if not isinstance(self.model_visible_messages, tuple) or any(
            not isinstance(message, ModelVisibleMessage) for message in self.model_visible_messages
        ):
            raise TypeError("model_visible_messages must contain ModelVisibleMessage values")

    @property
    def root_query(self) -> str:
        """Stable answer-free task query consumed by retrieval and ``Z_theta(q)``."""

        return self.query

    @property
    def source_messages(self) -> tuple[ModelVisibleMessage, ...]:
        """Benchmark-native messages, distinct from the stable root query."""

        return self.model_visible_messages

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "environment_id": self.environment_id,
            "context_id": self.context_id,
            "available_tools": list(self.available_tools),
            "public_context": self.public_context,
            "query": self.query,
            "task_family": self.task_family,
            "task_id": self.task_id,
            "action_surface": (
                None if self.action_surface is None else self.action_surface.to_value()
            ),
            "budget_profile": (
                None if self.budget_profile is None else self.budget_profile.to_value()
            ),
            "model_visible_messages": [
                message.to_value() for message in self.model_visible_messages
            ],
        }

    @classmethod
    def from_value(cls, value: object) -> RolloutTask:
        legacy_fields = {
            "environment_id",
            "context_id",
            "available_tools",
            "public_context",
            "query",
            "task_family",
            "task_id",
        }
        if not isinstance(value, dict):
            raise ValueError("rollout task must be a JSON object")
        normalized = normalize_json(value)
        if not isinstance(normalized, dict):
            raise ValueError("rollout task must be a JSON object")
        current_fields = legacy_fields | {
            "action_surface",
            "budget_profile",
            "model_visible_messages",
        }
        if frozenset(normalized) not in {frozenset(legacy_fields), frozenset(current_fields)}:
            raise ValueError("rollout task has an incompatible field set")
        data = normalized
        messages = data.get("model_visible_messages", [])
        if not isinstance(messages, list):
            raise TypeError("model_visible_messages must be an array")
        raw_surface = data.get("action_surface")
        raw_profile = data.get("budget_profile")
        return cls(
            task_id=_require_text(data["task_id"], field="task_id"),
            environment_id=_require_text(data["environment_id"], field="environment_id"),
            task_family=_require_text(data["task_family"], field="task_family"),
            context_id=_require_text(data["context_id"], field="context_id"),
            query=_require_text(data["query"], field="query"),
            available_tools=_require_text_tuple(data["available_tools"], field="available_tools"),
            public_context=_require_json_value(
                data["public_context"],
                field="public_context",
            ),
            action_surface=(None if raw_surface is None else ActionSurface.from_value(raw_surface)),
            budget_profile=(
                None if raw_profile is None else RolloutBudgetProfile.from_value(raw_profile)
            ),
            model_visible_messages=tuple(
                ModelVisibleMessage.from_value(message) for message in messages
            ),
        )


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    """Complete immutable input for collecting one trajectory."""

    trajectory_id: str
    task: RolloutTask
    retrieved_skills: tuple[FullRetrievedSkillContext, ...]
    active_skill_ids: tuple[str, ...]
    library_version: str
    sampling_coordinate: ScientificSamplingCoordinate
    decoding: DecodingSnapshot
    epsilon_min: float
    condition_id: str
    initial_context_profile: InitialContextProfile

    def __post_init__(self) -> None:
        _require_text(self.trajectory_id, field="trajectory_id")
        _require_text(self.condition_id, field="condition_id")
        if not isinstance(self.initial_context_profile, InitialContextProfile):
            raise TypeError("initial_context_profile must be explicit and typed")
        if not isinstance(self.task, RolloutTask):
            raise ValueError("task must be a RolloutTask")
        if not isinstance(self.retrieved_skills, tuple):
            raise ValueError("retrieved_skills must be a tuple")
        skill_ids: list[str] = []
        for skill in self.retrieved_skills:
            _require_text(skill.metadata.version, field="retrieved skill version")
            skill_ids.append(skill.metadata.skill_id)
        if len(set(skill_ids)) != len(skill_ids):
            raise ValueError("retrieved skill IDs must be unique")
        if (
            not isinstance(self.active_skill_ids, tuple)
            or tuple(sorted(set(self.active_skill_ids))) != self.active_skill_ids
            or any(type(skill_id) is not str or not skill_id for skill_id in self.active_skill_ids)
        ):
            raise ValueError("active_skill_ids must be sorted unique non-empty text")
        if not set(skill_ids) <= set(self.active_skill_ids):
            raise ValueError("retrieved skills must belong to the pinned active library")
        _require_text(self.library_version, field="library_version")
        if not isinstance(self.sampling_coordinate, ScientificSamplingCoordinate):
            raise ValueError("sampling_coordinate must be ScientificSamplingCoordinate")
        if self.sampling_coordinate.task_id != self.task.task_id:
            raise ValueError("sampling coordinate belongs to another task")
        if not isinstance(self.decoding, DecodingSnapshot):
            raise ValueError("decoding must be a DecodingSnapshot")
        epsilon_min = _require_float(self.epsilon_min, field="epsilon_min")
        if epsilon_min <= 0.0:
            raise ValueError("epsilon_min must be positive")
        object.__setattr__(self, "epsilon_min", epsilon_min)
