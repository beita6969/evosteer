"""Exact Protocol-v3 configuration identities for the alternating TTB trainer.

Method, rollout, optimizer, execution, and checkpoint controls have separate
authorities.  In particular the literal full method must explicitly select
raw-softmax rollout, no gradient clipping, and zero AdamW weight decay.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self

from skillev.contracts import JsonValue, normalize_json
from skillev.contracts.action_wire import ACTION_WIRES, NATIVE_TOOL_WIRES
from skillev.contracts.skill_exposure import SKILL_EXPOSURES
from skillev.policy.interface import ModelInputWindow
from skillev.runtime.contracts import BudgetVector
from skillev.task_semantic_guidance import (
    LEGACY_TASK_SEMANTICS,
    validate_task_semantic_guidance,
)

from .stability import PolicyStabilityConfig

if TYPE_CHECKING:
    from skillev.rollout.context import CanonicalInitialContextAssembler

METHOD_FORMAT: Final = "skillev-method@3"
POLICY_ROLLOUT_CONFIG_FORMAT: Final = "skillev-policy-rollout@3"
THINKING_POLICY_ROLLOUT_CONFIG_FORMAT: Final = "skillev-policy-rollout@4"
WINDOWED_POLICY_ROLLOUT_CONFIG_FORMAT: Final = "skillev-policy-rollout@5"
DOMAIN_POLICY_ROLLOUT_CONFIG_FORMAT: Final = "skillev-policy-rollout@6"
INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT: Final = "skillev-policy-rollout@7"
SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT: Final = "skillev-policy-rollout@8"
SHARED_SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT: Final = "skillev-policy-rollout@9"
_INTERFACE_POLICY_FORMATS: Final = frozenset(
    {
        INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT,
        SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT,
        SHARED_SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT,
    }
)
OPTIMIZER_CONFIG_FORMAT: Final = "skillev-optimizer-config@3"
TRAINING_EXECUTION_CONFIG_FORMAT: Final = "skillev-training-execution@3"
CHECKPOINT_CONFIG_FORMAT: Final = "skillev-checkpoint-config@4"
PRIVATE_CHECKPOINT_STORAGE_BINDING_FORMAT: Final = "skillev-private-checkpoint-storage@1"
TRAINER_CONFIG_FORMAT: Final = "skillev-trainer-config@4"

_UINT64_LIMIT = 2**64


def _bool(value: object) -> bool:
    if type(value) is not bool:
        raise TypeError("expected boolean rollout setting")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return unicodedata.normalize("NFC", value)


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _uint64(value: object, *, field: str) -> int:
    if type(value) is not int or not 0 <= value < _UINT64_LIMIT:
        raise ValueError(f"{field} must be an unsigned 64-bit integer")
    return value


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


def _positive_float(value: object, *, field: str) -> float:
    normalized = _finite_float(value, field=field)
    if normalized <= 0.0:
        raise ValueError(f"{field} must be positive")
    return normalized


def _object(
    value: object,
    *,
    label: str,
    fields: frozenset[str],
    format_value: str,
) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    if set(normalized) != fields:
        raise ValueError(f"{label} has an incompatible field set")
    if normalized["format"] != format_value:
        raise ValueError(f"unsupported {label} format")
    return normalized


@dataclass(frozen=True, slots=True)
class TTBMethodConfig:
    """Only the parameters of the objective defined by ``idea.tex``."""

    epsilon_min: float
    temperature_beta: float
    format: str = METHOD_FORMAT

    def __post_init__(self) -> None:
        if self.format != METHOD_FORMAT:
            raise ValueError("unsupported TTB method format")
        object.__setattr__(
            self,
            "epsilon_min",
            _positive_float(self.epsilon_min, field="epsilon_min"),
        )
        object.__setattr__(
            self,
            "temperature_beta",
            _positive_float(self.temperature_beta, field="temperature_beta"),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "epsilon_min": self.epsilon_min,
            "format": self.format,
            "temperature_beta": self.temperature_beta,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        data = _object(
            value,
            label="TTB method config",
            fields=frozenset({"epsilon_min", "format", "temperature_beta"}),
            format_value=METHOD_FORMAT,
        )
        return cls(
            epsilon_min=_positive_float(data["epsilon_min"], field="epsilon_min"),
            temperature_beta=_positive_float(
                data["temperature_beta"],
                field="temperature_beta",
            ),
        )


@dataclass(frozen=True, slots=True)
class PolicyRolloutConfig:
    """Execution limits for the type-fixed raw-softmax rollout policy."""

    base_seed: int
    max_turns: int
    max_reasoning_tokens: int
    max_action_tokens: int
    per_rollout_maximum: BudgetVector
    format: str = POLICY_ROLLOUT_CONFIG_FORMAT
    reasoning_native_thinking: bool = False
    input_window: ModelInputWindow | None = None
    reasoning_by_domain: tuple[tuple[str, bool], ...] = ()
    phase_context: bool = False
    reasoning_tool_catalog: bool = False
    token_budget_notice: bool = False
    action_wire: str = "structured-action-json@3"
    skill_exposure: str = "full-inline"
    public_action_semantics: bool = False
    task_semantic_guidance: str = LEGACY_TASK_SEMANTICS
    hotpot_deliberation: bool = False

    def __post_init__(self) -> None:
        validate_task_semantic_guidance(self.task_semantic_guidance)
        if type(self.hotpot_deliberation) is not bool:
            raise TypeError("Hotpot deliberation must be boolean")
        if (self.task_semantic_guidance != LEGACY_TASK_SEMANTICS or self.hotpot_deliberation) and (
            self.format != SHARED_SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT
        ):
            raise ValueError("shared task guidance requires the explicit v9 condition")
        if self.task_semantic_guidance != LEGACY_TASK_SEMANTICS and (
            self.action_wire in NATIVE_TOOL_WIRES and not self.public_action_semantics
        ):
            raise ValueError("shared native task guidance requires public action semantics")
        if type(self.public_action_semantics) is not bool or (
            self.public_action_semantics
            and (
                self.format
                not in {
                    SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT,
                    SHARED_SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT,
                }
                or self.action_wire not in NATIVE_TOOL_WIRES
            )
        ):
            raise ValueError("public native semantics require the explicit v8 native condition")
        if (
            type(self.phase_context) is not bool
            or self.action_wire not in ACTION_WIRES
            or self.skill_exposure not in SKILL_EXPOSURES
        ):
            raise ValueError("unsupported rollout interface condition")
        if (
            self.phase_context
            or self.action_wire != "structured-action-json@3"
            or self.skill_exposure != "full-inline"
        ) and self.format not in _INTERFACE_POLICY_FORMATS:
            raise ValueError("interface changes require the v7 rollout condition")
        if self.action_wire in NATIVE_TOOL_WIRES and not self.phase_context:
            raise ValueError("native wire requires phase-aware context")
        if type(self.token_budget_notice) is not bool or (
            self.token_budget_notice and not self.phase_context
        ):
            raise ValueError("token budget notice requires phase context")
        if type(self.reasoning_tool_catalog) is not bool or (
            self.reasoning_tool_catalog
            and (not self.phase_context or self.action_wire not in NATIVE_TOOL_WIRES)
        ):
            raise ValueError("reasoning tool catalog requires native phase context")
        if type(self.reasoning_native_thinking) is not bool:
            raise TypeError("reasoning_native_thinking must be boolean")
        if self.reasoning_native_thinking and self.format not in {
            THINKING_POLICY_ROLLOUT_CONFIG_FORMAT,
            WINDOWED_POLICY_ROLLOUT_CONFIG_FORMAT,
            DOMAIN_POLICY_ROLLOUT_CONFIG_FORMAT,
            INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT,
            SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT,
            SHARED_SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT,
        }:
            raise ValueError("native thinking requires the declared rollout condition v4")
        if self.format not in {
            POLICY_ROLLOUT_CONFIG_FORMAT,
            THINKING_POLICY_ROLLOUT_CONFIG_FORMAT,
            WINDOWED_POLICY_ROLLOUT_CONFIG_FORMAT,
            DOMAIN_POLICY_ROLLOUT_CONFIG_FORMAT,
            INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT,
            SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT,
            SHARED_SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT,
        }:
            raise ValueError("unsupported policy rollout config format")
        if self.format not in _INTERFACE_POLICY_FORMATS and (self.input_window is not None) != (
            self.format
            in {WINDOWED_POLICY_ROLLOUT_CONFIG_FORMAT, DOMAIN_POLICY_ROLLOUT_CONFIG_FORMAT}
        ):
            raise ValueError("bounded inputs require the explicit windowed rollout condition")
        if not isinstance(self.reasoning_by_domain, tuple) or any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or type(pair[0]) is not str
            or not pair[0].strip()
            or type(pair[1]) is not bool
            for pair in self.reasoning_by_domain
        ):
            raise TypeError("domain reasoning modes must be immutable named booleans")
        domains = tuple(pair[0] for pair in self.reasoning_by_domain)
        if domains != tuple(sorted(set(domains))):
            raise ValueError("domain reasoning modes must have unique sorted domains")
        if self.format not in _INTERFACE_POLICY_FORMATS and bool(domains) != (
            self.format == DOMAIN_POLICY_ROLLOUT_CONFIG_FORMAT
        ):
            raise ValueError("per-domain thinking requires the explicit domain rollout condition")
        object.__setattr__(self, "base_seed", _uint64(self.base_seed, field="base_seed"))
        for field in (
            "max_turns",
            "max_reasoning_tokens",
            "max_action_tokens",
        ):
            object.__setattr__(
                self,
                field,
                _positive_int(getattr(self, field), field=field),
            )
        if not isinstance(self.per_rollout_maximum, BudgetVector):
            raise TypeError("per_rollout_maximum must be a BudgetVector")
        model_calls = 2 * self.max_turns
        expected_output_tokens = self.max_turns * (
            self.max_reasoning_tokens + self.max_action_tokens
        )
        maximum = self.per_rollout_maximum
        if (
            maximum.model_calls != model_calls
            or maximum.agent_turns != self.max_turns
            or maximum.tool_calls != self.max_turns
            or maximum.output_tokens != expected_output_tokens
            or maximum.input_tokens < model_calls
            or maximum.input_tokens % model_calls
            or maximum.wall_time_milliseconds < self.max_turns
            or maximum.wall_time_milliseconds % self.max_turns
        ):
            raise ValueError("per_rollout_maximum must exactly match rollout turns and call caps")
        if (
            self.input_window is not None
            and self.input_window.max_tokens != maximum.input_tokens // model_calls
        ):
            raise ValueError("input window differs from the reserved per-request input allowance")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            **({"token_budget_notice": True} if self.token_budget_notice else {}),
            **({"reasoning_tool_catalog": True} if self.reasoning_tool_catalog else {}),
            **(
                {
                    "task_semantic_guidance": self.task_semantic_guidance,
                    "hotpot_deliberation": self.hotpot_deliberation,
                }
                if self.format == SHARED_SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT
                else {}
            ),
            **(
                {"public_action_semantics": self.public_action_semantics}
                if self.format
                in {
                    SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT,
                    SHARED_SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT,
                }
                else {}
            ),
            **(
                {
                    "phase_context": self.phase_context,
                    "action_wire": self.action_wire,
                    "skill_exposure": self.skill_exposure,
                    "input_window": self.input_window.to_value() if self.input_window else None,
                    "reasoning_by_domain": normalize_json(dict(self.reasoning_by_domain)),
                }
                if self.format in _INTERFACE_POLICY_FORMATS
                else {}
            ),
            "base_seed": self.base_seed,
            "format": self.format,
            "max_action_tokens": self.max_action_tokens,
            "max_reasoning_tokens": self.max_reasoning_tokens,
            "max_turns": self.max_turns,
            "per_rollout_maximum": normalize_json(self.per_rollout_maximum.to_value()),
            **(
                {"reasoning_native_thinking": self.reasoning_native_thinking}
                if self.format != POLICY_ROLLOUT_CONFIG_FORMAT
                else {}
            ),
            **({"input_window": self.input_window.to_value()} if self.input_window else {}),
            **(
                {"reasoning_by_domain": normalize_json(dict(self.reasoning_by_domain))}
                if self.reasoning_by_domain
                else {}
            ),
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        format_value = value.get("format") if isinstance(value, dict) else None
        shared_semantic_format = format_value == SHARED_SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT
        semantic_format = (
            shared_semantic_format or format_value == SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT
        )
        interface_format = format_value in _INTERFACE_POLICY_FORMATS
        native_format = format_value in {
            THINKING_POLICY_ROLLOUT_CONFIG_FORMAT,
            WINDOWED_POLICY_ROLLOUT_CONFIG_FORMAT,
            DOMAIN_POLICY_ROLLOUT_CONFIG_FORMAT,
            INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT,
            SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT,
            SHARED_SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT,
        }
        domain_format = interface_format or format_value == DOMAIN_POLICY_ROLLOUT_CONFIG_FORMAT
        windowed = domain_format or format_value == WINDOWED_POLICY_ROLLOUT_CONFIG_FORMAT
        data = _object(
            value,
            label="policy rollout config",
            fields=frozenset(
                {
                    "base_seed",
                    "format",
                    "max_action_tokens",
                    "max_reasoning_tokens",
                    "max_turns",
                    "per_rollout_maximum",
                }
                | (
                    {"task_semantic_guidance", "hotpot_deliberation"}
                    if shared_semantic_format
                    else set()
                )
                | ({"public_action_semantics"} if semantic_format else set())
                | (
                    {"reasoning_tool_catalog"}
                    if interface_format
                    and isinstance(value, dict)
                    and "reasoning_tool_catalog" in value
                    else set()
                )
                | (
                    {"token_budget_notice"}
                    if interface_format
                    and isinstance(value, dict)
                    and "token_budget_notice" in value
                    else set()
                )
                | ({"reasoning_native_thinking"} if native_format else set())
                | ({"input_window"} if windowed else set())
                | ({"reasoning_by_domain"} if domain_format else set())
                | (
                    {"phase_context", "action_wire", "skill_exposure"}
                    if interface_format
                    else set()
                )
            ),
            format_value=(
                SHARED_SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT
                if shared_semantic_format
                else SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT
                if semantic_format
                else INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT
                if interface_format
                else DOMAIN_POLICY_ROLLOUT_CONFIG_FORMAT
                if domain_format
                else WINDOWED_POLICY_ROLLOUT_CONFIG_FORMAT
                if windowed
                else THINKING_POLICY_ROLLOUT_CONFIG_FORMAT
                if native_format
                else POLICY_ROLLOUT_CONFIG_FORMAT
            ),
        )
        modes = data.get("reasoning_by_domain", {})
        if not isinstance(modes, dict):
            raise TypeError("domain reasoning modes must be an object")
        return cls(
            task_semantic_guidance=_text(
                data["task_semantic_guidance"], field="task_semantic_guidance"
            )
            if shared_semantic_format
            else LEGACY_TASK_SEMANTICS,
            hotpot_deliberation=_bool(data["hotpot_deliberation"])
            if shared_semantic_format
            else False,
            public_action_semantics=_bool(data["public_action_semantics"])
            if semantic_format
            else False,
            phase_context=_bool(data["phase_context"]) if interface_format else False,
            reasoning_tool_catalog=_bool(data.get("reasoning_tool_catalog", False)),
            token_budget_notice=_bool(data.get("token_budget_notice", False)),
            action_wire=_text(data["action_wire"], field="action_wire")
            if interface_format
            else "structured-action-json@3",
            skill_exposure=_text(data["skill_exposure"], field="skill_exposure")
            if interface_format
            else "full-inline",
            reasoning_by_domain=tuple((k, _bool(v)) for k, v in sorted(modes.items())),
            base_seed=_uint64(data["base_seed"], field="base_seed"),
            max_turns=_positive_int(data["max_turns"], field="max_turns"),
            max_reasoning_tokens=_positive_int(
                data["max_reasoning_tokens"],
                field="max_reasoning_tokens",
            ),
            max_action_tokens=_positive_int(
                data["max_action_tokens"],
                field="max_action_tokens",
            ),
            per_rollout_maximum=BudgetVector.from_value(data["per_rollout_maximum"]),
            format=(
                SHARED_SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT
                if shared_semantic_format
                else SEMANTIC_POLICY_ROLLOUT_CONFIG_FORMAT
                if semantic_format
                else INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT
                if interface_format
                else DOMAIN_POLICY_ROLLOUT_CONFIG_FORMAT
                if domain_format
                else WINDOWED_POLICY_ROLLOUT_CONFIG_FORMAT
                if windowed
                else THINKING_POLICY_ROLLOUT_CONFIG_FORMAT
            )
            if native_format
            else POLICY_ROLLOUT_CONFIG_FORMAT,
            reasoning_native_thinking=_bool(data["reasoning_native_thinking"])
            if native_format
            else False,
            input_window=ModelInputWindow.from_value(data["input_window"])
            if windowed and data["input_window"] is not None
            else None,
        )

    @property
    def condition_id(self) -> str:
        if self.format not in _INTERFACE_POLICY_FORMATS:
            return "trained-skillev@1"
        return (
            (
                f"trained-skillev-interface@1/phase={int(self.phase_context)}"
                f"/wire={self.action_wire}/skills={self.skill_exposure}"
            )
            + ("/public-action-semantics@1" if self.public_action_semantics else "")
            + ("/token-budget-notice@1" if self.token_budget_notice else "")
            + ("/reasoning-tool-catalog@1" if self.reasoning_tool_catalog else "")
            + (
                f"/{self.task_semantic_guidance}/hotpot={int(self.hotpot_deliberation)}"
                if self.task_semantic_guidance != LEGACY_TASK_SEMANTICS
                else ""
            )
        )

    def context_assembler(self, *, maximum_h0_tokens: int) -> CanonicalInitialContextAssembler:
        """Single construction path for formal training and read-only probes."""
        from skillev.rollout.context import CanonicalInitialContextAssembler

        return CanonicalInitialContextAssembler(
            maximum_h0_tokens=maximum_h0_tokens,
            input_window=self.input_window,
            phase_context=self.phase_context,
            reasoning_tool_catalog=self.reasoning_tool_catalog,
            token_budget_notice=self.token_budget_notice,
            action_wire=self.action_wire,
            skill_exposure=self.skill_exposure,
            public_action_semantics=self.public_action_semantics,
            task_semantic_guidance=self.task_semantic_guidance,
            hotpot_deliberation=self.hotpot_deliberation,
        )


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """The literal full-method AdamW configuration."""

    adapter_learning_rate: float
    z_learning_rate: float
    weight_decay: float = 0.0
    format: str = OPTIMIZER_CONFIG_FORMAT
    backward_learning_rate: float | None = None
    stability: PolicyStabilityConfig | None = None

    def __post_init__(self) -> None:
        if self.format not in {OPTIMIZER_CONFIG_FORMAT, "skillev-optimizer-config@4"}:
            raise ValueError("unsupported optimizer config format")
        if self.format == OPTIMIZER_CONFIG_FORMAT and (
            self.stability is not None or self.backward_learning_rate is not None
        ):
            raise ValueError("stability and split learning rates require a new optimizer condition")
        if self.backward_learning_rate is not None:
            _positive_float(self.backward_learning_rate, field="backward_learning_rate")
        if self.stability is not None and not isinstance(self.stability, PolicyStabilityConfig):
            raise TypeError("stability configuration must be typed")
        object.__setattr__(
            self,
            "adapter_learning_rate",
            _positive_float(
                self.adapter_learning_rate,
                field="adapter_learning_rate",
            ),
        )
        object.__setattr__(
            self,
            "z_learning_rate",
            _positive_float(self.z_learning_rate, field="z_learning_rate"),
        )
        weight_decay = _finite_float(self.weight_decay, field="weight_decay")
        if weight_decay != 0.0:
            raise ValueError("full method requires zero AdamW weight decay")
        object.__setattr__(self, "weight_decay", weight_decay)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "adapter_learning_rate": self.adapter_learning_rate,
            **(
                {
                    "backward_learning_rate": self.backward_learning_rate,
                    "stability": None if self.stability is None else self.stability.to_value(),
                }
                if self.format == "skillev-optimizer-config@4"
                else {}
            ),
            "format": self.format,
            "weight_decay": self.weight_decay,
            "z_learning_rate": self.z_learning_rate,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        candidate = isinstance(value, dict) and value.get("format") == "skillev-optimizer-config@4"
        data = _object(
            value,
            label="optimizer config",
            fields=frozenset(
                {
                    "adapter_learning_rate",
                    "format",
                    "weight_decay",
                    "z_learning_rate",
                }
                | ({"backward_learning_rate", "stability"} if candidate else set())
            ),
            format_value="skillev-optimizer-config@4" if candidate else OPTIMIZER_CONFIG_FORMAT,
        )
        return cls(
            format="skillev-optimizer-config@4" if candidate else OPTIMIZER_CONFIG_FORMAT,
            backward_learning_rate=None
            if data.get("backward_learning_rate") is None
            else _positive_float(data["backward_learning_rate"], field="backward_learning_rate"),
            stability=None
            if data.get("stability") is None
            else PolicyStabilityConfig.from_value(data["stability"]),
            adapter_learning_rate=_positive_float(
                data["adapter_learning_rate"],
                field="adapter_learning_rate",
            ),
            z_learning_rate=_positive_float(
                data["z_learning_rate"],
                field="z_learning_rate",
            ),
            weight_decay=_finite_float(data["weight_decay"], field="weight_decay"),
        )


@dataclass(frozen=True, slots=True)
class TrainingExecutionConfig:
    """Fixed experiment and batch-population identity."""

    experiment_id: str
    batch_size: int
    format: str = TRAINING_EXECUTION_CONFIG_FORMAT

    def __post_init__(self) -> None:
        if self.format != TRAINING_EXECUTION_CONFIG_FORMAT:
            raise ValueError("unsupported training execution config format")
        object.__setattr__(
            self,
            "experiment_id",
            _text(self.experiment_id, field="experiment_id"),
        )
        object.__setattr__(
            self,
            "batch_size",
            _positive_int(self.batch_size, field="batch_size"),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "batch_size": self.batch_size,
            "experiment_id": self.experiment_id,
            "format": self.format,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        data = _object(
            value,
            label="training execution config",
            fields=frozenset({"batch_size", "experiment_id", "format"}),
            format_value=TRAINING_EXECUTION_CONFIG_FORMAT,
        )
        return cls(
            experiment_id=_text(data["experiment_id"], field="experiment_id"),
            batch_size=_positive_int(data["batch_size"], field="batch_size"),
        )


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Public checkpoint cadence; storage is a private execution binding."""

    every_n_steps: int
    format: str = CHECKPOINT_CONFIG_FORMAT

    def __post_init__(self) -> None:
        if self.format != CHECKPOINT_CONFIG_FORMAT:
            raise ValueError("unsupported checkpoint config format")
        object.__setattr__(
            self,
            "every_n_steps",
            _positive_int(self.every_n_steps, field="every_n_steps"),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "every_n_steps": self.every_n_steps,
            "format": self.format,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        data = _object(
            value,
            label="checkpoint config",
            fields=frozenset({"every_n_steps", "format"}),
            format_value=CHECKPOINT_CONFIG_FORMAT,
        )
        return cls(
            every_n_steps=_positive_int(data["every_n_steps"], field="every_n_steps"),
        )


@dataclass(frozen=True, slots=True)
class PrivateCheckpointStorageBinding:
    """Private artifact root for checkpoints and runtime snapshots.

    This is intentionally not a member of :class:`TrainerConfig`: directory
    paths describe deployment placement rather than a scientific control and
    must not enter public identities, event payloads, or cross-arm equality.
    """

    directory: str
    format: str = PRIVATE_CHECKPOINT_STORAGE_BINDING_FORMAT

    def __post_init__(self) -> None:
        if self.format != PRIVATE_CHECKPOINT_STORAGE_BINDING_FORMAT:
            raise ValueError("unsupported private checkpoint storage binding format")
        object.__setattr__(self, "directory", _text(self.directory, field="directory"))

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "directory": self.directory,
            "format": self.format,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        data = _object(
            value,
            label="private checkpoint storage binding",
            fields=frozenset({"directory", "format"}),
            format_value=PRIVATE_CHECKPOINT_STORAGE_BINDING_FORMAT,
        )
        return cls(directory=_text(data["directory"], field="directory"))


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Complete v3 trainer provenance assembled from five disjoint controls."""

    method: TTBMethodConfig
    rollout: PolicyRolloutConfig
    optimizer: OptimizerConfig
    execution: TrainingExecutionConfig
    checkpoint: CheckpointConfig
    format: str = TRAINER_CONFIG_FORMAT

    def __post_init__(self) -> None:
        if self.format != TRAINER_CONFIG_FORMAT:
            raise ValueError("unsupported trainer config format")
        expected = (
            (self.method, TTBMethodConfig, "method"),
            (self.rollout, PolicyRolloutConfig, "rollout"),
            (self.optimizer, OptimizerConfig, "optimizer"),
            (self.execution, TrainingExecutionConfig, "execution"),
            (self.checkpoint, CheckpointConfig, "checkpoint"),
        )
        for value, expected_type, field in expected:
            if not isinstance(value, expected_type):
                raise TypeError(f"{field} must be {expected_type.__name__}")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "checkpoint": normalize_json(self.checkpoint.to_value()),
            "execution": normalize_json(self.execution.to_value()),
            "format": self.format,
            "method": normalize_json(self.method.to_value()),
            "optimizer": normalize_json(self.optimizer.to_value()),
            "rollout": normalize_json(self.rollout.to_value()),
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        data = _object(
            value,
            label="trainer config",
            fields=frozenset(
                {"checkpoint", "execution", "format", "method", "optimizer", "rollout"}
            ),
            format_value=TRAINER_CONFIG_FORMAT,
        )
        return cls(
            method=TTBMethodConfig.from_value(data["method"]),
            rollout=PolicyRolloutConfig.from_value(data["rollout"]),
            optimizer=OptimizerConfig.from_value(data["optimizer"]),
            execution=TrainingExecutionConfig.from_value(data["execution"]),
            checkpoint=CheckpointConfig.from_value(data["checkpoint"]),
        )


def conservative_rollout_maximum(
    *,
    max_turns: int,
    max_reasoning_tokens: int,
    max_action_tokens: int,
    max_model_input_tokens: int,
    max_tool_wall_time_milliseconds: int,
) -> BudgetVector:
    """Build the exact conservative reservation envelope for one rollout."""

    turns = _positive_int(max_turns, field="max_turns")
    reasoning = _positive_int(max_reasoning_tokens, field="max_reasoning_tokens")
    action = _positive_int(max_action_tokens, field="max_action_tokens")
    model_input = _positive_int(max_model_input_tokens, field="max_model_input_tokens")
    tool_wall_time = _positive_int(
        max_tool_wall_time_milliseconds,
        field="max_tool_wall_time_milliseconds",
    )
    model_calls = 2 * turns
    return BudgetVector(
        input_tokens=model_calls * model_input,
        output_tokens=turns * (reasoning + action),
        model_calls=model_calls,
        agent_turns=turns,
        tool_calls=turns,
        wall_time_milliseconds=turns * tool_wall_time,
    )


__all__ = [
    "CHECKPOINT_CONFIG_FORMAT",
    "METHOD_FORMAT",
    "OPTIMIZER_CONFIG_FORMAT",
    "POLICY_ROLLOUT_CONFIG_FORMAT",
    "PRIVATE_CHECKPOINT_STORAGE_BINDING_FORMAT",
    "TRAINER_CONFIG_FORMAT",
    "TRAINING_EXECUTION_CONFIG_FORMAT",
    "CheckpointConfig",
    "OptimizerConfig",
    "PolicyRolloutConfig",
    "PrivateCheckpointStorageBinding",
    "TTBMethodConfig",
    "TrainerConfig",
    "TrainingExecutionConfig",
    "conservative_rollout_maximum",
]
