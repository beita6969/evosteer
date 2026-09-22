"""Independent, executable interventions for base and trained evaluation arms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

from skillev.task_semantic_guidance import (
    LEGACY_TASK_SEMANTICS,
    validate_task_semantic_guidance,
)

ARM_V1_FORMAT = "skillev-step0-integrity@1"
ARM_V2_FORMAT = "skillev-step0-integrity@2"
ARM_V3_FORMAT = "skillev-step0-integrity@3"
ARM_V4_FORMAT = "skillev-policy-integrity@4"
ARM_V5_FORMAT = "skillev-policy-integrity@5"
ARM_V6_FORMAT = "skillev-policy-integrity@6"
ARM_V1_FIELDS = frozenset(
    {
        "format",
        "condition_id",
        "optimizer_steps",
        "native_thinking",
        "reasoning_pass",
        "agent_topology",
        "skill_mode",
        "handwritten_policy_mode",
        "demonstration_mode",
        "semantic_action_pruning",
        "automatic_catalog_actions",
    }
)
ARM_V2_FIELDS = ARM_V1_FIELDS | {"precomputed_task_context", "legacy"}
ARM_V3_FIELDS = ARM_V2_FIELDS | {"tool_call_mode"}
ARM_V4_FIELDS = ARM_V3_FIELDS | {"policy_id"}
ARM_V5_FIELDS = ARM_V4_FIELDS | {"skill_library_id", "skill_retrieval_rule"}
ARM_V6_FIELDS = ARM_V5_FIELDS | {"task_semantic_guidance", "hotpot_deliberation"}


class ToolCallMode(StrEnum):
    PLAIN_TEXT = "plain-text"
    QWEN_XML = "qwen-xml"


class SkillMode(StrEnum):
    OFF = "off"
    LIBRARY = "library"
    GENERIC_TEXT = "generic-text"
    CAPABILITY_RETRIEVED_TEXT = "capability-retrieved-text"
    LEGACY_BENCHMARK_ROUTED = "legacy-benchmark-routed"


class ReasoningPass(StrEnum):
    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED = "required"


class AgentTopology(StrEnum):
    SINGLE = "single-controller"
    TWO_PASS = "two-pass-single"  # noqa: S105 -- topology label, not a password
    MULTI_AGENT = "multi-agent"


@dataclass(frozen=True, slots=True)
class InferenceArm:
    arm_id: str
    optimizer_steps: int = 0
    native_thinking: bool = False
    reasoning_pass: ReasoningPass = ReasoningPass.DISABLED
    skill_mode: SkillMode = SkillMode.OFF
    agent_topology: AgentTopology = AgentTopology.SINGLE
    handwritten_policy: bool = False
    semantic_action_pruning: bool = False
    demonstrations: bool = False
    automatic_catalog_actions: bool = False
    precomputed_task_context: bool = False
    legacy: bool = False
    tool_call_mode: ToolCallMode = ToolCallMode.PLAIN_TEXT
    task_semantic_guidance: str = field(default=LEGACY_TASK_SEMANTICS, kw_only=True)
    hotpot_deliberation: bool = field(default=False, kw_only=True)
    policy_id: str | None = field(default=None, kw_only=True)
    skill_library_id: str | None = field(default=None, kw_only=True)
    skill_retrieval_rule: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        validate_task_semantic_guidance(self.task_semantic_guidance)
        if type(self.hotpot_deliberation) is not bool:
            raise TypeError("Hotpot deliberation must be boolean")
        if self.hotpot_deliberation and self.task_semantic_guidance == LEGACY_TASK_SEMANTICS:
            raise ValueError("explicit Hotpot semantics require the shared task-guidance arm")
        if (
            type(self.arm_id) is not str
            or not self.arm_id.strip()
            or type(self.optimizer_steps) is not int
            or self.optimizer_steps < 0
        ):
            raise ValueError("arm identity and update count are invalid")
        for name in (
            "native_thinking",
            "handwritten_policy",
            "semantic_action_pruning",
            "demonstrations",
            "automatic_catalog_actions",
            "precomputed_task_context",
            "legacy",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError("arm switches must be actual booleans")
        if not isinstance(self.reasoning_pass, ReasoningPass):
            raise TypeError("reasoning_pass must be typed")
        if not isinstance(self.skill_mode, SkillMode):
            raise TypeError("skill_mode must be typed")
        if not isinstance(self.agent_topology, AgentTopology):
            raise TypeError("agent_topology must be typed")
        if not isinstance(self.tool_call_mode, ToolCallMode):
            raise TypeError("tool_call_mode must be typed")
        if self.policy_id is not None and (
            type(self.policy_id) is not str or not self.policy_id.strip()
        ):
            raise ValueError("policy_id must be non-empty text or absent")
        if not self.legacy and (self.optimizer_steps > 0) != (self.policy_id is not None):
            raise ValueError("trained arms require an explicit policy; Step-0 is adapter-free")
        if self.skill_mode is SkillMode.LIBRARY:
            from skillev.evolution.skill_access import ACTIVE_APPLICABILITY_RULE

            if not self.skill_library_id or self.skill_retrieval_rule != ACTIVE_APPLICABILITY_RULE:
                raise ValueError("library arms require their explicit library and retrieval rule")
        elif self.skill_library_id is not None or self.skill_retrieval_rule is not None:
            raise ValueError("explicit skill libraries require library mode")
        if not self.legacy and any(
            (
                self.handwritten_policy,
                self.semantic_action_pruning,
                self.automatic_catalog_actions,
                self.precomputed_task_context,
                self.demonstrations,
                self.skill_mode is SkillMode.LEGACY_BENCHMARK_ROUTED,
            )
        ):
            raise ValueError("clean arms cannot contain legacy interventions")

    def validate_live_topology(self) -> None:
        """Historical arms remain readable, but new executions have one answer owner.

        The upstream SkillFlow Executor uses its separate baseline runtime; the
        retired solver/researcher pair is not an implementation of that role.
        """
        if self.agent_topology is AgentTopology.MULTI_AGENT:
            raise ValueError("live evaluation requires one owner without consulting agents")

    def validate_primary_no_skill(self) -> None:
        if self.optimizer_steps != 0 or self.skill_mode is not SkillMode.OFF or self.legacy:
            raise ValueError("arm is not a clean optimizer-step-zero no-skill condition")

    def to_value(self) -> dict[str, object]:
        """Keep Step-0 serialization unchanged; trained policies are explicit v4 arms."""
        value: dict[str, object] = {
            "format": ARM_V2_FORMAT,
            "condition_id": self.arm_id,
            "optimizer_steps": self.optimizer_steps,
            "native_thinking": self.native_thinking,
            "reasoning_pass": self.reasoning_pass.value,
            "agent_topology": self.agent_topology.value,
            "skill_mode": self.skill_mode.value,
            "handwritten_policy_mode": "legacy" if self.handwritten_policy else "off",
            "demonstration_mode": "legacy" if self.demonstrations else "off",
            "semantic_action_pruning": self.semantic_action_pruning,
            "automatic_catalog_actions": self.automatic_catalog_actions,
            "precomputed_task_context": self.precomputed_task_context,
            "legacy": self.legacy,
        }
        if self.tool_call_mode is not ToolCallMode.PLAIN_TEXT:
            value.update(format=ARM_V3_FORMAT, tool_call_mode=self.tool_call_mode.value)
        if self.policy_id is not None:
            value.update(
                format=ARM_V4_FORMAT,
                tool_call_mode=self.tool_call_mode.value,
                policy_id=self.policy_id,
            )
        if self.skill_mode is SkillMode.LIBRARY:
            value.update(
                format=ARM_V5_FORMAT,
                tool_call_mode=self.tool_call_mode.value,
                policy_id=self.policy_id,
                skill_library_id=self.skill_library_id,
                skill_retrieval_rule=self.skill_retrieval_rule,
            )
        if self.task_semantic_guidance != LEGACY_TASK_SEMANTICS:
            value.update(
                format=ARM_V6_FORMAT,
                tool_call_mode=self.tool_call_mode.value,
                policy_id=self.policy_id,
                skill_library_id=self.skill_library_id,
                skill_retrieval_rule=self.skill_retrieval_rule,
                task_semantic_guidance=self.task_semantic_guidance,
                hotpot_deliberation=self.hotpot_deliberation,
            )
        return value


def legacy_arm() -> InferenceArm:
    """Old constructor behavior remains explicitly identifiable, not relabelled clean."""
    return InferenceArm(
        arm_id="legacy-step0-benchmark-routed@10",
        native_thinking=True,
        reasoning_pass=ReasoningPass.REQUIRED,
        skill_mode=SkillMode.LEGACY_BENCHMARK_ROUTED,
        agent_topology=AgentTopology.TWO_PASS,
        handwritten_policy=True,
        semantic_action_pruning=True,
        demonstrations=True,
        automatic_catalog_actions=True,
        precomputed_task_context=True,
        legacy=True,
    )


@dataclass(slots=True)
class InterventionCounts:
    skill_access_episodes: int = 0
    skill_retrieval_requests: int = 0
    skill_retrieved: int = 0
    skill_no_match: int = 0
    skill_budget_skips: int = 0
    skill_context_omissions: int = 0
    skill_discovery_calls: int = 0
    skill_read_calls: int = 0
    skill_blocks_injected: int = 0
    skill_body_tokens: int = 0
    skill_calls: int = 0
    handwritten_decisions: int = 0
    legal_actions_removed_by_policy: int = 0
    automatic_catalog_actions: int = 0
    demonstration_tokens: int = 0
    extra_model_calls_for_serialization: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    communication_repairs: int = 0
    peer_model_calls: int = 0
    peer_messages: int = 0

    def require_arm(self, arm: InferenceArm) -> None:
        if not arm.legacy and self.extra_model_calls_for_serialization:
            raise ValueError("clean answers cannot use an additional model serializer")
        if not arm.legacy and any(
            (
                self.handwritten_decisions,
                self.legal_actions_removed_by_policy,
                self.automatic_catalog_actions,
            )
        ):
            raise ValueError("observed rule intervention in a clean arm")
        if arm.skill_mode is SkillMode.OFF and any(
            getattr(self, name) for name in self.__dataclass_fields__ if name.startswith("skill_")
        ):
            raise ValueError("observed skill intervention in no-skill arm")
        if not arm.demonstrations and self.demonstration_tokens:
            raise ValueError("observed demonstration in demos-off arm")


def validate_paired_intervention(left: InferenceArm, right: InferenceArm) -> str:
    """Change one axis: topology, skill access, or forward weights, never a bundle."""
    if left.legacy or right.legacy:
        raise ValueError("historical composite arms cannot enter a clean comparison")
    differences = {
        name for name in left.__dataclass_fields__ if getattr(left, name) != getattr(right, name)
    } - {"arm_id"}
    if not differences:
        return "replication"
    for intervention, allowed in (
        ("topology", {"agent_topology"}),
        ("skill-access", {"skill_mode", "skill_library_id", "skill_retrieval_rule"}),
        ("forward-policy", {"optimizer_steps", "policy_id"}),
    ):
        if differences <= allowed:
            if intervention == "forward-policy" and left.policy_id == right.policy_id:
                raise ValueError("different optimizer steps cannot name the same policy")
            return intervention
    raise ValueError("the paired intervention changes more than one matched axis")


def strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _strict_text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{name} must be non-empty text")
    return value


def _strict_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value


def _strict_mode(value: object, name: str) -> bool:
    if type(value) is not str:
        raise TypeError(f"{name} must be text")
    if value == "off":
        return False
    if value == "legacy":
        return True
    raise ValueError(f"{name} must be 'off' or 'legacy'")


def _decode_versioned_arm(data: Mapping[str, object]) -> InferenceArm:
    format_value = data.get("format")
    if type(format_value) is not str:
        raise TypeError("format must be text")
    if format_value == ARM_V1_FORMAT:
        expected_fields = ARM_V1_FIELDS
    elif format_value == ARM_V2_FORMAT:
        expected_fields = ARM_V2_FIELDS
    elif format_value == ARM_V3_FORMAT:
        expected_fields = ARM_V3_FIELDS
    elif format_value == ARM_V4_FORMAT:
        expected_fields = ARM_V4_FIELDS
    elif format_value == ARM_V5_FORMAT:
        expected_fields = ARM_V5_FIELDS
    elif format_value == ARM_V6_FORMAT:
        expected_fields = ARM_V6_FIELDS
    else:
        raise ValueError("unsupported integrity arm configuration version")
    if set(data) != expected_fields:
        raise ValueError("integrity arm has missing or unknown fields")
    if format_value == ARM_V1_FORMAT:
        precomputed_task_context = False
        legacy = False
    else:
        precomputed_task_context = strict_bool(
            data["precomputed_task_context"], "precomputed_task_context"
        )
        legacy = strict_bool(data["legacy"], "legacy")
    try:
        reasoning_pass = ReasoningPass(_strict_text(data["reasoning_pass"], "reasoning_pass"))
        agent_topology = AgentTopology(_strict_text(data["agent_topology"], "agent_topology"))
        skill_mode = SkillMode(_strict_text(data["skill_mode"], "skill_mode"))
        tool_call_mode = (
            ToolCallMode(_strict_text(data["tool_call_mode"], "tool_call_mode"))
            if format_value in {ARM_V3_FORMAT, ARM_V4_FORMAT, ARM_V5_FORMAT, ARM_V6_FORMAT}
            else ToolCallMode.PLAIN_TEXT
        )
    except ValueError as error:
        raise ValueError("integrity arm contains an unsupported enum value") from error
    return InferenceArm(
        task_semantic_guidance=_strict_text(
            data["task_semantic_guidance"], "task_semantic_guidance"
        )
        if format_value == ARM_V6_FORMAT
        else LEGACY_TASK_SEMANTICS,
        hotpot_deliberation=strict_bool(data["hotpot_deliberation"], "hotpot_deliberation")
        if format_value == ARM_V6_FORMAT
        else False,
        arm_id=_strict_text(data["condition_id"], "condition_id"),
        optimizer_steps=_strict_nonnegative_int(data["optimizer_steps"], "optimizer_steps"),
        native_thinking=strict_bool(data["native_thinking"], "native_thinking"),
        reasoning_pass=reasoning_pass,
        agent_topology=agent_topology,
        skill_mode=skill_mode,
        handwritten_policy=_strict_mode(data["handwritten_policy_mode"], "handwritten_policy_mode"),
        demonstrations=_strict_mode(data["demonstration_mode"], "demonstration_mode"),
        semantic_action_pruning=strict_bool(
            data["semantic_action_pruning"], "semantic_action_pruning"
        ),
        automatic_catalog_actions=strict_bool(
            data["automatic_catalog_actions"], "automatic_catalog_actions"
        ),
        precomputed_task_context=precomputed_task_context,
        legacy=legacy,
        tool_call_mode=tool_call_mode,
        policy_id=_strict_text(data["policy_id"], "policy_id")
        if format_value == ARM_V4_FORMAT
        or (format_value in {ARM_V5_FORMAT, ARM_V6_FORMAT} and data["policy_id"] is not None)
        else None,
        skill_library_id=_strict_text(data["skill_library_id"], "skill_library_id")
        if format_value in {ARM_V5_FORMAT, ARM_V6_FORMAT} and data["skill_library_id"] is not None
        else None,
        skill_retrieval_rule=_strict_text(data["skill_retrieval_rule"], "skill_retrieval_rule")
        if format_value in {ARM_V5_FORMAT, ARM_V6_FORMAT}
        and data["skill_retrieval_rule"] is not None
        else None,
    )


def decode_integrity_arm(value: Mapping[str, object]) -> InferenceArm:
    """Decode a complete versioned arm from YAML or broker JSON transport."""
    return _decode_versioned_arm(value)


def load_integrity_arm(path: Path) -> InferenceArm:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, Mapping) or not all(type(key) is str for key in data):
        raise ValueError("integrity arm configuration must be a text-keyed mapping")
    return decode_integrity_arm(data)
