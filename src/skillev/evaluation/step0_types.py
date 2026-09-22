"""Versioned Step-0 evaluation types with explicit evaluation-arm selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .step0_conditions import ReasoningAuthorityMode
from .step0_integrity import InferenceArm, legacy_arm


class StepZeroActionMode(StrEnum):
    COMPLETION = "completion"
    TRIVIA_SEARCH = "trivia-search"
    WEB_SHOP = "webshop"
    ALF_WORLD = "alfworld"
    SCIENCE_WORLD = "scienceworld"


class StepZeroCompletionMode(StrEnum):
    """Backward-compatible benchmark terminal routes."""

    PASSTHROUGH = "passthrough"
    SHORT_ANSWER = "short-answer"
    AIME_BOXED_INTEGER = "aime-boxed-integer"
    NATURAL_LANGUAGE = "natural-language"
    PYTHON_SOURCE = "python-source"


class StepZeroReasoningMode(StrEnum):
    STANDARD = "standard"
    QWEN_THINKING = "qwen-thinking"


AIME_STEP_ZERO_REASONING_TOKEN_FLOOR = 81_920


@dataclass(frozen=True, slots=True, kw_only=True)
class StepZeroTaskBinding:
    """Answer-free routing fixed before generation or scoring."""

    task_id: str
    benchmark_id: str
    task_family: str
    panel_index: int
    action_mode: StepZeroActionMode
    completion_mode: StepZeroCompletionMode = StepZeroCompletionMode.PASSTHROUGH
    reasoning_mode: StepZeroReasoningMode = StepZeroReasoningMode.STANDARD
    reasoning_authority_mode: ReasoningAuthorityMode = ReasoningAuthorityMode.FROZEN_DETERMINISTIC
    root_query: str | None = None
    maximum_trivia_searches: int = 1
    max_reasoning_tokens: int | None = None
    max_action_tokens: int | None = None
    maximum_h0_tokens: int | None = None
    skill_instruction_token_budget: int = 1024
    context_id: str | None = None
    owner_finish_allowed: bool = False

    def __post_init__(self) -> None:
        for name in ("task_id", "benchmark_id", "task_family"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if self.root_query is not None and not self.root_query.strip():
            raise ValueError("root_query must be non-empty or null")
        if self.context_id is not None and (
            type(self.context_id) is not str or not self.context_id.strip()
        ):
            raise ValueError("retrieval context must be non-empty or null")
        if type(self.panel_index) is not int or self.panel_index < 0:
            raise ValueError("panel_index must be non-negative")
        if not isinstance(self.action_mode, StepZeroActionMode):
            raise TypeError("action_mode must be StepZeroActionMode")
        if self.action_mode is StepZeroActionMode.TRIVIA_SEARCH and self.root_query is None:
            raise ValueError("Trivia search requires the stable answer-free question as root_query")
        if not isinstance(self.completion_mode, StepZeroCompletionMode):
            raise TypeError("completion_mode must be StepZeroCompletionMode")
        if not isinstance(self.reasoning_mode, StepZeroReasoningMode):
            raise TypeError("reasoning_mode must be StepZeroReasoningMode")
        if not isinstance(self.reasoning_authority_mode, ReasoningAuthorityMode):
            raise TypeError("reasoning_authority_mode must be ReasoningAuthorityMode")
        if type(self.maximum_trivia_searches) is not int or self.maximum_trivia_searches < 1:
            raise ValueError("maximum_trivia_searches must be a positive integer")
        if (
            self.action_mode is not StepZeroActionMode.TRIVIA_SEARCH
            and self.maximum_trivia_searches != 1
        ):
            raise ValueError("only TriviaQA bindings may change the search budget")
        for name in ("max_reasoning_tokens", "max_action_tokens", "maximum_h0_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be positive or null")
        if (
            type(self.skill_instruction_token_budget) is not int
            or self.skill_instruction_token_budget < 0
        ):
            raise ValueError("skill_instruction_token_budget must be non-negative")


@dataclass(frozen=True, slots=True)
class ArchitectureInferenceState:
    """Training state applied to the otherwise matched architecture runner.

    Step-0 remains the default.  A post-training evaluation must opt in to a
    non-zero state and identify the learned policy explicitly; this keeps the
    benchmark prompts, tools, and scorers unchanged while preventing a loaded
    LoRA from being reported as adapter-free.
    """

    method_id: str = "skillev-bayesian-improve-step-zero@1"
    optimizer_steps: int = 0
    forward_adapter_active: bool = False
    backward_adapter_active: bool = False
    posterior_active: bool = False
    calibration_active: bool = False
    operator_active: bool = False
    action_policy_authority: str = (
        "retrieved-skill-orchestration-plus-adapter-free-evaluation-matched-base"
    )

    def __post_init__(self) -> None:
        if not self.method_id.strip() or not self.action_policy_authority.strip():
            raise ValueError("architecture inference identities must be non-empty")
        if type(self.optimizer_steps) is not int or self.optimizer_steps < 0:
            raise ValueError("optimizer_steps must be a non-negative integer")
        flags = (
            self.forward_adapter_active,
            self.backward_adapter_active,
            self.posterior_active,
            self.calibration_active,
            self.operator_active,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("architecture inference activation flags must be boolean")
        if self.optimizer_steps == 0 and any(flags):
            raise ValueError("Step-0 cannot activate trained architecture state")
        if self.optimizer_steps > 0 and not self.forward_adapter_active:
            raise ValueError("trained inference requires the learned forward adapter")


@dataclass(frozen=True, slots=True)
class StepZeroArchitectureConfig:
    max_turns: int
    max_reasoning_tokens: int
    max_action_tokens: int
    maximum_h0_tokens: int
    base_seed: int
    response_model: str
    service_instance_id: str
    controller_id: str = "bayesian-improve-seeded-controller@1"
    seed_library_id: str = "step0-exact-eight-answer-free-seeds@1"
    inference_state: ArchitectureInferenceState = field(default_factory=ArchitectureInferenceState)
    arm: InferenceArm = field(kw_only=True)

    def __post_init__(self) -> None:
        for name in (
            "max_turns",
            "max_reasoning_tokens",
            "max_action_tokens",
            "maximum_h0_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be positive")
        if type(self.base_seed) is not int or not 0 <= self.base_seed < 2**64:
            raise ValueError("base_seed must be an unsigned 64-bit integer")
        for name in (
            "response_model",
            "service_instance_id",
            "controller_id",
            "seed_library_id",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.arm, InferenceArm):
            raise TypeError("arm must be an InferenceArm")
        if not self.arm.legacy and self.arm.optimizer_steps != self.inference_state.optimizer_steps:
            raise ValueError("arm update count differs from loaded policy state")
        if not isinstance(self.inference_state, ArchitectureInferenceState):
            raise TypeError("inference_state must be an ArchitectureInferenceState")

    @classmethod
    def legacy_reproduction(
        cls,
        *,
        max_turns: int,
        max_reasoning_tokens: int,
        max_action_tokens: int,
        maximum_h0_tokens: int,
        base_seed: int,
        response_model: str,
        service_instance_id: str,
        controller_id: str = "bayesian-improve-seeded-controller@1",
        seed_library_id: str = "step0-exact-eight-answer-free-seeds@1",
        inference_state: ArchitectureInferenceState | None = None,
    ) -> StepZeroArchitectureConfig:
        """Build a configuration for explicitly named historical reproduction only."""
        return cls(
            max_turns=max_turns,
            max_reasoning_tokens=max_reasoning_tokens,
            max_action_tokens=max_action_tokens,
            maximum_h0_tokens=maximum_h0_tokens,
            base_seed=base_seed,
            response_model=response_model,
            service_instance_id=service_instance_id,
            controller_id=controller_id,
            seed_library_id=seed_library_id,
            inference_state=(
                ArchitectureInferenceState() if inference_state is None else inference_state
            ),
            arm=legacy_arm(),
        )


@dataclass(frozen=True, slots=True)
class StepZeroDiagnostics:
    generations: int
    public_catalog_actions: int
    wire_invalid: int
    semantic_invalid: int
    retrieval_abstentions: int
    retrieved_skill_ids: tuple[str, ...]
