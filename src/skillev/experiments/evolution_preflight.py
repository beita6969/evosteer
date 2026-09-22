"""Result-blind F2/F3 reachability and complete-Phi budget planning.

The proof in this module is deliberately constructive rather than empirical.
It uses only public synthetic skills and synthetic diagnostics, executes the
production detector and decision policy, and derives a closed structural
resource envelope from the exact finite run plan.  It is a pre-B1 planning input: it contains no
benchmark row, reward, answer, verifier payload, or model result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from skillev.calibration import CalibrationConfig
from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.diagnostics import DiagnosticsConfig
from skillev.evolution import (
    RETAIN_AUTHORING_SAMPLING,
    AuthoringSamplingConfig,
    EvolutionConfig,
)
from skillev.rollout import AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP
from skillev.runtime import BudgetVector, SkillDocument, SkillLibraryState, require_seed_library
from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan
from skillev.training.planning import FixedAttemptBudgetPlan

from ._evolution_preflight_reachability import (
    EvolutionReachabilityProof,
    build_evolution_reachability,
)
from ._evolution_preflight_seed import planned_seed_documents
from .protocol import FIXED_SEED

EVOLUTION_PREFLIGHT_FORMAT: Final = "skillev-f2-f3-evolution-preflight@2"
EVOLUTION_PREFLIGHT_CLAIM_SCOPE: Final = (
    "result-blind structural reachability and coordinate-wise complete-Phi capacity proof; "
    "not a benchmark result, empirical trigger-rate estimate, B1 freeze, or B2 authorization"
)


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _uint64(value: object, *, field: str) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError(f"{field} must be an unsigned 64-bit integer")
    return value


def _object(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or frozenset(normalized) != fields:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


@dataclass(frozen=True, slots=True)
class EvolutionPreflightInputs:
    seed_documents: tuple[SkillDocument, ...]
    diagnostics_config: DiagnosticsConfig
    calibration_config: CalibrationConfig
    evolution_config: EvolutionConfig
    run_plan: ExactAttemptRunPlan
    authoring_sampling: AuthoringSamplingConfig
    scientific_seed: int
    batch_size: int
    max_turns: int
    max_reasoning_tokens: int
    max_action_tokens: int
    max_model_input_tokens: int
    max_tool_wall_time_milliseconds: int
    maximum_h0_tokens: int
    maximum_generate_authority_matches_per_task_per_cycle: int
    minimum_required_committed_cycles: int

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostics_config, DiagnosticsConfig):
            raise TypeError("diagnostics_config must be DiagnosticsConfig")
        if not isinstance(self.calibration_config, CalibrationConfig):
            raise TypeError("calibration_config must be CalibrationConfig")
        if not isinstance(self.evolution_config, EvolutionConfig):
            raise TypeError("evolution_config must be EvolutionConfig")
        if not isinstance(self.run_plan, ExactAttemptRunPlan):
            raise TypeError("run_plan must be ExactAttemptRunPlan")
        if not isinstance(self.authoring_sampling, AuthoringSamplingConfig):
            raise TypeError("authoring_sampling must be AuthoringSamplingConfig")
        if not isinstance(self.seed_documents, tuple) or any(
            not isinstance(item, SkillDocument) for item in self.seed_documents
        ):
            raise TypeError("seed_documents must contain SkillDocument values")
        if len(self.seed_documents) < self.evolution_config.required_consecutive_drops + 1:
            raise ValueError("seed library is too small for the fixed entropy witness")
        require_seed_library(self.seed_documents)
        _uint64(self.scientific_seed, field="scientific_seed")
        if self.calibration_config.default_k != self.evolution_config.k:
            raise ValueError("calibration and evolution confidence multipliers differ")
        for field in (
            "batch_size",
            "max_turns",
            "max_reasoning_tokens",
            "max_action_tokens",
            "max_model_input_tokens",
            "max_tool_wall_time_milliseconds",
            "maximum_h0_tokens",
            "maximum_generate_authority_matches_per_task_per_cycle",
            "minimum_required_committed_cycles",
        ):
            _positive_int(getattr(self, field), field=field)
        if self.minimum_required_committed_cycles > self.run_plan.maximum_cycles:
            raise ValueError("minimum required cycles exceed the exact run plan")
        if self.maximum_h0_tokens > self.max_model_input_tokens:
            raise ValueError("maximum H0 tokens exceed the per-call model input bound")
        if self.run_plan.phase_search_steps < (
            self.minimum_batches_per_phase * self.run_plan.maximum_cycles
        ):
            raise ValueError("phase-search slots cannot contain the planned maximum cycles")
        if (
            self.evolution_config.entropy_window + self.evolution_config.required_consecutive_drops
            > 2 * self.diagnostics_config.window_size
        ):
            raise ValueError("entropy witness must fit inside the residual triggering window")

    @property
    def initial_library(self) -> SkillLibraryState:
        return SkillLibraryState.from_seed_documents(self.seed_documents)

    @property
    def minimum_batches_per_phase(self) -> int:
        return max(
            2 * self.diagnostics_config.window_size,
            self.evolution_config.entropy_window + self.evolution_config.required_consecutive_drops,
        )

    @property
    def minimum_entropy_invoking_edges(self) -> int:
        return self.evolution_config.required_consecutive_drops + 1

    @property
    def triggering_edge_capacity(self) -> int:
        return 2 * self.diagnostics_config.window_size * self.batch_size * self.max_turns

    @property
    def triggering_trajectory_capacity(self) -> int:
        return 2 * self.diagnostics_config.window_size * self.batch_size

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "authoring_sampling": self.authoring_sampling.to_value(),
            "batch_size": self.batch_size,
            "calibration_config": self.calibration_config.to_value(),
            "diagnostics_config": self.diagnostics_config.to_value(),
            "evolution_config": self.evolution_config.to_value(),
            "max_action_tokens": self.max_action_tokens,
            "max_model_input_tokens": self.max_model_input_tokens,
            "max_reasoning_tokens": self.max_reasoning_tokens,
            "max_tool_wall_time_milliseconds": self.max_tool_wall_time_milliseconds,
            "max_turns": self.max_turns,
            "maximum_h0_tokens": self.maximum_h0_tokens,
            "maximum_generate_authority_matches_per_task_per_cycle": (
                self.maximum_generate_authority_matches_per_task_per_cycle
            ),
            "minimum_required_committed_cycles": self.minimum_required_committed_cycles,
            "run_plan": self.run_plan.to_value(),
            "scientific_seed": self.scientific_seed,
            "seed_documents": [item.to_value() for item in self.seed_documents],
            "seed_library_state_hash": self.initial_library.state_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> EvolutionPreflightInputs:
        data = _object(
            value,
            fields=frozenset(
                {
                    "authoring_sampling",
                    "batch_size",
                    "calibration_config",
                    "diagnostics_config",
                    "evolution_config",
                    "max_action_tokens",
                    "max_model_input_tokens",
                    "max_reasoning_tokens",
                    "max_tool_wall_time_milliseconds",
                    "max_turns",
                    "maximum_h0_tokens",
                    "maximum_generate_authority_matches_per_task_per_cycle",
                    "minimum_required_committed_cycles",
                    "run_plan",
                    "scientific_seed",
                    "seed_documents",
                    "seed_library_state_hash",
                }
            ),
            label="evolution preflight inputs",
        )
        raw_documents = data["seed_documents"]
        if not isinstance(raw_documents, list):
            raise TypeError("seed_documents must be an array")
        result = cls(
            seed_documents=tuple(SkillDocument.from_value(item) for item in raw_documents),
            diagnostics_config=DiagnosticsConfig.from_value(data["diagnostics_config"]),
            calibration_config=CalibrationConfig.from_value(data["calibration_config"]),
            evolution_config=EvolutionConfig.from_value(data["evolution_config"]),
            run_plan=ExactAttemptRunPlan.from_value(data["run_plan"]),
            authoring_sampling=AuthoringSamplingConfig.from_value(data["authoring_sampling"]),
            scientific_seed=_uint64(data["scientific_seed"], field="scientific_seed"),
            batch_size=_positive_int(data["batch_size"], field="batch_size"),
            max_turns=_positive_int(data["max_turns"], field="max_turns"),
            max_reasoning_tokens=_positive_int(
                data["max_reasoning_tokens"], field="max_reasoning_tokens"
            ),
            max_action_tokens=_positive_int(data["max_action_tokens"], field="max_action_tokens"),
            max_model_input_tokens=_positive_int(
                data["max_model_input_tokens"], field="max_model_input_tokens"
            ),
            max_tool_wall_time_milliseconds=_positive_int(
                data["max_tool_wall_time_milliseconds"],
                field="max_tool_wall_time_milliseconds",
            ),
            maximum_h0_tokens=_positive_int(data["maximum_h0_tokens"], field="maximum_h0_tokens"),
            maximum_generate_authority_matches_per_task_per_cycle=_positive_int(
                data["maximum_generate_authority_matches_per_task_per_cycle"],
                field="maximum_generate_authority_matches_per_task_per_cycle",
            ),
            minimum_required_committed_cycles=_positive_int(
                data["minimum_required_committed_cycles"],
                field="minimum_required_committed_cycles",
            ),
        )
        if data["seed_library_state_hash"] != result.initial_library.state_hash:
            raise ValueError("seed library state hash differs from exact documents")
        return result


@dataclass(frozen=True, slots=True)
class EvolutionCycleBounds:
    cycle_ordinal: int
    active_skills_before: int
    split_eligible_skills_before: int
    triggering_trajectory_capacity: int
    triggering_edge_capacity: int
    minimum_entropy_invoking_edges: int
    retain_maximum: int
    refine_maximum: int
    split_maximum: int
    prune_maximum: int
    generate_group_maximum: int
    total_action_maximum: int
    authoring_call_maximum: int
    authoring_prompt_token_maximum: int
    authoring_output_token_maximum: int
    net_library_growth_maximum: int
    active_skills_after_maximum: int

    def __post_init__(self) -> None:
        for field in (
            "cycle_ordinal",
            "active_skills_before",
            "split_eligible_skills_before",
            "triggering_trajectory_capacity",
            "triggering_edge_capacity",
            "minimum_entropy_invoking_edges",
            "active_skills_after_maximum",
        ):
            _positive_int(getattr(self, field), field=field)
        for field in (
            "retain_maximum",
            "refine_maximum",
            "split_maximum",
            "prune_maximum",
            "generate_group_maximum",
            "total_action_maximum",
            "authoring_call_maximum",
            "authoring_prompt_token_maximum",
            "authoring_output_token_maximum",
            "net_library_growth_maximum",
        ):
            _non_negative_int(getattr(self, field), field=field)
        if self.active_skills_after_maximum < self.active_skills_before:
            raise ValueError("maximum active library cannot shrink across a planned cycle")
        if self.active_skills_after_maximum > (
            self.active_skills_before + self.net_library_growth_maximum
        ):
            raise ValueError("cycle library-size envelope is inconsistent")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "active_skills_after_maximum": self.active_skills_after_maximum,
            "active_skills_before": self.active_skills_before,
            "authoring_call_maximum": self.authoring_call_maximum,
            "authoring_output_token_maximum": self.authoring_output_token_maximum,
            "authoring_prompt_token_maximum": self.authoring_prompt_token_maximum,
            "cycle_ordinal": self.cycle_ordinal,
            "generate_group_maximum": self.generate_group_maximum,
            "minimum_entropy_invoking_edges": self.minimum_entropy_invoking_edges,
            "net_library_growth_maximum": self.net_library_growth_maximum,
            "prune_maximum": self.prune_maximum,
            "refine_maximum": self.refine_maximum,
            "retain_maximum": self.retain_maximum,
            "split_maximum": self.split_maximum,
            "split_eligible_skills_before": self.split_eligible_skills_before,
            "total_action_maximum": self.total_action_maximum,
            "triggering_edge_capacity": self.triggering_edge_capacity,
            "triggering_trajectory_capacity": self.triggering_trajectory_capacity,
        }


@dataclass(frozen=True, slots=True)
class EvolutionPreflightProof:
    inputs: EvolutionPreflightInputs
    reachability: EvolutionReachabilityProof
    cycle_bounds: tuple[EvolutionCycleBounds, ...]
    phi_per_cycle_maximum: BudgetVector
    cycle_specific_total_phi_maximum: BudgetVector
    total_phi_maximum: BudgetVector
    closure_tail_budget: BudgetVector
    attempt_budget: BudgetVector
    maximum_library_size: int
    maximum_applicable_skills_per_task: int
    maximum_complete_rendered_skill_block_tokens: int
    maximum_complete_rendered_skill_block_tokens_in_h0: int
    available_h0_task_and_wrapper_tokens: int
    b1_generate_authority_admission_required: bool = True
    b1_exact_h0_admission_required: bool = True
    b1_minimum_cycle_terminal_admission_required: bool = True
    claim_scope: str = EVOLUTION_PREFLIGHT_CLAIM_SCOPE
    format: str = EVOLUTION_PREFLIGHT_FORMAT

    def __post_init__(self) -> None:
        if self.format != EVOLUTION_PREFLIGHT_FORMAT:
            raise ValueError("unsupported evolution preflight format")
        if self.claim_scope != EVOLUTION_PREFLIGHT_CLAIM_SCOPE:
            raise ValueError("evolution preflight claim scope is fixed")
        if not isinstance(self.inputs, EvolutionPreflightInputs):
            raise TypeError("inputs must be EvolutionPreflightInputs")
        if not isinstance(self.reachability, EvolutionReachabilityProof):
            raise TypeError("reachability must be EvolutionReachabilityProof")
        if len(self.cycle_bounds) != self.inputs.run_plan.maximum_cycles:
            raise ValueError("cycle bounds differ from the exact run plan")
        if any(not isinstance(item, EvolutionCycleBounds) for item in self.cycle_bounds):
            raise TypeError("cycle_bounds must contain EvolutionCycleBounds")
        for field in (
            "phi_per_cycle_maximum",
            "cycle_specific_total_phi_maximum",
            "total_phi_maximum",
            "closure_tail_budget",
            "attempt_budget",
        ):
            if not isinstance(getattr(self, field), BudgetVector):
                raise TypeError(f"{field} must be BudgetVector")
        _positive_int(self.maximum_library_size, field="maximum_library_size")
        _positive_int(
            self.maximum_applicable_skills_per_task,
            field="maximum_applicable_skills_per_task",
        )
        _positive_int(
            self.maximum_complete_rendered_skill_block_tokens,
            field="maximum_complete_rendered_skill_block_tokens",
        )
        _positive_int(
            self.maximum_complete_rendered_skill_block_tokens_in_h0,
            field="maximum_complete_rendered_skill_block_tokens_in_h0",
        )
        _positive_int(
            self.available_h0_task_and_wrapper_tokens,
            field="available_h0_task_and_wrapper_tokens",
        )
        if self.b1_generate_authority_admission_required is not True:
            raise ValueError(
                "B1 must verify its task projection bounds matching Generate authorities"
            )
        if self.b1_exact_h0_admission_required is not True:
            raise ValueError("B1 must exact-tokenize every scheduled H0 before freeze")
        if self.b1_minimum_cycle_terminal_admission_required is not True:
            raise ValueError("B1 must make the minimum committed-cycle count terminal")
        if self.cycle_bounds[-1].active_skills_after_maximum != self.maximum_library_size:
            raise ValueError("maximum library size differs from cycle recurrence")
        expected_total_phi = self.phi_per_cycle_maximum.scale(self.inputs.run_plan.maximum_cycles)
        if self.total_phi_maximum != expected_total_phi:
            raise ValueError("total Phi budget differs from the exact cycle envelope")
        exact_calls = sum(item.authoring_call_maximum for item in self.cycle_bounds)
        expected_cycle_specific = BudgetVector(
            input_tokens=(exact_calls * self.inputs.evolution_config.max_authoring_prompt_tokens),
            output_tokens=(
                exact_calls * self.inputs.evolution_config.max_authoring_completion_tokens
            ),
            model_calls=exact_calls,
        )
        if self.cycle_specific_total_phi_maximum != expected_cycle_specific:
            raise ValueError("cycle-specific Phi total differs from structural bounds")
        if self.attempt_budget != self.fixed_attempt_budget_plan.required():
            raise ValueError("attempt budget differs from rollout plus complete Phi")
        if self.maximum_complete_rendered_skill_block_tokens != AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP:
            raise ValueError("complete rendered skill-block limit differs from the frozen cap")
        if self.maximum_complete_rendered_skill_block_tokens_in_h0 != (
            self.maximum_applicable_skills_per_task
            * self.maximum_complete_rendered_skill_block_tokens
        ):
            raise ValueError("H0 complete rendered skill-block envelope is inconsistent")
        expected_applicable = len(self.inputs.seed_documents) + (
            self.inputs.run_plan.maximum_cycles
            * self.inputs.maximum_generate_authority_matches_per_task_per_cycle
        )
        if self.maximum_applicable_skills_per_task != expected_applicable:
            raise ValueError("per-task applicable-skill envelope differs from its B1 constraint")
        if self.maximum_h0_tokens != (
            self.maximum_complete_rendered_skill_block_tokens_in_h0
            + self.available_h0_task_and_wrapper_tokens
        ):
            raise ValueError("H0 task/wrapper capacity differs from the configured cap")

    @property
    def maximum_h0_tokens(self) -> int:
        return self.inputs.maximum_h0_tokens

    @property
    def fixed_attempt_budget_plan(self) -> FixedAttemptBudgetPlan:
        return FixedAttemptBudgetPlan(
            batch_count=self.inputs.run_plan.total_training_steps,
            batch_size=self.inputs.batch_size,
            max_turns=self.inputs.max_turns,
            reasoning_call_maximum=BudgetVector(
                input_tokens=self.inputs.max_model_input_tokens,
                output_tokens=self.inputs.max_reasoning_tokens,
                model_calls=1,
            ),
            action_call_maximum=BudgetVector(
                input_tokens=self.inputs.max_model_input_tokens,
                output_tokens=self.inputs.max_action_tokens,
                model_calls=1,
                agent_turns=1,
            ),
            tool_call_maximum=BudgetVector(
                tool_calls=1,
                wall_time_milliseconds=self.inputs.max_tool_wall_time_milliseconds,
            ),
            maximum_cycles=self.inputs.run_plan.maximum_cycles,
            phi_per_cycle_maximum=self.phi_per_cycle_maximum,
        )

    def _body_value(self) -> dict[str, JsonValue]:
        return {
            "attempt_budget": self.attempt_budget.to_value(),
            "b1_generate_authority_admission_required": (
                self.b1_generate_authority_admission_required
            ),
            "b1_exact_h0_admission_required": self.b1_exact_h0_admission_required,
            "b1_minimum_cycle_terminal_admission_required": (
                self.b1_minimum_cycle_terminal_admission_required
            ),
            "claim_scope": self.claim_scope,
            "closure_tail_budget": self.closure_tail_budget.to_value(),
            "cycle_specific_total_phi_maximum": (self.cycle_specific_total_phi_maximum.to_value()),
            "cycle_bounds": [item.to_value() for item in self.cycle_bounds],
            "format": self.format,
            "inputs": self.inputs.to_value(),
            "maximum_h0_tokens": self.maximum_h0_tokens,
            "maximum_applicable_skills_per_task": self.maximum_applicable_skills_per_task,
            "maximum_complete_rendered_skill_block_tokens": (
                self.maximum_complete_rendered_skill_block_tokens
            ),
            "maximum_complete_rendered_skill_block_tokens_in_h0": (
                self.maximum_complete_rendered_skill_block_tokens_in_h0
            ),
            "maximum_library_size": self.maximum_library_size,
            "phi_per_cycle_maximum": self.phi_per_cycle_maximum.to_value(),
            "reachability": self.reachability.to_value(),
            "available_h0_task_and_wrapper_tokens": self.available_h0_task_and_wrapper_tokens,
            "total_phi_maximum": self.total_phi_maximum.to_value(),
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self._body_value())

    def to_value(self) -> dict[str, JsonValue]:
        return {**self._body_value(), "content_hash": self.content_hash}

    @classmethod
    def from_value(cls, value: object) -> EvolutionPreflightProof:
        data = _object(
            value,
            fields=frozenset(
                {
                    "attempt_budget",
                    "b1_generate_authority_admission_required",
                    "b1_exact_h0_admission_required",
                    "b1_minimum_cycle_terminal_admission_required",
                    "claim_scope",
                    "closure_tail_budget",
                    "cycle_specific_total_phi_maximum",
                    "content_hash",
                    "cycle_bounds",
                    "format",
                    "inputs",
                    "available_h0_task_and_wrapper_tokens",
                    "maximum_h0_tokens",
                    "maximum_applicable_skills_per_task",
                    "maximum_complete_rendered_skill_block_tokens",
                    "maximum_complete_rendered_skill_block_tokens_in_h0",
                    "maximum_library_size",
                    "phi_per_cycle_maximum",
                    "reachability",
                    "total_phi_maximum",
                }
            ),
            label="evolution preflight proof",
        )
        inputs = EvolutionPreflightInputs.from_value(data["inputs"])
        expected = build_evolution_preflight(inputs)
        if data != expected.to_value():
            raise ValueError("evolution preflight differs from its constructive proof")
        return expected


def _cycle_bounds(inputs: EvolutionPreflightInputs) -> tuple[EvolutionCycleBounds, ...]:
    """Return coordinate-wise structural maxima for each planned cycle.

    The maxima intentionally do not assume that one particular posterior or
    action mix is typical.  One canonical action can credit at most one skill,
    so authoring calls and net positive products are bounded by the scientific
    edge population.  Generate authority is constant within a trajectory, so
    mixed-domain Generate groups are additionally bounded by the trajectory
    population, not the edge population.  A Prune is call-free and may target
    every active skill; concurrent Generate proposals can keep the post-cycle
    library non-empty.  The strict entropy witness consumes at least
    ``drops + 1`` covered edges.
    """

    trajectory_capacity = inputs.triggering_trajectory_capacity
    edge_capacity = inputs.triggering_edge_capacity
    entropy_edges = inputs.minimum_entropy_invoking_edges
    generate_maximum = min(trajectory_capacity, edge_capacity - entropy_edges)
    # State is (active skill count, still multi-family/split-eligible count).
    # Generate and Split children are sealed to one task family.  Retain and
    # Refine preserve applicability, so only the initial wildcard lineages can
    # ever consume the one-time Split growth opportunity.
    states = {(len(inputs.seed_documents), len(inputs.seed_documents))}
    output: list[EvolutionCycleBounds] = []
    for ordinal in range(1, inputs.run_plan.maximum_cycles + 1):
        active_maximum = max(active for active, _ in states)
        split_eligible_maximum = max(split_eligible for _, split_eligible in states)
        observed_skill_maximum = min(active_maximum, edge_capacity)
        calls = min(edge_capacity, observed_skill_maximum + generate_maximum)
        split_maximum = min(split_eligible_maximum, edge_capacity - generate_maximum)
        growth = split_maximum + generate_maximum

        next_states: set[tuple[int, int]] = set()
        for active, split_eligible in states:
            for generate_count in range(generate_maximum + 1):
                maximum_split_count = min(
                    split_eligible,
                    edge_capacity - generate_count,
                )
                for split_count in range(maximum_split_count + 1):
                    next_states.add(
                        (
                            active + generate_count + split_count,
                            split_eligible - split_count,
                        )
                    )
        active_after_maximum = max(active for active, _ in next_states)
        bound = EvolutionCycleBounds(
            cycle_ordinal=ordinal,
            active_skills_before=active_maximum,
            split_eligible_skills_before=split_eligible_maximum,
            triggering_trajectory_capacity=trajectory_capacity,
            triggering_edge_capacity=edge_capacity,
            minimum_entropy_invoking_edges=entropy_edges,
            retain_maximum=observed_skill_maximum,
            refine_maximum=observed_skill_maximum,
            split_maximum=split_maximum,
            prune_maximum=active_maximum,
            generate_group_maximum=generate_maximum,
            total_action_maximum=active_maximum + generate_maximum,
            authoring_call_maximum=calls,
            authoring_prompt_token_maximum=(
                calls * inputs.evolution_config.max_authoring_prompt_tokens
            ),
            authoring_output_token_maximum=(
                calls * inputs.evolution_config.max_authoring_completion_tokens
            ),
            net_library_growth_maximum=growth,
            active_skills_after_maximum=active_after_maximum,
        )
        output.append(bound)
        states = next_states
    return tuple(output)


def _rollout_budget(
    inputs: EvolutionPreflightInputs,
    *,
    batch_count: int,
) -> BudgetVector:
    plan = FixedAttemptBudgetPlan(
        batch_count=batch_count,
        batch_size=inputs.batch_size,
        max_turns=inputs.max_turns,
        reasoning_call_maximum=BudgetVector(
            input_tokens=inputs.max_model_input_tokens,
            output_tokens=inputs.max_reasoning_tokens,
            model_calls=1,
        ),
        action_call_maximum=BudgetVector(
            input_tokens=inputs.max_model_input_tokens,
            output_tokens=inputs.max_action_tokens,
            model_calls=1,
            agent_turns=1,
        ),
        tool_call_maximum=BudgetVector(
            tool_calls=1,
            wall_time_milliseconds=inputs.max_tool_wall_time_milliseconds,
        ),
        maximum_cycles=0,
        phi_per_cycle_maximum=BudgetVector(),
    )
    return plan.required()


def build_evolution_preflight(
    inputs: EvolutionPreflightInputs,
) -> EvolutionPreflightProof:
    """Construct and execute the result-blind F2/F3 proof."""

    if not isinstance(inputs, EvolutionPreflightInputs):
        raise TypeError("build_evolution_preflight requires EvolutionPreflightInputs")
    bounds = _cycle_bounds(inputs)
    calls = max(item.authoring_call_maximum for item in bounds)
    cycle_specific_calls = sum(item.authoring_call_maximum for item in bounds)
    # Every initial lineage contributes at most one active descendant to one
    # task: Retain/Refine are one-for-one, while Split children are sealed to
    # disjoint task families.  Generate is sealed to its exact
    # (task_family, context_id, available_tools) authority.  The remaining
    # multiplier is therefore an explicit B1 admission constraint on the
    # frozen public task projections, not an empirical assumption hidden in
    # this bound.
    maximum_applicable = len(inputs.seed_documents) + (
        inputs.run_plan.maximum_cycles
        * inputs.maximum_generate_authority_matches_per_task_per_cycle
    )
    rendered_skill_block_tokens = AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP
    rendered_skill_block_envelope = maximum_applicable * rendered_skill_block_tokens
    if rendered_skill_block_envelope >= inputs.maximum_h0_tokens:
        raise ValueError("complete-Phi rendered skill blocks leave no H0 task/wrapper capacity")
    phi_per_cycle = BudgetVector(
        input_tokens=calls * inputs.evolution_config.max_authoring_prompt_tokens,
        output_tokens=calls * inputs.evolution_config.max_authoring_completion_tokens,
        model_calls=calls,
    )
    fixed_plan = FixedAttemptBudgetPlan(
        batch_count=inputs.run_plan.total_training_steps,
        batch_size=inputs.batch_size,
        max_turns=inputs.max_turns,
        reasoning_call_maximum=BudgetVector(
            input_tokens=inputs.max_model_input_tokens,
            output_tokens=inputs.max_reasoning_tokens,
            model_calls=1,
        ),
        action_call_maximum=BudgetVector(
            input_tokens=inputs.max_model_input_tokens,
            output_tokens=inputs.max_action_tokens,
            model_calls=1,
            agent_turns=1,
        ),
        tool_call_maximum=BudgetVector(
            tool_calls=1,
            wall_time_milliseconds=inputs.max_tool_wall_time_milliseconds,
        ),
        maximum_cycles=inputs.run_plan.maximum_cycles,
        phi_per_cycle_maximum=phi_per_cycle,
    )
    return EvolutionPreflightProof(
        inputs=inputs,
        reachability=build_evolution_reachability(inputs),
        cycle_bounds=bounds,
        phi_per_cycle_maximum=phi_per_cycle,
        cycle_specific_total_phi_maximum=BudgetVector(
            input_tokens=(
                cycle_specific_calls * inputs.evolution_config.max_authoring_prompt_tokens
            ),
            output_tokens=(
                cycle_specific_calls * inputs.evolution_config.max_authoring_completion_tokens
            ),
            model_calls=cycle_specific_calls,
        ),
        total_phi_maximum=phi_per_cycle.scale(inputs.run_plan.maximum_cycles),
        closure_tail_budget=_rollout_budget(
            inputs,
            batch_count=inputs.run_plan.closure_steps,
        ),
        attempt_budget=fixed_plan.required(),
        maximum_library_size=bounds[-1].active_skills_after_maximum,
        maximum_applicable_skills_per_task=maximum_applicable,
        maximum_complete_rendered_skill_block_tokens=rendered_skill_block_tokens,
        maximum_complete_rendered_skill_block_tokens_in_h0=rendered_skill_block_envelope,
        available_h0_task_and_wrapper_tokens=(
            inputs.maximum_h0_tokens - rendered_skill_block_envelope
        ),
        b1_generate_authority_admission_required=True,
        b1_minimum_cycle_terminal_admission_required=True,
    )


def build_planned_evolution_preflight() -> EvolutionPreflightProof:
    """Build the one result-blind planning proof intended for the B1 input."""

    return build_evolution_preflight(
        EvolutionPreflightInputs(
            seed_documents=planned_seed_documents(),
            diagnostics_config=DiagnosticsConfig(),
            calibration_config=CalibrationConfig(),
            evolution_config=EvolutionConfig(generate_min_absolute_log_importance=0.1),
            run_plan=ExactAttemptRunPlan(
                # The reviewed active IID population is traversed once:
                # 4,606 phase-search steps followed by the two final-library
                # closure steps, for 4,608 total training episodes.
                phase_search_steps=4606,
                closure_steps=2,
                maximum_cycles=2,
            ),
            authoring_sampling=RETAIN_AUTHORING_SAMPLING,
            scientific_seed=FIXED_SEED,
            batch_size=1,
            max_turns=15,
            max_reasoning_tokens=1024,
            max_action_tokens=512,
            max_model_input_tokens=65536,
            max_tool_wall_time_milliseconds=300_000,
            maximum_h0_tokens=32768,
            maximum_generate_authority_matches_per_task_per_cycle=1,
            minimum_required_committed_cycles=1,
        )
    )


def write_planned_evolution_preflight(path: Path) -> EvolutionPreflightProof:
    """Write the canonical public proof; this does not freeze or run B1."""

    if not isinstance(path, Path):
        raise TypeError("evolution preflight path must be pathlib.Path")
    proof = build_planned_evolution_preflight()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(proof.to_value()) + "\n", encoding="utf-8")
    return proof


def load_evolution_preflight(path: Path) -> EvolutionPreflightProof:
    if not isinstance(path, Path):
        raise TypeError("evolution preflight path must be pathlib.Path")
    return EvolutionPreflightProof.from_value(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "EVOLUTION_PREFLIGHT_CLAIM_SCOPE",
    "EVOLUTION_PREFLIGHT_FORMAT",
    "EvolutionCycleBounds",
    "EvolutionPreflightInputs",
    "EvolutionPreflightProof",
    "EvolutionReachabilityProof",
    "build_evolution_preflight",
    "build_planned_evolution_preflight",
    "load_evolution_preflight",
    "planned_seed_documents",
    "write_planned_evolution_preflight",
]
