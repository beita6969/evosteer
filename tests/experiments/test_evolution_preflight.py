from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

from skillev.calibration import CalibrationConfig
from skillev.contracts import FailureMode, HorizonBucket, TokenBucket, stable_hash
from skillev.diagnostics import (
    BatchDiagnostics,
    ComparableResidualWindows,
    DiagnosticsConfig,
    SkillFlowStat,
)
from skillev.evolution import (
    RETAIN_AUTHORING_SAMPLING,
    AuthoringActionKind,
    AuthoringEdgeEvidence,
    EvidencePack,
    EvolutionConfig,
    EvolutionDecision,
    FullEvolutionPolicy,
    PhaseTransitionDetected,
    PhaseTransitionDetector,
)
from skillev.experiments.evolution_preflight import (
    EvolutionPreflightProof,
    build_planned_evolution_preflight,
    planned_seed_documents,
    write_planned_evolution_preflight,
)
from skillev.experiments.protocol import FIXED_SEED
from skillev.rollout import AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP
from skillev.runtime import BudgetVector, ExactAttemptRunPlan
from skillev.training import conservative_rollout_maximum


def _diagnostic(
    step: int,
    *,
    counts: dict[str, int],
    config: DiagnosticsConfig,
    library_version: str,
    stagnant: bool,
) -> BatchDiagnostics:
    current = 0.99 if stagnant else 0.5
    return BatchDiagnostics(
        batch_id=f"reachability-batch-{step}",
        optimizer_step=step,
        library_version=library_version,
        trajectories=(),
        skill_flows=tuple(
            SkillFlowStat(
                skill_id=skill_id,
                invoking_trajectory_count=1,
                invoking_edge_count=count,
                log_skill_flow=0.0,
            )
            for skill_id, count in sorted(counts.items())
        ),
        residual_window=ComparableResidualWindows(
            library_version=library_version,
            window_size=config.window_size,
            previous_window_mean_delta_squared=1.0,
            current_window_mean_delta_squared=current,
            relative_improvement=1.0 - current,
            rho=config.stagnation_rho,
            stagnant=stagnant,
        ),
        config=config,
    )


def _reachable_phase() -> PhaseTransitionDetected:
    diagnostics = DiagnosticsConfig()
    evolution = EvolutionConfig(generate_min_absolute_log_importance=0.1)
    library_version = "reachability-library-v1"
    detector = PhaseTransitionDetector.fresh(
        evolution_config=evolution,
        diagnostics_config=diagnostics,
        library_version=library_version,
    )

    result = None
    # At steps 98, 99, and 100 the exact 50-batch windows contain equal
    # populations of (A, B, C), then (A, B), and finally A alone.  The
    # resulting entropies are log(3), log(2), and zero under the production
    # entropy window; the last batch also supplies the residual conjunct.
    for step in range(1, 101):
        counts = {
            49: {"skill-a": 1},
            50: {"skill-b": 1},
            51: {"skill-c": 1},
        }.get(step, {})
        result = detector.observe(
            _diagnostic(
                step,
                counts=counts,
                config=diagnostics,
                library_version=library_version,
                stagnant=step == 100,
            )
        )

    assert isinstance(result, PhaseTransitionDetected)
    return result


def _reachable_decision(phase: PhaseTransitionDetected) -> EvolutionDecision:
    exemplar = AuthoringEdgeEvidence(
        edge_id="reachability-trajectory:1",
        task_family="synthetic/reachability",
        context_id="synthetic-context",
        action_kind=AuthoringActionKind.TOOL,
        tool_or_skill_name="synthetic-tool",
        argument_schema_id=stable_hash({"query": "string"}),
        observation_status=FailureMode.SUCCESS,
        token_bucket=TokenBucket.LE_1K,
        horizon_bucket=HorizonBucket.LE_3,
        absolute_log_importance=1.0,
        log_importance_quantile=1.0,
        invoked_skill_ids=(),
        available_tools=("synthetic-tool",),
    )
    pack = EvidencePack(
        phase_event=phase.event,
        skills=(),
        uncovered_importance_edges=(exemplar.edge_id,),
        uncovered_exemplars=(exemplar,),
    )
    return FullEvolutionPolicy().decide(
        pack, EvolutionConfig(generate_min_absolute_log_importance=0.1)
    )


def test_planned_preflight_is_canonical_and_exactly_round_trips(tmp_path: Path) -> None:
    proof = build_planned_evolution_preflight()
    restored = EvolutionPreflightProof.from_value(proof.to_value())
    unsigned = dict(proof.to_value())
    content_hash = unsigned.pop("content_hash")

    assert restored == proof
    assert restored.content_hash == proof.content_hash == content_hash
    assert content_hash == stable_hash(unsigned)

    output = tmp_path / "proof.json"
    write_planned_evolution_preflight(output)
    assert json.loads(output.read_text(encoding="utf-8")) == proof.to_value()


def test_planned_inputs_use_production_defaults_and_public_seed_library() -> None:
    proof = build_planned_evolution_preflight()
    inputs = proof.inputs

    assert inputs.diagnostics_config == DiagnosticsConfig()
    assert inputs.calibration_config == CalibrationConfig()
    assert inputs.evolution_config == EvolutionConfig(generate_min_absolute_log_importance=0.1)
    assert inputs.run_plan == ExactAttemptRunPlan(
        phase_search_steps=4606,
        closure_steps=2,
        maximum_cycles=2,
    )
    assert inputs.batch_size == 1
    assert inputs.max_turns == 15
    assert inputs.max_reasoning_tokens == 1024
    assert inputs.max_action_tokens == 512
    assert inputs.max_model_input_tokens == 65536
    assert inputs.max_tool_wall_time_milliseconds == 300_000
    assert inputs.maximum_h0_tokens == 32768
    assert inputs.maximum_generate_authority_matches_per_task_per_cycle == 1
    assert inputs.minimum_required_committed_cycles == 1
    assert inputs.authoring_sampling == RETAIN_AUTHORING_SAMPLING
    assert inputs.scientific_seed == FIXED_SEED

    documents = planned_seed_documents()
    assert len(documents) == 3
    assert inputs.seed_documents == documents
    assert len({item.manifest.skill_id for item in documents}) == 3
    assert all(
        any(not requirement.immutable for requirement in item.requirements) for item in documents
    )
    assert all(item.manifest.license_id == "CC0-1.0" for item in documents)


def test_production_detector_and_full_policy_have_constructive_reachability() -> None:
    detected = _reachable_phase()

    assert len(detected.triggering_window) == 2 * DiagnosticsConfig().window_size
    entropies = tuple(item.entropy for item in detected.event.entropy_series)
    assert len(entropies) == (
        EvolutionConfig(generate_min_absolute_log_importance=0.1).required_consecutive_drops + 1
    )
    assert all(current < previous for previous, current in pairwise(entropies))
    assert all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
        for actual, expected in zip(
            entropies,
            (math.log(3.0), math.log(2.0), -0.0),
            strict=True,
        )
    )

    decision = _reachable_decision(detected)
    assert tuple(item.evidence.evidence_type for item in decision.proposals) == ("generate",)

    reachability = build_planned_evolution_preflight().reachability
    assert reachability.phase_transition_detected is True
    assert reachability.trigger_optimizer_step == 100
    assert reachability.triggering_window_batch_count == len(detected.triggering_window)
    assert reachability.entropy_values == entropies
    assert reachability.decision_nonempty is True
    assert reachability.decision_proposal_types == ("generate",)
    assert reachability.verified_no_op_reachable is True


def test_budget_envelope_uses_tight_complete_phi_bounds() -> None:
    proof = build_planned_evolution_preflight()
    inputs = proof.inputs
    run_plan = inputs.run_plan
    evolution = inputs.evolution_config

    assert len(proof.cycle_bounds) == run_plan.maximum_cycles == 2
    assert proof.maximum_library_size == 206
    assert proof.maximum_h0_tokens == inputs.maximum_h0_tokens == 32768
    assert inputs.max_model_input_tokens == 65536
    assert proof.maximum_applicable_skills_per_task == 5
    assert proof.maximum_complete_rendered_skill_block_tokens == 3_400
    assert proof.maximum_complete_rendered_skill_block_tokens == AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP
    assert proof.maximum_complete_rendered_skill_block_tokens_in_h0 == 17_000
    assert proof.available_h0_task_and_wrapper_tokens == 15_768
    assert proof.b1_generate_authority_admission_required is True
    assert proof.b1_exact_h0_admission_required is True
    assert proof.b1_minimum_cycle_terminal_admission_required is True
    assert proof.phi_per_cycle_maximum == BudgetVector(
        input_tokens=1_687_552,
        output_tokens=843_776,
        model_calls=206,
    )
    assert proof.total_phi_maximum == proof.phi_per_cycle_maximum.scale(run_plan.maximum_cycles)
    assert proof.cycle_specific_total_phi_maximum == BudgetVector(
        input_tokens=2_531_328,
        output_tokens=1_265_664,
        model_calls=309,
    )
    per_rollout = conservative_rollout_maximum(
        max_turns=inputs.max_turns,
        max_reasoning_tokens=inputs.max_reasoning_tokens,
        max_action_tokens=inputs.max_action_tokens,
        max_model_input_tokens=inputs.max_model_input_tokens,
        max_tool_wall_time_milliseconds=inputs.max_tool_wall_time_milliseconds,
    )
    assert proof.closure_tail_budget == per_rollout.scale(
        run_plan.closure_steps * inputs.batch_size
    )
    assert proof.closure_tail_budget == BudgetVector(
        input_tokens=3_932_160,
        output_tokens=46_080,
        model_calls=60,
        agent_turns=30,
        tool_calls=30,
        wall_time_milliseconds=9_000_000,
    )
    assert tuple(item.triggering_trajectory_capacity for item in proof.cycle_bounds) == (
        100,
        100,
    )
    assert tuple(item.triggering_edge_capacity for item in proof.cycle_bounds) == (
        1500,
        1500,
    )
    assert tuple(item.minimum_entropy_invoking_edges for item in proof.cycle_bounds) == (3, 3)
    assert tuple(item.active_skills_before for item in proof.cycle_bounds) == (3, 106)
    assert tuple(item.split_eligible_skills_before for item in proof.cycle_bounds) == (3, 3)
    assert tuple(item.retain_maximum for item in proof.cycle_bounds) == (3, 106)
    assert tuple(item.refine_maximum for item in proof.cycle_bounds) == (3, 106)
    assert tuple(item.split_maximum for item in proof.cycle_bounds) == (3, 3)
    assert tuple(item.prune_maximum for item in proof.cycle_bounds) == (3, 106)
    assert tuple(item.generate_group_maximum for item in proof.cycle_bounds) == (100, 100)
    assert tuple(item.total_action_maximum for item in proof.cycle_bounds) == (103, 206)
    assert tuple(item.authoring_call_maximum for item in proof.cycle_bounds) == (103, 206)
    assert tuple(item.authoring_prompt_token_maximum for item in proof.cycle_bounds) == (
        843_776,
        1_687_552,
    )
    assert tuple(item.authoring_output_token_maximum for item in proof.cycle_bounds) == (
        421_888,
        843_776,
    )
    assert tuple(item.net_library_growth_maximum for item in proof.cycle_bounds) == (
        103,
        103,
    )
    assert tuple(item.active_skills_after_maximum for item in proof.cycle_bounds) == (
        106,
        206,
    )
    assert all(
        item.authoring_prompt_token_maximum
        == item.authoring_call_maximum * evolution.max_authoring_prompt_tokens
        for item in proof.cycle_bounds
    )
    assert all(
        item.authoring_output_token_maximum
        == item.authoring_call_maximum * evolution.max_authoring_completion_tokens
        for item in proof.cycle_bounds
    )

    fixed = proof.fixed_attempt_budget_plan
    assert fixed.phi_per_cycle_maximum == proof.phi_per_cycle_maximum
    assert fixed.maximum_cycles == run_plan.maximum_cycles
    assert fixed.required() == proof.attempt_budget
    assert proof.attempt_budget == BudgetVector(
        input_tokens=9_063_071_744,
        output_tokens=107_855_872,
        model_calls=138_652,
        agent_turns=69_120,
        tool_calls=69_120,
        wall_time_milliseconds=20_736_000_000,
    )
