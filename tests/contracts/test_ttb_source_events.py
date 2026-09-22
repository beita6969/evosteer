from __future__ import annotations

import math
from dataclasses import replace

import pytest

from skillev.contracts import (
    AuthoringUsageValue,
    EdgeLogprobRecord,
    EntropyObservation,
    EvolutionCycleCommitted,
    EvolutionMutationValue,
    EvolutionNoOpCommitted,
    EvolutionPhaseOpened,
    GenerateActionRecord,
    GenerateEvidence,
    InitialContext,
    LibraryInitialized,
    PhaseTransitionEvent,
    PosteriorBatchUpdate,
    RunCursorValue,
    SuccessRule,
    TerminalReward,
    TrainingStepCommit,
    TrainingStepReportValue,
    TrajectoryResidual,
    TrajectoryStep,
    TTBBatchStats,
    WindowStats,
    build_trajectory_record,
    canonical_json,
    parse_canonical_json,
    stable_hash,
)

NOW = "2026-07-25T00:00:00+00:00"


class CharacterTokenizer:
    @property
    def tokenizer_id(self) -> str:
        return "character@1"

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _trajectory():
    action = (
        '{"arguments":{},"kind":"skill","name":"debug-skill",'
        '"resource_id":"debug.skill","skill_id":"skill-a"}'
    )
    step = TrajectoryStep(
        index=1,
        reasoning_text="reason",
        action_text=action,
        action_token_ids=tuple(CharacterTokenizer().encode(action)),
        action_token_count=len(action),
        observation_text="complete",
        observation_status="success",
        invoked_skill_ids=("skill-a",),
        forward_prefix_hash=stable_hash({"prefix": "forward"}),
        hindsight_prefix_hash=stable_hash({"prefix": "backward"}),
    )
    reward = TerminalReward(
        value=1.0,
        success=True,
        success_rule=SuccessRule.R_EQUALS_ONE,
        success_threshold=None,
        native_metric_name="exact",
        native_payload={"correct": True},
        environment_id="debug",
        verifier_version="v1",
    )
    return build_trajectory_record(
        tokenizer=CharacterTokenizer(),
        trajectory_id="trajectory-1",
        environment_id="debug",
        task_family="debug",
        initial_context=InitialContext(
            query="q",
            retrieved_skill_ids=("skill-a",),
            active_skill_ids=("skill-a",),
            meta={
                "environment_id": "debug",
                "task_family": "debug",
                "task_id": "task-1",
            },
            assembler_version="assembler@1",
            assembled_hash=stable_hash({"h0": "q"}),
            assembled_token_count=1,
        ),
        steps=(step,),
        horizon=1,
        reward=reward,
        shifted_reward=1.01,
        epsilon_min=0.01,
        tokenizer_id=CharacterTokenizer().tokenizer_id,
        decoding_snapshot_id="decoding@1",
        created_at=NOW,
    )


def _training_step() -> TrainingStepCommit:
    record = _trajectory()
    edge = EdgeLogprobRecord(
        trajectory_id=record.trajectory_id,
        step_index=1,
        forward_logprob_per_token=-0.2,
        backward_logprob_per_token=-0.3,
        step_importance=0.1,
        forward_adapter_version="forward@0",
        backward_adapter_version="backward@0",
        scoring_stack_id="training-stack",
    )
    log_reward = math.log(record.shifted_reward)
    delta = 0.1 - log_reward
    residual = TrajectoryResidual(
        trajectory_id=record.trajectory_id,
        log_z=0.0,
        sum_forward=-0.2,
        sum_backward=-0.3,
        log_shifted_reward=log_reward,
        raw_reward=1.0,
        temperature_beta=1.0,
        delta=delta,
        horizon=1,
    )
    stats = TTBBatchStats(
        batch_id="batch-1",
        optimizer_step=1,
        library_version="library-v1",
        residuals=(residual,),
        batch_loss=delta**2,
        mean_reward=1.0,
        created_at=NOW,
    )
    report = TrainingStepReportValue(
        optimizer_step=1,
        batch_id="batch-1",
        torch_batch_loss=delta**2,
        audited_batch_loss=delta**2,
        mean_reward=1.0,
        grad_norm_forward=0.1,
        grad_norm_backward=0.1,
        grad_norm_z=0.1,
        forward_adapter_version="forward@1",
        backward_adapter_version="backward@1",
        z_version="z@1",
        started_at=NOW,
        completed_at=NOW,
    )
    return TrainingStepCommit(
        batch_id="batch-1",
        optimizer_step=1,
        policy_snapshot_before="policy@0",
        policy_snapshot_after="policy@1",
        library_version="library-v1",
        records=(record,),
        edge_records=(edge,),
        stats=stats,
        posterior_batch=PosteriorBatchUpdate(batch_id="batch-1", updates=()),
        report=report,
        run_cursor_after=RunCursorValue(
            run_plan_hash=stable_hash({"fixture": "run-plan"}),
            completed_training_steps=1,
            committed_cycles=0,
            committed_actions=0,
        ),
    )


def _phase() -> PhaseTransitionEvent:
    previous = WindowStats(
        start_optimizer_step=1,
        end_optimizer_step=2,
        batch_count=2,
        mean_squared_residual=1.0,
        member_batch_ids=("batch-1", "batch-2"),
    )
    current = WindowStats(
        start_optimizer_step=3,
        end_optimizer_step=4,
        batch_count=2,
        mean_squared_residual=0.99,
        member_batch_ids=("batch-3", "batch-4"),
    )
    return PhaseTransitionEvent(
        event_id="phase-1",
        triggered_at_step=4,
        library_version="library-v1",
        previous_window=previous,
        current_window=current,
        relative_improvement=0.01,
        rho=0.05,
        residual_condition_met=True,
        entropy_series=(
            EntropyObservation(2, 0.9, 3, 3),
            EntropyObservation(3, 0.8, 3, 3),
            EntropyObservation(4, 0.7, 2, 2),
        ),
        required_consecutive_drops=2,
        entropy_condition_met=True,
        triggered=True,
    )


def _cycle() -> EvolutionCycleCommitted:
    proposal_hash = stable_hash({"fixture": "generate-proposal"})
    action = GenerateActionRecord(
        action_id="action-1",
        phase_event_id="phase-1",
        library_version_before="library-v1",
        library_version_after="library-v2",
        produced_skill_id="skill-new",
        lineage_ref="lineage-1",
        evidence=GenerateEvidence(("edge-1",), 0.1, 0.9, "absolute-log-density-ratio@1"),
        proposal_content_hash=proposal_hash,
        rationale_text="Uncovered important edge.",
    )
    mutation = EvolutionMutationValue(
        library_version_before="library-v1",
        library_version_after="library-v2",
        new_documents=({"skill_id": "skill-new", "content": "new skill"},),
        active_skill_ids_after=("skill-a", "skill-new"),
        actions=(action,),
    )
    return EvolutionCycleCommitted(
        phase_event_id="phase-1",
        optimizer_step=4,
        mutation=mutation,
        decision_content_hash=stable_hash({"fixture": "decision"}),
        proposal_content_hashes=(proposal_hash,),
        authoring_reservation_ids=("authoring-reservation-1",),
        authoring_usage=AuthoringUsageValue(
            input_tokens=12,
            output_tokens=4,
            model_calls=1,
        ),
        z_reset_seed=17,
        z_version_after_reset="z-reset-0000000000000011@0",
        library_version_before="library-v1",
        library_version_after="library-v2",
        run_cursor_after=RunCursorValue(
            run_plan_hash=stable_hash({"fixture": "run-plan"}),
            completed_training_steps=4,
            committed_cycles=1,
            committed_actions=1,
        ),
    )


@pytest.mark.parametrize(
    "value",
    [
        _training_step(),
        LibraryInitialized(
            documents=({"skill_id": "skill-a", "content": "seed"},),
            active_skill_ids=("skill-a",),
            library_version="library-v1",
            initial_optimizer_step=0,
            method_identity_hash=stable_hash({"fixture": "method"}),
            run_cursor=RunCursorValue(
                run_plan_hash=stable_hash({"fixture": "run-plan"}),
                completed_training_steps=0,
                committed_cycles=0,
                committed_actions=0,
            ),
        ),
        EvolutionPhaseOpened(
            phase_event=_phase(),
            run_cursor_at_phase=RunCursorValue(
                run_plan_hash=stable_hash({"fixture": "run-plan"}),
                completed_training_steps=4,
                committed_cycles=0,
                committed_actions=0,
            ),
        ),
        EvolutionNoOpCommitted(
            phase_event_id="phase-1",
            optimizer_step=4,
            decision_reason="no-frozen-threshold-admits-an-evolution-action",
            library_version="library-v1",
            run_cursor_after=RunCursorValue(
                run_plan_hash=stable_hash({"fixture": "run-plan"}),
                completed_training_steps=4,
                committed_cycles=0,
                committed_actions=0,
            ),
        ),
        _cycle(),
    ],
)
def test_source_event_values_round_trip_exactly(value: object) -> None:
    payload = parse_canonical_json(canonical_json(value.to_value()))  # type: ignore[attr-defined]
    rebuilt = type(value).from_value(payload)

    assert rebuilt == value
    assert rebuilt.content_hash == value.content_hash


def test_training_step_commit_binds_all_members_to_one_batch() -> None:
    commit = _training_step()

    with pytest.raises(ValueError):
        replace(
            commit,
            posterior_batch=PosteriorBatchUpdate(batch_id="other", updates=()),
        )
    with pytest.raises(ValueError):
        replace(commit, report=replace(commit.report, optimizer_step=2))


def test_training_source_replay_rejects_tampered_skill_credit() -> None:
    payload = _training_step().to_value()
    records = payload["records"]
    assert isinstance(records, list)
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, dict)
    steps = record["steps"]
    assert isinstance(steps, list)
    assert len(steps) == 1
    step = steps[0]
    assert isinstance(step, dict)
    step["invoked_skill_ids"] = []

    with pytest.raises(ValueError):
        TrainingStepCommit.from_value(payload)


def test_cycle_commit_binds_mutation_and_library_versions() -> None:
    cycle = _cycle()

    with pytest.raises(ValueError):
        replace(cycle, library_version_after="library-v3")


def test_source_events_do_not_self_attest_with_nested_hash_ids() -> None:
    payloads = (
        _training_step().to_value(),
        EvolutionPhaseOpened(
            _phase(),
            RunCursorValue(
                run_plan_hash=stable_hash({"fixture": "run-plan"}),
                completed_training_steps=4,
                committed_cycles=0,
                committed_actions=0,
            ),
        ).to_value(),
        _cycle().to_value(),
    )

    forbidden = {
        "config_summary_hash",
        "cycle_id",
        "descriptor_id",
        "mutation_id",
        "optimizer_artifact_hash",
        "state_fingerprint",
        "trainer_state_hash",
    }
    assert all(forbidden.isdisjoint(payload) for payload in payloads)
