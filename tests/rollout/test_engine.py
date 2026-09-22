from __future__ import annotations

import asyncio
import math
import traceback
from dataclasses import replace
from pathlib import Path

import pytest

from skillev.contracts import JsonValue, TerminalReward, canonical_json
from skillev.rollout import (
    EnvironmentObservation,
    GenerationInfrastructureError,
    GenerationPhase,
    NoSubmissionReason,
    NoTerminalSubmission,
    PolicySnapshot,
    RolloutArtifact,
    RolloutInfrastructureError,
    RolloutInfrastructureKind,
    RolloutTermination,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import (
    ActionKind,
    BudgetVector,
    EnvironmentMethodFailedError,
    EnvironmentMethodTimeoutError,
    EventType,
    ReservationState,
    StructuredAction,
)
from skillev.runtime.event_log_reader import read_event_history
from skillev.scoring import (
    ScoringConfig,
    render_forward_prefix,
    render_forward_prefix_from_parts,
    render_hindsight_prefix,
    render_reasoning_prefix,
    score_trajectory,
)
from tests.rollout.engine_fakes import (
    FIXED_TIME,
    PRIVATE_CANARY,
    ByteTokenizer,
    FakeScoringBackbone,
    FakeTerminalEvaluator,
    GenerationScript,
    ScriptedEnvironment,
    default_request,
    default_snapshot,
    make_harness,
)


def _action_text(action: StructuredAction) -> str:
    return canonical_json(action.to_value())


def _skill_action_text() -> str:
    return _action_text(
        StructuredAction(
            kind=ActionKind.SKILL,
            name="lookup",
            arguments={"query": "public"},
            resource_id="public-tool",
            skill_id="model-selected-skill",
        )
    )


def _complete_action_text(value: object) -> str:
    return _action_text(
        StructuredAction(
            kind=ActionKind.COMPLETE,
            name="complete",
            arguments={"value": value},
        )
    )


def _two_step_scripts(tokenizer: ByteTokenizer) -> list[GenerationScript]:
    return [
        GenerationScript.text(tokenizer, "", stop_token_ids=(900,)),
        GenerationScript.text(tokenizer, _skill_action_text(), stop_token_ids=(901,)),
        GenerationScript.text(tokenizer, "finish", stop_token_ids=(902,)),
        GenerationScript.text(
            tokenizer,
            _complete_action_text({"answer": 6}),
            stop_token_ids=(903,),
        ),
    ]


def _one_step_scripts(
    tokenizer: ByteTokenizer,
    action_text: str,
) -> list[GenerationScript]:
    return [
        GenerationScript.text(tokenizer, "reason", stop_token_ids=(900,)),
        GenerationScript.text(tokenizer, action_text, stop_token_ids=(901,)),
    ]


def _tool_observation() -> EnvironmentObservation:
    return EnvironmentObservation(
        public_value={"result": 6},
        observation_status="success",
        invoked_skill_ids=("model-selected-skill",),
        budget_usage=BudgetVector(tool_calls=1),
    )


COMPLETION_VALIDATOR_CANARY = "PRIVATE-COMPLETION-VALIDATOR-/secret/path"


class FailingCompletionValidatorEnvironment(ScriptedEnvironment):
    def validate_completion(self, submission: JsonValue) -> bool:
        self.completion_checks.append(submission)
        raise RuntimeError(COMPLETION_VALIDATOR_CANARY)


class BrokenTerminalEvaluator(FakeTerminalEvaluator):
    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        self.requests.append(request)
        raise AssertionError("PRIVATE evaluator assertion")


def test_wrapped_action_executes_once_and_keeps_every_sampled_token(tmp_path: Path) -> None:
    tokenizer = ByteTokenizer()
    action = "Action:\n```json\n" + _complete_action_text({"answer": "own answer"}) + "\n```"
    scripts = _one_step_scripts(tokenizer, action)
    harness = make_harness(tmp_path, tokenizer=tokenizer, scripts=scripts)

    artifact = asyncio.run(harness.engine.run(harness.request))

    assert artifact.manifest.termination is RolloutTermination.COMPLETED
    final_evidence = harness.evaluator.requests[0].last_action
    assert final_evidence is not None
    assert final_evidence.text == action
    assert final_evidence.parse_status == "valid"
    assert artifact.record.horizon == 1
    step = artifact.record.steps[0]
    assert step.action_text == action
    assert step.action_token_ids == scripts[1].content_token_ids
    assert harness.environment.completion_checks == [{"answer": "own answer"}]
    assert len(harness.evaluator.requests) == 1
    assert (
        len(harness.generator.requests) == 2
    )  # Reasoning and the original action; no repair call.


def test_two_pass_two_step_artifact_has_exact_canonical_token_identity_and_scores(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    scripts = _two_step_scripts(tokenizer)
    action_spans = (scripts[1].content_token_ids, scripts[3].content_token_ids)
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=scripts,
        environment_results=[_tool_observation()],
        max_turns=2,
    )

    artifact = asyncio.run(harness.engine.run(harness.request))

    assert artifact.record.horizon == 2
    assert artifact.manifest.termination is RolloutTermination.COMPLETED
    assert tuple(request.phase for request in harness.generator.requests) == (
        GenerationPhase.REASONING,
        GenerationPhase.ACTION,
        GenerationPhase.REASONING,
        GenerationPhase.ACTION,
    )
    assert len(harness.generator.requests) == 2 * artifact.record.horizon
    assert all(
        request.expected_policy_snapshot_id == artifact.manifest.policy_snapshot.snapshot_id
        for request in harness.generator.requests
    )
    assert len({request.seed for request in harness.generator.requests}) == 4

    for index, step in enumerate(artifact.record.steps, start=1):
        previous = artifact.record.steps[: index - 1]
        reasoning_request = harness.generator.requests[2 * (index - 1)]
        action_request = harness.generator.requests[2 * (index - 1) + 1]
        reasoning_prompt = render_reasoning_prefix(
            artifact.initial_context.text,
            previous,
            index,
        )
        forward_from_parts = render_forward_prefix_from_parts(
            artifact.initial_context.text,
            previous,
            index,
            step.reasoning_text,
        )
        full_forward = render_forward_prefix(
            artifact.initial_context.text,
            artifact.record.steps,
            index,
        )
        assert reasoning_request.input_ids == (0, *reasoning_prompt.text.encode("utf-8"))
        assert action_request.input_ids == (0, *forward_from_parts.text.encode("utf-8"))
        assert reasoning_request.input_ids != tuple(tokenizer.encode(reasoning_prompt.text))
        assert action_request.input_ids != tuple(tokenizer.encode(forward_from_parts.text))
        assert full_forward.text == forward_from_parts.text
        assert full_forward.prefix_hash == forward_from_parts.prefix_hash
        assert step.forward_prefix_hash == forward_from_parts.prefix_hash
        assert step.action_token_ids == action_spans[index - 1]
        assert step.action_token_count == len(action_spans[index - 1])

    assert artifact.record.steps[0].reasoning_text == ""
    assert (
        artifact.record.steps[0].action_token_count != artifact.record.steps[1].action_token_count
    )
    assert all(
        stop_id not in step.action_token_ids
        for stop_id, step in zip((901, 903), artifact.record.steps, strict=True)
    )
    second_hindsight = render_hindsight_prefix(
        artifact.initial_context.text,
        artifact.record.steps,
        2,
    )
    assert second_hindsight.prefix_hash == artifact.record.steps[1].hindsight_prefix_hash
    assert "finish" not in second_hindsight.text
    assert artifact.record.steps[0].invoked_skill_ids == ("model-selected-skill",)
    assert len(harness.environment.calls) == 1
    assert harness.ledger.settled.agent_turns == artifact.record.horizon
    assert harness.ledger.settled.model_calls == 2 * artifact.record.horizon
    assert harness.ledger.settled.tool_calls == 1

    round_tripped = RolloutArtifact.from_value(
        artifact.to_value(),
        tokenizer=tokenizer,
    )
    assert round_tripped == artifact
    assert round_tripped.record.content_hash == artifact.record.content_hash

    external_score = score_trajectory(
        FakeScoringBackbone(tokenizer),
        artifact.record,
        artifact.initial_context.text,
        ScoringConfig(temperature_beta=1.25),
    )
    assert math.isfinite(float(external_score.delta.detach().item()))
    assert math.isfinite(float(external_score.loss.detach().item()))
    external_score.loss.backward()


@pytest.mark.parametrize(
    ("action_text", "expected_status"),
    [
        ("this is not JSON", "parse_error"),
        (canonical_json({"kind": "tool"}), "schema_invalid"),
    ],
)
def test_invalid_action_is_recorded_without_environment_call_or_resampling(
    tmp_path: Path,
    action_text: str,
    expected_status: str,
) -> None:
    tokenizer = ByteTokenizer()
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=_one_step_scripts(tokenizer, action_text),
        max_turns=1,
    )

    artifact = asyncio.run(harness.engine.run(harness.request))

    assert len(harness.generator.requests) == 2
    assert harness.environment.calls == []
    assert artifact.record.horizon == 1
    assert artifact.record.steps[0].action_text == action_text
    assert artifact.record.steps[0].observation_status == expected_status
    assert artifact.manifest.termination is RolloutTermination.HORIZON_EXHAUSTED
    final_evidence = harness.evaluator.requests[0].last_action
    assert final_evidence is not None
    assert final_evidence.text == action_text
    assert final_evidence.step_index == 1
    assert final_evidence.observation_status == expected_status
    assert final_evidence.parse_status == expected_status.replace("_", "-")


def test_unavailable_skill_action_is_recorded_without_environment_credit(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    unavailable_action = _action_text(
        StructuredAction(
            kind=ActionKind.SKILL,
            name="lookup",
            arguments={"query": "public"},
            resource_id="public-tool",
            skill_id="skill-not-retrieved",
        )
    )
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=_one_step_scripts(tokenizer, unavailable_action),
        max_turns=1,
    )

    artifact = asyncio.run(harness.engine.run(harness.request))

    assert harness.environment.calls == []
    assert artifact.record.steps[0].observation_status == "schema_invalid"
    assert artifact.record.steps[0].invoked_skill_ids == ()


def test_environment_credit_mismatch_rejects_the_complete_rollout(tmp_path: Path) -> None:
    tokenizer = ByteTokenizer()
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=_one_step_scripts(tokenizer, _skill_action_text()),
        environment_results=[
            EnvironmentObservation(
                public_value={"result": "public"},
                observation_status="success",
                invoked_skill_ids=("other-skill",),
                budget_usage=BudgetVector(tool_calls=1),
            )
        ],
        max_turns=1,
    )

    with pytest.raises(RolloutInfrastructureError) as captured:
        asyncio.run(harness.engine.run(harness.request))

    assert (
        captured.value.failure.kind
        is RolloutInfrastructureKind.ENVIRONMENT_SKILL_INVOCATION_MISMATCH
    )
    assert len(harness.environment.calls) == 1
    event_types = {event.event_type for event in read_event_history(harness.emitter.log.path)}
    assert EventType.ROLLOUT_REJECTED in event_types
    assert EventType.TERMINAL_REWARD_RECORDED not in event_types


def test_invalid_completion_shape_consumes_step_then_valid_completion_finishes(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    scripts = [
        GenerationScript.text(tokenizer, "draft"),
        GenerationScript.text(tokenizer, _complete_action_text({"draft": 6})),
        GenerationScript.text(tokenizer, "repair"),
        GenerationScript.text(tokenizer, _complete_action_text({"answer": 6})),
    ]
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=scripts,
        max_turns=2,
    )

    artifact = asyncio.run(harness.engine.run(harness.request))

    assert len(harness.generator.requests) == 4
    assert harness.environment.calls == []
    assert harness.environment.completion_checks == [{"draft": 6}, {"answer": 6}]
    assert tuple(step.observation_status for step in artifact.record.steps) == (
        "schema_invalid",
        "success",
    )
    assert artifact.manifest.termination is RolloutTermination.COMPLETED


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (
            EnvironmentMethodTimeoutError(
                budget_usage=BudgetVector(tool_calls=1, wall_time_milliseconds=5)
            ),
            "timeout",
        ),
        (
            EnvironmentMethodFailedError(
                budget_usage=BudgetVector(tool_calls=1, wall_time_milliseconds=5)
            ),
            "tool_error",
        ),
    ],
)
def test_typed_tool_method_failure_is_public_observation_not_rollout_rejection(
    tmp_path: Path,
    failure: Exception,
    expected_status: str,
) -> None:
    tokenizer = ByteTokenizer()
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=_one_step_scripts(tokenizer, _skill_action_text()),
        environment_results=[failure],
        max_turns=1,
    )

    artifact = asyncio.run(harness.engine.run(harness.request))

    assert artifact.record.steps[0].observation_status == expected_status
    assert artifact.manifest.termination is RolloutTermination.HORIZON_EXHAUSTED
    serialized = canonical_json(
        {
            "events": [event.to_value() for event in read_event_history(harness.emitter.log.path)],
            "record": artifact.record.to_value(),
        }
    )
    assert "environment method" not in serialized


@pytest.mark.parametrize(
    "failure",
    [AssertionError("PRIVATE assertion"), TypeError("PRIVATE type error")],
)
def test_unknown_environment_error_propagates_without_reclassification(
    tmp_path: Path,
    failure: Exception,
) -> None:
    tokenizer = ByteTokenizer()
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=_one_step_scripts(tokenizer, _skill_action_text()),
        environment_results=[failure],
        max_turns=1,
    )

    with pytest.raises(type(failure), match="PRIVATE"):
        asyncio.run(harness.engine.run(harness.request))

    assert harness.generator.ended_episodes == [harness.request.trajectory_id]
    formatted = "".join(traceback.format_exception(failure))
    assert "engine_fakes.py" in formatted
    assert "in execute" in formatted
    reserved_entries = [
        entry for entry in harness.ledger.entries if entry.state is ReservationState.RESERVED
    ]
    assert len(reserved_entries) == 1
    assert reserved_entries[0].settlement is None
    event_types = tuple(event.event_type for event in read_event_history(harness.emitter.log.path))
    assert EventType.ROLLOUT_REJECTED not in event_types
    assert EventType.TERMINAL_REWARD_RECORDED not in event_types
    assert "PRIVATE" not in canonical_json(
        [event.to_value() for event in read_event_history(harness.emitter.log.path)]
    )


def test_typed_generation_infrastructure_error_propagates_without_reclassification(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=[
            GenerationScript(
                content_token_ids=(),
                error=GenerationInfrastructureError("PRIVATE generator detail"),
            )
        ],
        max_turns=1,
    )

    with pytest.raises(GenerationInfrastructureError, match="PRIVATE generator detail"):
        asyncio.run(harness.engine.run(harness.request))

    assert harness.generator.ended_episodes == [harness.request.trajectory_id]
    events = read_event_history(harness.emitter.log.path)
    assert "PRIVATE generator detail" not in canonical_json([event.to_value() for event in events])
    assert any(entry.state is ReservationState.RESERVED for entry in harness.ledger.entries)


def test_unknown_generation_error_propagates_without_bounded_rejection(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=[
            GenerationScript(
                content_token_ids=(),
                error=AssertionError("PRIVATE generator assertion"),
            )
        ],
        max_turns=1,
    )

    with pytest.raises(AssertionError, match="PRIVATE generator assertion"):
        asyncio.run(harness.engine.run(harness.request))

    events = read_event_history(harness.emitter.log.path)
    assert EventType.ROLLOUT_REJECTED not in {event.event_type for event in events}
    assert any(entry.state is ReservationState.RESERVED for entry in harness.ledger.entries)


def test_stop_only_action_is_a_scored_candidate_failure(tmp_path: Path) -> None:
    stop_ids = (201,)
    tokenizer = ByteTokenizer(decode_overrides={stop_ids: "<stop>"})
    evaluator = FakeTerminalEvaluator(value=0.0)
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=[
            GenerationScript.text(tokenizer, "reason"),
            GenerationScript(content_token_ids=(), stop_token_ids=stop_ids),
        ],
        evaluator=evaluator,
        max_turns=2,
    )

    artifact = asyncio.run(harness.engine.run(harness.request))

    assert len(harness.generator.requests) == 2
    assert harness.environment.calls == []
    assert artifact.manifest.termination is RolloutTermination.NO_VALID_COMPLETE_ACTION
    assert artifact.record.horizon == 1
    assert artifact.record.steps[0].action_text == "<stop>"
    assert artifact.record.steps[0].action_token_ids == stop_ids
    assert artifact.record.steps[0].observation_status == "parse_error"
    assert evaluator.requests[0].evaluation_input == NoTerminalSubmission(
        NoSubmissionReason.NO_VALID_COMPLETE_ACTION
    )
    event_types = {event.event_type for event in read_event_history(harness.emitter.log.path)}
    assert EventType.ROLLOUT_REJECTED not in event_types
    assert EventType.TERMINAL_REWARD_RECORDED in event_types


def test_action_with_no_sampled_tokens_is_rejected_without_record(tmp_path: Path) -> None:
    tokenizer = ByteTokenizer()
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=[
            GenerationScript.text(tokenizer, "reason"),
            GenerationScript(content_token_ids=(), stop_token_ids=()),
        ],
        max_turns=1,
    )

    with pytest.raises(RolloutInfrastructureError) as captured:
        asyncio.run(harness.engine.run(harness.request))

    assert captured.value.failure.kind is RolloutInfrastructureKind.EMPTY_ACTION
    assert harness.environment.calls == []
    event_types = {event.event_type for event in read_event_history(harness.emitter.log.path)}
    assert EventType.TERMINAL_REWARD_RECORDED not in event_types


def test_many_to_one_invalid_json_commits_sampled_ids_as_agent_outcome(
    tmp_path: Path,
) -> None:
    generated_action_ids = (777,)
    action_text = "not json"
    tokenizer = ByteTokenizer(decode_overrides={generated_action_ids: action_text})
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=[
            GenerationScript.text(tokenizer, "reason"),
            GenerationScript(
                content_token_ids=generated_action_ids,
                stop_token_ids=(901,),
            ),
        ],
        max_turns=1,
    )

    artifact = asyncio.run(harness.engine.run(harness.request))

    assert harness.generator.requests[-1].phase is GenerationPhase.ACTION
    assert generated_action_ids in tokenizer.decoded_ids
    assert harness.environment.calls == []
    step = artifact.record.steps[0]
    assert step.action_text == action_text
    assert step.action_token_ids is generated_action_ids
    assert step.observation_status == "parse_error"
    event_types = {event.event_type for event in read_event_history(harness.emitter.log.path)}
    assert EventType.TERMINAL_REWARD_RECORDED in event_types


def test_many_to_one_valid_completion_commits_original_sampled_ids(
    tmp_path: Path,
) -> None:
    generated_action_ids = (778,)
    action_text = _complete_action_text({"answer": 6})
    tokenizer = ByteTokenizer(decode_overrides={generated_action_ids: action_text})
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=[
            GenerationScript.text(tokenizer, "reason"),
            GenerationScript(content_token_ids=generated_action_ids, stop_token_ids=(901,)),
        ],
        max_turns=1,
    )

    artifact = asyncio.run(harness.engine.run(harness.request))

    step = artifact.record.steps[0]
    assert step.action_text == action_text
    assert step.action_token_ids is generated_action_ids
    assert harness.environment.completion_checks == [{"answer": 6}]
    assert artifact.manifest.termination is RolloutTermination.COMPLETED
    event_types = {event.event_type for event in read_event_history(harness.emitter.log.path)}
    assert EventType.TERMINAL_REWARD_RECORDED in event_types


def test_stale_returned_snapshot_is_rejected_after_exactly_one_action_pass(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    current = default_snapshot(tokenizer)
    stale = PolicySnapshot.create(
        backbone_id=current.backbone_id,
        forward_adapter_version="forward@stale",
        tokenizer_id=tokenizer.tokenizer_id,
        backend_id=current.backend_id,
        initial_trainable_state_hash=current.initial_trainable_state_hash,
    )
    scripts = _one_step_scripts(tokenizer, _complete_action_text({"answer": 6}))
    scripts[1] = replace(scripts[1], policy_snapshot_id=stale.snapshot_id)
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        snapshot=current,
        scripts=scripts,
        max_turns=1,
    )

    with pytest.raises(RolloutInfrastructureError) as captured:
        asyncio.run(harness.engine.run(harness.request))

    assert captured.value.failure.kind is RolloutInfrastructureKind.POLICY_SNAPSHOT_MISMATCH
    assert len(harness.generator.requests) == 2
    assert harness.evaluator.requests == []


def test_snapshot_change_before_terminal_evaluation_rejects_completed_prefix(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    current = default_snapshot(tokenizer)
    changed = PolicySnapshot.create(
        backbone_id=current.backbone_id,
        forward_adapter_version="forward@2",
        tokenizer_id=tokenizer.tokenizer_id,
        backend_id=current.backend_id,
        initial_trainable_state_hash=current.initial_trainable_state_hash,
    )
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        snapshot=current,
        snapshot_sequence=(current, current, changed),
        scripts=_one_step_scripts(tokenizer, _complete_action_text({"answer": 6})),
        max_turns=1,
    )

    with pytest.raises(RolloutInfrastructureError) as captured:
        asyncio.run(harness.engine.run(harness.request))

    assert captured.value.failure.kind is RolloutInfrastructureKind.POLICY_SNAPSHOT_MISMATCH
    assert len(harness.generator.requests) == 2
    assert harness.evaluator.requests == []


def test_evaluator_infrastructure_error_propagates_without_zero_reward_or_record(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    evaluator = FakeTerminalEvaluator(value=0.0, fail=True)
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=_one_step_scripts(tokenizer, _complete_action_text({"answer": 6})),
        evaluator=evaluator,
        max_turns=1,
    )

    with pytest.raises(TerminalEvaluatorError, match="private evaluator"):
        asyncio.run(harness.engine.run(harness.request))

    assert len(evaluator.requests) == 1
    assert harness.generator.ended_episodes == [harness.request.trajectory_id]
    event_types = {event.event_type for event in read_event_history(harness.emitter.log.path)}
    assert EventType.TERMINAL_REWARD_RECORDED not in event_types


def test_unknown_evaluator_error_propagates_without_rejection_or_record(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    evaluator = BrokenTerminalEvaluator(value=0.0)
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=_one_step_scripts(tokenizer, _complete_action_text({"answer": 6})),
        evaluator=evaluator,
        max_turns=1,
    )

    with pytest.raises(AssertionError, match="PRIVATE evaluator assertion"):
        asyncio.run(harness.engine.run(harness.request))

    assert harness.generator.ended_episodes == [harness.request.trajectory_id]

    event_types = {event.event_type for event in read_event_history(harness.emitter.log.path)}
    assert EventType.ROLLOUT_REJECTED not in event_types
    assert EventType.TERMINAL_REWARD_RECORDED not in event_types
    assert "PRIVATE evaluator assertion" not in canonical_json(
        [event.to_value() for event in read_event_history(harness.emitter.log.path)]
    )


def test_completion_validator_infrastructure_error_propagates(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    environment = FailingCompletionValidatorEnvironment([_tool_observation()])
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=_two_step_scripts(tokenizer),
        environment=environment,
        max_turns=2,
    )

    with pytest.raises(RuntimeError, match=COMPLETION_VALIDATOR_CANARY):
        asyncio.run(harness.engine.run(harness.request))

    assert harness.evaluator.requests == []
    assert environment.completion_checks == [{"answer": 6}]
    assert len(harness.generator.requests) == 4
    assert len(environment.calls) == 1

    events = read_event_history(harness.emitter.log.path)
    event_types = tuple(event.event_type for event in events)
    assert event_types.count(EventType.ROLLOUT_STEP_COMMITTED) == 1
    assert event_types.count(EventType.AGENT_STEP_RECORDED) == 1
    assert EventType.TERMINAL_REWARD_RECORDED not in event_types
    public_surface = canonical_json([event.to_value() for event in events])
    assert COMPLETION_VALIDATOR_CANARY not in public_surface


def test_horizon_reward_is_raw_then_shifted_once(tmp_path: Path) -> None:
    tokenizer = ByteTokenizer()
    evaluator = FakeTerminalEvaluator(value=0.0)
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=_one_step_scripts(tokenizer, "invalid JSON"),
        evaluator=evaluator,
        max_turns=1,
    )

    artifact = asyncio.run(harness.engine.run(harness.request))

    assert evaluator.requests[0].termination is RolloutTermination.HORIZON_EXHAUSTED
    assert evaluator.requests[0].evaluation_input == NoTerminalSubmission(
        NoSubmissionReason.HORIZON_EXHAUSTED
    )
    assert artifact.record.reward.value == 0.0
    assert artifact.record.shifted_reward == harness.request.epsilon_min
    assert artifact.record.shifted_reward == (
        artifact.record.reward.value + harness.request.epsilon_min
    )


@pytest.mark.parametrize(
    ("environment_id", "task_family"),
    [
        ("other-environment", "debug-family"),
        ("debug-environment", "other-family"),
    ],
)
def test_task_environment_identity_mismatch_rejects_before_generation(
    tmp_path: Path,
    environment_id: str,
    task_family: str,
) -> None:
    tokenizer = ByteTokenizer()
    request = default_request(environment_id=environment_id, task_family=task_family)
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=[],
        request=request,
        max_turns=1,
    )

    with pytest.raises(RolloutInfrastructureError) as captured:
        asyncio.run(harness.engine.run(request))

    assert captured.value.failure.kind is RolloutInfrastructureKind.INITIAL_CONTEXT_MISMATCH
    assert harness.generator.requests == []


def test_private_evaluator_canary_never_reaches_generation_or_public_history(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=_two_step_scripts(tokenizer),
        environment_results=[_tool_observation()],
        max_turns=2,
    )

    artifact = asyncio.run(harness.engine.run(harness.request))

    generator_inputs = [
        tokenizer.decode(request.input_ids) for request in harness.generator.requests
    ]
    public_observations = [step.observation_text for step in artifact.record.steps]
    assert all(PRIVATE_CANARY not in text for text in generator_inputs)
    assert all(PRIVATE_CANARY not in text for text in public_observations)
    events = read_event_history(harness.emitter.log.path)
    terminal_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type is EventType.TERMINAL_REWARD_RECORDED
    )
    assert PRIVATE_CANARY not in canonical_json(
        [event.to_value() for event in events[:terminal_index]]
    )
    assert artifact.record.reward.native_payload["private_diagnostic"] == PRIVATE_CANARY


def test_fixed_inputs_and_clock_produce_identical_artifacts(tmp_path: Path) -> None:
    first_tokenizer = ByteTokenizer()
    first = make_harness(
        tmp_path,
        tokenizer=first_tokenizer,
        scripts=_two_step_scripts(first_tokenizer),
        environment_results=[_tool_observation()],
        max_turns=2,
        event_name="first.jsonl",
    )
    second_tokenizer = ByteTokenizer()
    second = make_harness(
        tmp_path,
        tokenizer=second_tokenizer,
        scripts=_two_step_scripts(second_tokenizer),
        environment_results=[_tool_observation()],
        max_turns=2,
        event_name="second.jsonl",
    )

    first_artifact = asyncio.run(first.engine.run(first.request))
    second_artifact = asyncio.run(second.engine.run(second.request))

    assert first_artifact == second_artifact
    assert first_artifact.manifest.started_at == FIXED_TIME
    assert first_artifact.manifest.completed_at == FIXED_TIME


def test_provisional_channel_receives_executed_steps_before_terminal_evaluation(tmp_path):
    tokenizer = ByteTokenizer()
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=_two_step_scripts(tokenizer),
        environment_results=[_tool_observation()],
        max_turns=2,
    )
    received = []

    async def accept(value):
        assert not harness.evaluator.requests
        assert value.step.index == len(received) + 1
        assert len(value.previous_steps) == len(received)
        received.append(value)

    harness.engine._provisional_sink = accept
    artifact = asyncio.run(harness.engine.run(harness.request))
    assert tuple(value.step for value in received) == artifact.record.steps
    assert all(value.initial_text == artifact.initial_context.text for value in received)
    assert all(value.policy == artifact.manifest.policy_snapshot for value in received)
    assert len(harness.evaluator.requests) == 1


def test_provisional_channel_failure_never_produces_a_terminal_failure_label(tmp_path):
    tokenizer = ByteTokenizer()
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        scripts=_two_step_scripts(tokenizer),
        environment_results=[_tool_observation()],
        max_turns=2,
    )

    async def fail(_):
        raise RuntimeError("gradient worker unavailable")

    harness.engine._provisional_sink = fail
    with pytest.raises(RuntimeError):
        asyncio.run(harness.engine.run(harness.request))
    assert not harness.evaluator.requests
