"""Actual synthetic official resets feed committed contrasts, never actor prompts."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace as N

import pytest
from skillev_private.benchmarks.alfworld import PrivateALFWorldTerminalEvaluator
from skillev_private.benchmarks.alfworld_official import OfficialALFWorldEpisodeFactory
from skillev_private.benchmarks.protocol_v13_training_sessions import (
    _SourceBoundEvaluator,
    _StaticEvaluator,
)

from skillev.contracts import TerminalReward, canonical_json
from skillev.evolution.public_execution import PublicExecutionSnippet
from skillev.evolution.trajectory_contrast import (
    SourceTrajectoryIdentity,
    attach_source_contrasts,
    same_source_contrasts,
)
from skillev.rollout import CanonicalInitialContextAssembler, ModelVisibleMessage
from skillev.scoring.rendering import (
    render_forward_prefix_from_parts,
    render_hindsight_prefix_from_parts,
)
from tests.benchmarks.test_alfworld_official_bridge import (
    INITIAL_COMMANDS,
    INITIAL_OBSERVATION,
    PRIVATE_CANARY,
    _case,
    _private_task,
    _ScriptedOfficialFactory,
)
from tests.benchmarks.test_alfworld_official_bridge import (
    _request as alf_request,
)
from tests.benchmarks.test_protocol_v13_training_sessions import _record, _request
from tests.evaluation.test_native_source_bridge import record as native_record
from tests.evolution.test_authoring_retriever_v3 import _exemplar
from tests.rollout.test_native_phase_context import PhaseTokenizer


def occurrence(original, identity):
    return replace(
        original,
        episode=replace(original.episode, episode_id=identity),
        input=replace(original.input, task_id=identity),
    )


def artifact(identity, reward, *, action="look"):
    return N(
        record=N(
            trajectory_id=identity,
            reward=reward,
            initial_context=N(meta={}),
            steps=(N(action_text=action, observation_text="Public step observation."),),
        ),
        manifest=N(
            policy_snapshot=N(snapshot_id="synthetic-policy"),
            library_version="synthetic-library",
            condition_id="synthetic-condition",
            decoding_snapshot_id="synthetic-decoding",
        ),
    )


def static_reward(record):
    delegate = _StaticEvaluator(record, record.input)
    evaluator = _SourceBoundEvaluator(delegate, record, record.input)
    request = _request(record.input.task_id, "public guess")
    before = asyncio.run(delegate.evaluate(request))
    after = asyncio.run(evaluator.evaluate(request))
    assert replace(after, native_payload=before.native_payload) == before
    return after


def alf_reward(identity, *, seed=None, observation=None, commands=None):
    original = _case()
    public = replace(
        original.public,
        task_id=identity,
        seed=original.public.seed if seed is None else seed,
        public_context={
            "admissible_commands": list(INITIAL_COMMANDS if commands is None else commands),
            "initial_observation": INITIAL_OBSERVATION if observation is None else observation,
        },
    )
    case = replace(original, public=public, private_task=_private_task(public))
    factory = _ScriptedOfficialFactory(
        public, reset_observation_override=observation, reset_commands_override=commands
    )
    session = OfficialALFWorldEpisodeFactory(factory).create(case)
    public_task = public.to_rollout_task()
    original_record = native_record("alfworld", "alfworld-success", {"hidden_goal": PRIVATE_CANARY})
    record = replace(
        original_record,
        episode=replace(
            original_record.episode, episode_id=identity, source_id=case.private_task.game_id
        ),
        input=public_task,
    )
    delegate = PrivateALFWorldTerminalEvaluator(
        public, session.outcome_view, session.observed_reset_json
    )
    evaluator = _SourceBoundEvaluator(delegate, record, public_task)
    # No environment step is needed to preserve a valid initial-state binding;
    # an unsubmitted terminal reward remains exactly its native zero result.
    from skillev.rollout import NoSubmissionReason, NoTerminalSubmission, RolloutTermination

    request = replace(
        alf_request(public, None),
        termination=RolloutTermination.HORIZON_EXHAUSTED,
        evaluation_input=NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED),
    )
    before = asyncio.run(delegate.evaluate(request))
    reward = asyncio.run(evaluator.evaluate(request))
    assert replace(reward, native_payload=before.native_payload) == before
    assert factory.envs[0].reset_calls == [public.seed]
    assert factory.envs[0].actions == []
    asyncio.run(session.cleanup())
    return reward, session, public_task


def test_actual_verified_alf_reset_reaches_reward_once_without_private_goal_or_prompt_changes():
    reward, session, public_task = alf_reward("synthetic-left")
    captured = json.loads(session.observed_reset_json)
    assert captured["observation_text"] == INITIAL_OBSERVATION
    assert captured["admissible_commands"] == list(INITIAL_COMMANDS)
    assert captured["instruction_text"] == public_task.query
    binding = reward.native_payload["training_evidence_source"]["reset_binding"]
    assert binding["kind"] == "alfworld-observed-reset"
    assert binding["observed_reset"] == captured
    assert PRIVATE_CANARY not in json.dumps(binding)
    assert "reset_binding" not in json.dumps(public_task.to_value())
    other, _, _ = alf_reward("synthetic-right")
    contrasts = same_source_contrasts(
        (artifact("left", reward), artifact("right", other, action="inventory"))
    )
    assert len(contrasts) == 1
    public = contrasts[0].authoring_value()
    assert public["reset_evidence"]["kind"] == "alfworld-observed-reset"
    assert "observed_reset" not in public
    assert INITIAL_OBSERVATION not in json.dumps(public)
    assert public["left_complete_result_reference"] == "trajectory:left"


@pytest.mark.parametrize(
    "change",
    [{"seed": 1730}, {"observation": "A genuinely different room."}, {"commands": ("inventory",)}],
)
def test_independently_verified_but_different_alf_initial_states_do_not_match(change):
    left, _, _ = alf_reward("synthetic-left")
    right, _, _ = alf_reward("synthetic-right", **change)
    assert not same_source_contrasts(
        (artifact("left", left), artifact("right", right, action="inventory"))
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"seed_override": 1730},
        {"reset_observation_override": "Wrong actual reset."},
        {"reset_commands_override": ("inventory",)},
    ],
)
def test_factory_mismatch_never_produces_an_observed_reset_binding(changes):
    case = _case()
    factory = _ScriptedOfficialFactory(case.public, **changes)
    with pytest.raises(ValueError):
        OfficialALFWorldEpisodeFactory(factory).create(case)
    assert all(len(env.reset_calls) <= 1 for env in factory.envs)


def test_static_task_matches_across_occurrences_and_authors_get_only_references():
    original = _record()
    marker = "SYNTHETIC_PUBLIC_SCAFFOLD_NOT_FOR_DUPLICATION " * 80
    original = replace(original, input=replace(original.input, query=marker))
    left = occurrence(original, "occurrence-left")
    right = occurrence(original, "occurrence-right")
    lreward, rreward = static_reward(left), static_reward(right)
    assert TerminalReward.from_value(lreward.to_value()) == lreward
    binding = lreward.native_payload["training_evidence_source"]["reset_binding"]
    assert binding["kind"] == "static-no-environment-reset"
    assert binding["observed_reset"] is None
    assert "task_id" not in binding["public_task"]
    assert binding == rreward.native_payload["training_evidence_source"]["reset_binding"]
    artifacts = (artifact("left", lreward), artifact("right", rreward, action="submit"))
    edge = replace(
        _exemplar(),
        edge_id="left:1",
        public_execution=PublicExecutionSnippet("look", "Public step observation."),
    )
    (attached,) = attach_source_contrasts(artifacts, (edge,))
    assert len(attached.public_execution.contrasts_json) == 1
    value = attached.public_execution.to_value()
    assert marker not in canonical_json(value)
    assert "private answer" not in canonical_json(value)
    assert "public_task" not in canonical_json(value)
    assert PublicExecutionSnippet.from_value(value) == attached.public_execution


@pytest.mark.parametrize("change", ["context", "scaffold", "seed", "conversation"])
def test_static_projection_keeps_actual_public_context_scaffold_seed_and_dialogue(change):
    original = _record()
    left = occurrence(original, "left")
    right = occurrence(original, "right")
    if change == "scaffold":
        right = replace(
            right, input=replace(right.input, query="A different public function signature.")
        )
    elif change == "conversation":
        right = replace(
            right,
            input=replace(
                right.input,
                model_visible_messages=(ModelVisibleMessage("user", "Different public dialogue."),),
            ),
        )
    else:
        context = {**right.input.public_context, change: "different public value"}
        right = replace(right, input=replace(right.input, public_context=context))
    assert not same_source_contrasts(
        (
            artifact("left", static_reward(left)),
            artifact("right", static_reward(right), action="other"),
        )
    )


def test_static_initial_projection_is_captured_before_actions_and_f_b_inputs_are_unchanged():
    record = _record()
    tokenizer = PhaseTokenizer()
    assembler = CanonicalInitialContextAssembler(maximum_h0_tokens=30000)
    args = {
        "task": record.input,
        "retrieved_skills": (),
        "active_skill_ids": (),
        "library_version": "synthetic-library",
        "tokenizer": tokenizer,
    }
    initial = assembler.assemble(**args)
    delegate = _StaticEvaluator(record, record.input)
    evaluator = _SourceBoundEvaluator(delegate, record, record.input)
    after = asyncio.run(evaluator.evaluate(_request(record.input.task_id, "public guess")))
    assert assembler.assemble(**args).text == initial.text
    for render in (render_forward_prefix_from_parts, render_hindsight_prefix_from_parts):
        assert (
            render(initial.text, (), 1, "public suffix").text
            == render(assembler.assemble(**args).text, (), 1, "public suffix").text
        )
    record.input.public_context["late_mutation"] = "later public state"
    later = asyncio.run(evaluator.evaluate(_request(record.input.task_id, "public guess")))
    assert (
        later.native_payload["training_evidence_source"]["reset_binding"]
        == after.native_payload["training_evidence_source"]["reset_binding"]
    )


def test_missing_old_evidence_is_not_backfilled_or_reinterpreted():
    record = _record()
    delegate = _StaticEvaluator(record, record.input)
    old = asyncio.run(
        _SourceBoundEvaluator(delegate, record).evaluate(
            _request(record.input.task_id, "public guess")
        )
    )
    assert "reset_binding" not in old.native_payload["training_evidence_source"]
    assert SourceTrajectoryIdentity.from_artifact(artifact("old", old)) is None
