"""Synthetic development guidance; no generated demonstrations or benchmark results."""

import asyncio
import json
from dataclasses import replace

import pytest
from skillev_private.experiments import development_collection as development
from skillev_private.experiments.skill_practice import (
    CONTEXT_KEY,
    INSTRUCTION,
    PROFILE,
    PROFILE_V2,
    PROFILE_V3,
    PracticeSessionFactory,
    practice_controls,
)

from skillev.rollout.context import CanonicalInitialContextAssembler
from tests.training.fakes import make_public_tasks
from tests.training.test_development_collection import request_fixture, write
from tests.v3_helpers import CharacterTokenizer


def test_guidance_is_explicit_before_calls_and_not_autonomous_evidence(tmp_path, monkeypatch):
    request, _, _ = request_fixture(tmp_path)
    records, plain = development.selection(request)
    assert "development_practice" not in plain
    request["development_practice"] = PROFILE
    practiced, frozen = development.selection(request)
    assert practiced == records  # native source and hidden evaluator input are untouched
    assert frozen["development_practice"] == practice_controls(PROFILE)
    report = development.aggregate(tmp_path, frozen, elapsed_seconds=1)
    assert not report["autonomous_skill_choice_measurement"]
    assert report["training_evidence_writes"] == 0
    assert report["development_practice"]["read_quota"] is None
    assert report["development_practice"]["reward_bonus"] == 0
    request["comparison_reference"] = write(tmp_path / "plain.json", plain)
    with pytest.raises(ValueError):
        development.selection(request)
    request.pop("comparison_reference")
    request["development_practice"] = "unknown-practice"
    monkeypatch.setattr(development, "observe_service", lambda *_: pytest.fail("unexpected HTTP"))
    with pytest.raises(ValueError):
        asyncio.run(development.run(request, tmp_path / "not-started"))
    assert not (tmp_path / "not-started").exists()


def test_practice_preserves_hydrated_goal_native_bundle_and_original_budget():
    tasks = make_public_tasks(2)
    native = tuple(replace(t, query=f"actual public reset goal {i}") for i, t in enumerate(tasks))
    used = []
    bundles = [object(), object()]

    class Native:
        async def prepare_tasks(self, selected):
            assert selected == tasks
            return native

        def create(self, task):
            assert task in native
            used.append(task)
            return bundles[native.index(task)]

    factory = PracticeSessionFactory(Native(), PROFILE)
    guided = asyncio.run(factory.prepare_tasks(tasks))
    for index, (old, new) in enumerate(zip(native, guided, strict=True)):
        assert replace(new, public_context=old.public_context) == old
        assert CONTEXT_KEY not in old.public_context
        assert factory.create(new) is bundles[index]
        with pytest.raises(ValueError):
            factory.create(replace(new, query="different public goal"))
    assert used == list(native)

    h0 = CanonicalInitialContextAssembler(maximum_h0_tokens=100_000).assemble(
        task=guided[0],
        retrieved_skills=(),
        active_skill_ids=(),
        library_version="synthetic-empty-library",
        tokenizer=CharacterTokenizer(),
    )
    assert INSTRUCTION in h0.text
    assert native[0].query in h0.text
    assert h0.contract.query == native[0].query
    # The note enters the real scoring context, not an unscored hidden teacher.
    assert PROFILE in json.dumps(guided[0].public_context)


def test_prepare_failure_does_not_create_or_replace_an_episode():
    tasks = make_public_tasks(1)

    class Failing:
        async def prepare_tasks(self, selected):
            raise RuntimeError("synthetic native preparation failure")

        def create(self, task):
            pytest.fail("failed native preparation must not be bypassed")

    factory = PracticeSessionFactory(Failing(), PROFILE)
    with pytest.raises(RuntimeError):
        asyncio.run(factory.prepare_tasks(tasks))
    assert not factory.prepared


@pytest.mark.parametrize("profile", [PROFILE, PROFILE_V2, PROFILE_V3])
def test_practice_reaches_actual_typed_alfworld_reasoning_and_action_messages(profile):
    from skillev.policy.phase_context import phase_chat_messages
    from skillev.scoring.rendering import (
        render_forward_prefix_from_parts,
        render_hindsight_prefix_from_parts,
        render_reasoning_prefix,
    )
    from skillev.task_semantic_guidance import PUBLIC_TASK_SEMANTICS_V8
    from tests.rollout.test_native_phase_context import PhaseTokenizer
    from tests.rollout.test_typed_alfworld_history import typed_request

    request = typed_request()

    class Native:
        async def prepare_tasks(self, tasks):
            return tasks

    task = asyncio.run(PracticeSessionFactory(Native(), profile).prepare_tasks((request.task,)))[0]
    h0 = CanonicalInitialContextAssembler(
        maximum_h0_tokens=30_000,
        phase_context=True,
        reasoning_tool_catalog=True,
        action_wire="native-single-tool-call@3",
        public_action_semantics=True,
        task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V8,
    ).assemble(
        task=task,
        retrieved_skills=(),
        active_skill_ids=(),
        library_version=request.library_version,
        tokenizer=PhaseTokenizer(),
        decoding=request.decoding,
    )
    for prefix in (
        render_reasoning_prefix(h0.text, (), 1),
        render_forward_prefix_from_parts(h0.text, (), 1, "synthetic owner draft"),
        render_hindsight_prefix_from_parts(h0.text, (), 1, "real public observation"),
    ):
        messages, _ = phase_chat_messages(prefix.text)
        assert practice_controls(profile)["instruction"] in "\n".join(
            m["content"] for m in messages
        )
        assert request.task.query in "\n".join(m["content"] for m in messages)

        if profile in (PROFILE_V2, PROFILE_V3):
            visible = "\n".join(m["content"] for m in messages)
            assert "fictional-not-current-execution" in visible
            for example in practice_controls(profile)["teaching_material"]["examples"]:
                assert example["case"] in visible
            assert all(m["role"] not in {"assistant", "tool"} for m in messages)


def test_v1_stays_exact_and_v2_comparison_is_an_explicit_new_condition(tmp_path):
    original_instruction = (
        "This is a training-development skill-use practice episode, not an evaluation. "
        "Before committing to a long solution, compare the visible skill summaries with "
        "the task's needs. When a document offers a relevant procedure you have not read, "
        "try reading it early, then apply the parts that actually help the task. "
        "You choose the document and all subsequent actions. If no listed procedure "
        "would help, solve directly. If a read turns out to be inapplicable, disregard it. "
        "Do not reread an already visible document as though it performed another check. "
        "Reading is not execution, verification or task success. There is no call-count "
        "target or extra reward; the original token, turn and tool-call budgets still apply."
    )
    assert practice_controls(PROFILE) == {
        "profile": PROFILE,
        "instruction": original_instruction,
        "autonomous_use_evidence": False,
        "formal_training_condition": False,
        "read_quota": None,
        "reward_bonus": 0,
    }
    request, _, _ = request_fixture(tmp_path)
    request["development_practice"] = PROFILE
    records, old = development.selection(request)
    request["comparison_reference"] = write(tmp_path / "old-practice.json", old)
    request["development_practice"] = PROFILE_V2
    with pytest.raises(ValueError):
        development.selection(request)
    request.pop("comparison_reference")
    current, new = development.selection(request)
    assert current == records
    assert (
        new["development_practice"]["teaching_material"][
            "counts_as_invocation_or_application_evidence"
        ]
        is False
    )
    request["arm"] = "initial-library"
    _, other_arm = development.selection(request)
    assert other_arm["development_practice"] == new["development_practice"]


@pytest.mark.parametrize("profile", [PROFILE_V2, PROFILE_V3])
def test_fictional_examples_are_real_h0_conditions_not_calls_or_scoring_targets(tmp_path, profile):
    from skillev.policy.interface import AdapterRole
    from skillev.rollout import EnvironmentObservation, RolloutArtifact
    from skillev.runtime import BudgetVector
    from skillev.scoring.edge_plan import prepare_edge_plan
    from skillev.task_semantic_guidance import PUBLIC_TASK_SEMANTICS_V8
    from skillev.training.invocation_evidence import invocation_execution_links
    from tests.rollout.engine_fakes import GenerationScript, make_harness
    from tests.rollout.test_native_phase_context import PhaseTokenizer
    from tests.rollout.test_native_tool_wire import call
    from tests.rollout.test_typed_alfworld_history import INITIAL, typed_request

    original = typed_request()

    class Native:
        async def prepare_tasks(self, tasks):
            return tasks

    task = asyncio.run(PracticeSessionFactory(Native(), profile).prepare_tasks((original.task,)))[0]
    request = replace(original, task=task)
    tokenizer = PhaseTokenizer()
    action = call("act", command="look")
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        request=request,
        max_turns=1,
        context_assembler=CanonicalInitialContextAssembler(
            maximum_h0_tokens=30000,
            phase_context=True,
            reasoning_tool_catalog=True,
            action_wire="native-single-tool-call@3",
            public_action_semantics=True,
            task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V8,
        ),
        scripts=[
            GenerationScript.text(tokenizer, "actual synthetic reasoning"),
            GenerationScript.text(tokenizer, action),
        ],
        environment_results=[
            EnvironmentObservation(
                public_value=INITIAL,
                observation_status="success",
                budget_usage=BudgetVector(tool_calls=1),
            )
        ],
    )
    artifact = asyncio.run(harness.engine.run(request))
    if profile == PROFILE_V3:
        # The first real owner reasoning input already has the request. Nothing
        # has inserted a read or a body as history, even when the owner acts directly.
        first_input = bytes(harness.generator.requests[0].input_ids).decode()
        assert practice_controls(profile)["instruction"] in first_input
        assert "requested-teaching-demonstration" in first_input
        assert "actual synthetic reasoning" not in first_input
        assert len(harness.generator.requests) == 2
    restored = RolloutArtifact.from_value(artifact.to_value(), tokenizer=tokenizer)
    plan = prepare_edge_plan(tokenizer, restored.record, restored.initial_context.text)
    assert len(restored.record.steps) == 1
    assert restored.record.steps[0].action_text == action
    assert restored.record.steps[0].invoked_skill_ids == ()
    assert not invocation_execution_links(restored.record)
    for role in (AdapterRole.FORWARD_POLICY, AdapterRole.BACKWARD_POLICY):
        edge = plan.edge(1, role)
        assert b"fictional-not-current-execution" in bytes(edge.prefix_ids)
        assert edge.action_ids == tuple(tokenizer.encode(action))
    assert (
        plan.edge(1, AdapterRole.FORWARD_POLICY).prefix_ids
        == harness.generator.requests[1].input_ids
    )
    assert b"fictional-not-current-execution" in bytes(harness.generator.requests[0].input_ids)
    assert b"actual synthetic reasoning" not in bytes(
        plan.edge(1, AdapterRole.BACKWARD_POLICY).prefix_ids
    )
    assert restored.record.initial_context.retrieved_skill_ids == ()
    assert restored.record.initial_context.active_skill_ids == ()


@pytest.mark.parametrize("previous", [PROFILE, PROFILE_V2])
def test_requested_teaching_is_distinct_from_optional_practice_and_identical_across_arms(
    tmp_path, previous
):
    request, _, _ = request_fixture(tmp_path)
    request["development_practice"] = previous
    original_records, prior = development.selection(request)
    assert "demonstration" not in prior["development_practice"]
    request["comparison_reference"] = write(tmp_path / "prior.json", prior)
    request["development_practice"] = PROFILE_V3
    with pytest.raises(ValueError):
        development.selection(request)
    request.pop("comparison_reference")
    records, teaching = development.selection(request)
    assert records == original_records
    controls = teaching["development_practice"]
    assert controls["demonstration"]["kind"] == "requested-teaching-demonstration"
    assert not controls["autonomous_use_evidence"]
    assert not controls["formal_training_condition"]
    assert not controls["demonstration"]["iid_condition"]
    assert not controls["demonstration"]["framework_inserted_actions"]
    assert controls["read_quota"] is None
    assert controls["reward_bonus"] == 0
    request["arm"] = "initial-library"
    _, other = development.selection(request)
    assert other["development_practice"] == controls
    report = development.aggregate(tmp_path, teaching, elapsed_seconds=1)
    assert report["development_practice"] == controls
    assert not report["autonomous_skill_choice_measurement"]
    assert report["training_evidence_writes"] == 0
