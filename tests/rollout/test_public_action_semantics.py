"""Public meaning is explicit; legacy carrier text is not parsed heuristically."""

import json
from dataclasses import replace

import pytest
from skillev_private.benchmarks.protocol_v13_training_sessions import _action_contract
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig

from skillev.benchmarks.protocol_v10_action import COMPLETION_WIRE_INSTRUCTION
from skillev.benchmarks.reasoning_guidance import HOTPOT_DELIBERATION
from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.policy.phase_context import phase_chat_messages
from skillev.rollout import CanonicalInitialContextAssembler
from skillev.rollout.action_contract import ActionContract
from skillev.rollout.action_surface import (
    ACTION_SURFACE_FORMAT,
    ACTION_SURFACE_FORMAT_V3,
    ActionSurface,
    PublicActionInstructions,
)
from tests.rollout.engine_fakes import default_request
from tests.rollout.test_native_phase_context import PhaseTokenizer

PUBLIC_MEANINGS = (
    (Protocol13Benchmark.AIME_2026, "integer from 0 through 999"),
    (Protocol13Benchmark.HOTPOT_QA, "Connect relevant evidence"),
    (Protocol13Benchmark.TRIVIA_QA, "final answer in answer"),
    (Protocol13Benchmark.HEALTHBENCH, "complete response to the conversation"),
    (Protocol13Benchmark.HUMAN_EVAL, "Preserve indentation"),
    (Protocol13Benchmark.MBPP_PLUS, "Python implementation requested by the public task"),
    (Protocol13Benchmark.ALF_WORLD, "admissible_commands"),
)


def public_native_context(domain, *, enabled=True, deliberation=True):
    surface, _ = _action_contract(domain, 100, hotpot_deliberation=deliberation)
    request = default_request()
    task = replace(
        request.task,
        query="Synthetic public task",
        public_context={"benchmark_id": domain.value},
        action_surface=surface,
    )
    assembler = CanonicalInitialContextAssembler(
        maximum_h0_tokens=20000,
        phase_context=True,
        action_wire="native-single-tool-call@1",
        public_action_semantics=enabled,
    )
    context = assembler.assemble(
        task=task,
        retrieved_skills=(),
        active_skill_ids=(),
        library_version="synthetic-library",
        tokenizer=PhaseTokenizer(),
    )
    return context, task


@pytest.mark.parametrize(("domain", "meaning"), PUBLIC_MEANINGS)
def test_every_domain_semantics_reaches_reasoning_and_action_without_json_envelope(domain, meaning):
    context, _ = public_native_context(domain)
    reasoning, reasoning_tools = phase_chat_messages(context.text + "Reasoning:\n")
    action, tools = phase_chat_messages(context.text + "Action:\n")
    assert meaning in json.dumps(reasoning)
    assert meaning in json.dumps(action)
    assert meaning in json.dumps(tools)
    assert reasoning_tools is None
    assert COMPLETION_WIRE_INSTRUCTION not in json.dumps([reasoning, action, tools])
    assert '"kind":"complete"' not in json.dumps([reasoning, action, tools])
    assert context.contract.meta["public_action_semantics"] == "public-action-semantics@1"
    assert context.contract.assembler_version.endswith("public-action-semantics@1")


def test_hotpot_setting_changes_public_semantics_but_old_native_condition_stays_unchanged():
    on, _ = public_native_context(Protocol13Benchmark.HOTPOT_QA)
    off, _ = public_native_context(Protocol13Benchmark.HOTPOT_QA, deliberation=False)
    legacy, _ = public_native_context(Protocol13Benchmark.HOTPOT_QA, enabled=False)
    assert HOTPOT_DELIBERATION in on.text
    assert HOTPOT_DELIBERATION not in json.dumps(phase_chat_messages(off.text + "Action:\n"))
    assert HOTPOT_DELIBERATION not in json.dumps(phase_chat_messages(legacy.text + "Action:\n"))


@pytest.mark.parametrize(("domain", "_meaning"), PUBLIC_MEANINGS)
def test_old_json_projection_is_byte_identical_and_surface_roundtrips(domain, _meaning):
    _, task = public_native_context(domain)
    current = task.action_surface
    legacy = replace(current, format=ACTION_SURFACE_FORMAT, instructions=None)
    assert ActionSurface.from_value(legacy.to_value()) == legacy
    assert ActionSurface.from_value(current.to_value()) == current
    assembler = CanonicalInitialContextAssembler(maximum_h0_tokens=20000)
    kwargs = {
        "retrieved_skills": (),
        "active_skill_ids": (),
        "library_version": "synthetic-library",
        "tokenizer": PhaseTokenizer(),
    }
    assert (
        assembler.assemble(task=task, **kwargs).text
        == assembler.assemble(task=replace(task, action_surface=legacy), **kwargs).text
    )
    assert (
        ActionContract.freeze(current).render_public_instruction()
        == ActionContract.freeze(legacy).render_public_instruction()
    )


def test_semantic_json_words_are_preserved_and_unpartitioned_prose_is_not_guessed():
    _, task = public_native_context(Protocol13Benchmark.TRIVIA_QA)
    meaning = "Explain JSON objects and root fields as the requested subject."
    surface = replace(
        task.action_surface,
        format=ACTION_SURFACE_FORMAT_V3,
        public_instructions=(),
        instructions=PublicActionInstructions((meaning,), (COMPLETION_WIRE_INSTRUCTION,)),
    )
    assert ActionContract.freeze(surface).render_native_semantics() == (meaning,)
    legacy = replace(surface, format=ACTION_SURFACE_FORMAT, instructions=None)
    with pytest.raises(ValueError):
        ActionContract.freeze(legacy).render_native_semantics()


def test_semantic_candidate_is_explicit_and_legacy_config_does_not_enable_it(tmp_path):
    old = BayesianFormalConfig(
        format="skillev-bayesian-formal-training@4",
        phase_context=True,
        action_wire="native-single-tool-call@1",
    )
    candidate = replace(
        old, format="skillev-bayesian-formal-training@5", public_action_semantics=True
    )
    import yaml

    path = tmp_path / "semantic-candidate.yaml"
    path.write_text(yaml.safe_dump(candidate.to_value()))
    assert BayesianFormalConfig.load(path) == candidate
    rollout = candidate.application_config("semantic-candidate").trainer.rollout
    assert type(rollout).from_value(rollout.to_value()) == rollout
    assert rollout.public_action_semantics
    assert candidate.condition != old.condition
    assert rollout.condition_id != old.application_config("old").trainer.rollout.condition_id
    assert "public_action_semantics" not in old.to_value()
    assert not old.application_config("old").trainer.rollout.public_action_semantics
    with pytest.raises(ValueError):
        replace(old, public_action_semantics=True)
