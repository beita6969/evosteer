import asyncio
import json
from dataclasses import replace

import pytest

from skillev.contracts import canonical_json
from skillev.policy.phase_context import TYPED_PUBLIC_HISTORY, PhaseContextSpec, phase_chat_messages
from skillev.rollout import CanonicalInitialContextAssembler, DecodingSnapshot
from skillev.rollout.action_surface import RolloutBudgetProfile
from skillev.scoring.rendering import (
    render_forward_prefix_from_parts,
    render_hindsight_prefix_from_parts,
    render_reasoning_prefix,
)
from tests.rollout.engine_fakes import GenerationScript, default_request, make_harness
from tests.rollout.test_native_phase_context import PhaseTokenizer
from tests.rollout.test_native_tool_wire import call, contract


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("catalog", [False, True])
def test_notices_use_each_phase_cap_and_survive_persisted_h0_without_changing_old_prefixes(
    typed, catalog
):
    old = PhaseContextSpec(
        action_wire="native-single-tool-call@3",
        tools_json=canonical_json(list(contract().to_native_tools())),
        reasoning_tool_catalog=catalog,
        history_format=TYPED_PUBLIC_HISTORY if typed else None,
    )
    new = replace(old, reasoning_token_cap=123, action_token_cap=456)
    assert PhaseContextSpec.split(new.wrap("public task"))[0] == new
    assert PhaseContextSpec.split(old.wrap("public task"))[0] == old
    assert "token_cap" not in old.wrap("public task")
    reasoning = render_reasoning_prefix(new.wrap("public task"), (), 1)
    assert "123 tokens" in phase_chat_messages(reasoning.text)[0][0]["content"]
    for renderer, text in (
        (render_forward_prefix_from_parts, "private current R"),
        (render_hindsight_prefix_from_parts, "public observation"),
    ):
        before = phase_chat_messages(renderer(old.wrap("public task"), (), 1, text).text)
        after = phase_chat_messages(renderer(new.wrap("public task"), (), 1, text).text)
        assert after[1] == before[1]
        assert "456 tokens" in after[0][0]["content"]
        assert "not a submitted answer" in after[0][0]["content"]
        # Apart from the new system notice, the original model-visible records are unchanged.
        assert after[0][1:] == before[0][1:]
    backward = phase_chat_messages(
        render_hindsight_prefix_from_parts(new.wrap("public task"), (), 1, "observation").text
    )
    assert "private current R" not in json.dumps(backward)


def test_actual_decoding_caps_do_not_change_capped_output_or_next_turn_behavior(tmp_path):
    tokenizer = PhaseTokenizer()
    draft = "r" * 12
    unfinished = '<tool_call>{"name":"submit_answer","arguments":{"answer":"' + "x" * 144
    assert len(tokenizer.encode(unfinished)) < 220
    action_cap = len(tokenizer.encode(unfinished))
    request = default_request()
    request = replace(
        request,
        task=replace(
            request.task,
            action_surface=contract().surface,
            budget_profile=RolloutBudgetProfile("different-global-profile", 2, 999, 888),
        ),
        retrieved_skills=(),
        active_skill_ids=(),
        decoding=DecodingSnapshot.create(
            max_reasoning_tokens=12,
            max_action_tokens=action_cap,
            base_seed=17,
            action_boundary_version="native-model-stop@1",
        ),
    )
    results = []
    for enabled in (False, True):
        scripts = [
            replace(
                GenerationScript.text(tokenizer, draft, stop_token_ids=()), finish_reason="length"
            ),
            replace(
                GenerationScript.text(tokenizer, unfinished, stop_token_ids=()),
                finish_reason="length",
            ),
            GenerationScript.text(tokenizer, "next"),
            GenerationScript.text(
                tokenizer, call("submit_answer", answer="public synthetic response")
            ),
        ]
        harness = make_harness(
            tmp_path,
            tokenizer=tokenizer,
            request=request,
            scripts=scripts,
            event_name=f"notice-{enabled}.jsonl",
            context_assembler=CanonicalInitialContextAssembler(
                maximum_h0_tokens=20000,
                phase_context=True,
                action_wire="native-single-tool-call@3",
                token_budget_notice=enabled,
            ),
        )
        artifact = asyncio.run(harness.engine.run(request))
        assert len(artifact.record.steps) == 2
        assert artifact.record.steps[0].reasoning_text == draft
        assert artifact.record.steps[0].action_text == unfinished
        assert artifact.record.steps[0].observation_status != "success"
        assert harness.environment.completion_checks == [{"answer": "public synthetic response"}]
        assert [item.max_new_tokens for item in harness.generator.requests] == [
            12,
            action_cap,
            12,
            action_cap,
        ]
        spec, _ = PhaseContextSpec.split(artifact.initial_context.text)
        assert spec.reasoning_token_cap == (12 if enabled else None)
        assert spec.action_token_cap == (action_cap if enabled else None)
        results.append(
            [
                (step.reasoning_text, step.action_text, step.observation_status)
                for step in artifact.record.steps
            ]
        )
    assert results[0] == results[1]


def test_notice_requires_actual_decoding_not_a_profile_guess():
    request = default_request()
    with pytest.raises(ValueError):
        CanonicalInitialContextAssembler(
            maximum_h0_tokens=20000, phase_context=True, token_budget_notice=True
        ).assemble(
            task=request.task,
            retrieved_skills=(),
            active_skill_ids=(),
            library_version=request.library_version,
            tokenizer=PhaseTokenizer(),
        )
