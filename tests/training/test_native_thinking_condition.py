"""The seven-domain owner thinks in c_t; action sampling and TTB stay aligned."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from skillev_private.benchmarks.protocol_v13_training_sessions import _action_contract
from skillev_private.experiments.protocol_v13_training_debug import _application_config

from skillev.benchmarks.reasoning_guidance import HOTPOT_DELIBERATION
from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.policy.interface import (
    THINKING_ROLLOUT_PROMPT_ENCODER_VERSION,
    encode_reasoning_prompt,
)
from skillev.rollout import DecodingSnapshot
from skillev.scoring.edge_plan import prepare_edge_plan
from skillev.training.config import PolicyRolloutConfig
from tests.rollout.engine_fakes import (
    ByteTokenizer,
    GenerationScript,
    ScriptedEnvironment,
    default_request,
    make_harness,
)

SEVEN = tuple(b for b in Protocol13Benchmark if b is not Protocol13Benchmark.WEB_SHOP)


class ThinkingTokenizer(ByteTokenizer):
    def encode_rollout_prompt_with_thinking(self, text):
        return [1, *text.encode()]


@pytest.mark.parametrize("benchmark", SEVEN)
def test_native_reasoning_does_not_relabel_or_rescore_an_action_thinking_prefix(
    tmp_path, benchmark
):
    tokenizer = ThinkingTokenizer()
    request = default_request(task_family=benchmark.value)
    request = replace(
        request,
        decoding=DecodingSnapshot.create(
            max_reasoning_tokens=request.decoding.max_reasoning_tokens,
            max_action_tokens=request.decoding.max_action_tokens,
            base_seed=request.decoding.base_seed,
            prompt_encoder_version=THINKING_ROLLOUT_PROMPT_ENCODER_VERSION,
        ),
    )
    reasoning = "Public reasoning.</think>Public conclusion."
    action = '{"kind":"complete","name":"complete","arguments":{"value":{"answer":"public"}}}'
    scripts = [
        GenerationScript.text(tokenizer, reasoning),
        GenerationScript.text(tokenizer, action),
    ]
    harness = make_harness(
        tmp_path,
        request=request,
        tokenizer=tokenizer,
        scripts=scripts,
        environment=ScriptedEnvironment([], task_family=benchmark.value),
    )
    artifact = asyncio.run(harness.engine.run(request))
    sent = harness.generator.requests
    assert len(sent) == 2
    assert sent[0].input_ids[0] == 1  # Actually selected the native-thinking encoder.
    assert sent[1].input_ids[0] == 0
    assert artifact.record.steps[0].reasoning_text == reasoning
    assert artifact.record.steps[0].action_token_ids == scripts[1].content_token_ids
    plan = prepare_edge_plan(tokenizer, artifact.record, artifact.initial_context.text)
    assert plan.edges[0].prefix_ids == sent[1].input_ids
    assert plan.edges[0].action_ids == scripts[1].content_token_ids
    assert artifact.manifest.prompt_encoder_version == THINKING_ROLLOUT_PROMPT_ENCODER_VERSION
    assert DecodingSnapshot.from_value(request.decoding.to_value()) == request.decoding


@pytest.mark.parametrize("benchmark", SEVEN)
def test_guidance_is_opt_in_hotpot_only_with_unchanged_action_and_budget_contract(benchmark):
    before, budget = _action_contract(
        benchmark, 50 if benchmark is Protocol13Benchmark.ALF_WORLD else None
    )
    after, new_budget = _action_contract(
        benchmark,
        50 if benchmark is Protocol13Benchmark.ALF_WORLD else None,
        hotpot_deliberation=True,
    )
    assert new_budget == budget
    if benchmark is Protocol13Benchmark.HOTPOT_QA:
        assert after.public_instructions[0] == HOTPOT_DELIBERATION + before.public_instructions[0]
        assert after.public_instructions[1:] == before.public_instructions[1:]
        assert (
            replace(
                after,
                public_instructions=before.public_instructions,
                instructions=before.instructions,
            )
            == before
        )
    else:
        assert after == before


def test_current_debug_configuration_explicitly_freezes_thinking_and_legacy_stays_off():
    app, _ = _application_config(run_id="thinking-condition", steps=4)
    config = app.trainer.rollout
    assert config.reasoning_native_thinking
    assert PolicyRolloutConfig.from_value(config.to_value()) == config
    legacy = replace(config, format="skillev-policy-rollout@3", reasoning_native_thinking=False)
    assert not PolicyRolloutConfig.from_value(legacy.to_value()).reasoning_native_thinking
    with pytest.raises(ValueError):
        replace(legacy, reasoning_native_thinking=True)
    with pytest.raises(TypeError):
        encode_reasoning_prompt(ByteTokenizer(), "public", native_thinking=True)
