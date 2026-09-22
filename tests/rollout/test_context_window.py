"""Input overflow is a context window, never a new terminal outcome."""

import asyncio
from dataclasses import replace

import pytest

from skillev.diagnostics.rollout_progress import RolloutProgress, bind_progress
from skillev.policy.interface import (
    THINKING_ROLLOUT_PROMPT_ENCODER_VERSION,
    ModelInputWindow,
    encode_policy_prompt,
)
from skillev.rollout import CanonicalInitialContextAssembler, DecodingSnapshot, RolloutArtifact
from skillev.scoring import ScoringDirection, edge_logprob_mean, render_hindsight_prefix
from skillev.scoring.edge_plan import PreparedEdgePlan, prepare_edge_plan
from tests.rollout.engine_fakes import (
    FakeScoringBackbone,
    GenerationScript,
    default_request,
    make_harness,
)
from tests.rollout.test_engine import _skill_action_text, _tool_observation
from tests.training.test_native_thinking_condition import ThinkingTokenizer


def test_window_keeps_original_order_and_uses_all_available_capacity():
    window = ModelInputWindow(12)
    assert window.apply(list(range(10)), list(range(4))) == tuple(range(10))
    # A normal H0 is retained whole; the oldest history is removed.
    assert window.apply(list(range(20)), list(range(4))) == (*range(4), *range(12, 20))
    # Oversized H0: preserve task/system head, interfaces at H0 tail, recent suffix.
    assert window.apply(list(range(30)), list(range(20))) == (0, 1, 2, 17, 18, 19, *range(24, 30))
    assert window.apply(list(range(20)), list(range(20))) == (0, 1, 2, *range(11, 20))
    assert ModelInputWindow.from_value(window.to_value()) == window


@pytest.mark.parametrize("oversized_h0", [False, True])
def test_real_engine_continues_over_65536_and_scoring_uses_exact_sent_tokens(
    tmp_path, oversized_h0
):
    tokenizer = ThinkingTokenizer()
    request = default_request()
    # No task answers/private fixtures: exercise the same lengths as the observed fault.
    if oversized_h0:
        request = replace(request, task=replace(request.task, query="public query " * 6000))
    request = replace(
        request,
        decoding=DecodingSnapshot.create(
            max_reasoning_tokens=128,
            max_action_tokens=4096,
            base_seed=17,
            prompt_encoder_version=THINKING_ROLLOUT_PROMPT_ENCODER_VERSION,
        ),
    )
    answer = '{"kind":"complete","name":"complete","arguments":{"value":{"answer":"public"}}}'
    scripts = [
        GenerationScript.text(tokenizer, text)
        for text in ("reason", _skill_action_text(), "finish", answer)
    ]
    window = ModelInputWindow(65_536)
    harness = make_harness(
        tmp_path,
        scripts=scripts,
        tokenizer=tokenizer,
        request=request,
        context_assembler=CanonicalInitialContextAssembler(
            maximum_h0_tokens=65_536, input_window=window
        ),
        environment_results=[replace(_tool_observation(), public_value={"detail": "x" * 68_000})],
    )
    harness.engine._reasoning_call_maximum = replace(
        harness.engine._reasoning_call_maximum, input_tokens=65_536
    )
    harness.engine._action_call_maximum = replace(
        harness.engine._action_call_maximum, input_tokens=65_536
    )
    row = RolloutProgress(
        canonical_position=0,
        trajectory_id=request.trajectory_id,
        task_domain=request.task.task_family,
        batch_id="batch",
        policy_snapshot_id=harness.generator.snapshot().snapshot_id,
        library_version=request.library_version,
        max_turns=2,
    )
    with bind_progress(row):
        artifact = asyncio.run(harness.engine.run(request))
    sent = harness.generator.requests
    assert len(sent) == 4
    assert artifact.record.horizon == 2
    assert len(harness.evaluator.requests) == 1
    assert artifact.record.reward.success
    assert all(len(r.input_ids) <= 65_536 for r in sent)
    assert all(r.input_ids[0] == (1 if i % 2 == 0 else 0) for i, r in enumerate(sent))
    assert len(sent[2].input_ids) == len(sent[3].input_ids) == 65_536
    phases = row.snapshot()["phases"]
    assert all(p["truncated_input_tokens"] > 0 for p in phases[2:])
    assert all(
        p["original_input_tokens"] == p["input_tokens"] + p["truncated_input_tokens"]
        for p in phases
    )
    assert len(artifact.record.steps[0].observation_text) > 65_536
    restored = RolloutArtifact.from_value(artifact.to_value(), tokenizer=tokenizer)
    assert restored == artifact
    assert ModelInputWindow.from_meta(restored.record.initial_context.meta) == window
    plan = prepare_edge_plan(tokenizer, restored.record, restored.initial_context.text)
    for index, edge in enumerate(plan.edges[:2]):
        assert edge.prefix_ids == sent[index * 2 + 1].input_ids
        assert edge.action_ids == scripts[index * 2 + 1].content_token_ids
    assert all(len(e.prefix_ids) <= 65_536 for e in plan.edges)
    assert (
        PreparedEdgePlan.from_wire_edges(
            plan.wire_edges(),
            record=restored.record,
            initial_text=restored.initial_context.text,
            tokenizer_id=tokenizer.tokenizer_id,
        )
        == plan
    )
    calls = []

    class RecordingBackbone(FakeScoringBackbone):
        def score(self, prefix_ids, action_ids, role):
            calls.append((prefix_ids, action_ids))
            return super().score(prefix_ids, action_ids, role)

    backbone = RecordingBackbone(tokenizer)
    for direction in (ScoringDirection.FORWARD, ScoringDirection.HINDSIGHT):
        for i in (1, 2):
            edge_logprob_mean(
                backbone, restored.record, restored.initial_context.text, i, direction
            )
    assert calls == [(e.prefix_ids, e.action_ids) for e in plan.edges]
    hindsight = render_hindsight_prefix(restored.initial_context.text, restored.record.steps, 1)
    assert (
        plan.edges[2].prefix_ids
        == encode_policy_prompt(
            tokenizer, hindsight.text, initial_text=restored.initial_context.text, window=window
        ).ids
    )
