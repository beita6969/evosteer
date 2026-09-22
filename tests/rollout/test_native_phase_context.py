import asyncio
import json
from dataclasses import replace

import pytest

from skillev.policy.interface import AdapterRole
from skillev.policy.phase_context import PhaseContextSpec, phase_chat_messages
from skillev.policy.versions import TrainableVersions
from skillev.rollout import CanonicalInitialContextAssembler, DecodingSnapshot, RolloutArtifact
from skillev.rollout.provisional import ProvisionalStep
from skillev.scoring.edge_plan import prepare_edge_plan
from skillev.scoring.rendering import (
    render_forward_prefix_from_parts,
    render_hindsight_prefix_from_parts,
)
from skillev.training.provisional_math import prepare_provisional_step
from tests.rollout.engine_fakes import (
    ByteTokenizer,
    FakeScoringBackbone,
    GenerationScript,
    default_request,
    make_harness,
)
from tests.rollout.test_native_tool_wire import call, contract


class PhaseTokenizer(ByteTokenizer):
    def encode_rollout_prompt(self, text):
        messages, tools = phase_chat_messages(text)
        return list(json.dumps([messages, tools], ensure_ascii=False).encode())


@pytest.mark.parametrize(
    "wire", ["native-single-tool-call@1", "native-single-tool-call@2", "native-single-tool-call@3"]
)
@pytest.mark.parametrize("reasoning_tool_catalog", [False, True])
@pytest.mark.parametrize("token_budget_notice", [False, True])
@pytest.mark.parametrize(
    ("public_action_semantics", "shared_semantics"), [(False, False), (True, False), (True, True)]
)
def test_native_episode_raw_span_and_restored_sealed_provisional_prefixes(
    tmp_path,
    public_action_semantics,
    shared_semantics,
    wire,
    reasoning_tool_catalog,
    token_budget_notice,
):
    tokenizer = PhaseTokenizer()
    action = call("submit_answer", answer='多行\n"quoted"\\path')
    if not wire.endswith("@1"):
        action = "Here is my submission.\n" + action
    if wire.endswith("@3"):
        action = "{(public note) completed response}\n" + action
    request = default_request()
    surface = contract().surface
    if public_action_semantics:
        from skillev_private.benchmarks.protocol_v13_training_sessions import _action_contract

        from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark

        surface, _ = _action_contract(Protocol13Benchmark.HOTPOT_QA, None, hotpot_deliberation=True)
    request = replace(
        request,
        task=replace(
            request.task,
            action_surface=surface,
            public_context={"benchmark_id": "hotpotqa"}
            if shared_semantics
            else request.task.public_context,
        ),
        retrieved_skills=(),
        active_skill_ids=(),
        decoding=DecodingSnapshot.create(
            max_reasoning_tokens=128,
            max_action_tokens=4096,
            base_seed=17,
            action_boundary_version="native-model-stop@1",
        ),
    )
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        request=request,
        scripts=[
            GenerationScript.text(tokenizer, "current reasoning"),
            GenerationScript.text(tokenizer, action),
        ],
        context_assembler=CanonicalInitialContextAssembler(
            maximum_h0_tokens=10000,
            phase_context=True,
            reasoning_tool_catalog=reasoning_tool_catalog,
            token_budget_notice=token_budget_notice,
            action_wire=wire,
            public_action_semantics=public_action_semantics,
            task_semantic_guidance="public-task-semantics@1" if shared_semantics else "legacy",
            hotpot_deliberation=shared_semantics,
        ),
    )
    artifact = asyncio.run(harness.engine.run(request))
    from skillev_private.experiments.ttb_update_attribution import _edge_categories

    # The diagnostic gradient categories must use the same persisted carrier
    # as actual execution, not label every native call as invalid legacy JSON.
    categories = _edge_categories(artifact.record)
    assert len(categories) == 1
    assert categories[0].startswith("structure-True/")
    step = artifact.record.steps[0]
    assert step.action_text == action
    assert step.action_token_ids == tuple(tokenizer.encode(action))
    assert harness.environment.completion_checks == [{"answer": '多行\n"quoted"\\path'}]
    assert len(harness.generator.requests) == 2
    restored = RolloutArtifact.from_value(artifact.to_value(), tokenizer=tokenizer)
    plan = prepare_edge_plan(tokenizer, restored.record, restored.initial_context.text)
    assert (
        plan.edge(1, AdapterRole.FORWARD_POLICY).prefix_ids
        == harness.generator.requests[1].input_ids
    )
    provisional = prepare_provisional_step(
        FakeScoringBackbone(tokenizer),
        ProvisionalStep(
            restored.record.trajectory_id,
            restored.manifest.task_id,
            restored.manifest.policy_snapshot,
            restored.manifest.library_version,
            restored.record.initial_context.query,
            restored.initial_context.text,
            (),
            step,
        ),
        TrainableVersions("f@1", "b@1", "z@1"),
    )
    assert provisional.forward.prefix_ids == plan.edge(1, AdapterRole.FORWARD_POLICY).prefix_ids
    assert provisional.backward.prefix_ids == plan.edge(1, AdapterRole.BACKWARD_POLICY).prefix_ids
    assert b"current reasoning" not in bytes(provisional.backward.prefix_ids)
    assert (
        provisional.forward.action_ids == provisional.backward.action_ids == step.action_token_ids
    )


def test_phase_only_systems_and_backward_noninterference():
    initial = PhaseContextSpec().wrap("Public task\n")
    left = render_forward_prefix_from_parts(initial, (), 1, "reason one")
    right = render_forward_prefix_from_parts(initial, (), 1, "reason two")
    backward = render_hindsight_prefix_from_parts(initial, (), 1, "public observation")
    assert phase_chat_messages(left.text)[0][0] == phase_chat_messages(right.text)[0][0]
    assert "reason one" not in json.dumps(phase_chat_messages(backward.text))
    changed = render_hindsight_prefix_from_parts(initial, (), 1, "different observation")
    assert phase_chat_messages(backward.text) != phase_chat_messages(changed.text)
    assert (
        phase_chat_messages(initial + "Reasoning:\n")[0][0] != phase_chat_messages(left.text)[0][0]
    )
