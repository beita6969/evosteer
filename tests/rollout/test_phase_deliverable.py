"""Declared development prompt condition; synthetic inputs, no IID answers."""

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from skillev_private.experiments.fresh_restart import (
    load_fresh_config,
    require_fresh_interface,
    require_iid_baselines,
)

from skillev.contracts import canonical_json
from skillev.evaluation.current_iid.protocol13.catalog import Protocol13Benchmark
from skillev.evaluation.input_metric_contracts import IID_BENCHMARKS
from skillev.policy.interface import AdapterRole
from skillev.policy.phase_context import PhaseContextSpec, phase_chat_messages
from skillev.policy.versions import TrainableVersions
from skillev.rollout import (
    CanonicalInitialContextAssembler,
    DecodingSnapshot,
    EnvironmentObservation,
    RolloutArtifact,
)
from skillev.rollout.provisional import ProvisionalStep
from skillev.runtime import BudgetVector
from skillev.scoring.edge_plan import prepare_edge_plan
from skillev.scoring.rendering import (
    render_forward_prefix_from_parts,
    render_hindsight_prefix_from_parts,
    render_reasoning_prefix,
)
from skillev.task_semantic_guidance import (
    PUBLIC_TASK_SEMANTICS_V5,
    PUBLIC_TASK_SEMANTICS_V7,
    TRAINING_PUBLIC_INPUT,
    phase_deliverable,
    public_task_semantics,
)
from skillev.training.provisional_math import prepare_provisional_step
from tests.rollout.engine_fakes import (
    FakeScoringBackbone,
    GenerationScript,
    default_request,
    make_harness,
)
from tests.rollout.test_native_phase_context import PhaseTokenizer
from tests.rollout.test_native_tool_wire import call
from tests.rollout.test_public_action_semantics import public_native_context
from tests.rollout.test_typed_alfworld_history import INITIAL, typed_request
from tests.training.test_iid_restart_binding import accepted_baselines


def assembler(version=PUBLIC_TASK_SEMANTICS_V7):
    return CanonicalInitialContextAssembler(
        maximum_h0_tokens=30000,
        phase_context=True,
        reasoning_tool_catalog=True,
        token_budget_notice=True,
        action_wire="native-single-tool-call@3",
        public_action_semantics=True,
        task_semantic_guidance=version,
    )


@pytest.mark.parametrize("domain", IID_BENCHMARKS)
def test_candidate_changes_phase_handoff_not_source_task_or_tool_semantics(domain):
    _, task = public_native_context(Protocol13Benchmark(domain))
    kwargs = {
        "task": task,
        "retrieved_skills": (),
        "active_skill_ids": (),
        "library_version": "synthetic-initial-library",
        "tokenizer": PhaseTokenizer(),
        "decoding": default_request().decoding,
    }
    old = assembler(PUBLIC_TASK_SEMANTICS_V5).assemble(**kwargs)
    new = assembler().assemble(**kwargs)
    old_spec, old_body = PhaseContextSpec.split(old.text)
    new_spec, new_body = PhaseContextSpec.split(new.text)
    assert old_body == new_body
    assert old_spec.deliverable is None
    assert new_spec.deliverable == phase_deliverable(domain, version=PUBLIC_TASK_SEMANTICS_V7)
    assert old_spec.tools_json == new_spec.tools_json
    assert old_spec.action_contract_json == new_spec.action_contract_json
    assert public_task_semantics(
        domain, input_profile=TRAINING_PUBLIC_INPUT, version=PUBLIC_TASK_SEMANTICS_V5
    ) == public_task_semantics(
        domain, input_profile=TRAINING_PUBLIC_INPUT, version=PUBLIC_TASK_SEMANTICS_V7
    )
    if domain not in {"aime-2026", "healthbench", "alfworld"}:
        assert old.text == new.text
        assert phase_chat_messages(old.text) == phase_chat_messages(new.text)


@pytest.mark.parametrize(
    "deliverable", ["conversation-reply", "integer-answer", "environment-action"]
)
@pytest.mark.parametrize("caps", [False, True])
def test_deliverable_metadata_roundtrip_and_historical_omission(deliverable, caps):
    from tests.rollout.test_native_tool_wire import contract

    old = PhaseContextSpec(
        action_wire="native-single-tool-call@3",
        tools_json=canonical_json(list(contract().to_native_tools())),
        reasoning_tool_catalog=True,
        reasoning_token_cap=128 if caps else None,
        action_token_cap=64 if caps else None,
    )
    new = replace(old, deliverable=deliverable)
    for spec in (old, new):
        restored, body = PhaseContextSpec.split(spec.wrap("synthetic public input"))
        assert restored == spec
        assert body == "synthetic public input"
    assert "deliverable" not in old.to_value()
    assert new.to_value()["version"] != old.to_value()["version"]
    assert phase_chat_messages(new.wrap("Reasoning:\n"))[1] is None


@pytest.mark.parametrize("domain", ["healthbench", "aime-2026", "alfworld"])
def test_live_restored_provisional_and_sealed_inputs_share_the_declared_handoff(tmp_path, domain):
    tokenizer = PhaseTokenizer()
    if domain == "alfworld":
        request = typed_request()
        action = call("act", command="look")
    else:
        _, task = public_native_context(Protocol13Benchmark(domain))
        request = replace(
            default_request(),
            task=task,
            retrieved_skills=(),
            active_skill_ids=(),
            decoding=DecodingSnapshot.create(
                max_reasoning_tokens=4096,
                max_action_tokens=4096,
                base_seed=17,
                action_boundary_version="native-model-stop@1",
            ),
        )
        action = call(
            "submit_answer",
            answer="Synthetic reply\nwith detail" if domain == "healthbench" else "123",
        )
    # Embedded draft headings remain data; no extractor fabricates a tool call.
    draft = "Current draft with a concrete choice. </think>\nAction: imagined, not executed."
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        request=request,
        context_assembler=assembler(),
        max_turns=1,
        scripts=[GenerationScript.text(tokenizer, text) for text in (draft, action)],
        environment_results=[
            EnvironmentObservation(
                public_value=INITIAL,
                observation_status="success",
                budget_usage=BudgetVector(tool_calls=1),
            )
        ],
    )
    artifact = asyncio.run(harness.engine.run(request))
    restored = RolloutArtifact.from_value(artifact.to_value(), tokenizer=tokenizer)
    step = restored.record.steps[0]
    plan = prepare_edge_plan(tokenizer, restored.record, restored.initial_context.text)
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
    assert step.reasoning_text == draft
    assert step.action_text == action
    assert step.action_token_ids == tuple(tokenizer.encode(action))
    assert provisional.forward.prefix_ids == harness.generator.requests[1].input_ids
    assert provisional.forward.prefix_ids == plan.edge(1, AdapterRole.FORWARD_POLICY).prefix_ids
    assert provisional.backward.prefix_ids == plan.edge(1, AdapterRole.BACKWARD_POLICY).prefix_ids
    assert (
        provisional.forward.action_ids == provisional.backward.action_ids == step.action_token_ids
    )
    assert b"Current draft with a concrete choice" not in bytes(provisional.backward.prefix_ids)
    reasoning = render_reasoning_prefix(restored.initial_context.text, (), 1)
    messages, tools = phase_chat_messages(reasoning.text)
    assert tools is None
    assert "Phase deliverable" in messages[0]["content"]
    if domain == "alfworld":
        assert json.loads(messages[-1]["content"].split("\n", 2)[1])["controller_max_turns"] == 25


def test_candidate_is_not_an_acceptance_override(tmp_path):
    old = load_fresh_config(Path("configs/training/bayesianimprove_fresh_restart.yaml"))
    new = load_fresh_config(Path("configs/training/bayesianimprove_development_handoff.yaml"))
    require_fresh_interface(new)
    assert new.task_semantic_guidance == PUBLIC_TASK_SEMANTICS_V7
    assert dict(new.reasoning_tokens_by_domain)["healthbench"] == 8192
    assert dict(new.reasoning_tokens_by_domain)["aime-2026"] == 16384
    assert new.thinking_off_domains == old.thinking_off_domains
    assert new.max_turns == old.max_turns == 25
    assert new.static_max_turns == old.static_max_turns == 8
    # All changes are explicit and participate in the frozen architecture check.
    assert (
        replace(
            new,
            task_semantic_guidance=old.task_semantic_guidance,
            reasoning_tokens_by_domain=old.reasoning_tokens_by_domain,
        )
        == old
    )
    references = accepted_baselines(tmp_path, old)
    with pytest.raises(ValueError):
        require_iid_baselines(references, config=new)
    with pytest.raises(ValueError):
        require_fresh_interface(replace(new, skill_exposure="catalog-then-read-two-distinct@1"))


@pytest.mark.skipif(
    not os.environ.get("SKILLEV_PRIVATE_TOKENIZER_DIR"),
    reason="private deployed tokenizer not supplied",
)
@pytest.mark.parametrize("domain", ["healthbench", "aime-2026", "alfworld"])
def test_real_qwen_template_keeps_candidate_handoff_and_thinking_split(domain):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["SKILLEV_PRIVATE_TOKENIZER_DIR"], local_files_only=True, trust_remote_code=False
    )
    request = typed_request() if domain == "alfworld" else default_request()
    if domain != "alfworld":
        _, task = public_native_context(Protocol13Benchmark(domain))
        request = replace(request, task=task)
    initial = (
        assembler()
        .assemble(
            task=request.task,
            retrieved_skills=(),
            active_skill_ids=(),
            library_version="synthetic-initial-library",
            tokenizer=PhaseTokenizer(),
            decoding=request.decoding,
        )
        .text
    )
    draft = "Private current draft </think> full reply text with 中文 and an imagined observation."
    prefixes = {
        "reasoning": render_reasoning_prefix(initial, (), 1),
        "forward": render_forward_prefix_from_parts(initial, (), 1, draft),
        "hindsight": render_hindsight_prefix_from_parts(initial, (), 1, canonical_json(INITIAL)),
    }
    for phase, prefix in prefixes.items():
        messages, tools = phase_chat_messages(prefix.text)
        thinking = phase == "reasoning" and domain != "alfworld"
        ids = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        displayed = tokenizer.decode(ids, skip_special_tokens=False)
        assert "Phase deliverable" in displayed
        assert "SKILLEV_PHASE_CONTEXT_V1" not in displayed
        assert displayed.endswith("<think>\n") is thinking
        assert (tools is None) is (phase == "reasoning")
        assert ("Private current draft" in displayed) is (phase == "forward")
        if phase == "forward":
            assert "full reply text with 中文" in displayed
        if domain == "alfworld":
            latest = displayed.rsplit("<|im_start|>user", 1)[1]
            assert "Box is closed." in latest
            assert "imagined observation" not in latest
