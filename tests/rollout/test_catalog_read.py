import asyncio
import json
from dataclasses import replace

import pytest

from skillev.contracts import canonical_json
from skillev.contracts.skill_exposure import (
    CATALOG_EXPOSURES,
    PROACTIVE_CATALOG_EXPOSURE,
    TWO_SKILL_CATALOG_EXPOSURE,
)
from skillev.rollout import CanonicalInitialContextAssembler, DecodingSnapshot
from skillev.rollout.catalog import CatalogReadEnvironment
from skillev.runtime import ActionKind, StructuredAction
from tests.rollout.engine_fakes import (
    ByteTokenizer,
    FakeTerminalEvaluator,
    GenerationScript,
    ScriptedEnvironment,
    default_request,
    make_harness,
)
from tests.rollout.test_native_tool_wire import call, contract


@pytest.mark.parametrize("reward_value", [0.0, 0.75])
def test_proactive_hint_is_visible_but_does_not_inject_reads_or_change_reward(
    tmp_path, reward_value
):
    from skillev.policy.phase_context import phase_chat_messages
    from skillev.scoring.rendering import (
        render_forward_prefix_from_parts,
        render_hindsight_prefix_from_parts,
        render_reasoning_prefix,
    )
    from skillev.training.metrics_telemetry import skill_target_facts

    tokenizer = ByteTokenizer()
    request = default_request()
    skill = replace(
        request.retrieved_skills[0],
        content=canonical_json(
            {
                "title": "Public method",
                "summary": "Inspect a public invariant when relevant",
                "applicability": {},
                "instructions": "UNREAD_BODY_ONLY",
            }
        ),
    )
    request = replace(
        request,
        task=replace(request.task, action_surface=contract().surface),
        retrieved_skills=(skill,),
        decoding=DecodingSnapshot.create(
            max_reasoning_tokens=128,
            max_action_tokens=4096,
            base_seed=17,
            action_boundary_version="native-model-stop@1",
        ),
    )
    assembler = CanonicalInitialContextAssembler(
        maximum_h0_tokens=20000,
        phase_context=True,
        reasoning_tool_catalog=True,
        action_wire="native-single-tool-call@3",
        skill_exposure=PROACTIVE_CATALOG_EXPOSURE,
    )
    assembled = assembler.assemble(
        task=request.task,
        retrieved_skills=request.retrieved_skills,
        active_skill_ids=request.active_skill_ids,
        library_version=request.library_version,
        tokenizer=tokenizer,
    )
    assert assembled.contract.meta["skill_exposure"] == PROACTIVE_CATALOG_EXPOSURE
    for prefix in (
        render_reasoning_prefix(assembled.text, (), 1),
        render_forward_prefix_from_parts(assembled.text, (), 1, "CURRENT_REASONING"),
        render_hindsight_prefix_from_parts(assembled.text, (), 1, "PUBLIC_OBSERVATION"),
    ):
        rendered = json.dumps(phase_chat_messages(prefix.text))
        assert "Use relevant skills proactively" in rendered
        assert "no required number" in rendered
        assert "UNREAD_BODY_ONLY" not in rendered
    assert "CURRENT_REASONING" not in json.dumps(
        phase_chat_messages(
            render_hindsight_prefix_from_parts(assembled.text, (), 1, "PUBLIC_OBSERVATION").text
        )
    )
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        request=request,
        max_turns=1,
        evaluator=FakeTerminalEvaluator(value=reward_value),
        context_assembler=assembler,
        environment=CatalogReadEnvironment(
            ScriptedEnvironment([]), (skill,), request.library_version
        ),
        scripts=[
            GenerationScript.text(tokenizer, "Solve directly with the known public method"),
            GenerationScript.text(tokenizer, call("submit_answer", answer="synthetic")),
        ],
    )
    artifact = asyncio.run(harness.engine.run(request))
    assert artifact.record.horizon == 1
    assert artifact.record.reward.value == reward_value
    assert artifact.record.steps[0].invoked_skill_ids == ()
    assert harness.ledger.settled.tool_calls == 0
    assert harness.ledger.settled.agent_turns == 1
    assert skill_target_facts([artifact.record.to_value()]) == {}


@pytest.mark.parametrize("exposure", sorted(CATALOG_EXPOSURES))
def test_catalog_exposes_summary_then_exact_pinned_body_without_model(exposure):
    request = default_request()
    document = canonical_json(
        {
            "title": "Public strategy",
            "summary": "When the task needs it",
            "applicability": {"task_families": ["debug-family"]},
            "instructions": "BODY_NOT_IN_CATALOG",
            "requirements": [],
        }
    )
    skill = replace(request.retrieved_skills[0], content=document)
    assembler = CanonicalInitialContextAssembler(maximum_h0_tokens=10000, skill_exposure=exposure)
    assembled = assembler.assemble(
        task=request.task,
        retrieved_skills=(skill,),
        active_skill_ids=request.active_skill_ids,
        library_version=request.library_version,
        tokenizer=ByteTokenizer(),
    )
    assert "BODY_NOT_IN_CATALOG" not in assembled.text
    assert "Public strategy" in assembled.text
    environment = CatalogReadEnvironment(ScriptedEnvironment([]), (skill,), request.library_version)
    action = StructuredAction(
        ActionKind.SKILL, "invoke", {}, "skill-runtime", skill.metadata.skill_id
    )
    observation = asyncio.run(environment.execute(action, step_index=1))
    assert observation.public_value["content"] == document
    assert observation.invoked_skill_ids == (skill.metadata.skill_id,)
    missing = asyncio.run(
        environment.execute(replace(action, skill_id="outside-catalog"), step_index=2)
    )
    assert not missing.invoked_skill_ids
    with pytest.raises(ValueError):
        CanonicalInitialContextAssembler(maximum_h0_tokens=8, skill_exposure=exposure).assemble(
            task=request.task,
            retrieved_skills=(skill,),
            active_skill_ids=request.active_skill_ids,
            library_version=request.library_version,
            tokenizer=ByteTokenizer(),
        )


@pytest.mark.parametrize(
    "wire", ["native-single-tool-call@1", "native-single-tool-call@2", "native-single-tool-call@3"]
)
@pytest.mark.parametrize("skill_count", [0, 1, 2])
def test_two_skill_prompt_and_native_tool_description_agree_without_body_exposure(
    wire, skill_count
):
    request = default_request()
    prototype = request.retrieved_skills[0]
    skills = tuple(
        replace(
            prototype,
            metadata=replace(prototype.metadata, skill_id=f"skill-{i}"),
            content=canonical_json(
                {
                    "title": "Visible strategy",
                    "summary": "Public description",
                    "applicability": {},
                    "instructions": "BODY_ONLY_AFTER_READ",
                }
            ),
        )
        for i in range(skill_count)
    )
    task = replace(request.task, action_surface=contract().surface)
    kwargs = {
        "task": task,
        "retrieved_skills": skills,
        "active_skill_ids": tuple(skill.metadata.skill_id for skill in skills),
        "library_version": request.library_version,
        "tokenizer": ByteTokenizer(),
    }
    assembler = CanonicalInitialContextAssembler(
        maximum_h0_tokens=20000,
        phase_context=True,
        action_wire=wire,
        skill_exposure=TWO_SKILL_CATALOG_EXPOSURE,
    )
    assembled = assembler.assemble(**kwargs)
    assert "at least two DIFFERENT skills" in assembled.text
    assert "consume the existing turn" in assembled.text
    assert "BODY_ONLY_AFTER_READ" not in assembled.text
    assert "skill is optional" not in assembled.text
    assert "optional, not a required step" not in assembled.text
    assert assembled.contract.meta["skill_exposure"] == TWO_SKILL_CATALOG_EXPOSURE
    if not skill_count:
        assert "No applicable skills" in assembled.text
    legacy = CanonicalInitialContextAssembler(
        maximum_h0_tokens=20000,
        phase_context=True,
        action_wire=wire,
        skill_exposure="catalog-then-read@1",
    ).assemble(**kwargs)
    assert "at least two DIFFERENT skills" not in legacy.text
    assert json.dumps(assembled.contract.to_value()) != json.dumps(legacy.contract.to_value())


@pytest.mark.parametrize("exposure", sorted(CATALOG_EXPOSURES))
@pytest.mark.parametrize("reward_value", [0.75, 0.0])
def test_two_skill_reads_reach_later_model_requests_and_still_consume_turns(
    tmp_path, exposure, reward_value
):
    tokenizer = ByteTokenizer()
    request = default_request()
    prototype = request.retrieved_skills[0]
    skills = tuple(
        replace(
            prototype,
            metadata=replace(prototype.metadata, skill_id=f"skill-{i}"),
            content=canonical_json(
                {
                    "title": f"Strategy {i}",
                    "summary": "Public method",
                    "applicability": {},
                    "instructions": f"BODY_READ_{i}",
                }
            ),
        )
        for i in range(2)
    )
    request = replace(
        request,
        task=replace(request.task, action_surface=contract().surface),
        retrieved_skills=skills,
        active_skill_ids=tuple(s.metadata.skill_id for s in skills),
        decoding=DecodingSnapshot.create(
            max_reasoning_tokens=128,
            max_action_tokens=4096,
            base_seed=17,
            action_boundary_version="native-model-stop@1",
        ),
    )
    texts = [
        "choose first",
        call("read_skill", skill_id="skill-0"),
        "choose another",
        call("read_skill", skill_id="skill-1"),
        "solve",
        call("submit_answer", answer="synthetic"),
    ]
    harness = make_harness(
        tmp_path,
        tokenizer=tokenizer,
        request=request,
        max_turns=3,
        evaluator=FakeTerminalEvaluator(value=reward_value),
        context_assembler=CanonicalInitialContextAssembler(
            maximum_h0_tokens=20000,
            phase_context=True,
            action_wire="native-single-tool-call@3",
            skill_exposure=exposure,
        ),
        environment=CatalogReadEnvironment(
            ScriptedEnvironment([]), skills, request.library_version
        ),
        scripts=[GenerationScript.text(tokenizer, text) for text in texts],
    )
    artifact = asyncio.run(harness.engine.run(request))
    assert artifact.record.horizon == 3
    assert harness.ledger.settled.agent_turns == 3
    assert harness.ledger.settled.tool_calls == 2
    assert [step.invoked_skill_ids for step in artifact.record.steps] == [
        ("skill-0",),
        ("skill-1",),
        (),
    ]
    prompts = [tokenizer.decode(request.input_ids) for request in harness.generator.requests]
    assert "BODY_READ_0" not in prompts[0]
    assert "BODY_READ_0" in prompts[2]
    assert "BODY_READ_1" not in prompts[2]
    assert "BODY_READ_0" in prompts[4]
    assert "BODY_READ_1" in prompts[4]

    from skillev.rollout import RolloutArtifact
    from skillev.training.evidence_context import TrajectoryEvidenceContext
    from skillev.training.skill_discovery_metrics import skill_discovery_facts

    restored = RolloutArtifact.from_value(artifact.to_value(), tokenizer=tokenizer)
    assert restored.skill_input_evidence == artifact.skill_input_evidence
    context = TrajectoryEvidenceContext.from_artifact(restored)
    assert context.visible_skill_ids == ("skill-0", "skill-1")
    assert context.body_visible_skill_ids == ("skill-0", "skill-1")
    first, second = context.invocation_links
    assert first.following_execution_steps == ()  # retain the original until-next-read meaning
    assert first.later_execution_steps == second.later_execution_steps == (3,)
    assert first.body_visible_execution_steps == second.body_visible_execution_steps == (3,)
    facts = skill_discovery_facts([artifact.record.to_value()])
    assert facts["skill_read_followed_by_action_count"] == 1
    assert facts["skill_reads_with_later_action"] == 2
    forward = [r for r in restored.skill_input_evidence if r["phase"] == "action"]
    assert forward[0]["visible_skill_body_refs"] == []
    assert {ref["read_step_index"] for ref in forward[2]["visible_skill_body_refs"]} == {1, 2}
    assert "BODY_READ" not in json.dumps(artifact.skill_input_evidence)
    if exposure == "catalog-then-read@1":
        assert "at least two" not in prompts[0]

    from tests.training.test_bayesian_chain import pipeline, restore
    from tests.v3_helpers import make_source

    projection, _ = pipeline()
    projection.commit(projection.preview(make_source(artifacts=(artifact,))))
    recovered = restore(projection)
    batch = recovered.posterior_provenance.batches[-1]
    assert len(batch.posterior.updates) == 2
    assert all(e.outcome == artifact.record.reward.success for e in batch.posterior.updates)
    assert batch.trajectory_contexts[0].skill_input_evidence == artifact.skill_input_evidence
