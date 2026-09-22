"""A dispatched native skill must survive record admission and evidence export."""

import asyncio
import json
from dataclasses import replace

import pytest

from skillev.rollout import CanonicalInitialContextAssembler, DecodingSnapshot, RolloutArtifact
from skillev.training.invocation_evidence import invocation_execution_links
from tests.rollout.engine_fakes import (
    ByteTokenizer,
    GenerationScript,
    default_request,
    make_harness,
)
from tests.rollout.test_engine import _tool_observation
from tests.rollout.test_native_tool_wire import call, contract


@pytest.mark.parametrize("wire", ["native-single-tool-call@2", "native-single-tool-call@3"])
@pytest.mark.parametrize("envelope", ["native", "commentary", "arguments", "bare", "fence"])
def test_native_skill_dispatch_record_and_evidence_share_one_decoder(tmp_path, envelope, wire):
    tokenizer = ByteTokenizer()
    request = default_request()
    skill_id = request.active_skill_ids[0]
    arguments = {"skill_id": skill_id}
    action = call("read_skill", **arguments)
    if envelope == "commentary":
        action = "I will consult an available method.\n" + action
    elif envelope == "arguments":
        action = json.dumps({"name": "read_skill", "arguments": arguments})
    elif envelope == "bare":
        action = json.dumps(arguments)
    elif envelope == "fence":
        action = "I will read this method.\n```json\n" + json.dumps(arguments) + "\n```"
    request = replace(
        request,
        task=replace(request.task, action_surface=contract().surface),
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
        context_assembler=CanonicalInitialContextAssembler(
            maximum_h0_tokens=10000, phase_context=True, action_wire=wire
        ),
        environment_results=[_tool_observation()],
        scripts=[
            GenerationScript.text(tokenizer, text)
            for text in ("plan", action, "finish", call("submit_answer", answer="synthetic"))
        ],
    )
    artifact = asyncio.run(harness.engine.run(request))
    restored = RolloutArtifact.from_value(artifact.to_value(), tokenizer=tokenizer)
    first = restored.record.steps[0]
    assert first.action_text == action
    assert first.action_token_ids == tuple(tokenizer.encode(action))
    assert first.invoked_skill_ids == (skill_id,)
    (link,) = invocation_execution_links(restored.record)
    assert link.admitted
    assert link.declared_skill_id == skill_id
    assert link.following_execution_steps == (2,)
    with pytest.raises(ValueError):
        replace(
            restored.record,
            steps=(replace(first, invoked_skill_ids=()), *restored.record.steps[1:]),
        )
    with pytest.raises(ValueError):
        replace(
            restored.record,
            steps=(replace(first, invoked_skill_ids=("not-the-called-skill",)),),
            horizon=1,
        )
