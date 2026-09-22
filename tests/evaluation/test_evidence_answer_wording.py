"""Optional owner wording guidance, never a reference-aware answer projection."""

import asyncio

import pytest

from skillev.evaluation.corpus_search import INPUT_PROFILE
from skillev.evaluation.input_metric_contracts import (
    CONTRACTS,
    IID_BENCHMARKS,
    OOD_BENCHMARKS,
    PublicTaskView,
)
from skillev.evaluation.scienceworld_commands import command_profile
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode, decode_integrity_arm
from skillev.task_semantic_guidance import (
    PUBLIC_TASK_SEMANTICS_V8,
    PUBLIC_TASK_SEMANTICS_V9,
    phase_deliverable,
    public_task_semantics,
)
from tests.evaluation.test_integrity_broker_boundary import runtime
from tests.rollout.test_native_phase_context import PhaseTokenizer
from tests.rollout.test_phase_deliverable import assembler
from tests.rollout.test_typed_alfworld_history import typed_request


@pytest.mark.parametrize("benchmark", [*IID_BENCHMARKS, *OOD_BENCHMARKS])
def test_only_declared_evidence_qa_meaning_changes(benchmark):
    spec = CONTRACTS[benchmark]
    for profile in spec.input_profiles:
        old = public_task_semantics(
            benchmark, input_profile=profile, version=PUBLIC_TASK_SEMANTICS_V8
        )
        new = public_task_semantics(
            benchmark, input_profile=profile, version=PUBLIC_TASK_SEMANTICS_V9
        )
        if benchmark == "musique" or (benchmark == "nq-open" and profile == INPUT_PROFILE):
            assert new.startswith(old)
            assert "original wording" in new
            assert "When the evidence contains" in new
        else:
            assert new == old
        assert phase_deliverable(benchmark, version=PUBLIC_TASK_SEMANTICS_V8) == phase_deliverable(
            benchmark, version=PUBLIC_TASK_SEMANTICS_V9
        )


def test_new_version_preserves_environment_interfaces_and_frozen_arm_identity():
    assert command_profile(PUBLIC_TASK_SEMANTICS_V9) == command_profile(PUBLIC_TASK_SEMANTICS_V8)
    request = typed_request()
    kwargs = {
        "task": request.task,
        "retrieved_skills": (),
        "active_skill_ids": (),
        "library_version": "synthetic-initial-library",
        "tokenizer": PhaseTokenizer(),
        "decoding": request.decoding,
    }
    old = assembler(PUBLIC_TASK_SEMANTICS_V8).assemble(**kwargs)
    new = assembler(PUBLIC_TASK_SEMANTICS_V9).assemble(**kwargs)
    # Only the declared version changes, not the ALFWorld public state/wire.
    assert new.text.replace(PUBLIC_TASK_SEMANTICS_V9, PUBLIC_TASK_SEMANTICS_V8) == old.text
    arm = InferenceArm("evidence-wording", task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V9)
    assert decode_integrity_arm(arm.to_value()) == arm


def test_guidance_reaches_isolated_owner_without_replacing_its_wrong_answer(tmp_path):
    entry = PublicTaskView.from_record(
        "synthetic-reading",
        "musique",
        {
            "question": "Where is the fictional workshop?",
            "context": "The fictional workshop is in Elmport.",
        },
    )
    instance = runtime(tmp_path, entry, ["Final answer: Elsewhere"])
    arm = InferenceArm(
        "evidence-wording",
        task_semantic_guidance=PUBLIC_TASK_SEMANTICS_V9,
        tool_call_mode=ToolCallMode.QWEN_XML,
    )
    final = asyncio.run(instance.generate(entry, arm, "wording"))
    assert final.text == "Elsewhere"
    assert final.intervention_counts["model_calls"] == 1
    assert final.intervention_counts["peer_model_calls"] == 0
    requests = instance.journal.traces(("wording", arm.arm_id, entry.task_id), "rendered-request")
    assert len(requests) == 1
    assert "original wording" in requests[0]["messages"][0]["content"]
    assert any(entry.render() in message["content"] for message in requests[0]["messages"])
    instance.journal.close()
