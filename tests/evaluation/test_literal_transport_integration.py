"""Lossless literal data through the isolated actor and authoritative broker."""

import asyncio

import pytest

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.sealed_candidates import EventOrigin
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from tests.evaluation.test_integrity_broker_boundary import runtime
from tests.evaluation.test_response_literal_boundaries import SOURCES, tool_call


def test_unsupported_consultation_with_native_examples_executes_neither(tmp_path):
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic task."})
    body = "Explain these examples:\n" + tool_call("click", {"target": "Blue Mug"})
    instance = runtime(tmp_path, entry, ["Message to solver:\n" + body, "Final answer: 42"])
    final = asyncio.run(instance.generate(entry, InferenceArm("single"), "synthetic"))
    scope = ("synthetic", "single", "case")
    assert final.text == r"\boxed{42}"
    assert final.intervention_counts["model_calls"] == 2
    assert final.intervention_counts["peer_model_calls"] == 0
    assert final.intervention_counts["tool_calls"] == 0
    assert final.intervention_counts["communication_repairs"] == 1
    assert not instance.journal.traces(scope, "agent-reply", origin=EventOrigin.ACTOR_DIAGNOSTIC)
    assert all(row.participant == "owner" for row in instance.journal.model_outputs(scope))
    instance.journal.close()


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("wrapper", ["{}", "Final code:\n{}", "```python\n{}\n```"])
def test_owner_literal_source_reaches_authoritative_candidate_without_a_repair(
    tmp_path, source, wrapper
):
    entry = PublicTaskView.from_record(
        "case", "humaneval", {"prompt": "def f():\n    # Synthetic public source request\n"}
    )
    response = wrapper.format(source)
    instance = runtime(tmp_path, entry, [response], calls=1)
    final = asyncio.run(instance.generate(entry, InferenceArm("single"), "synthetic"))
    assert final.text == source
    assert final.submission["raw_response"] == response
    assert final.intervention_counts["model_calls"] == 1
    assert final.intervention_counts["peer_model_calls"] == 0
    assert final.intervention_counts["communication_repairs"] == 0
    assert instance.journal.get("synthetic", "single", "case").text == source
    instance.journal.close()


def test_native_history_defaults_reach_public_archive_then_owner_submits(tmp_path):
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic task."})
    instance = runtime(
        tmp_path, entry, [tool_call("history", {"archive": "discussion"}), "42"], calls=2
    )
    final = asyncio.run(
        instance.generate(
            entry, InferenceArm("single", tool_call_mode=ToolCallMode.QWEN_XML), "synthetic"
        )
    )
    assert final.text == r"\boxed{42}"
    assert final.intervention_counts["model_calls"] == 2
    assert final.intervention_counts["tool_calls"] == 1
    assert final.intervention_counts["communication_repairs"] == 0
    history = instance.journal.traces(
        ("synthetic", "single", "case"), "history-read", origin=EventOrigin.ACTOR_DIAGNOSTIC
    )
    assert len(history) == 1
    assert history[0]["request"]["cursor"] == 0
    assert history[0]["request"]["limit"] == 4
    instance.journal.close()
