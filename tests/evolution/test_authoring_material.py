import json
from dataclasses import replace

import pytest

from skillev.evolution import AuthoringFailedError, RefineAuthoringRequest
from skillev.evolution.authoring_material import render_bounded_authoring_prompt
from skillev.evolution.evidence import authoring_edge_evidence
from skillev.evolution.public_execution import PublicExecutionSnippet
from tests.v3_helpers import (
    CharacterTokenizer,
    make_artifact,
    make_authoring_edge,
    make_skill_document,
)


def request():
    edges = tuple(
        replace(
            make_authoring_edge(f"edge-{index}"),
            public_execution=PublicExecutionSnippet("click[missing]", "not available"),
        )
        for index in range(500)
    )
    return RefineAuthoringRequest(
        make_skill_document("refine"),
        edges,
        json.dumps({"event_ids": [f"event-{i}" for i in range(10000)], "lcb": 0.15}),
        ("weak-context",),
        17,
    )


def test_large_evidence_is_bounded_without_changing_complete_request():
    value = request()
    prompt = render_bounded_authoring_prompt(
        value, tokenizer=CharacterTokenizer(), maximum_tokens=8192
    )
    assert len(prompt) <= 8192
    material = json.loads(prompt.split("\n")[1])["material"]
    assert material["projection"]["full_edge_count"] == 500
    assert 0 < material["projection"]["selected_edge_count"] < 500
    assert len(value.edge_exemplars) == 500
    assert material["evidence_summary"]["lcb"] == 0.15
    assert material["evidence_summary"]["event_ids"]["count"] == 10000


def test_mandatory_constraints_never_silently_truncated():
    with pytest.raises(AuthoringFailedError):
        render_bounded_authoring_prompt(
            request(), tokenizer=CharacterTokenizer(), maximum_tokens=50
        )


def test_public_edge_copies_execution_but_not_private_evaluator_material():
    item = make_artifact("public-error")
    record = replace(
        item.record,
        reward=replace(
            item.record.reward, native_payload={"private_answer": "PRIVATE_SENTINEL_ANSWER"}
        ),
    )
    evidence = authoring_edge_evidence(
        record, 1, absolute_log_importance=2, log_importance_quantile=1
    )
    assert evidence.public_execution.action_text == record.steps[0].action_text
    assert evidence.public_execution.observation_text == record.steps[0].observation_text
    assert "PRIVATE_SENTINEL_ANSWER" not in repr(evidence.to_value())
    changed = replace(
        evidence, public_execution=PublicExecutionSnippet("click[missing]", "tool unavailable")
    )
    assert evidence.to_value() != changed.to_value()


def test_authoring_action_kind_matches_runtime_wrapper_admission():
    from skillev.evolution.evidence import AuthoringActionKind

    item = make_artifact("wrapped-call")
    step = item.record.steps[0]
    wrapped = replace(step, action_text="Action:\n```json\n" + step.action_text + "\n```")
    record = replace(item.record, steps=(wrapped,))
    evidence = authoring_edge_evidence(
        record, 1, absolute_log_importance=2, log_importance_quantile=1
    )
    assert evidence.action_kind is AuthoringActionKind.SKILL
    assert evidence.public_execution.action_text == wrapped.action_text
