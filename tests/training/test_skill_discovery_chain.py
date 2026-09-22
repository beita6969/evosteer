import json
from dataclasses import replace

import pytest

from skillev.contracts import canonical_json
from skillev.diagnostics.skill_visibility import CatalogVisibility
from skillev.policy.interface import EncodedPolicyPrompt
from skillev.rollout.catalog import render_skill_catalog_entry
from skillev.training.evidence_context import TrajectoryEvidenceContext
from skillev.training.invocation_evidence import invocation_execution_links
from skillev.training.skill_discovery_metrics import skill_discovery_facts
from tests.rollout.engine_fakes import ByteTokenizer, default_request
from tests.training.test_evidence_reporting import artifact
from tests.training.test_metrics_telemetry import phase, setup_export


def catalog_record(*, calls=True):
    value = artifact("read-example", calls=calls)
    h0 = replace(
        value.record.initial_context,
        meta={**value.record.initial_context.meta, "skill_exposure": "catalog-then-read@1"},
    )
    return replace(
        value,
        record=replace(value.record, initial_context=h0),
        initial_context=replace(value.initial_context, contract=h0),
    )


def test_catalog_exposure_alone_is_neither_a_read_nor_an_invocation():
    value = catalog_record(calls=False)
    context = TrajectoryEvidenceContext.from_artifact(value)
    assert context.visible_skill_ids is None  # legacy retrieval IDs are not input evidence
    assert context.body_visible_skill_ids is None
    assert context.invocation_links == ()
    facts = skill_discovery_facts([value.record.to_value()])
    assert facts["skill_visible_id_count"] == 1
    assert facts["skill_body_read_count"] == 0
    assert facts["skill_invoked_ids"] == []


@pytest.mark.parametrize("fault", [None, "empty", "library", "skill", "malformed"])
def test_body_receipt_requires_the_actual_pinned_response(fault):
    value = catalog_record()
    response = {
        "status": "skill-read",
        "skill_id": "skill-alpha",
        "library_version": "library-v1",
        "version": "1",
        "content": "PRIVATE_BODY",
    }
    if fault == "empty":
        response["content"] = ""
    if fault == "library":
        response["library_version"] = "other-library"
    if fault == "skill":
        response["skill_id"] = "other-skill"
    text = "{" if fault == "malformed" else canonical_json(response)
    record = replace(value.record, steps=(replace(value.record.steps[0], observation_text=text),))
    link = invocation_execution_links(record)[0]
    assert link.body_returned is (fault is None)
    assert "PRIVATE_BODY" not in json.dumps(link.to_value())
    assert type(link).from_value(link.to_value()) == link
    facts = skill_discovery_facts([record.to_value()])
    assert facts["skill_body_read_count"] == int(fault is None)
    assert facts["skill_body_read_failed_count"] == int(fault is not None)
    assert facts["skill_read_followed_by_action_count"] == 0


def test_actual_window_visibility_checks_complete_catalog_entries_not_mentions():
    prototype = default_request().retrieved_skills[0]
    skills = tuple(
        replace(
            prototype,
            metadata=replace(prototype.metadata, skill_id=f"skill-{i}"),
            content=canonical_json(
                {
                    "title": f"Procedure {i}",
                    "summary": "Task-specific guidance",
                    "applicability": {},
                    "instructions": "Not in catalog",
                }
            ),
        )
        for i in range(2)
    )
    tokenizer = ByteTokenizer()
    entries = [render_skill_catalog_entry(s, position=i + 1) for i, s in enumerate(skills)]
    visibility = CatalogVisibility.from_skills(skills)
    full = tuple(tokenizer.encode("Header\n" + "".join(entries)))
    assert visibility.metrics(EncodedPolicyPrompt(full, len(full)), tokenizer)[
        "catalog_visible_skill_ids"
    ] == ["skill-0", "skill-1"]
    # After H0 truncation only one complete entry survives. Mentioning the
    # other ID must not turn into an exposure or execution claim.
    admitted = tuple(tokenizer.encode("skill-0 was mentioned\n" + entries[1]))
    prompt = EncodedPolicyPrompt(admitted, len(full), (len(admitted),))
    report = visibility.metrics(prompt, tokenizer)
    assert report["catalog_visible_skill_ids"] == ["skill-1"]
    assert report["catalog_not_fully_visible_skill_ids"] == ["skill-0"]
    assert prompt.ids == admitted


@pytest.mark.parametrize("rule", [None, "residual-and-entropy", "zero-coverage-generate@1"])
def test_cold_start_and_natural_triggers_are_distinct_and_legacy_is_unknown(tmp_path, rule):
    store, collector = setup_export(tmp_path)
    collector.observe(phase(1, **({"trigger_rule": rule} if rule else {})))
    result = collector.phase(1)
    assert result["phase_trigger_count"] == 1
    assert result["cold_start_trigger_count"] == (
        int(rule == "zero-coverage-generate@1") if rule else None
    )
    assert result["natural_phase_trigger_count"] == (
        int(rule == "residual-and-entropy") if rule else None
    )
    assert result["library_mutation_count"] is None
    store.close()
