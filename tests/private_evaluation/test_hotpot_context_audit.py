from skillev_private.benchmarks.hotpot_context_audit import (
    ContextEvidenceStatus,
    audit_hotpot_context,
    parse_supporting_fact_refs,
)


def test_hotpot_supporting_fact_refs_accept_released_pair_and_mapping_shapes() -> None:
    refs = parse_supporting_fact_refs([["Title A", 1], {"title": "Title B", "sent_id": 2}])
    assert refs is not None
    assert tuple((item.title, item.sentence_index) for item in refs) == (
        ("Title A", 1),
        ("Title B", 2),
    )


def test_hotpot_flattened_context_is_partial_but_structurally_valid() -> None:
    passages = tuple(f"Title {index}. Public sentence {index}." for index in range(1, 11))
    rendered = "Context passages:\n\n" + "\n\n".join(passages) + "\n\nQuestion:\nWhy?"
    audit = audit_hotpot_context(
        task_id="hotpot-1",
        rendered_question=rendered,
        public_context=passages,
        supporting_facts=[["Title 2", 0], ["Title 8", 0]],
    )
    assert audit.evidence_status is ContextEvidenceStatus.PARTIALLY_VERIFIED
    assert audit.supporting_fact_present_count == 2
    assert audit.structurally_valid


def test_hotpot_second_question_rendering_is_rejected() -> None:
    passages = tuple("Why?" if index == 0 else f"Passage {index}" for index in range(10))
    audit = audit_hotpot_context(
        task_id="hotpot-2",
        rendered_question="\n".join((*passages, "Question:\nWhy?")),
        public_context=passages,
        supporting_facts=None,
    )
    assert not audit.passed
