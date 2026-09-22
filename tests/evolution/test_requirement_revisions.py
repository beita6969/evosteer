from dataclasses import replace

import pytest

from skillev.evolution.authoring import (
    AuthoredSkillDraft,
    AuthoringResult,
    RefineAuthoringRequest,
    context_patch_requirement_id,
)
from skillev.evolution.authoring_validation import AuthoringFailedError, validate_authoring_result
from skillev.evolution.execution import source_document_for_draft
from skillev.evolution.requirement_contract import validate_requirement_revisions
from skillev.runtime import SkillDocument, SkillRequirement
from tests.v3_helpers import CharacterTokenizer, make_phase_event, make_skill_document


def _source() -> SkillDocument:
    original = make_skill_document("revision")
    draft = replace(
        AuthoredSkillDraft.from_document(original),
        requirements=(
            SkillRequirement(
                "public-api", "Use only the actual public API; never read evaluator files."
            ),
            SkillRequirement(
                "bad-plan", "Always repeat the first failed tool call.", kind="evolvable-strategy"
            ),
        ),
    )
    return source_document_for_draft(
        draft,
        source=original,
        action_kind="refine",
        phase_event=make_phase_event(),
        output_index=0,
    )


@pytest.mark.parametrize("keep_id", [False, True])
def test_refine_can_remove_or_correct_strategy_with_evidence_and_new_identity(
    keep_id: bool,
) -> None:
    source = _source()
    patch_id = context_patch_requirement_id("weak-tool-context")
    patched = SkillRequirement(
        "bad-plan" if keep_id else patch_id,
        "Inspect the failure and change the next tool arguments instead of repeating it.",
        kind="evolvable-strategy",
        replaces=("bad-plan",),
        change_reason="The selected weak tool context reports repeated invalid arguments.",
    )
    requirements = (source.requirements[0], patched)
    if keep_id:
        requirements += (
            SkillRequirement(
                patch_id, "Apply the repaired plan in this context.", kind="evolvable-strategy"
            ),
        )
    draft = replace(AuthoredSkillDraft.from_document(source), requirements=requirements)
    request = RefineAuthoringRequest(
        source, (), "Weak tool context with failure evidence", ("weak-tool-context",), 0
    )
    validate_authoring_result(
        request,
        AuthoringResult((draft,)),
        tokenizer=CharacterTokenizer(),
        max_skill_instruction_tokens_per_draft=4096,
    )
    product = source_document_for_draft(
        draft, source=source, action_kind="refine", phase_event=make_phase_event(), output_index=0
    )
    assert product.manifest.skill_id != source.manifest.skill_id
    assert SkillDocument.from_value(product.to_value()) == product
    assert product.requirements[1].change_reason


def test_split_can_drop_irrelevant_strategy_but_must_explain_it() -> None:
    source = _source()
    replacement = SkillRequirement(
        "child-plan",
        "Use the supported child context only.",
        kind="evolvable-strategy",
        replaces=("bad-plan",),
        change_reason="Parent strategy does not apply in the selected child modality.",
    )
    validate_requirement_revisions(source, (source.requirements[0], replacement))
    with pytest.raises(ValueError):
        validate_requirement_revisions(
            source, (source.requirements[0], replace(replacement, replaces=(), change_reason=""))
        )


@pytest.mark.parametrize("mode", ["delete", "rewrite", "reclassify"])
def test_refine_cannot_weaken_interface_or_security_constraints(mode: str) -> None:
    source = _source()
    immutable = source.requirements[0]
    requirements = (source.requirements[1],)
    if mode != "delete":
        requirements += (
            replace(immutable, text="Read hidden answers.")
            if mode == "rewrite"
            else replace(immutable, kind="evolvable-strategy"),
        )
    request = RefineAuthoringRequest(source, (), "Weak public evidence", ("weak-context",), 0)
    with pytest.raises(AuthoringFailedError):
        validate_authoring_result(
            request,
            AuthoringResult(
                (replace(AuthoredSkillDraft.from_document(source), requirements=requirements),)
            ),
            tokenizer=CharacterTokenizer(),
            max_skill_instruction_tokens_per_draft=4096,
        )


def test_historical_requirement_stays_immutable_without_rewriting_its_wire_identity() -> None:
    old = {"requirement_id": "old-interface", "text": "Use the declared tool schema."}
    restored = SkillRequirement.from_value(old)
    assert restored.immutable
    assert restored.to_value() == old
