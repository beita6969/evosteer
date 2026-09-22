from __future__ import annotations

from dataclasses import replace

import pytest

from skillev.contracts import (
    SkillEvolutionAction,
    SkillEvolutionRecord,
    SkillVersionRef,
    stable_hash,
    validate_lineage,
)


def _ref(skill_id: str, version: int) -> SkillVersionRef:
    return SkillVersionRef(skill_id, version, stable_hash({"skill": skill_id, "version": version}))


def _record(
    action: SkillEvolutionAction,
    inputs: tuple[SkillVersionRef, ...],
    outputs: tuple[SkillVersionRef, ...],
    *,
    sequence_number: int = 1,
    previous_record_hash: str | None = None,
) -> SkillEvolutionRecord:
    return SkillEvolutionRecord(
        lineage_id="library-main",
        sequence_number=sequence_number,
        action=action,
        input_versions=inputs,
        output_versions=outputs,
        previous_record_hash=previous_record_hash,
        evidence_hash=stable_hash({"batch": sequence_number}),
        reason="flow-directed library update",
    )


@pytest.mark.parametrize(
    ("action", "inputs", "outputs"),
    [
        (SkillEvolutionAction.RETAIN_COMPRESS, (_ref("alpha", 1),), (_ref("alpha", 2),)),
        (SkillEvolutionAction.REFINE, (_ref("alpha", 1),), (_ref("alpha", 2),)),
        (
            SkillEvolutionAction.SPLIT,
            (_ref("alpha", 1),),
            (_ref("alpha-a", 1), _ref("alpha-b", 1)),
        ),
        (SkillEvolutionAction.PRUNE, (_ref("alpha", 1),), ()),
        (SkillEvolutionAction.GENERATE, (), (_ref("alpha", 1),)),
    ],
)
def test_each_method_action_accepts_its_natural_shape(
    action: SkillEvolutionAction,
    inputs: tuple[SkillVersionRef, ...],
    outputs: tuple[SkillVersionRef, ...],
) -> None:
    assert _record(action, inputs, outputs).action is action


@pytest.mark.parametrize(
    ("action", "inputs", "outputs"),
    [
        (SkillEvolutionAction.RETAIN_COMPRESS, (), (_ref("alpha", 1),)),
        (SkillEvolutionAction.REFINE, (_ref("alpha", 1),), ()),
        (SkillEvolutionAction.SPLIT, (_ref("alpha", 1),), (_ref("alpha-a", 1),)),
        (SkillEvolutionAction.PRUNE, (_ref("alpha", 1),), (_ref("alpha", 2),)),
        (SkillEvolutionAction.GENERATE, (_ref("alpha", 1),), (_ref("beta", 1),)),
    ],
)
def test_actions_reject_incompatible_input_output_shapes(
    action: SkillEvolutionAction,
    inputs: tuple[SkillVersionRef, ...],
    outputs: tuple[SkillVersionRef, ...],
) -> None:
    with pytest.raises(ValueError):
        _record(action, inputs, outputs)


def test_record_round_trip_preserves_its_content_hash() -> None:
    record = _record(
        SkillEvolutionAction.SPLIT,
        (_ref("alpha", 1),),
        (_ref("alpha-a", 1), _ref("alpha-b", 1)),
    )
    rebuilt = SkillEvolutionRecord.from_value(record.to_value())

    assert rebuilt == record
    assert rebuilt.record_hash == record.record_hash


def test_records_form_a_hash_linked_predecessor_chain() -> None:
    first = _record(SkillEvolutionAction.GENERATE, (), (_ref("alpha", 1),))
    second = _record(
        SkillEvolutionAction.REFINE,
        (_ref("alpha", 1),),
        (_ref("alpha", 2),),
        sequence_number=2,
        previous_record_hash=first.record_hash,
    )

    assert second.follows(first)
    validate_lineage((first, second))

    disconnected = replace(second, previous_record_hash=stable_hash({"other": "record"}))
    assert not disconnected.follows(first)
    with pytest.raises(ValueError):
        validate_lineage((first, disconnected))


def test_skill_references_must_be_unique_and_sorted() -> None:
    alpha = _ref("alpha", 1)
    beta = _ref("beta", 1)

    with pytest.raises(ValueError):
        _record(SkillEvolutionAction.SPLIT, (alpha,), (beta, alpha))
    with pytest.raises(ValueError):
        _record(SkillEvolutionAction.SPLIT, (alpha,), (beta, beta))
