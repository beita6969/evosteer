import pytest
from skillev_private.direct_reference.skillflow_iid import (
    parse_skillflow_iid_record,
    trivia_aliases_from_record,
)


def _record(aliases: object):
    extra: dict[str, object] = {"source": "TriviaQA"}
    if aliases is not None:
        extra["aliases"] = aliases
    return parse_skillflow_iid_record(
        {
            "question": "Passage\nQuestion: Who?",
            "answer": "The Answer|Answer",
            "extra": extra,
        },
        source_position=0,
    )


def test_structured_trivia_aliases_must_agree_with_wire_after_normalization() -> None:
    aliases = trivia_aliases_from_record(_record(["the answer", "Answer"]))
    assert aliases.source == "validated-structured-and-wire"
    assert aliases.aliases == ("the answer", "Answer")


def test_structured_trivia_alias_disagreement_is_rejected() -> None:
    with pytest.raises(ValueError, match="disagree"):
        trivia_aliases_from_record(_record(["different"]))


def test_wire_provenance_remains_explicit_without_structured_aliases() -> None:
    aliases = trivia_aliases_from_record(_record(None))
    assert aliases.source == "skillflow-prepared-wire-first-five"
