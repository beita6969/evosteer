from skillev.evaluation.current_iid.protocol13.aime_extraction import (
    AIMEExtractionReason,
    extract_aime_source_final,
)
from skillev.evaluation.direct_baseline.parsing import (
    ParseReason,
    ParseStatus,
    parse_aime_source_integer,
)


def test_source_aime_parser_prefers_last_complete_box() -> None:
    parsed = extract_aime_source_final(r"Answer: 12\nthen \boxed{34}\nand \boxed{56}")
    assert parsed.value == 56
    assert parsed.reason is AIMEExtractionReason.BOXED
    public = parse_aime_source_integer(r"Answer: 12\nthen \boxed{34}\nand \boxed{56}")
    assert public.value == "56"
    assert public.reason is ParseReason.BOXED_ANSWER


def test_source_aime_parser_uses_only_explicit_fallback() -> None:
    assert extract_aime_source_final("work contains 123\nFinal answer: 42").value == 42
    assert extract_aime_source_final("work contains 123\n42").value is None
    assert parse_aime_source_integer("work contains 123\n42").status is ParseStatus.EMPTY


def test_source_aime_parser_rejects_malformed_complete_box_and_conflicts() -> None:
    malformed = extract_aime_source_final(r"Final answer: 42\n\boxed{42.0}")
    assert malformed.value is None
    assert malformed.reason is AIMEExtractionReason.INVALID_INTEGER
    conflicting = parse_aime_source_integer("Final answer: 42\nAnswer: 43")
    assert conflicting.status is ParseStatus.AMBIGUOUS
    assert conflicting.reason is ParseReason.CONFLICTING_FINALS
