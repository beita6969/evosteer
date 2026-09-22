from skillev.evaluation.direct_baseline.parsing import (
    ParseReason,
    parse_aime_boxed_integer,
)


def test_aime_boxed_integer_is_strict_and_uses_last_complete_box() -> None:
    assert parse_aime_boxed_integer(r"first \boxed{003}, final \boxed{999}").value == "999"
    assert parse_aime_boxed_integer(r"\boxed{1000}").value is None
    assert parse_aime_boxed_integer(r"\boxed{12.7}").value is None
    assert parse_aime_boxed_integer(r"\boxed{12").reason is ParseReason.INVALID_FORMAT


def test_aime_parser_never_salvages_the_private_reasoning_channel() -> None:
    visible_text = "I could not produce a final answer."
    private_reasoning = r"The answer is \boxed{123}."
    assert parse_aime_boxed_integer(visible_text).value is None
    assert parse_aime_boxed_integer(private_reasoning).value == "123"
