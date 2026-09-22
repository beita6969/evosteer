from skillev.evaluation.direct_baseline.parsing import (
    ParseStatus,
    parse_alfworld_source_action_strict,
    parse_webshop_source_action_strict,
)


def test_webshop_source_parser_accepts_one_bare_native_action() -> None:
    assert parse_webshop_source_action_strict("search[blue cotton shirt]").value == (
        "search[blue cotton shirt]"
    )
    assert parse_webshop_source_action_strict("click[Buy Now]").value == "click[Buy Now]"
    assert parse_webshop_source_action_strict("Action: click[Buy Now]").status is ParseStatus.EMPTY
    assert parse_webshop_source_action_strict("think\nclick[Buy Now]").status is ParseStatus.EMPTY


def test_alfworld_source_parser_accepts_one_bare_line_only() -> None:
    assert parse_alfworld_source_action_strict("go to cabinet 1").value == "go to cabinet 1"
    assert (
        parse_alfworld_source_action_strict("Action: go to cabinet 1").status is ParseStatus.EMPTY
    )
    assert (
        parse_alfworld_source_action_strict("reason\ngo to cabinet 1").status is ParseStatus.EMPTY
    )
