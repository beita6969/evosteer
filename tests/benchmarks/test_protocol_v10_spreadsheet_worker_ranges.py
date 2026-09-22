from __future__ import annotations

from skillev_private.benchmarks.protocol_v10_spreadsheet_worker import (
    _normalize_cell_range,
    _split_answer_positions,
    _upstream_answer_position_is_safe,
)


def test_answer_position_split_preserves_quoted_sheet_name_commas() -> None:
    assert _split_answer_positions("'b2b, sez, de'!A5:V10,'Sheet 2'!A:G") == (
        "'b2b, sez, de'!A5:V10",
        "'Sheet 2'!A:G",
    )
    assert _split_answer_positions("Sheet3'!A:G,'Sheet4'!A:G") == (
        "Sheet3'!A:G",
        "'Sheet4'!A:G",
    )


def test_noncanonical_ranges_expand_to_official_cell_rectangles() -> None:
    assert _normalize_cell_range("A:G", max_row=308, max_column=7) == "A1:G308"
    assert _normalize_cell_range("BD2:308", max_row=308, max_column=56) == "BD2:BD308"
    assert _normalize_cell_range("2:12", max_row=12, max_column=7) == "A2:G12"


def test_upstream_safe_detection_routes_only_supported_ranges_directly() -> None:
    assert _upstream_answer_position_is_safe("'Sheet1'!A1:B12")
    assert not _upstream_answer_position_is_safe("'Sheet3'!A:G,'Sheet4'!A:G")
    assert not _upstream_answer_position_is_safe("'b2b, sez, de'!A5:V10")
