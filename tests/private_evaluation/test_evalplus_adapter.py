import pytest
from skillev_private.benchmarks.evalplus_adapter import parse_evalplus_single_candidate


def test_evalplus_v020_native_schema_requires_one_candidate() -> None:
    verdict = parse_evalplus_single_candidate(
        "Mbpp/1",
        {
            "nfiles": 1,
            "base": [["success", [True]]],
            "plus": [["success", [True]]],
        },
    )
    assert verdict.base_plus_passed
    with pytest.raises(ValueError, match="one candidate"):
        parse_evalplus_single_candidate(
            "Mbpp/1",
            {
                "nfiles": 2,
                "base": [["success", []], ["success", []]],
                "plus": [["success", []], ["success", []]],
            },
        )
