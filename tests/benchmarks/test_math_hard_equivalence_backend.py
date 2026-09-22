from __future__ import annotations

import asyncio

import pytest
from skillev_private.benchmarks import (
    MathEquivalenceInfrastructureError,
    MathEquivalenceKind,
    MathEquivalenceRequest,
    SympyMathEquivalenceBackend,
    math_hard_official,
)


def _check(prediction: str, reference: str) -> MathEquivalenceKind:
    result = asyncio.run(
        SympyMathEquivalenceBackend().check(
            MathEquivalenceRequest(
                task_id="synthetic-math-task",
                prediction=prediction,
                reference_answer=reference,
            )
        )
    )
    return result.kind


def test_exact_match_uses_normalized_exact_path_without_symbolic_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_parser(value: str) -> object:
        raise AssertionError(value)

    monkeypatch.setattr(math_hard_official, "parse_latex", forbidden_parser)

    assert _check(" $ 7 $ ", "7") is MathEquivalenceKind.EXACT


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("0.5", r"\frac{1}{2}"),
        (r"(x+1)^2", r"x^2+2x+1"),
        (r"\sqrt{9}", "3"),
    ],
)
def test_sympy_official_latex_parser_proves_equivalence(
    prediction: str,
    reference: str,
) -> None:
    assert _check(prediction, reference) is MathEquivalenceKind.EQUIVALENT


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("5", "8"),
        (r"x^2", r"x^3"),
        (r"\frac{", r"\frac{1}{3}"),
    ],
)
def test_non_equivalent_or_unparseable_candidate_is_a_valid_miss(
    prediction: str,
    reference: str,
) -> None:
    assert _check(prediction, reference) is MathEquivalenceKind.NOT_EQUIVALENT


def test_unparseable_trusted_reference_is_infrastructure_failure_without_reference() -> None:
    private_reference = r"\frac{SYNTHETIC-PRIVATE-MARKER"

    with pytest.raises(MathEquivalenceInfrastructureError) as caught:
        _check("1", private_reference)

    assert private_reference not in str(caught.value)
    assert "SYNTHETIC-PRIVATE-MARKER" not in str(caught.value)


def test_result_contains_only_content_free_match_kind() -> None:
    request = MathEquivalenceRequest(
        task_id="synthetic-math-task",
        prediction="2/4",
        reference_answer="1/2",
    )

    result = asyncio.run(SympyMathEquivalenceBackend().check(request))

    assert result.kind is MathEquivalenceKind.EQUIVALENT
    assert not hasattr(result, "prediction")
    assert not hasattr(result, "reference_answer")
