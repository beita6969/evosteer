"""Exact and SymPy-official LaTeX equivalence backend for MATH-Hard."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from sympy import Basic, simplify  # type: ignore[import-untyped]
from sympy.core.sympify import SympifyError  # type: ignore[import-untyped]
from sympy.parsing.latex import parse_latex  # type: ignore[import-untyped]
from sympy.parsing.latex.errors import LaTeXParsingError  # type: ignore[import-untyped]

from .code_math import (
    MathEquivalenceInfrastructureError,
    MathEquivalenceKind,
    MathEquivalenceRequest,
    MathEquivalenceResult,
)

_WHITESPACE = re.compile(r"\s+")


def _exact_form(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    if len(normalized) >= 2 and normalized[0] == "$" and normalized[-1] == "$":
        normalized = normalized[1:-1]
    return _WHITESPACE.sub("", normalized)


def _parse_reference(reference: str) -> Basic:
    try:
        expression = parse_latex(reference)
    except (LaTeXParsingError, SympifyError, TypeError, ValueError) as error:
        raise MathEquivalenceInfrastructureError(
            "trusted MATH-Hard reference could not be parsed"
        ) from error
    if not isinstance(expression, Basic):
        raise MathEquivalenceInfrastructureError(
            "trusted MATH-Hard reference produced an invalid expression"
        )
    return expression


def _parse_prediction(prediction: str) -> Basic | None:
    try:
        expression = parse_latex(prediction)
    except (LaTeXParsingError, SympifyError, TypeError, ValueError):
        return None
    return expression if isinstance(expression, Basic) else None


def _symbolically_equal(prediction: Basic, reference: Basic) -> bool:
    if prediction == reference:
        return True
    try:
        difference = simplify(prediction - reference)
    except (ArithmeticError, NotImplementedError, TypeError, ValueError):
        difference = None
    if difference == 0:
        return True
    if isinstance(difference, Basic) and difference.equals(0) is True:
        return True
    try:
        return prediction.equals(reference) is True
    except (ArithmeticError, NotImplementedError, TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class SympyMathEquivalenceBackend:
    """Return exact or symbolic equivalence without exposing reference text."""

    async def check(self, request: MathEquivalenceRequest) -> MathEquivalenceResult:
        if not isinstance(request, MathEquivalenceRequest):
            raise TypeError("MATH-Hard backend requires MathEquivalenceRequest")
        if _exact_form(request.prediction) == _exact_form(request.reference_answer):
            return MathEquivalenceResult(MathEquivalenceKind.EXACT)

        reference = _parse_reference(request.reference_answer)
        prediction = _parse_prediction(request.prediction)
        if prediction is None:
            return MathEquivalenceResult(MathEquivalenceKind.NOT_EQUIVALENT)
        kind = (
            MathEquivalenceKind.EQUIVALENT
            if _symbolically_equal(prediction, reference)
            else MathEquivalenceKind.NOT_EQUIVALENT
        )
        return MathEquivalenceResult(kind)


__all__ = ["SympyMathEquivalenceBackend"]
