"""Shared deterministic extraction of the final boxed MATH reference."""

from __future__ import annotations


def last_boxed_answer(solution: str) -> str:
    start = solution.rfind(r"\boxed")
    if start < 0:
        raise ValueError("MATH solution lacks a boxed answer")
    opening = solution.find("{", start)
    if opening < 0:
        raise ValueError("MATH boxed answer lacks braces")
    depth, cursor = 1, opening + 1
    while cursor < len(solution) and depth:
        depth += (solution[cursor] == "{") - (solution[cursor] == "}")
        cursor += 1
    if depth:
        raise ValueError("MATH boxed answer has unbalanced braces")
    return solution[opening + 1 : cursor - 1].strip()
