from __future__ import annotations

import pytest

from skillev.contracts import (
    CanonicalizationError,
    canonical_json,
    parse_canonical_json,
    stable_hash,
)


def test_object_order_does_not_change_canonical_json_or_hash() -> None:
    first = {"beta": [2, 1], "alpha": {"value": True}}
    second = {"alpha": {"value": True}, "beta": [2, 1]}

    assert canonical_json(first) == canonical_json(second)
    assert stable_hash(first) == stable_hash(second)


def test_unicode_and_negative_zero_have_one_representation() -> None:
    decomposed = {"e\u0301": -0.0}
    composed = {"é": 0.0}

    assert canonical_json(decomposed) == canonical_json(composed)
    assert stable_hash(decomposed) == stable_hash(composed)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), b"bytes", {1: "bad"}])
def test_unsupported_values_are_rejected(value: object) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json(value)


def test_keys_that_collide_after_unicode_normalization_are_rejected() -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json({"é": 1, "e\u0301": 2})


def test_parser_requires_the_input_to_already_be_canonical() -> None:
    value = {"alpha": 1, "beta": ["two"]}
    text = canonical_json(value)

    assert parse_canonical_json(text) == value
    with pytest.raises(CanonicalizationError):
        parse_canonical_json('{ "alpha": 1, "beta": ["two"] }')
