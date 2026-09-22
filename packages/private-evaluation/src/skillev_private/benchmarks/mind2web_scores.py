"""Bounded, non-executing projection of official Mind2Web candidate scores."""

from __future__ import annotations

import math
import pickletools
from collections.abc import Callable, Collection, Iterator
from pathlib import Path
from typing import Final

import numpy as np

MIND2WEB_CANDIDATE_TOP_K: Final = 50


class Mind2WebScoreFormatError(ValueError):
    """Raised when the pinned score artifact has another data shape."""


def _pickle_ops(path: Path) -> Iterator[tuple[str, object]]:
    try:
        with path.open("rb") as stream:
            for opcode, argument, _ in pickletools.genops(stream):
                if opcode.name != "FRAME":
                    yield opcode.name, argument
    except (OSError, ValueError) as error:
        raise Mind2WebScoreFormatError("Mind2Web score artifact is not a valid pickle") from error


class _OpcodeStream:
    def __init__(self, source: Iterator[tuple[str, object]]) -> None:
        self._source = source
        self._pending: tuple[str, object] | None = None

    def take(self) -> tuple[str, object]:
        if self._pending is not None:
            item = self._pending
            self._pending = None
            return item
        try:
            return next(self._source)
        except StopIteration as error:
            raise Mind2WebScoreFormatError("Mind2Web score artifact ended early") from error

    def put_back(self, item: tuple[str, object]) -> None:
        if self._pending is not None:
            raise AssertionError("Mind2Web opcode stream already has one pending item")
        self._pending = item


def _expect(
    ops: _OpcodeStream,
    opcode_name: str,
    argument: object | None = None,
) -> object:
    actual_name, actual_argument = ops.take()
    if actual_name != opcode_name or (argument is not None and actual_argument != argument):
        raise Mind2WebScoreFormatError("Mind2Web score artifact has another pickle shape")
    return actual_argument


def _text_opcode(item: tuple[str, object], *, label: str) -> str:
    opcode_name, argument = item
    if (
        opcode_name not in {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8"}
        or type(argument) is not str
        or not argument
        or "\x00" in argument
    ):
        raise Mind2WebScoreFormatError(f"Mind2Web score {label} is invalid")
    return argument


def _read_dict_items(
    ops: _OpcodeStream,
    *,
    read_key: Callable[[tuple[str, object]], str],
    read_value: Callable[[_OpcodeStream], None],
) -> None:
    """Consume protocol-4 dict items without constructing the dictionary."""

    first = ops.take()
    if first[0] != "MARK":
        read_key(first)
        read_value(ops)
        _expect(ops, "SETITEM")
        return
    while True:
        item = ops.take()
        if item[0] == "SETITEMS":
            following = ops.take()
            if following[0] == "MARK":
                continue
            ops.put_back(following)
            return
        read_key(item)
        read_value(ops)


def load_mind2web_candidate_rankings(
    path: Path,
    *,
    action_ids: Collection[str] | None = None,
    top_k: int = MIND2WEB_CANDIDATE_TOP_K,
) -> dict[str, tuple[str, ...]]:
    """Read only the pickle's score map and retain at most ``top_k`` IDs per action.

    The official artifact is a protocol-4 pickle containing two built-in
    mappings: ``scores`` and ``ranks``.  Calling :func:`pickle.load` would both
    execute pickle reducers and materialize roughly ten million score entries.
    This reader interprets the observed built-in score-map opcodes directly,
    never imports a pickle global, and discards every non-top-k score before
    advancing to the next action.  Ranking uses the official candidate
    generator's ``numpy.argsort(-scores)`` operation over pickle insertion
    order.
    """

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("Mind2Web score path must be absolute")
    if not path.is_file():
        raise FileNotFoundError(path)
    if type(top_k) is not int or top_k < 1:
        raise ValueError("Mind2Web top_k must be a positive integer")
    requested: frozenset[str] | None
    if action_ids is None:
        requested = None
    else:
        requested = frozenset(action_ids)
        if not requested or any(
            type(action_id) is not str or not action_id.strip() or "\x00" in action_id
            for action_id in requested
        ):
            raise ValueError("Mind2Web action_ids must contain non-empty text")

    ops = _OpcodeStream(_pickle_ops(path))
    _expect(ops, "PROTO", 4)
    _expect(ops, "EMPTY_DICT")
    _expect(ops, "MEMOIZE")
    _expect(ops, "MARK")
    _expect(ops, "SHORT_BINUNICODE", "scores")
    _expect(ops, "MEMOIZE")
    _expect(ops, "EMPTY_DICT")
    _expect(ops, "MEMOIZE")

    result: dict[str, tuple[str, ...]] = {}
    seen_actions: set[str] = set()
    current_action: str | None = None

    def read_action(item: tuple[str, object]) -> str:
        nonlocal current_action
        action_id = _text_opcode(item, label="action identity")
        if action_id in seen_actions:
            raise Mind2WebScoreFormatError("Mind2Web score action identity is duplicated")
        seen_actions.add(action_id)
        current_action = action_id
        return action_id

    def read_candidates(candidate_ops: _OpcodeStream) -> None:
        if current_action is None:
            raise AssertionError("Mind2Web candidate map has no action identity")
        action_id = current_action
        _expect(candidate_ops, "MEMOIZE")
        _expect(candidate_ops, "EMPTY_DICT")
        _expect(candidate_ops, "MEMOIZE")
        retain = requested is None or action_id in requested
        candidate_ids: list[str] = []
        scores: list[float] = []

        seen_candidates: set[str] = set()
        current_candidate: str | None = None

        def read_candidate(item: tuple[str, object]) -> str:
            nonlocal current_candidate
            candidate_id = _text_opcode(item, label="candidate identity")
            if candidate_id in seen_candidates:
                raise Mind2WebScoreFormatError("Mind2Web score candidate identity is duplicated")
            seen_candidates.add(candidate_id)
            current_candidate = candidate_id
            return candidate_id

        def read_score(score_ops: _OpcodeStream) -> None:
            if current_candidate is None:
                raise AssertionError("Mind2Web score has no candidate identity")
            _expect(score_ops, "MEMOIZE")
            score = _expect(score_ops, "BINFLOAT")
            if type(score) is not float or not math.isfinite(score):
                raise Mind2WebScoreFormatError("Mind2Web candidate score must be finite")
            if retain:
                candidate_ids.append(current_candidate)
                scores.append(score)

        _read_dict_items(
            candidate_ops,
            read_key=read_candidate,
            read_value=read_score,
        )
        if retain:
            order = np.argsort(-np.asarray(scores, dtype=np.float64))
            result[action_id] = tuple(candidate_ids[int(index)] for index in order[:top_k])

    _read_dict_items(
        ops,
        read_key=read_action,
        read_value=read_candidates,
    )
    if requested is not None:
        missing = requested.difference(result)
        if missing:
            raise Mind2WebScoreFormatError(
                "Mind2Web score artifact is missing requested action identities"
            )
    return result


__all__ = [
    "MIND2WEB_CANDIDATE_TOP_K",
    "Mind2WebScoreFormatError",
    "load_mind2web_candidate_rankings",
]
