from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from skillev_private.benchmarks.mind2web_scores import (
    Mind2WebScoreFormatError,
    load_mind2web_candidate_rankings,
)


def _score_payload() -> bytes:
    return pickle.dumps(
        {
            "scores": {
                "annotation-1_action-1": {
                    "node-c": 0.1,
                    "node-b": 0.9,
                    "node-a": 0.8,
                },
                "annotation-2_action-2": {
                    "node-e": -0.2,
                    "node-d": 0.5,
                },
            },
            "ranks": {
                "annotation-1_action-1": {
                    "node-c": 2,
                    "node-b": 1,
                    "node-a": 0,
                },
                "annotation-2_action-2": {
                    "node-e": 1,
                    "node-d": 0,
                },
            },
        },
        protocol=4,
    )


def test_streams_only_requested_top_k_in_official_score_order(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "scores.pkl").resolve()
    path.write_bytes(_score_payload())

    rankings = load_mind2web_candidate_rankings(
        path,
        action_ids=("annotation-1_action-1",),
        top_k=2,
    )

    assert rankings == {"annotation-1_action-1": ("node-b", "node-a")}


def test_rejects_missing_actions_and_another_builtin_shape(tmp_path: Path) -> None:
    path = (tmp_path / "scores.pkl").resolve()
    path.write_bytes(_score_payload())
    with pytest.raises(Mind2WebScoreFormatError):
        load_mind2web_candidate_rankings(
            path,
            action_ids=("missing_action",),
        )

    path.write_bytes(pickle.dumps({"not_scores": {}}, protocol=4))
    with pytest.raises(Mind2WebScoreFormatError):
        load_mind2web_candidate_rankings(path)


def test_reader_never_executes_pickle_reducers(tmp_path: Path) -> None:
    marker = (tmp_path / "pickle-reducer-ran").resolve()

    class _Reducer:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            expression = f"open({str(marker)!r}, 'w').write('unsafe')"
            return eval, (expression,)

    path = (tmp_path / "scores.pkl").resolve()
    path.write_bytes(
        pickle.dumps(
            {"scores": _Reducer(), "ranks": {}},
            protocol=4,
        )
    )

    with pytest.raises(Mind2WebScoreFormatError):
        load_mind2web_candidate_rankings(path)
    assert not marker.exists()
