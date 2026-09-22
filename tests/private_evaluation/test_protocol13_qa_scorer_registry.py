from skillev_private.benchmarks.qa_metrics import score_hotpotqa_answers
from skillev_private.direct_reference.evaluators import (
    SCORER_REGISTRY,
    require_protocol13_scorer_profile,
    score_hotpotqa,
)
from skillev_private.direct_reference.skillflow_iid import (
    parse_skillflow_trivia_answers,
)


def test_hotpot_v2_registration_uses_special_answer_scorer() -> None:
    assert require_protocol13_scorer_profile("hotpotqa-official-em-f1@2") is score_hotpotqa
    assert SCORER_REGISTRY["hotpotqa-official-em-f1@2"] is score_hotpotqa


def test_hotpot_special_answers_are_mutually_exclusive() -> None:
    assert score_hotpotqa_answers("yes", ("yes",)).em == 1.0
    assert score_hotpotqa_answers("yes indeed", ("yes",)).em == 0.0
    assert score_hotpotqa_answers("no", ("yes",)).f1 == 0.0
    assert score_hotpotqa_answers("noanswer", ("ordinary answer",)).f1 == 0.0


def test_trivia_preserves_every_unique_alias_present_on_the_wire() -> None:
    assert parse_skillflow_trivia_answers("a|b|c|d|e|f|f") == (
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    )
