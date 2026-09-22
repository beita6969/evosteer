from skillev_private.benchmarks.qa_metrics import score_hotpotqa_answers
from skillev_private.direct_reference.skillflow_iid import parse_skillflow_trivia_answers


def test_hotpot_special_answers_are_mutually_exclusive() -> None:
    score = score_hotpotqa_answers("yes indeed", ("yes",))
    assert score.em == 0.0
    assert score.f1 == 0.0


def test_trivia_wire_preserves_more_than_five_unique_aliases() -> None:
    aliases = parse_skillflow_trivia_answers("a|b|c|d|e|f|a")
    assert aliases == ("a", "b", "c", "d", "e", "f")
