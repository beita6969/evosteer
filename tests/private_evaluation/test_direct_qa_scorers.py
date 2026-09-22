from skillev_private.benchmarks.qa_metrics import (
    normalize_hotpotqa_answer,
    normalize_triviaqa_answer,
    score_qa_answers,
)


def test_qa_scorer_uses_token_multisets_and_best_alias() -> None:
    metrics = score_qa_answers(
        "The New York York",
        ("NYC", "new york"),
        normalizer=normalize_hotpotqa_answer,
    )
    assert metrics.em == 0
    assert metrics.f1 == 0.8


def test_trivia_normalizer_preserves_alias_semantics() -> None:
    metrics = score_qa_answers(
        "New York City",
        ("NYC", "New_York_City"),
        normalizer=normalize_triviaqa_answer,
    )
    assert metrics.em == 1
    assert metrics.f1 == 1
