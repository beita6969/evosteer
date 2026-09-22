from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest
from skillev_private.benchmarks import (
    GPQA_DIAMOND_CSV_HEADER,
    PrivateGPQADiamondCase,
    PrivateHumanEvalCase,
    PrivateMathCase,
    PrivateMind2WebStepCase,
    PrivateStaticBenchmarkCase,
    StaticScoringRule,
    convert_atlas_nq_row,
    convert_atlas_triviaqa_row,
    convert_gpqa_diamond_row,
    convert_hotpotqa_row,
    convert_humaneval_row,
    convert_math_hard_row,
    convert_matharena_aime_2026_row,
    convert_medqa_us_4option_row,
    convert_mind2web_row,
    convert_musique_row,
)

from skillev.contracts import canonical_json

PRIVATE_CANARY = "PRIVATE-SOURCE-CONVERTER-CANARY"
REVISION = "synthetic-fixture@1"

PublicCase = (
    PrivateStaticBenchmarkCase
    | PrivateGPQADiamondCase
    | PrivateHumanEvalCase
    | PrivateMathCase
    | PrivateMind2WebStepCase
)


def _public_wire(case: PublicCase) -> str:
    return canonical_json(case.public.to_rollout_task().to_value())


def _hotpot_row() -> dict[str, object]:
    return {
        "id": "synthetic-hotpot-1",
        "question": "Which public entity connects the two documents?",
        "answer": PRIVATE_CANARY,
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {"title": ["Public A", "Public B"], "sent_id": [0, 0]},
        "context": {
            "title": ["Public A", "Public B"],
            "sentences": [["A public sentence."], ["Another public sentence."]],
        },
    }


def _medqa_row() -> dict[str, object]:
    return {
        "question": "Which public option should be selected?",
        "answer": "Public option bravo",
        "options": {
            "A": "Public option alpha",
            "B": "Public option bravo",
            "C": "Public option charlie",
            "D": "Public option delta",
        },
        "meta_info": "USMLE synthetic fixture",
        "answer_idx": "B",
    }


def _musique_row() -> dict[str, object]:
    return {
        "id": "2hop__11_22",
        "paragraphs": [
            {
                "idx": 0,
                "title": "Public first paragraph",
                "paragraph_text": "The first public context passage.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Public second paragraph",
                "paragraph_text": "The second public context passage.",
                "is_supporting": True,
            },
        ],
        "question": "What is the final public short answer?",
        "question_decomposition": [
            {
                "id": 11,
                "question": "private decomposition step one",
                "answer": "private intermediate",
                "paragraph_support_idx": 0,
            },
            {
                "id": 22,
                "question": "private decomposition step two",
                "answer": PRIVATE_CANARY,
                "paragraph_support_idx": 1,
            },
        ],
        "answer": PRIVATE_CANARY,
        "answer_aliases": ["private alias"],
        "answerable": True,
    }


def _gpqa_row() -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(GPQA_DIAMOND_CSV_HEADER, "")
    row.update(
        {
            "Question": "Which option follows from the public scientific premise?",
            "Correct Answer": "Correct-looking public choice",
            "Incorrect Answer 1": "Distractor one",
            "Incorrect Answer 2": "Distractor two",
            "Incorrect Answer 3": "Distractor three",
            "Explanation": "Private verifier explanation.",
            "Record ID": "synthetic-record-1",
            "High-level domain": "Physics",
            "Subdomain": "Synthetic mechanics",
            "Canary String": PRIVATE_CANARY,
        }
    )
    return row


def _mind2web_row() -> dict[str, object]:
    return {
        "annotation_id": "synthetic-annotation-1",
        "website": "public.example",
        "domain": "shopping",
        "subdomain": "retail",
        "confirmed_task": "Select the requested public item.",
        "action_reprs": [f"CLICK private gold representation {PRIVATE_CANARY}"],
        "actions": [
            {
                "action_uid": "synthetic-action-1",
                "raw_html": "<html><button>public</button></html>",
                "cleaned_html": "<button>public</button>",
                "operation": {
                    "op": "CLICK",
                    "original_op": "CLICK",
                    "value": PRIVATE_CANARY,
                },
                "pos_candidates": [
                    {
                        "tag": "button",
                        "is_original_target": True,
                        "is_top_level_target": True,
                        "backend_node_id": "node-z",
                        "attributes": '{"name":"public target-looking button"}',
                    }
                ],
                "neg_candidates": [
                    {
                        "tag": "button",
                        "backend_node_id": "node-a",
                        "attributes": '{"name":"public distractor"}',
                    }
                ],
            }
        ],
    }


def test_hotpot_converter_validates_nested_source_and_hides_labels() -> None:
    case = convert_hotpotqa_row(_hotpot_row(), dataset_revision=REVISION, split="dev")

    assert isinstance(case, PrivateStaticBenchmarkCase)
    assert case.target.scoring_rule is StaticScoringRule.TOKEN_F1
    assert case.target.accepted_answers == (PRIVATE_CANARY,)
    assert PRIVATE_CANARY not in _public_wire(case)
    assert "supporting_facts" not in _public_wire(case)

    broken = _hotpot_row()
    broken["supporting_facts"] = {"title": ["Missing title"], "sent_id": [0]}
    with pytest.raises(ValueError):
        convert_hotpotqa_row(broken, dataset_revision=REVISION, split="dev")


def test_hotpot_converter_accepts_official_support_and_sentence_shapes() -> None:
    row = _hotpot_row()
    row["context"] = {
        "title": ["Public A", "Public B"],
        "sentences": [
            ["Repeated sentence.", "", "Repeated sentence."],
            ["Another public sentence."],
        ],
    }
    row["supporting_facts"] = {
        "title": ["Public A", "Public A"],
        "sent_id": [0, 902],
    }

    case = convert_hotpotqa_row(row, dataset_revision=REVISION, split="dev")

    assert case.target.accepted_answers == (PRIVATE_CANARY,)


def test_atlas_trivia_and_nq_exact_rows_split_private_answers() -> None:
    trivia = convert_atlas_triviaqa_row(
        {
            "question": "A public trivia question?",
            "answers": [PRIVATE_CANARY, "private alias"],
            "target": PRIVATE_CANARY,
        },
        dataset_revision=REVISION,
        split="dev",
    )
    nq = convert_atlas_nq_row(
        {"question": "A public NQ question?", "answers": [PRIVATE_CANARY]},
        dataset_revision=REVISION,
        split="test",
    )

    assert trivia.target.scoring_rule is StaticScoringRule.TOKEN_F1
    assert nq.target.scoring_rule is StaticScoringRule.EXACT_TEXT
    assert PRIVATE_CANARY not in _public_wire(trivia)
    assert PRIVATE_CANARY not in _public_wire(nq)
    title_cased_target = convert_atlas_triviaqa_row(
        {"question": "q", "answers": ["USA"], "target": "Usa"},
        dataset_revision=REVISION,
        split="dev",
    )
    assert title_cased_target.target.accepted_answers == ("USA",)


def test_medqa_converter_keeps_choices_public_and_correct_label_private() -> None:
    case = convert_medqa_us_4option_row(_medqa_row(), dataset_revision=REVISION, split="train")

    assert case.target.accepted_answers == ("B",)
    assert case.target.scoring_rule is StaticScoringRule.OPTION
    public_wire = _public_wire(case)
    assert "Public option bravo" in public_wire
    assert "answer_idx" not in public_wire

    broken = _medqa_row()
    broken["answer"] = "Public option alpha"
    with pytest.raises(ValueError):
        convert_medqa_us_4option_row(broken, dataset_revision=REVISION, split="train")

    duplicate_options = _medqa_row()
    duplicate_options["options"] = {
        "A": "Repeated option",
        "B": "Repeated option",
        "C": "Public option charlie",
        "D": "Public option delta",
    }
    duplicate_options["answer"] = "Repeated option"
    duplicate = convert_medqa_us_4option_row(
        duplicate_options,
        dataset_revision=REVISION,
        split="train",
    )
    assert duplicate.target.accepted_answers == ("B",)


def test_musique_converter_exposes_paragraphs_but_not_support_or_decomposition() -> None:
    case = convert_musique_row(_musique_row(), dataset_revision=REVISION, split="validation")

    assert case.target.accepted_answers == (PRIVATE_CANARY, "private alias")
    public_wire = _public_wire(case)
    assert "The first public context passage." in public_wire
    assert "is_supporting" not in public_wire
    assert "question_decomposition" not in public_wire
    assert PRIVATE_CANARY not in public_wire

    broken = _musique_row()
    decomposition = list(broken["question_decomposition"])
    decomposition[0] = {**decomposition[0], "paragraph_support_idx": 1}
    paragraphs = list(broken["paragraphs"])
    paragraphs[1] = {**paragraphs[1], "is_supporting": False}
    broken["question_decomposition"] = decomposition
    broken["paragraphs"] = paragraphs
    with pytest.raises(ValueError):
        convert_musique_row(broken, dataset_revision=REVISION, split="validation")


def test_gpqa_converter_pins_full_header_and_never_exposes_canary() -> None:
    case = convert_gpqa_diamond_row(_gpqa_row(), dataset_revision=REVISION, split="train")

    assert isinstance(case, PrivateGPQADiamondCase)
    assert case.target.canary_string == PRIVATE_CANARY
    assert case.target.to_static_target().accepted_answers == (case.target.correct_option,)
    assert PRIVATE_CANARY not in _public_wire(case)
    assert "Private verifier explanation" not in _public_wire(case)
    options = case.public.public_context["options"]
    assert isinstance(options, dict)
    assert options[case.target.correct_option] == "Correct-looking public choice"

    extra = _gpqa_row()
    extra["unexpected"] = "schema drift"
    with pytest.raises(ValueError):
        convert_gpqa_diamond_row(extra, dataset_revision=REVISION, split="train")
    wrong_cell_type = _gpqa_row()
    wrong_cell_type["Non-Expert Validator Accuracy"] = 0.5
    with pytest.raises(TypeError):
        convert_gpqa_diamond_row(wrong_cell_type, dataset_revision=REVISION, split="train")


def test_gpqa_converter_preserves_duplicate_official_distractor_slots() -> None:
    duplicate_distractor = _gpqa_row()
    duplicate_distractor["Incorrect Answer 3"] = duplicate_distractor["Incorrect Answer 2"]

    case = convert_gpqa_diamond_row(
        duplicate_distractor,
        dataset_revision=REVISION,
        split="train",
    )

    options = case.public.public_context["options"]
    assert isinstance(options, dict)
    assert len(options) == 4
    assert len(set(options.values())) == 3

    ambiguous_correct = _gpqa_row()
    ambiguous_correct["Incorrect Answer 1"] = ambiguous_correct["Correct Answer"]
    with pytest.raises(ValueError):
        convert_gpqa_diamond_row(
            ambiguous_correct,
            dataset_revision=REVISION,
            split="train",
        )


def test_humaneval_converter_keeps_solution_and_tests_private_without_execution() -> None:
    case = convert_humaneval_row(
        {
            "task_id": "HumanEval/synthetic-1",
            "prompt": 'def public_function(value: int) -> int:\n    """Public task."""\n',
            "canonical_solution": f"    return value  # {PRIVATE_CANARY}\n",
            "test": f"def check(candidate):\n    assert candidate(1) == 1  # {PRIVATE_CANARY}\n",
            "entry_point": "public_function",
        },
        dataset_revision=REVISION,
        split="test",
    )

    assert isinstance(case, PrivateHumanEvalCase)
    assert PRIVATE_CANARY in case.target.test
    assert PRIVATE_CANARY not in _public_wire(case)
    assert not hasattr(case.target, "execute")


def test_math_converter_strictly_extracts_last_balanced_box_without_scoring() -> None:
    case = convert_math_hard_row(
        {
            "problem": "Compute the public synthetic expression.",
            "level": "Level 5",
            "type": "Algebra",
            "solution": (
                f"Private reasoning {PRIVATE_CANARY}; first \\boxed{{1}}, "
                r"final \boxed {\frac{2}{3}}."
            ),
        },
        dataset_revision=REVISION,
        split="test",
    )

    assert isinstance(case, PrivateMathCase)
    assert case.target.boxed_answer == r"\frac{2}{3}"
    assert PRIVATE_CANARY not in _public_wire(case)
    with pytest.raises(ValueError):
        convert_math_hard_row(
            {
                "problem": "public problem",
                "level": "Level 5",
                "type": "Algebra",
                "solution": "private reasoning without a boxed answer",
            },
            dataset_revision=REVISION,
            split="test",
        )


def test_matharena_aime_2026_converter_keeps_integer_answer_private() -> None:
    case = convert_matharena_aime_2026_row(
        {
            "problem_idx": 7,
            "answer": 42,
            "problem": "Find the requested public competition integer.",
        },
        dataset_revision=REVISION,
        split="train",
    )

    assert case.public.task_id == "aime-2026/07"
    assert case.target.scoring_rule is StaticScoringRule.INTEGER
    assert case.target.accepted_answers == ("42",)
    assert "accepted_answers" not in _public_wire(case)
    assert case.target.score("042") == 1.0

    with pytest.raises(ValueError):
        convert_matharena_aime_2026_row(
            {"problem_idx": 31, "answer": 42, "problem": "public"},
            dataset_revision=REVISION,
            split="train",
        )


def test_mind2web_converter_strips_candidate_labels_and_gold_action() -> None:
    row = _mind2web_row()
    action_id = "synthetic-annotation-1_synthetic-action-1"
    cases = convert_mind2web_row(
        row,
        dataset_revision=REVISION,
        split="test_domain",
        candidate_rankings={action_id: ("node-a", "node-z")},
    )

    assert len(cases) == 1
    case = cases[0]
    assert isinstance(case, PrivateMind2WebStepCase)
    assert case.target.positive_backend_node_ids == ("node-z",)
    assert case.target.value == PRIVATE_CANARY
    public_wire = _public_wire(case)
    assert PRIVATE_CANARY not in public_wire
    assert "is_original_target" not in public_wire
    assert "is_top_level_target" not in public_wire
    candidates = case.public.public_context["candidates"]
    assert isinstance(candidates, list)
    assert [candidate["backend_node_id"] for candidate in candidates] == ["node-a", "node-z"]

    no_positive = _mind2web_row()
    actions = list(no_positive["actions"])
    actions[0] = {**actions[0], "pos_candidates": []}
    no_positive["actions"] = actions
    unavailable_case = convert_mind2web_row(
        no_positive,
        dataset_revision=REVISION,
        split="test_domain",
        candidate_rankings={action_id: ("node-a",)},
    )[0]
    assert unavailable_case.target.positive_backend_node_ids == ()
    assert unavailable_case.public.public_context["candidates"] == [
        {
            "attributes": '{"name":"public distractor"}',
            "backend_node_id": "node-a",
            "tag": "button",
        }
    ]


def test_mind2web_converter_keeps_repeated_positional_action_representations() -> None:
    row = _mind2web_row()
    actions = list(row["actions"])
    actions.append({**actions[0], "action_uid": "synthetic-action-2"})
    row["actions"] = actions
    row["action_reprs"] = ["Repeated action representation"] * 2

    cases = convert_mind2web_row(
        row,
        dataset_revision=REVISION,
        split="test_domain",
        candidate_rankings={
            "synthetic-annotation-1_synthetic-action-1": ("node-a", "node-z"),
            "synthetic-annotation-1_synthetic-action-2": ("node-a", "node-z"),
        },
    )

    assert tuple(case.target.action_repr for case in cases) == (
        "Repeated action representation",
        "Repeated action representation",
    )


def test_mind2web_converter_uses_official_ranked_top_k_without_reencoding_targets() -> None:
    row = _mind2web_row()
    action_id = "synthetic-annotation-1_synthetic-action-1"

    case = convert_mind2web_row(
        row,
        dataset_revision=REVISION,
        split="test_domain",
        candidate_rankings={action_id: ("node-z", "node-a")},
    )[0]

    candidates = case.public.public_context["candidates"]
    assert isinstance(candidates, list)
    assert [candidate["backend_node_id"] for candidate in candidates] == [
        "node-z",
        "node-a",
    ]
    assert case.public.public_context["candidate_count"] == 2
    assert case.public.public_context["candidate_top_k"] == 50
    assert case.target.positive_backend_node_ids == ("node-z",)

    unavailable = convert_mind2web_row(
        row,
        dataset_revision=REVISION,
        split="test_domain",
        candidate_rankings={action_id: ("node-a",)},
    )[0]
    assert unavailable.target.positive_backend_node_ids == ()


Converter = Callable[..., object]


@pytest.mark.parametrize(
    ("converter", "row", "split"),
    [
        (convert_hotpotqa_row, _hotpot_row(), "dev"),
        (
            convert_atlas_triviaqa_row,
            {"question": "q", "answers": ["a"], "target": "a"},
            "dev",
        ),
        (convert_atlas_nq_row, {"question": "q", "answers": ["a"]}, "test"),
        (convert_medqa_us_4option_row, _medqa_row(), "train"),
        (convert_musique_row, _musique_row(), "validation"),
        (convert_gpqa_diamond_row, _gpqa_row(), "train"),
        (
            convert_humaneval_row,
            {
                "task_id": "HumanEval/synthetic",
                "prompt": "def f():\n",
                "canonical_solution": "    return 1\n",
                "test": "def check(candidate): assert candidate() == 1",
                "entry_point": "f",
            },
            "test",
        ),
        (
            convert_math_hard_row,
            {
                "problem": "public",
                "level": "Level 5",
                "type": "Algebra",
                "solution": r"private \\boxed{1}",
            },
            "test",
        ),
        (
            convert_matharena_aime_2026_row,
            {"problem_idx": 1, "answer": 0, "problem": "public"},
            "train",
        ),
    ],
)
def test_every_source_converter_rejects_unknown_fields(
    converter: Converter,
    row: Mapping[str, object],
    split: str,
) -> None:
    drifted = dict(row)
    drifted["unexpected_field"] = "must not be ignored"

    with pytest.raises(ValueError):
        converter(drifted, dataset_revision=REVISION, split=split)


def test_mind2web_source_converter_rejects_unknown_fields_with_required_ranking() -> None:
    row = _mind2web_row()
    row["unexpected_field"] = "must not be ignored"

    with pytest.raises(ValueError):
        convert_mind2web_row(
            row,
            dataset_revision=REVISION,
            split="test_domain",
            candidate_rankings={"synthetic-annotation-1_synthetic-action-1": ("node-a", "node-z")},
        )


def test_mind2web_production_converter_requires_rankings_at_the_call_boundary() -> None:
    with pytest.raises(TypeError):
        convert_mind2web_row(  # type: ignore[call-arg]
            _mind2web_row(),
            dataset_revision=REVISION,
            split="test_domain",
        )
