"""Exact-schema converters for privately stored benchmark source rows.

Only synthetic fixtures for these functions belong in Git.  Real rows are
read in ignored run directories, converted there, and split into a model-
visible :class:`~skillev.benchmarks.BenchmarkPublicItem` plus verifier-only
target state.  Every converter accepts one named upstream wire shape; missing
or extra fields are rejected rather than guessed or adapted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from skillev.benchmarks import BenchmarkPublicItem
from skillev.contracts import JsonValue, normalize_json, stable_hash

from .mind2web_scores import MIND2WEB_CANDIDATE_TOP_K
from .source_cases import (
    PrivateGPQADiamondCase,
    PrivateGPQADiamondTarget,
    PrivateHumanEvalCase,
    PrivateHumanEvalTarget,
    PrivateMathCase,
    PrivateMathTarget,
    PrivateMind2WebStepCase,
    PrivateMind2WebStepTarget,
)
from .static import PrivateStaticBenchmarkCase, PrivateStaticTarget, StaticScoringRule

# Exact header of the privately acquired official Idavidrein/gpqa Diamond CSV.
# The values never enter a public/model-facing serialization.
GPQA_DIAMOND_CSV_HEADER = (
    "Pre-Revision Question",
    "Pre-Revision Correct Answer",
    "Pre-Revision Incorrect Answer 1",
    "Pre-Revision Incorrect Answer 2",
    "Pre-Revision Incorrect Answer 3",
    "Pre-Revision Explanation",
    "Self-reported question-writing time (minutes)",
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
    "Explanation",
    "Revision Comments (from Question Writer)",
    "Subdomain",
    "Writer's Difficulty Estimate",
    "Extra Revised Question",
    "Extra Revised Explanation",
    "Extra Revised Correct Answer",
    "Extra Revised Incorrect Answer 1",
    "Extra Revised Incorrect Answer 2",
    "Extra Revised Incorrect Answer 3",
    "Non-Expert Validator Accuracy",
    "Majority Non-Expert Vals Incorrect",
    "Expert Validator Accuracy",
    "Record ID",
    "High-level domain",
    "Question Writer",
    "Feedback_EV_1",
    "Validator Revision Suggestion_EV_1",
    "Is First Validation_EV_1",
    "Post hoc agreement_EV_1",
    "Sufficient Expertise?_EV_1",
    "Understand the question?_EV_1",
    "Question Difficulty_EV_1",
    "Validator Answered Correctly_EV_1",
    "Self-reported time (minutes)_EV_1",
    "Probability Correct_EV_1",
    "Manual Correctness Adjustment_EV_1",
    "Expert Validator_EV_1",
    "Feedback_EV_2",
    "Validator Revision Suggestion_EV_2",
    "Is First Validation_EV_2",
    "Post hoc agreement_EV_2",
    "Sufficient Expertise?_EV_2",
    "Understand the question?_EV_2",
    "Question Difficulty_EV_2",
    "Validator Answered Correctly_EV_2",
    "Self-reported time (minutes)_EV_2",
    "Probability Correct_EV_2",
    "Manual Correctness Adjustment_EV_2",
    "Expert Validator_EV_2",
    "Feedback_NEV_1",
    "Validator Answered Correctly_NEV_1",
    "Explanation_NEV_1",
    "Self-reported time (minutes)_NEV_1",
    "Websites visited_NEV_1",
    "Probability Correct_NEV_1",
    "Manual Correctness Adjustment_NEV_1",
    "Non-Expert Validator_NEV_1",
    "Feedback_NEV_2",
    "Validator Answered Correctly_NEV_2",
    "Explanation_NEV_2",
    "Self-reported time (minutes)_NEV_2",
    "Websites visited_NEV_2",
    "Probability Correct_NEV_2",
    "Manual Correctness Adjustment_NEV_2",
    "Non-Expert Validator_NEV_2",
    "Feedback_NEV_3",
    "Validator Answered Correctly_NEV_3",
    "Explanation_NEV_3",
    "Self-reported time (minutes)_NEV_3",
    "Websites visited_NEV_3",
    "Probability Correct_NEV_3",
    "Manual Correctness Adjustment_NEV_3",
    "Non-Expert Validator_NEV_3",
    "Expert Validator Disagreement Category",
    "Canary String",
)

_HOTPOT_FIELDS = frozenset(
    {"id", "question", "answer", "type", "level", "supporting_facts", "context"}
)
_ATLAS_TRIVIA_FIELDS = frozenset({"question", "answers", "target"})
_ATLAS_NQ_FIELDS = frozenset({"question", "answers"})
_MEDQA_FIELDS = frozenset({"question", "answer", "options", "meta_info", "answer_idx"})
_MUSIQUE_FIELDS = frozenset(
    {
        "id",
        "paragraphs",
        "question",
        "question_decomposition",
        "answer",
        "answer_aliases",
        "answerable",
    }
)
_HUMANEVAL_FIELDS = frozenset({"task_id", "prompt", "canonical_solution", "test", "entry_point"})
_MATH_FIELDS = frozenset({"problem", "level", "type", "solution"})
_AIME_2026_FIELDS = frozenset({"answer", "problem", "problem_idx"})
_MIND2WEB_FIELDS = frozenset(
    {
        "annotation_id",
        "website",
        "domain",
        "subdomain",
        "confirmed_task",
        "action_reprs",
        "actions",
    }
)
_MIND2WEB_SPLITS = frozenset({"train", "test_task", "test_website", "test_domain"})


def _row(
    value: object,
    *,
    fields: frozenset[str],
    source: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{source} row must be a mapping")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{source} row keys must be text")
    data = dict(cast(Mapping[str, object], value))
    if set(data) != fields:
        raise ValueError(f"{source} row has an incompatible field set")
    return data


def _object(
    value: object,
    *,
    fields: frozenset[str],
    field: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{field} must be an object")
    data = cast(dict[str, object], value)
    if any(type(key) is not str for key in data) or set(data) != fields:
        raise ValueError(f"{field} has an incompatible field set")
    return data


def _array(value: object, *, field: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{field} must be an array")
    return cast(list[object], value)


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be text")
    normalized = normalize_json(value)
    if type(normalized) is not str:
        raise TypeError(f"{field} must normalize to text")
    if "\x00" in normalized:
        raise ValueError(f"{field} cannot contain NUL")
    if not allow_empty and not normalized.strip():
        raise ValueError(f"{field} must be non-empty text")
    return normalized


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field} must be a boolean")
    return value


def _text_array(
    value: object,
    *,
    field: str,
    allow_empty_array: bool = False,
    allow_empty_items: bool = False,
    require_unique: bool = True,
) -> tuple[str, ...]:
    items = _array(value, field=field)
    if not allow_empty_array and not items:
        raise ValueError(f"{field} must be non-empty")
    texts = tuple(
        _text(item, field=f"{field} item", allow_empty=allow_empty_items) for item in items
    )
    if require_unique and len(set(texts)) != len(texts):
        raise ValueError(f"{field} must contain unique text")
    return texts


def _dataset_identity(dataset_revision: object, split: object) -> tuple[str, str]:
    return (
        _text(dataset_revision, field="dataset_revision"),
        _text(split, field="split"),
    )


def _derived_task_id(
    *,
    benchmark_id: str,
    dataset_revision: str,
    split: str,
    public_identity: object,
) -> str:
    digest = stable_hash(
        {
            "benchmark_id": benchmark_id,
            "dataset_revision": dataset_revision,
            "public_identity": public_identity,
            "split": split,
        }
    ).removeprefix("sha256:")
    return f"{benchmark_id}/{digest}"


def _static_case(
    public: BenchmarkPublicItem,
    *,
    scoring_rule: StaticScoringRule,
    accepted_answers: tuple[str, ...],
) -> PrivateStaticBenchmarkCase:
    return PrivateStaticBenchmarkCase(
        public=public,
        target=PrivateStaticTarget(
            task_id=public.task_id,
            scoring_rule=scoring_rule,
            accepted_answers=accepted_answers,
        ),
    )


def convert_hotpotqa_row(
    row: Mapping[str, object],
    *,
    dataset_revision: str,
    split: str,
) -> PrivateStaticBenchmarkCase:
    """Convert one exact Hugging Face HotpotQA parquet-like row."""

    revision, split_name = _dataset_identity(dataset_revision, split)
    data = _row(row, fields=_HOTPOT_FIELDS, source="HotpotQA")
    source_id = _text(data["id"], field="HotpotQA id")
    question = _text(data["question"], field="HotpotQA question")
    answer = _text(data["answer"], field="HotpotQA answer")
    question_type = _text(data["type"], field="HotpotQA type")
    level = _text(data["level"], field="HotpotQA level")
    if question_type not in {"bridge", "comparison"}:
        raise ValueError("HotpotQA type must be bridge or comparison")
    if level not in {"easy", "medium", "hard"}:
        raise ValueError("HotpotQA level must be easy, medium, or hard")

    context = _object(
        data["context"],
        fields=frozenset({"title", "sentences"}),
        field="HotpotQA context",
    )
    titles = _text_array(context["title"], field="HotpotQA context.title")
    sentence_groups = _array(context["sentences"], field="HotpotQA context.sentences")
    if len(sentence_groups) != len(titles):
        raise ValueError("HotpotQA context titles and sentence groups must align")
    sentences_by_title: dict[str, tuple[str, ...]] = {}
    for title, sentences in zip(titles, sentence_groups, strict=True):
        sentence_tuple = _text_array(
            sentences,
            field=f"HotpotQA context[{title}] sentences",
            allow_empty_items=True,
            require_unique=False,
        )
        sentences_by_title[title] = sentence_tuple

    supporting = _object(
        data["supporting_facts"],
        fields=frozenset({"title", "sent_id"}),
        field="HotpotQA supporting_facts",
    )
    supporting_titles = _text_array(
        supporting["title"],
        field="HotpotQA supporting_facts.title",
        require_unique=False,
    )
    supporting_ids = _array(supporting["sent_id"], field="HotpotQA supporting_facts.sent_id")
    if len(supporting_titles) != len(supporting_ids):
        raise ValueError("HotpotQA supporting fact titles and sentence IDs must align")
    supporting_pairs: list[tuple[str, int]] = []
    for title, raw_sentence_id in zip(supporting_titles, supporting_ids, strict=True):
        sentence_id = _integer(raw_sentence_id, field="HotpotQA supporting sentence ID")
        if title not in sentences_by_title:
            raise ValueError("HotpotQA supporting fact title does not address its context")
        supporting_pairs.append((title, sentence_id))
    if len(set(supporting_pairs)) != len(supporting_pairs):
        raise ValueError("HotpotQA supporting facts must be unique")

    public = BenchmarkPublicItem(
        benchmark_id="hotpotqa",
        dataset_revision=revision,
        split=split_name,
        task_id=f"hotpotqa/{source_id}",
        task_family=f"hotpotqa/{question_type}",
        query=question,
        public_context={
            "answer_format": "short-text",
            "level": level,
            "question_type": question_type,
            "source_format": "huggingface-hotpotqa-parquet@1",
        },
    )
    return _static_case(
        public,
        scoring_rule=StaticScoringRule.TOKEN_F1,
        accepted_answers=(answer,),
    )


def convert_atlas_triviaqa_row(
    row: Mapping[str, object],
    *,
    dataset_revision: str,
    split: str,
) -> PrivateStaticBenchmarkCase:
    """Convert one exact Atlas ``prepare_qa.py`` TriviaQA JSONL row."""

    revision, split_name = _dataset_identity(dataset_revision, split)
    data = _row(row, fields=_ATLAS_TRIVIA_FIELDS, source="Atlas TriviaQA")
    question = _text(data["question"], field="Atlas TriviaQA question")
    answers = _text_array(data["answers"], field="Atlas TriviaQA answers")
    _text(data["target"], field="Atlas TriviaQA target")
    task_id = _derived_task_id(
        benchmark_id="triviaqa",
        dataset_revision=revision,
        split=split_name,
        public_identity=question,
    )
    public = BenchmarkPublicItem(
        benchmark_id="triviaqa",
        dataset_revision=revision,
        split=split_name,
        task_id=task_id,
        task_family="triviaqa/open-domain",
        query=question,
        public_context={
            "answer_format": "short-text",
            "source_format": "atlas-prepare-qa-trivia@1",
        },
    )
    return _static_case(
        public,
        scoring_rule=StaticScoringRule.TOKEN_F1,
        accepted_answers=answers,
    )


def convert_atlas_nq_row(
    row: Mapping[str, object],
    *,
    dataset_revision: str,
    split: str,
) -> PrivateStaticBenchmarkCase:
    """Convert one exact Atlas ``prepare_qa.py`` NQ-Open JSONL row."""

    revision, split_name = _dataset_identity(dataset_revision, split)
    data = _row(row, fields=_ATLAS_NQ_FIELDS, source="Atlas NQ")
    question = _text(data["question"], field="Atlas NQ question")
    answers = _text_array(data["answers"], field="Atlas NQ answers")
    task_id = _derived_task_id(
        benchmark_id="nq-open",
        dataset_revision=revision,
        split=split_name,
        public_identity=question,
    )
    public = BenchmarkPublicItem(
        benchmark_id="nq-open",
        dataset_revision=revision,
        split=split_name,
        task_id=task_id,
        task_family="nq-open/open-domain",
        query=question,
        public_context={
            "answer_format": "short-text",
            "source_format": "atlas-prepare-qa-nq@1",
        },
    )
    return _static_case(
        public,
        scoring_rule=StaticScoringRule.EXACT_TEXT,
        accepted_answers=answers,
    )


def convert_medqa_us_4option_row(
    row: Mapping[str, object],
    *,
    dataset_revision: str,
    split: str,
) -> PrivateStaticBenchmarkCase:
    """Convert one exact official MedQA US four-option JSONL row."""

    revision, split_name = _dataset_identity(dataset_revision, split)
    data = _row(row, fields=_MEDQA_FIELDS, source="MedQA US four-option")
    question = _text(data["question"], field="MedQA question")
    answer = _text(data["answer"], field="MedQA answer")
    answer_idx = _text(data["answer_idx"], field="MedQA answer_idx")
    meta_info = _text(data["meta_info"], field="MedQA meta_info")
    options_data = _object(
        data["options"],
        fields=frozenset({"A", "B", "C", "D"}),
        field="MedQA options",
    )
    options = {
        label: _text(options_data[label], field=f"MedQA option {label}")
        for label in ("A", "B", "C", "D")
    }
    if answer_idx not in options:
        raise ValueError("MedQA answer_idx must be A, B, C, or D")
    if answer != options[answer_idx]:
        raise ValueError("MedQA answer text must equal the indexed option")
    task_id = _derived_task_id(
        benchmark_id="medqa",
        dataset_revision=revision,
        split=split_name,
        public_identity={"options": options, "question": question},
    )
    public = BenchmarkPublicItem(
        benchmark_id="medqa",
        dataset_revision=revision,
        split=split_name,
        task_id=task_id,
        task_family="medqa/us-four-option",
        query=question,
        public_context={
            "answer_format": "option-letter",
            "meta_info": meta_info,
            "options": normalize_json(options),
            "source_format": "medqa-us-four-options-jsonl@1",
        },
    )
    return _static_case(
        public,
        scoring_rule=StaticScoringRule.OPTION,
        accepted_answers=(answer_idx,),
    )


def convert_musique_row(
    row: Mapping[str, object],
    *,
    dataset_revision: str,
    split: str,
) -> PrivateStaticBenchmarkCase:
    """Convert one exact official MuSiQue-Answerable JSONL row."""

    revision, split_name = _dataset_identity(dataset_revision, split)
    data = _row(row, fields=_MUSIQUE_FIELDS, source="MuSiQue")
    source_id = _text(data["id"], field="MuSiQue id")
    question = _text(data["question"], field="MuSiQue question")
    answer = _text(data["answer"], field="MuSiQue answer")
    aliases = _text_array(
        data["answer_aliases"],
        field="MuSiQue answer_aliases",
        allow_empty_array=True,
    )
    if answer in aliases:
        raise ValueError("MuSiQue answer aliases must exclude the canonical answer")
    if not _boolean(data["answerable"], field="MuSiQue answerable"):
        raise ValueError("this converter accepts only MuSiQue-Answerable rows")

    paragraphs_raw = _array(data["paragraphs"], field="MuSiQue paragraphs")
    if len(paragraphs_raw) < 2:
        raise ValueError("MuSiQue requires at least two paragraphs")
    public_paragraphs: list[dict[str, JsonValue]] = []
    supporting_indices: set[int] = set()
    paragraph_indices: list[int] = []
    for position, raw_paragraph in enumerate(paragraphs_raw):
        paragraph = _object(
            raw_paragraph,
            fields=frozenset({"idx", "title", "paragraph_text", "is_supporting"}),
            field=f"MuSiQue paragraph {position}",
        )
        index = _integer(paragraph["idx"], field="MuSiQue paragraph idx")
        paragraph_indices.append(index)
        if _boolean(paragraph["is_supporting"], field="MuSiQue paragraph is_supporting"):
            supporting_indices.add(index)
        public_paragraphs.append(
            {
                "idx": index,
                "paragraph_text": _text(
                    paragraph["paragraph_text"], field="MuSiQue paragraph_text"
                ),
                "title": _text(paragraph["title"], field="MuSiQue paragraph title"),
            }
        )
    if paragraph_indices != list(range(len(paragraphs_raw))):
        raise ValueError("MuSiQue paragraph indices must be ordered and contiguous")

    decomposition_raw = _array(
        data["question_decomposition"], field="MuSiQue question_decomposition"
    )
    if not 2 <= len(decomposition_raw) <= 4:
        raise ValueError("MuSiQue question decomposition must have two to four steps")
    decomposition_ids: list[int] = []
    for position, raw_step in enumerate(decomposition_raw):
        step = _object(
            raw_step,
            fields=frozenset({"id", "question", "answer", "paragraph_support_idx"}),
            field=f"MuSiQue decomposition step {position}",
        )
        decomposition_ids.append(_integer(step["id"], field="MuSiQue decomposition step id"))
        _text(step["question"], field="MuSiQue decomposition question")
        _text(step["answer"], field="MuSiQue decomposition answer")
        support_idx = _integer(step["paragraph_support_idx"], field="MuSiQue paragraph_support_idx")
        if support_idx not in supporting_indices:
            raise ValueError("MuSiQue decomposition must point to a supporting paragraph")
    if len(set(decomposition_ids)) != len(decomposition_ids):
        raise ValueError("MuSiQue decomposition step identities must be unique")

    public = BenchmarkPublicItem(
        benchmark_id="musique",
        dataset_revision=revision,
        split=split_name,
        task_id=f"musique/{source_id}",
        task_family=f"musique/{len(decomposition_raw)}-hop",
        query=question,
        public_context={
            "answer_format": "short-text",
            "paragraphs": normalize_json(public_paragraphs),
            "source_format": "musique-official-jsonl@1",
        },
    )
    return _static_case(
        public,
        scoring_rule=StaticScoringRule.TOKEN_F1,
        accepted_answers=(answer, *aliases),
    )


def convert_gpqa_diamond_row(
    row: Mapping[str, object],
    *,
    dataset_revision: str,
    split: str,
) -> PrivateGPQADiamondCase:
    """Convert one exact row from the official GPQA Diamond CSV."""

    revision, split_name = _dataset_identity(dataset_revision, split)
    data = _row(
        row,
        fields=frozenset(GPQA_DIAMOND_CSV_HEADER),
        source="GPQA Diamond",
    )
    # ``csv.DictReader`` is the required ingestion boundary.  Validate every
    # CSV cell, including unused private provenance, instead of accepting a
    # pandas NaN or silently changing source schema.
    normalized = {
        field: _text(data[field], field=f"GPQA {field}", allow_empty=True)
        for field in GPQA_DIAMOND_CSV_HEADER
    }
    for required in (
        "Question",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
        "Explanation",
        "Record ID",
        "High-level domain",
        "Subdomain",
        "Canary String",
    ):
        if not normalized[required].strip():
            raise ValueError(f"GPQA {required} must be non-empty")

    question = normalized["Question"]
    correct_answer = normalized["Correct Answer"]
    choices = (
        correct_answer,
        normalized["Incorrect Answer 1"],
        normalized["Incorrect Answer 2"],
        normalized["Incorrect Answer 3"],
    )
    # The pinned official Diamond CSV contains a small number of rows where
    # two distractor cells are byte-identical.  They remain two official
    # option slots; dropping either row or synthesizing a replacement would
    # change the frozen population.  Only an answer duplicated by a
    # distractor is ambiguous for the private option-label target.
    if correct_answer in choices[1:]:
        raise ValueError("GPQA correct answer must differ from every distractor")
    ordered_choices = tuple(
        sorted(
            choices,
            key=lambda choice: (
                stable_hash(
                    {
                        "algorithm": "gpqa-result-blind-choice-order@1",
                        "choice": choice,
                        "question": question,
                    }
                ),
                choice,
            ),
        )
    )
    options = dict(zip(("A", "B", "C", "D"), ordered_choices, strict=True))
    correct_option = next(label for label, choice in options.items() if choice == correct_answer)
    task_id = f"gpqa-diamond/{normalized['Record ID']}"
    public = BenchmarkPublicItem(
        benchmark_id="gpqa-diamond",
        dataset_revision=revision,
        split=split_name,
        task_id=task_id,
        task_family=(f"gpqa-diamond/{normalized['High-level domain']}/{normalized['Subdomain']}"),
        query=question,
        public_context={
            "answer_format": "option-letter",
            "options": normalize_json(options),
            "source_format": "idavidrein-gpqa-diamond-csv@1",
        },
    )
    return PrivateGPQADiamondCase(
        public=public,
        target=PrivateGPQADiamondTarget(
            task_id=task_id,
            correct_option=correct_option,
            explanation=normalized["Explanation"],
            canary_string=normalized["Canary String"],
        ),
    )


def convert_humaneval_row(
    row: Mapping[str, object],
    *,
    dataset_revision: str,
    split: str,
) -> PrivateHumanEvalCase:
    """Split one official HumanEval JSONL row without executing its tests."""

    revision, split_name = _dataset_identity(dataset_revision, split)
    data = _row(row, fields=_HUMANEVAL_FIELDS, source="HumanEval")
    task_id = _text(data["task_id"], field="HumanEval task_id")
    prompt = _text(data["prompt"], field="HumanEval prompt")
    canonical_solution = _text(data["canonical_solution"], field="HumanEval canonical_solution")
    test = _text(data["test"], field="HumanEval test")
    entry_point = _text(data["entry_point"], field="HumanEval entry_point")
    public = BenchmarkPublicItem(
        benchmark_id="humaneval",
        dataset_revision=revision,
        split=split_name,
        task_id=task_id,
        task_family="humaneval/python",
        query=prompt,
        public_context={
            "answer_format": "python-completion",
            "entry_point": entry_point,
            "source_format": "openai-humaneval-jsonl@1",
        },
    )
    return PrivateHumanEvalCase(
        public=public,
        target=PrivateHumanEvalTarget(
            task_id=task_id,
            canonical_solution=canonical_solution,
            test=test,
            entry_point=entry_point,
        ),
    )


def _last_boxed_answer(solution: str) -> str:
    marker = r"\boxed"
    positions: list[int] = []
    cursor = 0
    while True:
        position = solution.find(marker, cursor)
        if position < 0:
            break
        positions.append(position)
        cursor = position + len(marker)
    if not positions:
        raise ValueError("MATH solution must contain a boxed final answer")

    extracted: list[str] = []
    for position in positions:
        opening = position + len(marker)
        while opening < len(solution) and solution[opening].isspace():
            opening += 1
        if opening >= len(solution) or solution[opening] != "{":
            raise ValueError("MATH boxed answer must use a braced expression")
        depth = 1
        end = opening + 1
        while end < len(solution) and depth:
            if solution[end] == "{":
                depth += 1
            elif solution[end] == "}":
                depth -= 1
            end += 1
        if depth:
            raise ValueError("MATH boxed answer has unbalanced braces")
        answer = solution[opening + 1 : end - 1].strip()
        if not answer:
            raise ValueError("MATH boxed answer cannot be empty")
        extracted.append(answer)
    return extracted[-1]


def convert_math_hard_row(
    row: Mapping[str, object],
    *,
    dataset_revision: str,
    split: str,
) -> PrivateMathCase:
    """Convert one exact MATH parquet-like Level-5 row without scoring it."""

    revision, split_name = _dataset_identity(dataset_revision, split)
    data = _row(row, fields=_MATH_FIELDS, source="MATH")
    problem = _text(data["problem"], field="MATH problem")
    level = _text(data["level"], field="MATH level")
    subject = _text(data["type"], field="MATH type")
    solution = _text(data["solution"], field="MATH solution")
    if level != "Level 5":
        raise ValueError("MATH-Hard converter accepts only Level 5 rows")
    boxed_answer = _last_boxed_answer(solution)
    task_id = _derived_task_id(
        benchmark_id="math-hard",
        dataset_revision=revision,
        split=split_name,
        public_identity={"problem": problem, "subject": subject},
    )
    public = BenchmarkPublicItem(
        benchmark_id="math-hard",
        dataset_revision=revision,
        split=split_name,
        task_id=task_id,
        task_family=f"math-hard/{subject}",
        query=problem,
        public_context={
            "answer_format": "latex-expression",
            "level": level,
            "source_format": "math-parquet@1",
            "subject": subject,
        },
    )
    return PrivateMathCase(
        public=public,
        target=PrivateMathTarget(
            task_id=task_id,
            solution=solution,
            boxed_answer=boxed_answer,
        ),
    )


def convert_matharena_aime_2026_row(
    row: Mapping[str, object],
    *,
    dataset_revision: str,
    split: str,
) -> PrivateStaticBenchmarkCase:
    """Convert one exact ``MathArena/aime_2026`` parquet-like row."""

    revision, split_name = _dataset_identity(dataset_revision, split)
    data = _row(row, fields=_AIME_2026_FIELDS, source="MathArena AIME 2026")
    problem_index = _integer(
        data["problem_idx"],
        field="MathArena AIME 2026 problem_idx",
        minimum=1,
    )
    if problem_index > 30:
        raise ValueError("MathArena AIME 2026 problem_idx must lie in [1, 30]")
    answer = _integer(data["answer"], field="MathArena AIME 2026 answer")
    if answer > 999:
        raise ValueError("MathArena AIME 2026 answer must lie in [0, 999]")
    problem = _text(data["problem"], field="MathArena AIME 2026 problem")
    task_id = f"aime-2026/{problem_index:02d}"
    public = BenchmarkPublicItem(
        benchmark_id="aime-2026",
        dataset_revision=revision,
        split=split_name,
        task_id=task_id,
        task_family="aime-2026/integer-answer",
        query=problem,
        public_context={
            "answer_format": "integer-000-to-999",
            "problem_index": problem_index,
            "source_format": "matharena-aime-2026-parquet@1",
        },
    )
    return _static_case(
        public,
        scoring_rule=StaticScoringRule.INTEGER,
        accepted_answers=(str(answer),),
    )


def _mind2web_candidate(
    value: object,
    *,
    field: str,
    positive: bool,
) -> dict[str, JsonValue]:
    fields = {
        "tag",
        "backend_node_id",
        "attributes",
    }
    if positive:
        fields.update({"is_original_target", "is_top_level_target"})
    candidate = _object(
        value,
        fields=frozenset(fields),
        field=field,
    )
    public: dict[str, JsonValue] = {
        "attributes": _text(candidate["attributes"], field=f"{field}.attributes", allow_empty=True),
        "backend_node_id": _text(candidate["backend_node_id"], field=f"{field}.backend_node_id"),
        "tag": _text(candidate["tag"], field=f"{field}.tag"),
    }
    if positive:
        _boolean(candidate["is_original_target"], field=f"{field}.is_original_target")
        _boolean(candidate["is_top_level_target"], field=f"{field}.is_top_level_target")
    return public


def convert_mind2web_row(
    row: Mapping[str, object],
    *,
    dataset_revision: str,
    split: str,
    candidate_rankings: Mapping[str, tuple[str, ...]],
) -> tuple[PrivateMind2WebStepCase, ...]:
    """Convert one official Mind2Web split row into candidate-level step cases."""

    revision, split_name = _dataset_identity(dataset_revision, split)
    if split_name not in _MIND2WEB_SPLITS:
        raise ValueError("Mind2Web split must be train, test_task, test_website, or test_domain")
    data = _row(row, fields=_MIND2WEB_FIELDS, source="Mind2Web")
    annotation_id = _text(data["annotation_id"], field="Mind2Web annotation_id")
    website = _text(data["website"], field="Mind2Web website")
    domain = _text(data["domain"], field="Mind2Web domain")
    subdomain = _text(data["subdomain"], field="Mind2Web subdomain")
    task = _text(data["confirmed_task"], field="Mind2Web confirmed_task")
    action_reprs = _text_array(
        data["action_reprs"],
        field="Mind2Web action_reprs",
        require_unique=False,
    )
    actions = _array(data["actions"], field="Mind2Web actions")
    if not actions or len(actions) != len(action_reprs):
        raise ValueError("Mind2Web actions and action_reprs must align and be non-empty")

    cases: list[PrivateMind2WebStepCase] = []
    action_uids: list[str] = []
    for step_index, (raw_action, action_repr) in enumerate(
        zip(actions, action_reprs, strict=True), start=1
    ):
        action = _object(
            raw_action,
            fields=frozenset(
                {
                    "action_uid",
                    "raw_html",
                    "cleaned_html",
                    "operation",
                    "pos_candidates",
                    "neg_candidates",
                }
            ),
            field=f"Mind2Web action {step_index}",
        )
        action_uid = _text(action["action_uid"], field="Mind2Web action_uid")
        action_uids.append(action_uid)
        _text(action["raw_html"], field="Mind2Web raw_html")
        _text(action["cleaned_html"], field="Mind2Web cleaned_html")
        operation = _object(
            action["operation"],
            fields=frozenset({"op", "original_op", "value"}),
            field="Mind2Web operation",
        )
        op = _text(operation["op"], field="Mind2Web operation.op")
        if op not in {"CLICK", "TYPE", "SELECT"}:
            raise ValueError("Mind2Web operation.op must be CLICK, TYPE, or SELECT")
        original_op = _text(operation["original_op"], field="Mind2Web operation.original_op")
        operation_value = _text(
            operation["value"], field="Mind2Web operation.value", allow_empty=True
        )

        public_candidates: list[dict[str, JsonValue]] = []
        public_candidates_by_id: dict[str, dict[str, JsonValue]] = {}
        positive_node_ids: list[str] = []
        positive_raw = _array(action["pos_candidates"], field="Mind2Web pos_candidates")
        negative_raw = _array(action["neg_candidates"], field="Mind2Web neg_candidates")
        score_action_id = f"{annotation_id}_{action_uid}"
        ranked_node_ids = candidate_rankings.get(score_action_id)
        if ranked_node_ids is None:
            raise ValueError("Mind2Web candidate ranking is missing an action")
        if len(set(ranked_node_ids)) != len(ranked_node_ids):
            raise ValueError("Mind2Web candidate ranking repeats a backend node identity")
        ranked_node_id_set = frozenset(ranked_node_ids)
        all_node_ids: list[str] = []
        for position, raw_candidate in enumerate(positive_raw):
            candidate = _mind2web_candidate(
                raw_candidate,
                field=f"Mind2Web positive candidate {position}",
                positive=True,
            )
            node_id = cast(str, candidate["backend_node_id"])
            all_node_ids.append(node_id)
            if node_id in ranked_node_id_set:
                public_candidates_by_id[node_id] = candidate
                positive_node_ids.append(node_id)
        for position, raw_candidate in enumerate(negative_raw):
            candidate = _mind2web_candidate(
                raw_candidate,
                field=f"Mind2Web negative candidate {position}",
                positive=False,
            )
            node_id = cast(str, candidate["backend_node_id"])
            all_node_ids.append(node_id)
            if node_id in ranked_node_id_set:
                public_candidates_by_id[node_id] = candidate
        if len(set(all_node_ids)) != len(all_node_ids):
            raise ValueError("Mind2Web candidate backend node identities must be unique")
        missing_ranked_ids = set(ranked_node_ids).difference(public_candidates_by_id)
        if missing_ranked_ids:
            raise ValueError("Mind2Web ranking names a candidate absent from the action")
        public_candidates = [public_candidates_by_id[node_id] for node_id in ranked_node_ids]

        task_id = f"mind2web/{annotation_id}/{action_uid}"
        ranking_context: dict[str, JsonValue] = {
            "candidate_count": len(ranked_node_ids),
            "candidate_ranking": "mind2web-official-deberta-v3-base-scores@1",
            "candidate_top_k": MIND2WEB_CANDIDATE_TOP_K,
        }
        public = BenchmarkPublicItem(
            benchmark_id="mind2web",
            dataset_revision=revision,
            split=split_name,
            task_id=task_id,
            task_family=f"mind2web/{split_name}/{domain}",
            query=task,
            public_context={
                "action_count": len(actions),
                "answer_format": "browser-operation-and-element",
                "candidates": normalize_json(public_candidates),
                "domain": domain,
                "source_format": "mind2web-official-split@1",
                "step_index": step_index,
                "subdomain": subdomain,
                "website": website,
                **ranking_context,
            },
        )
        cases.append(
            PrivateMind2WebStepCase(
                public=public,
                target=PrivateMind2WebStepTarget(
                    task_id=task_id,
                    annotation_id=annotation_id,
                    action_uid=action_uid,
                    operation=op,
                    original_operation=original_op,
                    value=operation_value,
                    positive_backend_node_ids=tuple(sorted(positive_node_ids)),
                    action_repr=action_repr,
                ),
            )
        )
    if len(set(action_uids)) != len(action_uids):
        raise ValueError("Mind2Web action identities must be unique within a task")
    return tuple(cases)


__all__ = [
    "GPQA_DIAMOND_CSV_HEADER",
    "convert_atlas_nq_row",
    "convert_atlas_triviaqa_row",
    "convert_gpqa_diamond_row",
    "convert_hotpotqa_row",
    "convert_humaneval_row",
    "convert_math_hard_row",
    "convert_matharena_aime_2026_row",
    "convert_medqa_us_4option_row",
    "convert_mind2web_row",
    "convert_musique_row",
]
