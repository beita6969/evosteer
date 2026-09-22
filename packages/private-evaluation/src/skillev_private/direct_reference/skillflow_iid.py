"""Source-specific wire-format parsing for the released SkillFlow IID panel."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from skillev_private.benchmarks.qa_metrics import normalize_triviaqa_answer


@dataclass(frozen=True, slots=True)
class SkillFlowIIDRecord:
    source: str
    source_position: int
    rendered_question: str
    rendering_contract: str
    answer: str
    extra: dict[str, object]
    raw: dict[str, object]

    @property
    def question(self) -> str:
        return self.rendered_question


class HotpotRenderMode(StrEnum):
    RELEASED_VISIBLE_QUESTION = "released-visible-question"
    RAW_TEN_PASSAGE_CONTEXT = "raw-ten-passage-context"


@dataclass(frozen=True, slots=True)
class TriviaAliasSet:
    aliases: tuple[str, ...]
    source: str
    potentially_truncated: bool


def parse_skillflow_iid_record(raw: object, *, source_position: int) -> SkillFlowIIDRecord:
    if type(raw) is not dict:
        raise ValueError("SkillFlow IID record must be an object")
    value = cast(dict[str, object], raw)
    extra_value = value.get("extra")
    if type(extra_value) is not dict:
        raise ValueError("SkillFlow IID record lacks metadata")
    extra = cast(dict[str, object], extra_value)
    source = extra.get("source")
    question = value.get("question")
    answer = value.get("answer")
    if type(source) is not str or type(question) is not str or type(answer) is not str:
        raise ValueError("SkillFlow IID source, question, and answer must be text")
    if not source.strip() or not question.strip() or not answer.strip() or source_position < 0:
        raise ValueError("SkillFlow IID record fields must be non-empty")
    rendering_contract = {
        "HotpotQA": "skillflow-hotpotqa-rendered-distractor-validation@1",
        "TriviaQA": "skillflow-triviaqa-rc-validation-rendered@1",
        "AIME 2026": "skillflow-aime-2026-rendered@1",
        "MedQA": "skillflow-medqa-mixed-source-rendered@1",
        "WebShop": "skillflow-webshop-rendered@1",
        "ALFWorld": "skillflow-alfworld-rendered@1",
        "SWE-bench": "skillflow-swe-bench-rendered@1",
    }.get(source)
    if rendering_contract is None:
        raise ValueError("SkillFlow IID source has no rendering contract")
    return SkillFlowIIDRecord(
        source,
        source_position,
        question,
        rendering_contract,
        answer,
        extra,
        value,
    )


def parse_skillflow_trivia_wire_answers(value: str) -> TriviaAliasSet:
    aliases = tuple(dict.fromkeys(part.strip() for part in value.split("|") if part.strip()))
    if not aliases:
        raise ValueError("SkillFlow TriviaQA answer wire contains no aliases")
    if any("\x00" in alias for alias in aliases):
        raise ValueError("SkillFlow TriviaQA alias contains NUL")
    return TriviaAliasSet(
        aliases=aliases,
        source="skillflow-prepared-wire-first-five",
        potentially_truncated=len(aliases) >= 5,
    )


def parse_skillflow_trivia_answers(value: str) -> tuple[str, ...]:
    """Compatibility projection; new code should retain the typed provenance."""

    return parse_skillflow_trivia_wire_answers(value).aliases


def trivia_aliases_from_record(record: SkillFlowIIDRecord) -> TriviaAliasSet:
    """Use structured aliases only after proving agreement with the released wire."""

    if record.source != "TriviaQA":
        raise ValueError("TriviaQA alias provenance requires a TriviaQA record")
    wire = parse_skillflow_trivia_wire_answers(record.answer)
    raw_aliases = record.extra.get("aliases")
    if raw_aliases is None:
        return wire
    if not isinstance(raw_aliases, list):
        raise ValueError("TriviaQA aliases metadata must be a list")
    structured = tuple(
        dict.fromkeys(
            item.strip() for item in raw_aliases if isinstance(item, str) and item.strip()
        )
    )
    if not structured:
        raise ValueError("TriviaQA structured aliases are empty")
    normalized_wire = {normalize_triviaqa_answer(item) for item in wire.aliases}
    normalized_structured = {normalize_triviaqa_answer(item) for item in structured}
    if normalized_structured != normalized_wire:
        raise ValueError("TriviaQA wire and structured aliases disagree")
    return TriviaAliasSet(
        aliases=structured,
        source="validated-structured-and-wire",
        potentially_truncated=False,
    )


def render_skillflow_trivia_question_only(record: SkillFlowIIDRecord) -> str:
    """Discard the released RC passage preview and retain the final TriviaQA question."""

    if record.source != "TriviaQA":
        raise ValueError("question-only rendering requires a TriviaQA record")
    _prefix, marker, question = record.rendered_question.rpartition("Question:")
    if marker:
        if not question.strip():
            raise ValueError("TriviaQA rendered question has an empty final Question marker")
        return question.strip()
    return record.rendered_question.strip()


def render_skillflow_hotpotqa_full_context(record: SkillFlowIIDRecord) -> str:
    """Render the released Hotpot question with every model-visible distractor passage."""

    if record.source != "HotpotQA":
        raise ValueError("full-context rendering requires a HotpotQA record")
    raw_context = record.raw.get("context")
    if (
        type(raw_context) is not list
        or len(raw_context) != 10
        or any(type(passage) is not str or not passage.strip() for passage in raw_context)
    ):
        raise ValueError("HotpotQA requires ten non-empty public context passages")
    prefix, marker, question = record.rendered_question.rpartition("Question:")
    if not marker or not prefix.strip() or not question.strip():
        raise ValueError("HotpotQA rendered question lacks its final Question marker")
    passages = "\n\n".join(
        f"Passage {index}:\n{passage.strip()}"
        for index, passage in enumerate(cast(list[str], raw_context), start=1)
    )
    return f"Context passages:\n\n{passages}\n\nQuestion:\n{question.strip()}"


def render_hotpot_input(record: SkillFlowIIDRecord, *, mode: HotpotRenderMode) -> str:
    if mode is HotpotRenderMode.RELEASED_VISIBLE_QUESTION:
        if not record.question.strip():
            raise ValueError("released Hotpot question is empty")
        return record.question
    if mode is HotpotRenderMode.RAW_TEN_PASSAGE_CONTEXT:
        return render_skillflow_hotpotqa_full_context(record)
    raise AssertionError(mode)
