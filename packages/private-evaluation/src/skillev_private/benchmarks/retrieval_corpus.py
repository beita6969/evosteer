"""Strict, answer-free corpus preparation for retrieval benchmarks.

Parquet iteration remains an injected data-loading concern.  The row adapters
below validate one official source row at a time, then copy only model-visible
passage fields into the public retrieval contract.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Mapping
from typing import cast

from skillev.benchmarks import DocumentPassage
from skillev.contracts import normalize_json, stable_hash

ATLAS_WIKIPEDIA_TSV_FIELDS = ("id", "text", "title")
ATLAS_DPR_PASSAGE_ID_PREFIX = "atlas-dpr-wikipedia:"
_ATLAS_SOURCE_ID_WIDTH = 12


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be text")
    normalized = normalize_json(value)
    if type(normalized) is not str or normalized != value:
        raise ValueError(f"{field} must be canonical text")
    if "\x00" in value or (not allow_empty and not value.strip()):
        raise ValueError(f"{field} must be text without NUL")
    return value


def _official_source_text(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> str:
    """Project official display text into the canonical NFC contract domain.

    The pinned Atlas and HotpotQA sources contain canonically equivalent
    decomposed Unicode in otherwise valid public text.  That concrete source
    condition is an ingestion concern: normalize only Unicode composition,
    then enforce the same NUL/emptiness boundary as every public passage.
    """

    if type(value) is not str:
        raise TypeError(f"{field} must be text")
    normalized = normalize_json(value)
    if type(normalized) is not str:
        raise TypeError(f"{field} normalization did not produce text")
    if "\x00" in normalized or (not allow_empty and not normalized.strip()):
        raise ValueError(f"{field} must be text without NUL")
    return normalized


def _object(value: object, *, fields: frozenset[str], field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{field} must be an object")
    data = cast(dict[object, object], value)
    if any(type(key) is not str for key in data) or set(data) != fields:
        raise ValueError(f"{field} has an incompatible field set")
    return cast(dict[str, object], data)


def _array(value: object, *, field: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{field} must be an array")
    return cast(list[object], value)


def _passage_identity(
    *,
    source: str,
    dataset_revision: str,
    source_id: str,
    position: int,
    title: str,
    text: str,
) -> tuple[str, str]:
    document_id = stable_hash(
        {
            "dataset_revision": dataset_revision,
            "source": source,
            "source_id": source_id,
            "title": title,
        }
    )
    passage_id = stable_hash(
        {
            "document_id": document_id,
            "position": position,
            "source": source,
            "text": text,
        }
    )
    return passage_id, document_id


def _passages(values: list[DocumentPassage], *, source: str) -> tuple[DocumentPassage, ...]:
    if not values:
        raise ValueError(f"{source} row produced no public passages")
    passage_ids = tuple(passage.passage_id for passage in values)
    if len(set(passage_ids)) != len(passage_ids):
        raise ValueError(f"{source} row produced duplicate passage identities")
    return tuple(values)


def hotpotqa_passages_from_row(
    row: Mapping[str, object],
    *,
    dataset_revision: str,
    split: str,
) -> tuple[DocumentPassage, ...]:
    """Project one strict HotpotQA row without answer or support labels."""

    # The Atlas/DPR support-index builder imports this module on a
    # dependency-light CPU runtime.  Dataset-case converters eventually reach
    # the training session types (and therefore torch), whereas the streaming
    # Wikipedia projection below does not need them.  Import the row converter
    # only when this benchmark-specific projection is actually requested.
    from .converters import convert_hotpotqa_row

    case = convert_hotpotqa_row(
        row,
        dataset_revision=dataset_revision,
        split=split,
    )
    source_id = case.public.task_id.removeprefix("hotpotqa/")
    data = row
    context = _object(
        data["context"],
        fields=frozenset({"title", "sentences"}),
        field="HotpotQA context",
    )
    titles = _array(context["title"], field="HotpotQA context.title")
    sentence_groups = _array(context["sentences"], field="HotpotQA context.sentences")
    if len(titles) != len(sentence_groups):
        raise ValueError("HotpotQA context titles and sentence groups must align")
    passages: list[DocumentPassage] = []
    seen_titles: set[str] = set()
    for position, (raw_title, raw_sentences) in enumerate(
        zip(titles, sentence_groups, strict=True)
    ):
        title = _official_source_text(
            raw_title,
            field=f"HotpotQA context title {position}",
        )
        if title in seen_titles:
            raise ValueError("HotpotQA context titles must be unique")
        seen_titles.add(title)
        sentences = _array(raw_sentences, field=f"HotpotQA context[{title}] sentences")
        text = "".join(
            _official_source_text(
                sentence,
                field=f"HotpotQA context[{title}] sentence",
                allow_empty=True,
            )
            for sentence in sentences
        )
        text = _text(text, field=f"HotpotQA context[{title}] text")
        passage_id, document_id = _passage_identity(
            source="hotpotqa",
            dataset_revision=case.public.dataset_revision,
            source_id=source_id,
            position=position,
            title=title,
            text=text,
        )
        passages.append(DocumentPassage(passage_id, document_id, title, text))
    return _passages(passages, source="HotpotQA")


def musique_passages_from_row(
    row: Mapping[str, object],
    *,
    dataset_revision: str,
    split: str,
) -> tuple[DocumentPassage, ...]:
    """Project one strict MuSiQue row without answer/decomposition labels."""

    from .converters import convert_musique_row

    case = convert_musique_row(
        row,
        dataset_revision=dataset_revision,
        split=split,
    )
    source_id = case.public.task_id.removeprefix("musique/")
    data = row
    raw_paragraphs = _array(data["paragraphs"], field="MuSiQue paragraphs")
    passages: list[DocumentPassage] = []
    for position, raw_paragraph in enumerate(raw_paragraphs):
        paragraph = _object(
            raw_paragraph,
            fields=frozenset({"idx", "title", "paragraph_text", "is_supporting"}),
            field=f"MuSiQue paragraph {position}",
        )
        index = paragraph["idx"]
        if type(index) is not int or index != position:
            raise ValueError("MuSiQue paragraph indices must be ordered and contiguous")
        title = _text(paragraph["title"], field=f"MuSiQue paragraph {position} title")
        text = _text(
            paragraph["paragraph_text"],
            field=f"MuSiQue paragraph {position} text",
        )
        passage_id, document_id = _passage_identity(
            source="musique",
            dataset_revision=case.public.dataset_revision,
            source_id=source_id,
            position=position,
            title=title,
            text=text,
        )
        passages.append(DocumentPassage(passage_id, document_id, title, text))
    return _passages(passages, source="MuSiQue")


def iter_atlas_wikipedia_tsv(lines: Iterable[str]) -> Iterator[DocumentPassage]:
    """Stream a pinned ``id, text, title`` Atlas/DPR Wikipedia TSV."""

    reader = csv.DictReader(lines, delimiter="\t")
    if tuple(reader.fieldnames or ()) != ATLAS_WIKIPEDIA_TSV_FIELDS:
        raise ValueError("Atlas Wikipedia TSV header must be exactly id, text, title")
    previous_source_rowid: int | None = None
    previous_document: tuple[str, str] | None = None
    for row_number, row in enumerate(reader, start=2):
        if set(row) != set(ATLAS_WIKIPEDIA_TSV_FIELDS) or any(
            value is None for value in row.values()
        ):
            raise ValueError(f"Atlas Wikipedia TSV row {row_number} has incompatible fields")
        source_id = _text(row["id"], field=f"Atlas Wikipedia row {row_number} id")
        if not source_id.isascii() or not source_id.isdecimal():
            raise ValueError("Atlas Wikipedia TSV passage ids must be positive decimal integers")
        source_rowid = int(source_id)
        if source_rowid < 1:
            raise ValueError("Atlas Wikipedia TSV passage ids must be positive decimal integers")
        if previous_source_rowid is not None and source_rowid <= previous_source_rowid:
            raise ValueError("Atlas Wikipedia TSV passage ids must be strictly ordered")
        previous_source_rowid = source_rowid
        text = _official_source_text(row["text"], field=f"Atlas Wikipedia row {row_number} text")
        title = _official_source_text(row["title"], field=f"Atlas Wikipedia row {row_number} title")
        passage_id = f"{ATLAS_DPR_PASSAGE_ID_PREFIX}{source_rowid:0{_ATLAS_SOURCE_ID_WIDTH}d}"
        if previous_document is not None and title == previous_document[0]:
            document_id = previous_document[1]
        else:
            document_id = stable_hash(
                {
                    "format": "atlas-dpr-wikipedia-title@1",
                    "title": title,
                }
            )
            previous_document = (title, document_id)
        yield DocumentPassage(
            passage_id=passage_id,
            document_id=document_id,
            title=title,
            text=text,
            source_rowid=source_rowid,
        )


__all__ = [
    "ATLAS_DPR_PASSAGE_ID_PREFIX",
    "ATLAS_WIKIPEDIA_TSV_FIELDS",
    "hotpotqa_passages_from_row",
    "iter_atlas_wikipedia_tsv",
    "musique_passages_from_row",
]
