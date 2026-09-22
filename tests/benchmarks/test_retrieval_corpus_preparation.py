from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
import skillev_private.benchmarks.retrieval_corpus as retrieval_corpus
from skillev_private.benchmarks import (
    hotpotqa_passages_from_row,
    iter_atlas_wikipedia_tsv,
    musique_passages_from_row,
)
from skillev_private.benchmarks.retrieval_corpus import ATLAS_DPR_PASSAGE_ID_PREFIX

from skillev.benchmarks import (
    DocumentPassage,
    RetrievalIndex,
    build_retrieval_index,
)
from skillev.contracts import JsonValue, canonical_json

PRIVATE_RETRIEVAL_CANARY = "PRIVATE-RETRIEVAL-GOLD-CANARY"
REVISION = "retrieval-fixture@1"


def _hotpot_row() -> dict[str, object]:
    return {
        "id": "hotpot-source-1",
        "question": "Which public documents connect?",
        "answer": PRIVATE_RETRIEVAL_CANARY,
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {"title": ["Alpha", "Beta"], "sent_id": [0, 0]},
        "context": {
            "title": ["Alpha", "Beta"],
            "sentences": [
                ["Alpha contains the public zephyr fact."],
                ["Beta contains another public fact."],
            ],
        },
    }


def _musique_row() -> dict[str, object]:
    return {
        "id": "2hop__1_2",
        "paragraphs": [
            {
                "idx": 0,
                "title": "Gamma",
                "paragraph_text": "Gamma contains the public nebula fact.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Delta",
                "paragraph_text": "Delta is a public distractor passage.",
                "is_supporting": False,
            },
        ],
        "question": "What is the final public answer?",
        "question_decomposition": [
            {
                "id": 1,
                "question": "private decomposition one",
                "answer": "private intermediate",
                "paragraph_support_idx": 0,
            },
            {
                "id": 2,
                "question": PRIVATE_RETRIEVAL_CANARY,
                "answer": PRIVATE_RETRIEVAL_CANARY,
                "paragraph_support_idx": 0,
            },
        ],
        "answer": PRIVATE_RETRIEVAL_CANARY,
        "answer_aliases": ["private answer alias"],
        "answerable": True,
    }


def _wire(passages: tuple[DocumentPassage, ...]) -> str:
    return canonical_json([passage.to_value() for passage in passages])


def test_hotpot_row_projects_only_answer_free_passages_deterministically() -> None:
    first = hotpotqa_passages_from_row(
        _hotpot_row(),
        dataset_revision=REVISION,
        split="dev",
    )
    second = hotpotqa_passages_from_row(
        _hotpot_row(),
        dataset_revision=REVISION,
        split="dev",
    )

    assert first == second
    assert all(isinstance(passage, DocumentPassage) for passage in first)
    assert [passage.title for passage in first] == ["Alpha", "Beta"]
    assert [passage.text for passage in first] == [
        "Alpha contains the public zephyr fact.",
        "Beta contains another public fact.",
    ]
    wire = _wire(first)
    assert PRIVATE_RETRIEVAL_CANARY not in wire
    assert "supporting_facts" not in wire
    assert "sent_id" not in wire


def test_hotpot_row_rejects_schema_drift_and_duplicate_titles() -> None:
    extra = _hotpot_row()
    extra["unknown"] = "schema drift"
    with pytest.raises(ValueError):
        hotpotqa_passages_from_row(extra, dataset_revision=REVISION, split="dev")

    duplicate = _hotpot_row()
    duplicate["context"] = {
        "title": ["Alpha", "Alpha"],
        "sentences": [["First."], ["Second."]],
    }
    duplicate["supporting_facts"] = {"title": ["Alpha"], "sent_id": [0]}
    with pytest.raises(ValueError):
        hotpotqa_passages_from_row(duplicate, dataset_revision=REVISION, split="dev")


def test_hotpot_row_preserves_positional_empty_and_repeated_sentences() -> None:
    row = _hotpot_row()
    row["context"] = {
        "title": ["Alpha", "Beta"],
        "sentences": [
            ["Repeated.", "", "Cafe\u0301.", "Repeated."],
            ["Second public passage."],
        ],
    }
    row["supporting_facts"] = {
        "title": ["Alpha", "Alpha"],
        "sent_id": [0, 3],
    }

    passages = hotpotqa_passages_from_row(
        row,
        dataset_revision=REVISION,
        split="dev",
    )

    assert passages[0].text == "Repeated.Café.Repeated."


def test_musique_row_projects_paragraphs_without_private_labels() -> None:
    passages = musique_passages_from_row(
        _musique_row(),
        dataset_revision=REVISION,
        split="validation",
    )

    assert [passage.title for passage in passages] == ["Gamma", "Delta"]
    assert [passage.text for passage in passages] == [
        "Gamma contains the public nebula fact.",
        "Delta is a public distractor passage.",
    ]
    wire = _wire(passages)
    assert PRIVATE_RETRIEVAL_CANARY not in wire
    assert "is_supporting" not in wire
    assert "question_decomposition" not in wire
    assert "paragraph_support_idx" not in wire


def test_atlas_tsv_reader_is_streaming_strict_and_deterministic() -> None:
    source = (
        "id\ttext\ttitle\n"
        "1\tOrion contains a public asterism.\tOrion\n"
        "2\tLyra contains the star Vega.\tLyra\n"
    )

    first = tuple(iter_atlas_wikipedia_tsv(StringIO(source)))
    second = tuple(iter_atlas_wikipedia_tsv(StringIO(source)))

    assert first == second
    assert [passage.title for passage in first] == ["Orion", "Lyra"]
    assert [passage.source_rowid for passage in first] == [1, 2]
    assert [passage.passage_id for passage in first] == [
        f"{ATLAS_DPR_PASSAGE_ID_PREFIX}000000000001",
        f"{ATLAS_DPR_PASSAGE_ID_PREFIX}000000000002",
    ]
    assert [passage.passage_id for passage in first] == sorted(
        passage.passage_id for passage in first
    )
    assert len({passage.passage_id for passage in first}) == 2
    assert PRIVATE_RETRIEVAL_CANARY not in _wire(first)
    assert "source_rowid" not in _wire(first)

    with pytest.raises(ValueError):
        tuple(iter_atlas_wikipedia_tsv(StringIO("id\ttext\ttitle\textra\n1\tt\tx\ty\n")))
    with pytest.raises(ValueError):
        tuple(
            iter_atlas_wikipedia_tsv(
                StringIO("id\ttext\ttitle\n1\tfirst\tAlpha\n1\tsecond\tBeta\n")
            )
        )
    assert (
        len(
            tuple(
                iter_atlas_wikipedia_tsv(
                    StringIO("id\ttext\ttitle\n9\tfirst\tAlpha\n10\tsecond\tBeta\n")
                )
            )
        )
        == 2
    )
    with pytest.raises(ValueError):
        tuple(
            iter_atlas_wikipedia_tsv(
                StringIO("id\ttext\ttitle\n2\tfirst\tAlpha\n1\tsecond\tBeta\n")
            )
        )
    with pytest.raises(ValueError):
        tuple(iter_atlas_wikipedia_tsv(StringIO("id\ttext\ttitle\nwiki-1\tfirst\tAlpha\n")))


def test_atlas_tsv_reader_normalizes_official_decomposed_unicode_to_nfc() -> None:
    source = "id\ttext\ttitle\n1\tCafe\u0301 text.\tCafe\u0301\n"

    passages = tuple(iter_atlas_wikipedia_tsv(StringIO(source)))

    assert len(passages) == 1
    assert passages[0].title == "Caf\u00e9"
    assert passages[0].text == "Caf\u00e9 text."
    assert _wire(passages) == _wire(
        tuple(
            iter_atlas_wikipedia_tsv(StringIO("id\ttext\ttitle\n1\tCaf\u00e9 text.\tCaf\u00e9\n"))
        )
    )


def test_atlas_tsv_reader_reuses_contiguous_title_document_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stable_hash_calls = 0
    real_stable_hash = retrieval_corpus.stable_hash

    def count_stable_hash(value: JsonValue) -> str:
        nonlocal stable_hash_calls
        stable_hash_calls += 1
        return real_stable_hash(value)

    monkeypatch.setattr(retrieval_corpus, "stable_hash", count_stable_hash)
    passages = tuple(
        iter_atlas_wikipedia_tsv(
            StringIO(
                "id\ttext\ttitle\n"
                "1\tfirst public text\tSame title\n"
                "2\tsecond public text\tSame title\n"
            )
        )
    )

    assert passages[0].document_id == passages[1].document_id
    assert stable_hash_calls == 1


def test_prepared_passages_build_and_search_the_sqlite_index(tmp_path: Path) -> None:
    passages = (
        *hotpotqa_passages_from_row(
            _hotpot_row(),
            dataset_revision=REVISION,
            split="dev",
        ),
        *musique_passages_from_row(
            _musique_row(),
            dataset_revision=REVISION,
            split="validation",
        ),
        *tuple(
            iter_atlas_wikipedia_tsv(
                StringIO("id\ttext\ttitle\n1\tVega is in public Lyra.\tLyra\n")
            )
        ),
    )
    path = tmp_path / "prepared-retrieval.sqlite"
    manifest = build_retrieval_index(
        path,
        passages,
        corpus_name="synthetic-public-corpus",
        corpus_version=REVISION,
    )

    assert manifest.passage_count == len(passages)
    with RetrievalIndex.open(path) as index:
        zephyr = index.search("zephyr", limit=3)
        nebula = index.search("nebula", limit=3)
        vega = index.search("Vega", limit=3)

    assert [hit.title for hit in zephyr] == ["Alpha"]
    assert [hit.title for hit in nebula] == ["Gamma"]
    assert [hit.title for hit in vega] == ["Lyra"]
