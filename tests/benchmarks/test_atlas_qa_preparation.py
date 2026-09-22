from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from skillev_private.benchmarks.atlas_qa_preparation import (
    ATLAS_QA_PREPARATION_FORMAT,
    ATLAS_QA_SOURCE_COMMIT,
    prepare_atlas_qa,
    verify_atlas_qa_preparation,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _trivia(question: str, target: str) -> dict[str, object]:
    return {
        "Question": question,
        "Answer": {
            "Aliases": [target, f"{target} alias"],
            "Value": target,
        },
        "unused_private_source_field": "not copied",
    }


def _fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    index_root = root / "dataindex"
    for prefix, indices in {
        "TQA": {"train": [2, 0], "dev": [1], "test": [1, 0]},
        "NQ": {"train": [2], "dev": [0], "test": [1]},
    }.items():
        for split, values in indices.items():
            _write_json(index_root / f"{prefix}.{split}.idx.json", values)

    trivia_root = root / "triviaqa-unfiltered"
    _write_json(
        trivia_root / "unfiltered-web-train.json",
        {
            "Data": [
                _trivia("Train question zero?", "ZERO"),
                _trivia("Café question one?", "ONE"),
                _trivia("Train question two?", "Two"),
            ],
            "ignored_top_level_field": True,
        },
    )
    _write_json(
        trivia_root / "unfiltered-web-dev.json",
        {"Data": [_trivia("Test question zero?", "Test Zero"), _trivia("Test one?", "TEST")]},
    )

    nq_train = root / "nq" / "NQ-open.train.jsonl"
    nq_dev = root / "nq" / "NQ-open.dev.jsonl"
    _write_jsonl(
        nq_train,
        [
            {"question": "NQ train zero?", "answer": ["zero"]},
            {"question": "NQ train one?", "answer": ["one"]},
            {"question": "NQ train two?", "answer": ["two"]},
        ],
    )
    _write_jsonl(
        nq_dev,
        [
            {"question": "NQ test zero?", "answer": ["test zero"]},
            {"question": "NQ test one?", "answer": ["test one"]},
        ],
    )
    return index_root, trivia_root, nq_train, nq_dev


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_atlas_qa_streams_exact_split_order_and_preserves_sources(
    tmp_path: Path,
) -> None:
    index_root, trivia_root, nq_train, nq_dev = _fixture(tmp_path / "sources")
    source_files = tuple(
        sorted(
            (
                *index_root.glob("*.json"),
                *trivia_root.glob("*.json"),
                nq_train,
                nq_dev,
            )
        )
    )
    before = {path: _sha256(path) for path in source_files}
    output_root = tmp_path / "prepared"

    manifest = prepare_atlas_qa(
        dataindex_root=index_root,
        triviaqa_unfiltered_root=trivia_root,
        nq_train_jsonl=nq_train,
        nq_dev_jsonl=nq_dev,
        output_root=output_root,
    )

    assert manifest["format"] == ATLAS_QA_PREPARATION_FORMAT
    assert manifest["atlas_source_commit"] == ATLAS_QA_SOURCE_COMMIT
    assert {path: _sha256(path) for path in source_files} == before
    assert _rows(output_root / "triviaqa_data" / "train.jsonl") == [
        {
            "question": "Train question two?",
            "answers": ["Two", "Two alias"],
            "target": "Two",
        },
        {
            "question": "Train question zero?",
            "answers": ["ZERO", "ZERO alias"],
            "target": "Zero",
        },
    ]
    assert _rows(output_root / "triviaqa_data" / "dev.jsonl") == [
        {
            "question": "Café question one?",
            "answers": ["ONE", "ONE alias"],
            "target": "One",
        }
    ]
    assert _rows(output_root / "triviaqa_data" / "test.jsonl") == [
        {
            "question": "Test one?",
            "answers": ["TEST", "TEST alias"],
            "target": "Test",
        },
        {
            "question": "Test question zero?",
            "answers": ["Test Zero", "Test Zero alias"],
            "target": "Test Zero",
        },
    ]
    assert _rows(output_root / "nq_data" / "train.jsonl") == [
        {"question": "NQ train two?", "answers": ["two"]}
    ]
    assert _rows(output_root / "nq_data" / "dev.jsonl") == [
        {"question": "NQ train zero?", "answers": ["zero"]}
    ]
    assert _rows(output_root / "nq_data" / "test.jsonl") == [
        {"question": "NQ test one?", "answers": ["test one"]}
    ]
    assert (
        json.loads((output_root / "preparation.manifest.json").read_text(encoding="utf-8"))
        == manifest
    )
    for relative_path, expected_hash in manifest["output_sha256"].items():
        assert _sha256(output_root / relative_path) == expected_hash
    assert verify_atlas_qa_preparation(output_root) == manifest


def test_prepare_atlas_qa_is_deterministic_and_never_replaces_an_output(
    tmp_path: Path,
) -> None:
    index_root, trivia_root, nq_train, nq_dev = _fixture(tmp_path / "sources")
    arguments = {
        "dataindex_root": index_root,
        "triviaqa_unfiltered_root": trivia_root,
        "nq_train_jsonl": nq_train,
        "nq_dev_jsonl": nq_dev,
    }
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = prepare_atlas_qa(output_root=first_root, **arguments)
    second = prepare_atlas_qa(output_root=second_root, **arguments)

    assert first == second
    sentinel = first_root / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_atlas_qa(output_root=first_root, **arguments)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_prepare_atlas_qa_rejects_source_drift_before_publishing(
    tmp_path: Path,
) -> None:
    index_root, trivia_root, nq_train, nq_dev = _fixture(tmp_path / "sources")
    _write_jsonl(
        nq_train,
        [
            {"question": "NQ train zero?", "answer": ["zero"], "unexpected": True},
            {"question": "NQ train one?", "answer": ["one"]},
            {"question": "NQ train two?", "answer": ["two"]},
        ],
    )
    output_root = tmp_path / "prepared"

    with pytest.raises(ValueError):
        prepare_atlas_qa(
            dataindex_root=index_root,
            triviaqa_unfiltered_root=trivia_root,
            nq_train_jsonl=nq_train,
            nq_dev_jsonl=nq_dev,
            output_root=output_root,
        )
    assert not output_root.exists()
