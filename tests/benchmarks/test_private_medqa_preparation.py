from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from skillev_private.benchmarks.medqa_preparation import (
    MEDQA_PREPARATION_FORMAT,
    prepare_medqa_us_four_option,
    verify_medqa_us_four_option_preparation,
)

_REVISION = "27b02f66aac217933c9648a06f82e9f720377925"


def _row() -> dict[str, object]:
    return {
        "answer": "correct",
        "answer_idx": "B",
        "meta_info": "step1",
        "metamap_phrases": ["auxiliary", "phrases"],
        "options": {
            "A": "wrong a",
            "B": "correct",
            "C": "wrong c",
            "D": "wrong d",
        },
        "question": "Which option is correct?",
    }


def _source(root: Path, *, row: dict[str, object] | None = None) -> Path:
    root.mkdir()
    payload = json.dumps(_row() if row is None else row) + "\n"
    for split in ("dev", "test", "train"):
        (root / f"phrases_no_exclude_{split}.jsonl").write_text(
            payload,
            encoding="utf-8",
        )
    return root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepares_exact_five_field_wire_and_preserves_sources(
    tmp_path: Path,
) -> None:
    source = _source((tmp_path / "source").resolve())
    before = {path: _sha256(path) for path in source.iterdir()}
    output = (tmp_path / "prepared").resolve()

    manifest = prepare_medqa_us_four_option(
        source_root=source,
        output_root=output,
        dataset_revision=_REVISION,
    )

    assert manifest["format"] == MEDQA_PREPARATION_FORMAT
    assert {path: _sha256(path) for path in source.iterdir()} == before
    for split in ("dev", "test", "train"):
        row = json.loads((output / f"{split}.jsonl").read_text(encoding="utf-8"))
        assert set(row) == {
            "answer",
            "answer_idx",
            "meta_info",
            "options",
            "question",
        }
        assert "metamap_phrases" not in row
    assert (
        verify_medqa_us_four_option_preparation(
            output_root=output,
            dataset_revision=_REVISION,
        )
        == manifest
    )


def test_is_deterministic_and_never_replaces_an_output(tmp_path: Path) -> None:
    source = _source((tmp_path / "source").resolve())
    first_root = (tmp_path / "first").resolve()
    second_root = (tmp_path / "second").resolve()

    first = prepare_medqa_us_four_option(
        source_root=source,
        output_root=first_root,
        dataset_revision=_REVISION,
    )
    second = prepare_medqa_us_four_option(
        source_root=source,
        output_root=second_root,
        dataset_revision=_REVISION,
    )

    assert first == second
    with pytest.raises(FileExistsError):
        prepare_medqa_us_four_option(
            source_root=source,
            output_root=first_root,
            dataset_revision=_REVISION,
        )


def test_rejects_source_wire_drift_before_publication(tmp_path: Path) -> None:
    row = _row()
    del row["metamap_phrases"]
    source = _source((tmp_path / "source").resolve(), row=row)
    output = (tmp_path / "prepared").resolve()

    with pytest.raises(ValueError):
        prepare_medqa_us_four_option(
            source_root=source,
            output_root=output,
            dataset_revision=_REVISION,
        )
    assert not output.exists()


def test_verifier_rejects_changed_output(tmp_path: Path) -> None:
    source = _source((tmp_path / "source").resolve())
    output = (tmp_path / "prepared").resolve()
    prepare_medqa_us_four_option(
        source_root=source,
        output_root=output,
        dataset_revision=_REVISION,
    )
    path = output / "dev.jsonl"
    path.write_bytes(path.read_bytes() + b'{"changed":true}\n')

    with pytest.raises(ValueError):
        verify_medqa_us_four_option_preparation(
            output_root=output,
            dataset_revision=_REVISION,
        )
