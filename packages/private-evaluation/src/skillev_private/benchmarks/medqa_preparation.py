"""Stream the official MedQA US four-option wire into loader-ready JSONL.

The distributed four-option files contain one additional
``metamap_phrases`` field.  The production converter deliberately accepts only
the five benchmark fields that define the task and target, so preparation
removes that auxiliary annotation while preserving every accepted value.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import cast

from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash

from .acquisition import _read_published_canonical_record
from .converters import convert_medqa_us_4option_row

MEDQA_PREPARATION_FORMAT = "skillev-private-medqa-us-four-option-preparation@1"
_SPLITS = ("dev", "test", "train")
_SOURCE_FIELDS = frozenset(
    {
        "answer",
        "answer_idx",
        "meta_info",
        "metamap_phrases",
        "options",
        "question",
    }
)
_OUTPUT_FIELDS = ("answer", "answer_idx", "meta_info", "options", "question")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicate_rejecting_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("MedQA source contains a duplicate object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"MedQA source contains unsupported constant {value}")


def _prepare_split(
    *,
    source_path: Path,
    output_path: Path,
    dataset_revision: str,
    split: str,
) -> tuple[int, str, str]:
    input_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    row_count = 0
    with source_path.open("rb") as source, output_path.open("xb") as output:
        for raw_line in source:
            input_digest.update(raw_line)
            if not raw_line.strip():
                raise ValueError("MedQA source contains a blank JSONL record")
            try:
                value = json.loads(
                    raw_line,
                    object_pairs_hook=_duplicate_rejecting_object,
                    parse_constant=_reject_json_constant,
                )
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("MedQA source is invalid JSONL") from error
            if type(value) is not dict or set(value) != _SOURCE_FIELDS:
                raise ValueError("MedQA source record differs from the pinned official wire")
            source_row = cast(dict[str, object], value)
            row = {field: source_row[field] for field in _OUTPUT_FIELDS}
            convert_medqa_us_4option_row(
                row,
                dataset_revision=dataset_revision,
                split=split,
            )
            line = (canonical_json(normalize_json(row)) + "\n").encode()
            output.write(line)
            output_digest.update(line)
            row_count += 1
    if row_count == 0:
        raise ValueError("MedQA source split must not be empty")
    return row_count, input_digest.hexdigest(), output_digest.hexdigest()


def prepare_medqa_us_four_option(
    *,
    source_root: Path,
    output_root: Path,
    dataset_revision: str,
) -> dict[str, JsonValue]:
    """Prepare all three official MedQA US four-option splits once."""

    if (
        not isinstance(source_root, Path)
        or not source_root.is_absolute()
        or not source_root.is_dir()
    ):
        raise NotADirectoryError(source_root)
    if not isinstance(output_root, Path) or not output_root.is_absolute():
        raise ValueError("MedQA output_root must be an absolute Path")
    if output_root.exists():
        raise FileExistsError(output_root)
    if (
        type(dataset_revision) is not str
        or len(dataset_revision) != 40
        or any(character not in "0123456789abcdef" for character in dataset_revision)
    ):
        raise ValueError("MedQA dataset_revision must be a full lowercase Git commit")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.with_name(f".{output_root.name}.part-{uuid.uuid4().hex}")
    staging_root.mkdir()
    input_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for split in _SPLITS:
        source_name = f"phrases_no_exclude_{split}.jsonl"
        output_name = f"{split}.jsonl"
        count, input_hash, output_hash = _prepare_split(
            source_path=source_root / source_name,
            output_path=staging_root / output_name,
            dataset_revision=dataset_revision,
            split=split,
        )
        input_hashes[source_name] = input_hash
        output_hashes[output_name] = output_hash
        row_counts[output_name] = count

    manifest_without_hash: dict[str, JsonValue] = {
        "dataset_revision": dataset_revision,
        "format": MEDQA_PREPARATION_FORMAT,
        "input_sha256": normalize_json(input_hashes),
        "output_sha256": normalize_json(output_hashes),
        "row_counts": normalize_json(row_counts),
    }
    manifest = cast(
        dict[str, JsonValue],
        normalize_json(
            {
                **manifest_without_hash,
                "content_hash": stable_hash(manifest_without_hash),
            }
        ),
    )
    (staging_root / "preparation.manifest.json").write_text(
        canonical_json(manifest) + "\n",
        encoding="utf-8",
    )
    staging_root.rename(output_root)
    return manifest


def verify_medqa_us_four_option_preparation(
    *,
    output_root: Path,
    dataset_revision: str,
) -> dict[str, JsonValue]:
    """Verify and return one previously published MedQA preparation."""

    if not isinstance(output_root, Path) or not output_root.is_absolute():
        raise ValueError("MedQA output_root must be an absolute Path")
    if not output_root.is_dir():
        raise NotADirectoryError(output_root)
    raw_value = _read_published_canonical_record(
        output_root / "preparation.manifest.json",
        label="MedQA preparation manifest",
    )
    if not isinstance(raw_value, dict):
        raise TypeError("MedQA preparation manifest must be an object")
    value = raw_value
    expected_fields = {
        "content_hash",
        "dataset_revision",
        "format",
        "input_sha256",
        "output_sha256",
        "row_counts",
    }
    if set(value) != expected_fields:
        raise ValueError("MedQA preparation manifest has an invalid field set")
    if value["format"] != MEDQA_PREPARATION_FORMAT:
        raise ValueError("unsupported MedQA preparation manifest format")
    if value["dataset_revision"] != dataset_revision:
        raise ValueError("MedQA preparation uses a different dataset revision")
    content_hash = value["content_hash"]
    if type(content_hash) is not str:
        raise TypeError("MedQA preparation content_hash must be text")
    manifest_without_hash = dict(value)
    del manifest_without_hash["content_hash"]
    if stable_hash(manifest_without_hash) != content_hash:
        raise ValueError("MedQA preparation content hash differs")

    input_hashes = value["input_sha256"]
    output_hashes = value["output_sha256"]
    row_counts = value["row_counts"]
    expected_inputs = {f"phrases_no_exclude_{split}.jsonl" for split in _SPLITS}
    expected_outputs = {f"{split}.jsonl" for split in _SPLITS}
    if not isinstance(input_hashes, dict) or set(input_hashes) != expected_inputs:
        raise ValueError("MedQA preparation input identities differ")
    if not isinstance(output_hashes, dict) or set(output_hashes) != expected_outputs:
        raise ValueError("MedQA preparation output identities differ")
    if not isinstance(row_counts, dict) or set(row_counts) != expected_outputs:
        raise ValueError("MedQA preparation row counts differ")
    for output_name in expected_outputs:
        digest = output_hashes[output_name]
        row_count = row_counts[output_name]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("MedQA preparation output digest is invalid")
        if type(row_count) is not int or row_count < 1:
            raise ValueError("MedQA preparation row count must be positive")
        output_path = output_root / output_name
        if not output_path.is_file():
            raise FileNotFoundError(output_path)
        if _sha256_file(output_path) != digest:
            raise ValueError("MedQA prepared output differs from its manifest")
    return value


__all__ = [
    "MEDQA_PREPARATION_FORMAT",
    "prepare_medqa_us_four_option",
    "verify_medqa_us_four_option_preparation",
]
