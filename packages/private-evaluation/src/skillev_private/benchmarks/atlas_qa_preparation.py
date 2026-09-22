"""Memory-bounded, no-delete preparation of Atlas TriviaQA and NQ JSONL.

The production converters consume the exact wire emitted by
``facebookresearch/atlas@0ec8889:preprocessing/prepare_qa.py``.  The upstream
script removes downloaded archives and raw inputs after conversion.  Local
benchmark preparation instead keeps every source byte and publishes the six
standard split files into one previously absent output directory.

TriviaQA's unfiltered source is a large JSON object.  It is iterated with
``ijson`` so preparation retains only the selected minimal QA records rather
than materialising the full source in WSL memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import BinaryIO, cast

import ijson  # type: ignore[import-untyped]

from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash

from .acquisition import _read_published_canonical_record

ATLAS_QA_PREPARATION_FORMAT = "skillev-private-atlas-qa-preparation@1"
ATLAS_QA_SOURCE_COMMIT = "0ec8889492d5187b26c51b8d1781239a4cf6741e"
_SPLITS = ("train", "dev", "test")
_OUTPUT_PATHS = tuple(
    f"{benchmark}_data/{split}.jsonl" for benchmark in ("triviaqa", "nq") for split in _SPLITS
)
_INPUT_PATHS = (
    "dataindex/NQ.dev.idx.json",
    "dataindex/NQ.test.idx.json",
    "dataindex/NQ.train.idx.json",
    "dataindex/TQA.dev.idx.json",
    "dataindex/TQA.test.idx.json",
    "dataindex/TQA.train.idx.json",
    "nq/NQ-open.dev.jsonl",
    "nq/NQ-open.train.jsonl",
    "triviaqa/unfiltered-web-dev.json",
    "triviaqa/unfiltered-web-train.json",
)


class _HashingReader:
    """Update a digest while a streaming parser consumes one binary file."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self._stream.read(size)
        self._digest.update(data)
        return data

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, *, source: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{source} record must be an exact string-keyed object")
    return cast(dict[str, object], value)


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text without NUL")
    return value


def _text_array(value: object, *, field: str) -> list[str]:
    if type(value) is not list or not value:
        raise TypeError(f"{field} must be a non-empty text array")
    result = [_text(item, field=field) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _load_indices(path: Path) -> tuple[int, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        type(value) is not list
        or not value
        or any(type(index) is not int or index < 0 for index in value)
    ):
        raise ValueError(
            f"Atlas split index must be a non-empty array of non-negative integers: {path}"
        )
    indices = tuple(cast(list[int], value))
    if len(set(indices)) != len(indices):
        raise ValueError(f"Atlas split index must not contain duplicates: {path}")
    return indices


def _load_split_indices(index_root: Path, prefix: str) -> dict[str, tuple[int, ...]]:
    return {split: _load_indices(index_root / f"{prefix}.{split}.idx.json") for split in _SPLITS}


def _convert_trivia(value: object) -> dict[str, JsonValue]:
    row = _object(value, source="TriviaQA")
    question = _text(row.get("Question"), field="TriviaQA Question")
    answer = _object(row.get("Answer"), source="TriviaQA Answer")
    aliases = _text_array(answer.get("Aliases"), field="TriviaQA Answer.Aliases")
    target = _text(answer.get("Value"), field="TriviaQA Answer.Value")
    if target.isupper():
        target = target.title()
    return {
        "question": question,
        "answers": normalize_json(aliases),
        "target": target,
    }


def _convert_nq(value: object) -> dict[str, JsonValue]:
    row = _object(value, source="NQ-Open")
    if set(row) != {"question", "answer"}:
        raise ValueError("NQ-Open source record fields differ from the pinned official wire")
    return {
        "question": _text(row["question"], field="NQ-Open question"),
        "answers": normalize_json(_text_array(row["answer"], field="NQ-Open answer")),
    }


def _select_trivia(
    path: Path,
    selected_indices: set[int],
) -> tuple[dict[int, dict[str, JsonValue]], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    selected: dict[int, dict[str, JsonValue]] = {}
    with path.open("rb") as source:
        hashing_reader = _HashingReader(source)
        for index, value in enumerate(ijson.items(hashing_reader, "Data.item")):
            if index in selected_indices:
                selected[index] = _convert_trivia(value)
        source_hash = hashing_reader.hexdigest
    missing = selected_indices.difference(selected)
    if missing:
        raise ValueError(f"TriviaQA source is missing {len(missing)} selected rows")
    return selected, source_hash


def _select_nq(
    path: Path,
    selected_indices: set[int],
) -> tuple[dict[int, dict[str, JsonValue]], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    selected: dict[int, dict[str, JsonValue]] = {}
    with path.open("rb") as stream:
        for index, raw_line in enumerate(stream):
            digest.update(raw_line)
            if not raw_line.strip():
                raise ValueError("NQ-Open source contains a blank JSONL record")
            if index in selected_indices:
                selected[index] = _convert_nq(json.loads(raw_line))
    missing = selected_indices.difference(selected)
    if missing:
        raise ValueError(f"NQ-Open source is missing {len(missing)} selected rows")
    return selected, digest.hexdigest()


def _write_split(
    path: Path,
    indices: tuple[int, ...],
    selected: Mapping[int, dict[str, JsonValue]],
) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("xb") as stream:
        for index in indices:
            line = (json.dumps(selected[index], ensure_ascii=False) + "\n").encode("utf-8")
            stream.write(line)
            digest.update(line)
    return len(indices), digest.hexdigest()


def prepare_atlas_qa(
    *,
    dataindex_root: Path,
    triviaqa_unfiltered_root: Path,
    nq_train_jsonl: Path,
    nq_dev_jsonl: Path,
    output_root: Path,
) -> dict[str, JsonValue]:
    """Prepare all standard Atlas QA splits without deleting or replacing files."""

    paths = (
        dataindex_root,
        triviaqa_unfiltered_root,
        nq_train_jsonl,
        nq_dev_jsonl,
        output_root,
    )
    if any(not isinstance(path, Path) or not path.is_absolute() for path in paths):
        raise ValueError("Atlas QA preparation paths must be absolute")
    if not dataindex_root.is_dir():
        raise FileNotFoundError(dataindex_root)
    if not triviaqa_unfiltered_root.is_dir():
        raise FileNotFoundError(triviaqa_unfiltered_root)
    if output_root.exists():
        raise FileExistsError(output_root)

    trivia_indices = _load_split_indices(dataindex_root, "TQA")
    nq_indices = _load_split_indices(dataindex_root, "NQ")
    trivia_train_source = triviaqa_unfiltered_root / "unfiltered-web-train.json"
    trivia_dev_source = triviaqa_unfiltered_root / "unfiltered-web-dev.json"

    trivia_train_rows, trivia_train_hash = _select_trivia(
        trivia_train_source,
        set(trivia_indices["train"]) | set(trivia_indices["dev"]),
    )
    trivia_test_rows, trivia_dev_hash = _select_trivia(
        trivia_dev_source,
        set(trivia_indices["test"]),
    )
    nq_train_rows, nq_train_hash = _select_nq(
        nq_train_jsonl,
        set(nq_indices["train"]) | set(nq_indices["dev"]),
    )
    nq_test_rows, nq_dev_hash = _select_nq(nq_dev_jsonl, set(nq_indices["test"]))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.with_name(f".{output_root.name}.part-{uuid.uuid4().hex}")
    staging_root.mkdir()
    trivia_output = staging_root / "triviaqa_data"
    nq_output = staging_root / "nq_data"
    trivia_output.mkdir()
    nq_output.mkdir()

    row_counts: dict[str, int] = {}
    output_hashes: dict[str, str] = {}
    for split in _SPLITS:
        trivia_rows = trivia_test_rows if split == "test" else trivia_train_rows
        count, digest = _write_split(
            trivia_output / f"{split}.jsonl",
            trivia_indices[split],
            trivia_rows,
        )
        key = f"triviaqa_data/{split}.jsonl"
        row_counts[key] = count
        output_hashes[key] = digest

        nq_rows = nq_test_rows if split == "test" else nq_train_rows
        count, digest = _write_split(
            nq_output / f"{split}.jsonl",
            nq_indices[split],
            nq_rows,
        )
        key = f"nq_data/{split}.jsonl"
        row_counts[key] = count
        output_hashes[key] = digest

    input_hashes = {
        "dataindex/NQ.dev.idx.json": _sha256_file(dataindex_root / "NQ.dev.idx.json"),
        "dataindex/NQ.test.idx.json": _sha256_file(dataindex_root / "NQ.test.idx.json"),
        "dataindex/NQ.train.idx.json": _sha256_file(dataindex_root / "NQ.train.idx.json"),
        "dataindex/TQA.dev.idx.json": _sha256_file(dataindex_root / "TQA.dev.idx.json"),
        "dataindex/TQA.test.idx.json": _sha256_file(dataindex_root / "TQA.test.idx.json"),
        "dataindex/TQA.train.idx.json": _sha256_file(dataindex_root / "TQA.train.idx.json"),
        "nq/NQ-open.dev.jsonl": nq_dev_hash,
        "nq/NQ-open.train.jsonl": nq_train_hash,
        "triviaqa/unfiltered-web-dev.json": trivia_dev_hash,
        "triviaqa/unfiltered-web-train.json": trivia_train_hash,
    }
    manifest_without_hash: dict[str, JsonValue] = {
        "atlas_source_commit": ATLAS_QA_SOURCE_COMMIT,
        "format": ATLAS_QA_PREPARATION_FORMAT,
        "input_sha256": normalize_json(input_hashes),
        "output_sha256": normalize_json(output_hashes),
        "row_counts": normalize_json(row_counts),
    }
    manifest = {
        **manifest_without_hash,
        "content_hash": stable_hash(manifest_without_hash),
    }
    normalized_manifest = cast(dict[str, JsonValue], normalize_json(manifest))
    (staging_root / "preparation.manifest.json").write_text(
        canonical_json(normalized_manifest) + "\n",
        encoding="utf-8",
    )
    staging_root.rename(output_root)
    return normalized_manifest


def verify_atlas_qa_preparation(output_root: Path) -> dict[str, JsonValue]:
    """Verify and return one previously published Atlas QA preparation."""

    if not isinstance(output_root, Path) or not output_root.is_absolute():
        raise ValueError("Atlas QA output_root must be an absolute Path")
    if not output_root.is_dir():
        raise NotADirectoryError(output_root)
    raw_value = _read_published_canonical_record(
        output_root / "preparation.manifest.json",
        label="Atlas QA preparation manifest",
    )
    if not isinstance(raw_value, dict):
        raise TypeError("Atlas QA preparation manifest must be an object")
    value = raw_value
    expected_fields = {
        "atlas_source_commit",
        "content_hash",
        "format",
        "input_sha256",
        "output_sha256",
        "row_counts",
    }
    if set(value) != expected_fields:
        raise ValueError("Atlas QA preparation manifest has an invalid field set")
    if value["format"] != ATLAS_QA_PREPARATION_FORMAT:
        raise ValueError("unsupported Atlas QA preparation manifest format")
    if value["atlas_source_commit"] != ATLAS_QA_SOURCE_COMMIT:
        raise ValueError("Atlas QA preparation uses a different source commit")

    content_hash = value["content_hash"]
    if type(content_hash) is not str:
        raise TypeError("Atlas QA preparation content_hash must be text")
    manifest_without_hash = dict(value)
    del manifest_without_hash["content_hash"]
    if stable_hash(manifest_without_hash) != content_hash:
        raise ValueError("Atlas QA preparation content hash differs")

    input_hashes = value["input_sha256"]
    output_hashes = value["output_sha256"]
    row_counts = value["row_counts"]
    if not isinstance(input_hashes, dict) or set(input_hashes) != set(_INPUT_PATHS):
        raise ValueError("Atlas QA preparation input identities differ")
    if not isinstance(output_hashes, dict) or set(output_hashes) != set(_OUTPUT_PATHS):
        raise ValueError("Atlas QA preparation output identities differ")
    if not isinstance(row_counts, dict) or set(row_counts) != set(_OUTPUT_PATHS):
        raise ValueError("Atlas QA preparation row counts differ")
    for relative_path in _INPUT_PATHS:
        digest = input_hashes[relative_path]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Atlas QA preparation input digest is invalid")
    for relative_path in _OUTPUT_PATHS:
        digest = output_hashes[relative_path]
        row_count = row_counts[relative_path]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Atlas QA preparation output digest is invalid")
        if type(row_count) is not int or row_count < 1:
            raise ValueError("Atlas QA preparation row count must be positive")
        output_path = output_root / relative_path
        if not output_path.is_file():
            raise FileNotFoundError(output_path)
        if _sha256_file(output_path) != digest:
            raise ValueError("Atlas QA prepared output differs from its manifest")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare pinned Atlas TriviaQA and NQ JSONL without deleting source files"
    )
    parser.add_argument("--dataindex-root", required=True, type=Path)
    parser.add_argument("--triviaqa-unfiltered-root", required=True, type=Path)
    parser.add_argument("--nq-train-jsonl", required=True, type=Path)
    parser.add_argument("--nq-dev-jsonl", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = prepare_atlas_qa(
        dataindex_root=args.dataindex_root.resolve(),
        triviaqa_unfiltered_root=args.triviaqa_unfiltered_root.resolve(),
        nq_train_jsonl=args.nq_train_jsonl.resolve(),
        nq_dev_jsonl=args.nq_dev_jsonl.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATLAS_QA_PREPARATION_FORMAT",
    "ATLAS_QA_SOURCE_COMMIT",
    "main",
    "prepare_atlas_qa",
    "verify_atlas_qa_preparation",
]
