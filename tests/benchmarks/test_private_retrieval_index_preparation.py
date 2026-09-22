from __future__ import annotations

import gzip
import hashlib
import sqlite3
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest
import skillev_private.benchmarks.retrieval_preparation as preparation
from skillev_private.benchmarks.retrieval_corpus import iter_atlas_wikipedia_tsv

from skillev.benchmarks import DocumentPassage, RetrievalIndex, RetrievalIndexManifest
from skillev.contracts import canonical_json, parse_canonical_json

PASSAGE_CANARY = "PRIVATE-CORPUS-PASSAGE-CANARY"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_source(path: Path, *, second_text: str = "Lyra contains Vega.") -> str:
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as stream:
        stream.write("id\ttext\ttitle\n")
        stream.write(f"1\t{PASSAGE_CANARY} appears in Orion.\tOrion\n")
        stream.write(f"2\t{second_text}\tLyra\n")
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "atlas-wikipedia.tsv.gz",
        tmp_path / "atlas-wikipedia.sqlite",
        tmp_path / "atlas-wikipedia.manifest.json",
    )


def test_retrieval_preparation_import_does_not_require_model_libraries() -> None:
    source = f"""
import importlib.abc
import sys

sys.path[:0] = [
    {str(REPOSITORY_ROOT / "src")!r},
    {str(REPOSITORY_ROOT / "packages" / "private-evaluation" / "src")!r},
]
blocked = {{"torch", "transformers", "peft", "tokenizers"}}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.partition(".")[0] in blocked:
            raise RuntimeError(fullname)
        return None

sys.meta_path.insert(0, Blocker())
import skillev_private.benchmarks.retrieval_preparation
assert blocked.isdisjoint({{name.partition(".")[0] for name in sys.modules}})
"""
    completed = subprocess.run(  # noqa: S603 - fixed current interpreter and test source
        (sys.executable, "-I", "-c", source),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_preparation_streams_parser_once_and_emits_only_aggregate_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path, output_path, manifest_path = _paths(tmp_path)
    expected_raw_sha256 = _write_source(input_path)
    parser_calls = 0
    iterator_calls = 0

    class _OneShotPassages(Iterable[DocumentPassage]):
        def __init__(self, values: Iterator[DocumentPassage]) -> None:
            self._values = values

        def __iter__(self) -> Iterator[DocumentPassage]:
            nonlocal iterator_calls
            iterator_calls += 1
            if iterator_calls != 1:
                raise AssertionError("passage stream was iterated more than once")
            yield from self._values

    def strict_parser(lines: Iterable[str]) -> Iterable[DocumentPassage]:
        nonlocal parser_calls
        parser_calls += 1
        return _OneShotPassages(iter_atlas_wikipedia_tsv(lines))

    monkeypatch.setattr(preparation, "iter_atlas_wikipedia_tsv", strict_parser)
    manifest = preparation.prepare_atlas_retrieval_index(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        corpus_name="atlas-wikipedia",
        corpus_version="2018-12-20@pinned",
        expected_raw_sha256=expected_raw_sha256,
    )

    assert parser_calls == 1
    assert iterator_calls == 1
    assert manifest.input_raw_sha256 == expected_raw_sha256
    assert manifest.index_manifest.passage_count == 2
    raw_manifest = manifest_path.read_text(encoding="utf-8")
    assert raw_manifest == canonical_json(manifest.to_value()) + "\n"
    assert PASSAGE_CANARY not in raw_manifest
    assert "Orion" not in raw_manifest
    assert (
        preparation.RetrievalPreparationManifest.from_value(parse_canonical_json(raw_manifest[:-1]))
        == manifest
    )
    assert (
        preparation.verify_retrieval_preparation_output(
            index_path=output_path,
            manifest_path=manifest_path,
        )
        == manifest
    )
    with RetrievalIndex.open(output_path) as index:
        assert index.manifest == manifest.index_manifest
        assert [hit.title for hit in index.search(PASSAGE_CANARY, limit=1)] == ["Orion"]


def test_cli_builds_in_explicit_staging_directory_and_publishes_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path, output_path, manifest_path = _paths(tmp_path)
    staging_directory = tmp_path / "node-local"
    expected_raw_sha256 = _write_source(input_path)
    monkeypatch.setenv("SKILLEV_RETRIEVAL_STAGING_DIRECTORY", str(staging_directory))
    monkeypatch.setenv("SKILLEV_RETRIEVAL_EXPECTED_PASSAGE_COUNT", "2")

    exit_code = preparation.main(
        (
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
            "--corpus-name",
            "atlas-wikipedia",
            "--corpus-version",
            "2018-12-20@pinned",
            "--raw-sha256",
            expected_raw_sha256,
        )
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    printed = preparation.RetrievalPreparationManifest.from_value(
        parse_canonical_json(captured.out.rstrip("\n"))
    )
    assert "phase=ingest" in captured.err
    assert "rate_rows_per_second=" in captured.err
    assert "eta_seconds=" in captured.err
    assert printed == preparation.load_retrieval_preparation_manifest(manifest_path)
    assert not tuple(staging_directory.iterdir())
    with RetrievalIndex.open(output_path) as index:
        assert index.manifest == printed.index_manifest


def test_cross_filesystem_publication_does_not_clobber_racing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "node-local" / "built.sqlite"
    target = tmp_path / "shared" / "published.sqlite"
    source.parent.mkdir()
    target.parent.mkdir()
    source.write_bytes(b"completed-index")
    target.write_bytes(b"racing-index")
    monkeypatch.setattr(preparation, "_same_filesystem", lambda _source, _target: False)

    with pytest.raises(ValueError):
        preparation._publish_new_file(source, target)

    assert source.read_bytes() == b"completed-index"
    assert target.read_bytes() == b"racing-index"
    assert tuple(target.parent.iterdir()) == (target,)


def test_throughput_gate_rejects_before_publication(tmp_path: Path) -> None:
    input_path, output_path, manifest_path = _paths(tmp_path)
    expected_raw_sha256 = _write_source(input_path)

    with pytest.raises(RuntimeError):
        preparation.prepare_atlas_retrieval_index(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            corpus_name="atlas-wikipedia",
            corpus_version="2018-12-20@pinned",
            expected_raw_sha256=expected_raw_sha256,
            staging_directory=tmp_path / "node-local",
            expected_passage_count=2,
            minimum_ingest_rows_per_second=10**30,
        )

    assert not output_path.exists()
    assert not manifest_path.exists()


def test_performance_gate_has_explicit_prefix_identity_and_fixed_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path, output_path, manifest_path = _paths(tmp_path)
    expected_raw_sha256 = _write_source(input_path)
    captured: dict[str, object] = {}

    def fake_builder(
        path: Path,
        passages: Iterable[DocumentPassage],
        **kwargs: object,
    ) -> RetrievalIndexManifest:
        captured.update(kwargs)
        captured["source_rows_seen"] = len(tuple(passages))
        path.write_bytes(b"aggregate-only-test-index")
        return RetrievalIndexManifest.create_from_corpus_hash(
            corpus_name=str(kwargs["corpus_name"]),
            corpus_version=str(kwargs["corpus_version"]),
            passage_count=100_000,
            corpus_hash="sha256:" + "0" * 64,
        )

    monkeypatch.setattr(preparation, "build_retrieval_index_stream", fake_builder)
    prepared = preparation.prepare_atlas_retrieval_index(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        corpus_name="atlas-wikipedia",
        corpus_version="2018-12-20@pinned",
        expected_raw_sha256=expected_raw_sha256,
        staging_directory=tmp_path / "node-local",
        performance_gate_rows=100_000,
    )

    assert captured["expected_passage_count"] == 100_000
    assert captured["source_rows_seen"] == 2
    assert prepared.index_manifest.corpus_version.endswith("/lexical-fts5-gate-100000")

    another_input, another_output, another_manifest = (
        tmp_path / "another.tsv.gz",
        tmp_path / "another.sqlite",
        tmp_path / "another.manifest.json",
    )
    another_sha = _write_source(another_input)
    with pytest.raises(ValueError):
        preparation.prepare_atlas_retrieval_index(
            input_path=another_input,
            output_path=another_output,
            manifest_path=another_manifest,
            corpus_name="atlas-wikipedia",
            corpus_version="2018-12-20@pinned",
            expected_raw_sha256=another_sha,
            performance_gate_rows=10_000,
        )


def test_preparation_rejects_tampered_raw_gzip_before_index_build(tmp_path: Path) -> None:
    input_path, output_path, manifest_path = _paths(tmp_path)
    pinned = _write_source(input_path)
    _write_source(input_path, second_text="The source bytes changed after pinning.")

    with pytest.raises(ValueError):
        preparation.prepare_atlas_retrieval_index(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            corpus_name="atlas-wikipedia",
            corpus_version="2018-12-20@pinned",
            expected_raw_sha256=pinned,
        )

    assert not output_path.exists()
    assert not manifest_path.exists()


def test_existing_exact_target_is_verified_without_rebuilding_and_mismatch_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path, output_path, manifest_path = _paths(tmp_path)
    pinned = _write_source(input_path)
    first = preparation.prepare_atlas_retrieval_index(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        corpus_name="atlas-wikipedia",
        corpus_version="2018-12-20@pinned",
        expected_raw_sha256=pinned,
    )
    before_index = output_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    def unexpected_build(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("an exact existing target must not be rebuilt")

    monkeypatch.setattr(preparation, "build_retrieval_index_stream", unexpected_build)
    second = preparation.prepare_atlas_retrieval_index(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        corpus_name="atlas-wikipedia",
        corpus_version="2018-12-20@pinned",
        expected_raw_sha256=pinned,
    )

    assert second == first
    assert output_path.read_bytes() == before_index
    assert manifest_path.read_bytes() == before_manifest
    with pytest.raises(ValueError):
        preparation.prepare_atlas_retrieval_index(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            corpus_name="atlas-wikipedia",
            corpus_version="different-version",
            expected_raw_sha256=pinned,
        )
    assert output_path.read_bytes() == before_index
    assert manifest_path.read_bytes() == before_manifest


def test_existing_index_byte_tamper_and_unpaired_target_are_rejected(tmp_path: Path) -> None:
    input_path, output_path, manifest_path = _paths(tmp_path)
    pinned = _write_source(input_path)
    preparation.prepare_atlas_retrieval_index(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        corpus_name="atlas-wikipedia",
        corpus_version="2018-12-20@pinned",
        expected_raw_sha256=pinned,
    )
    connection = sqlite3.connect(output_path)
    try:
        connection.execute(
            "UPDATE passages SET text = ? WHERE passage_id = ("
            "SELECT passage_id FROM passages ORDER BY passage_id LIMIT 1)",
            ("tampered public passage",),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError):
        preparation.prepare_atlas_retrieval_index(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            corpus_name="atlas-wikipedia",
            corpus_version="2018-12-20@pinned",
            expected_raw_sha256=pinned,
        )

    manifest_path.unlink()
    with pytest.raises(ValueError):
        preparation.prepare_atlas_retrieval_index(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            corpus_name="atlas-wikipedia",
            corpus_version="2018-12-20@pinned",
            expected_raw_sha256=pinned,
        )
