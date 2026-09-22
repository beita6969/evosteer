from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

import skillev.benchmarks.retrieval as retrieval_module
from skillev.benchmarks.retrieval import (
    RETRIEVAL_BACKEND,
    DocumentPassage,
    QABenchmark,
    QARetrievalEnvironment,
    RetrievalBuildPhase,
    RetrievalBuildProgress,
    RetrievalIndex,
    RetrievalIndexManifest,
    SearchHit,
    audit_retrieval_index_content,
    build_retrieval_index,
    build_retrieval_index_stream,
)
from skillev.contracts import JsonValue, canonical_json
from skillev.runtime import (
    ActionKind,
    BudgetVector,
    EnvironmentMethodFailedError,
    EnvironmentObservation,
    StructuredAction,
)

PRIVATE_ANSWER_CANARY = "PRIVATE-GOLD-ANSWER-DO-NOT-EXPOSE"


def _passages() -> tuple[DocumentPassage, ...]:
    return (
        DocumentPassage(
            passage_id="passage-alpha",
            document_id="document-alpha",
            title="Mercury notes",
            text="Mercury follows an orbit around the Sun.",
        ),
        DocumentPassage(
            passage_id="passage-beta",
            document_id="document-beta",
            title="Venus notes",
            text="Venus follows an orbit around the Sun.",
        ),
        DocumentPassage(
            passage_id="tie-a",
            document_id="document-tie-a",
            title="Identical record",
            text="A quasar emits bright light.",
        ),
        DocumentPassage(
            passage_id="tie-b",
            document_id="document-tie-b",
            title="Identical record",
            text="A quasar emits bright light.",
        ),
    )


def _build(
    path: Path,
    passages: tuple[DocumentPassage, ...] | None = None,
) -> RetrievalIndexManifest:
    return build_retrieval_index(
        path,
        _passages() if passages is None else passages,
        corpus_name="public-qa-fixture",
        corpus_version="fixture@1",
    )


def _environment(index: RetrievalIndex) -> QARetrievalEnvironment:
    return QARetrievalEnvironment(
        index=index,
        benchmark=QABenchmark.HOTPOT_QA,
        dataset_revision="fixture@1",
        task_family="qa/multi-hop",
        resource_id="qa-retrieval",
    )


def _tool(
    name: str,
    arguments: JsonValue,
    *,
    resource_id: str = "qa-retrieval",
) -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.TOOL,
        name=name,
        arguments=arguments,
        resource_id=resource_id,
    )


def _skill(skill_id: str = "skill-public-search") -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.SKILL,
        name="apply-skill",
        arguments={},
        resource_id="skill-runtime",
        skill_id=skill_id,
    )


def _execute(
    environment: QARetrievalEnvironment,
    action: StructuredAction,
    *,
    step_index: int,
) -> EnvironmentObservation:
    return asyncio.run(
        environment.execute(
            action,
            step_index=step_index,
        )
    )


def test_retrieval_records_have_canonical_round_trips() -> None:
    passage = _passages()[0]
    hit = SearchHit(
        passage_id=passage.passage_id,
        document_id=passage.document_id,
        title=passage.title,
        snippet=passage.text,
        rank=1,
    )

    assert DocumentPassage.from_value(passage.to_value()) == passage
    assert SearchHit.from_value(hit.to_value()) == hit
    assert passage.content_hash == DocumentPassage.from_value(passage.to_value()).content_hash
    assert hit.content_hash == SearchHit.from_value(hit.to_value()).content_hash


def test_content_addressed_index_is_independent_of_input_order_and_path(tmp_path: Path) -> None:
    first_path = tmp_path / "first.sqlite"
    second_path = tmp_path / "second.sqlite"

    first_manifest = _build(first_path)
    second_manifest = _build(second_path, tuple(reversed(_passages())))

    assert isinstance(first_manifest, RetrievalIndexManifest)
    assert first_manifest == second_manifest
    assert RetrievalIndexManifest.from_value(first_manifest.to_value()) == first_manifest
    assert first_manifest.content_hash == second_manifest.content_hash
    assert first_manifest.index_id == second_manifest.index_id
    assert first_manifest.retrieval_backend == RETRIEVAL_BACKEND
    assert first_manifest.to_value()["retrieval_backend"] == "sqlite-fts5-lexical"
    with RetrievalIndex.open(first_path) as first, RetrievalIndex.open(second_path) as second:
        assert first.manifest == second.manifest
        assert first.search("orbit sun", limit=10) == second.search("orbit sun", limit=10)


def test_streaming_builder_consumes_once_and_preserves_canonical_identity(tmp_path: Path) -> None:
    class OneShotCorpus:
        def __init__(self) -> None:
            self.iterations = 0

        def __iter__(self) -> Iterator[DocumentPassage]:
            self.iterations += 1
            if self.iterations != 1:
                raise AssertionError("streaming corpus was traversed more than once")
            yield from reversed(_passages())

    corpus = OneShotCorpus()
    streamed_path = tmp_path / "streamed.sqlite"
    tuple_path = tmp_path / "tuple.sqlite"
    streamed = build_retrieval_index_stream(
        streamed_path,
        corpus,
        corpus_name="public-qa-fixture",
        corpus_version="fixture@1",
        insert_batch_size=2,
    )
    expected = _build(tuple_path)

    assert corpus.iterations == 1
    assert streamed == expected
    with RetrievalIndex.open(streamed_path) as index:
        assert index.manifest == expected
        assert index.read("passage-alpha") == _passages()[0]


def test_ephemeral_builder_applies_bulk_construction_pragmas(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "pragmas.sqlite")
    try:
        retrieval_module._configure_ephemeral_index_build(connection)

        assert connection.execute("PRAGMA journal_mode").fetchone() == ("off",)
        assert connection.execute("PRAGMA synchronous").fetchone() == (0,)
        assert connection.execute("PRAGMA cache_size").fetchone() == (-(2 * 1024 * 1024),)
        assert connection.execute("PRAGMA temp_store").fetchone() == (2,)
        assert connection.execute("PRAGMA locking_mode").fetchone() == ("exclusive",)
    finally:
        connection.close()


def test_explicit_staging_copy_publishes_from_target_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_directory = tmp_path / "node-local"
    target = tmp_path / "shared" / "retrieval.sqlite"
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    monkeypatch.setattr(retrieval_module, "_same_filesystem", lambda _source, _target: False)

    def record_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(retrieval_module.os, "replace", record_replace)
    manifest = build_retrieval_index_stream(
        target,
        _passages(),
        corpus_name="public-qa-fixture",
        corpus_version="fixture@1",
        staging_directory=staging_directory,
    )

    assert replacements
    publication_source, publication_target = replacements[-1]
    assert publication_source.parent == target.parent
    assert publication_target == target
    assert not tuple(staging_directory.iterdir())
    with RetrievalIndex.open(target) as index:
        assert index.manifest == manifest


def test_cross_filesystem_copy_failure_never_publishes_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_directory = tmp_path / "node-local"
    target = tmp_path / "shared" / "retrieval.sqlite"
    monkeypatch.setattr(retrieval_module, "_same_filesystem", lambda _source, _target: False)

    def interrupted_copy(_source: Path, destination: Path) -> None:
        destination.write_bytes(b"partial")
        raise OSError("injected copy interruption")

    monkeypatch.setattr(retrieval_module, "_copy_fsync", interrupted_copy)
    with pytest.raises(OSError):
        build_retrieval_index_stream(
            target,
            _passages(),
            corpus_name="public-qa-fixture",
            corpus_version="fixture@1",
            staging_directory=staging_directory,
        )

    assert not target.exists()
    assert not tuple(staging_directory.iterdir())
    assert not tuple(target.parent.iterdir())


def test_explicit_staging_does_not_change_deterministic_index(
    tmp_path: Path,
) -> None:
    default_target = tmp_path / "default.sqlite"
    staged_target = tmp_path / "published" / "staged.sqlite"
    staging_directory = tmp_path / "node-local"
    passages = tuple(reversed(_passages()))

    default_manifest = build_retrieval_index_stream(
        default_target,
        passages,
        corpus_name="public-qa-fixture",
        corpus_version="fixture@1",
        insert_batch_size=2,
    )
    staged_manifest = build_retrieval_index_stream(
        staged_target,
        passages,
        corpus_name="public-qa-fixture",
        corpus_version="fixture@1",
        insert_batch_size=2,
        staging_directory=staging_directory,
    )

    assert staged_manifest == default_manifest
    assert staged_target.read_bytes() == default_target.read_bytes()


def test_builder_uses_explicit_monotone_source_rowids_and_deferred_unique_index(
    tmp_path: Path,
) -> None:
    target = tmp_path / "source-rowids.sqlite"
    passages = tuple(
        DocumentPassage(
            passage_id=f"passage-{source_rowid}",
            document_id=f"document-{source_rowid}",
            title=f"Title {source_rowid}",
            text=f"Public text {source_rowid}.",
            source_rowid=source_rowid,
        )
        for source_rowid in (9, 10)
    )

    build_retrieval_index_stream(
        target,
        passages,
        corpus_name="public-qa-fixture",
        corpus_version="fixture@1",
        insert_batch_size=1,
    )

    connection = sqlite3.connect(target)
    try:
        assert connection.execute("SELECT rowid FROM passages ORDER BY rowid").fetchall() == [
            (9,),
            (10,),
        ]
        table_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'passages'"
        ).fetchone()
        unique_index = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'passages_passage_id_uq'"
        ).fetchone()
    finally:
        connection.close()

    assert table_sql is not None
    assert "passage_id TEXT NOT NULL UNIQUE" not in table_sql[0]
    assert unique_index is not None
    assert "UNIQUE INDEX" in unique_index[0]


def test_canonical_source_order_hashes_during_ingestion_without_second_corpus_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "single-pass-hash.sqlite"
    passages = tuple(
        DocumentPassage(
            passage_id=f"atlas-dpr-wikipedia:{source_rowid:012d}",
            document_id=f"document-{source_rowid}",
            title=f"Title {source_rowid}",
            text=f"Public text {source_rowid}.",
            source_rowid=source_rowid,
        )
        for source_rowid in (9, 10)
    )

    def unexpected_second_read(_rows: object) -> object:
        raise AssertionError("canonical Atlas source order must hash during ingestion")

    monkeypatch.setattr(retrieval_module, "_passages_from_rows", unexpected_second_read)
    manifest = build_retrieval_index_stream(
        target,
        passages,
        corpus_name="atlas-wikipedia",
        corpus_version="fixture@1",
    )

    assert manifest.passage_count == 2


def test_builder_reports_ingest_and_fts_rates_with_eta(tmp_path: Path) -> None:
    target = tmp_path / "progress.sqlite"
    progress: list[RetrievalBuildProgress] = []

    build_retrieval_index_stream(
        target,
        _passages(),
        corpus_name="public-qa-fixture",
        corpus_version="fixture@1",
        insert_batch_size=2,
        expected_passage_count=4,
        progress_callback=progress.append,
        progress_interval_rows=2,
    )

    assert [item.phase for item in progress] == [
        RetrievalBuildPhase.INGEST,
        RetrievalBuildPhase.INGEST,
        RetrievalBuildPhase.UNIQUE_INDEX,
        RetrievalBuildPhase.FTS_INDEX,
        RetrievalBuildPhase.FTS_INDEX,
        RetrievalBuildPhase.CORPUS_HASH,
        RetrievalBuildPhase.VERIFY,
        RetrievalBuildPhase.PUBLISH,
        RetrievalBuildPhase.COMPLETE,
    ]
    first_ingest = progress[0]
    first_fts = next(item for item in progress if item.phase is RetrievalBuildPhase.FTS_INDEX)
    assert first_ingest.rows_completed == 2
    assert first_ingest.rows_per_second > 0.0
    assert first_ingest.eta_seconds is not None
    assert first_ingest.eta_seconds > 0.0
    assert first_fts.rows_completed == 2
    assert first_fts.rows_per_second > 0.0
    assert first_fts.eta_seconds is not None
    assert first_fts.eta_seconds > 0.0
    assert progress[-1].rows_completed == 4
    assert progress[-1].eta_seconds is None


def test_expected_passage_count_mismatch_never_publishes(tmp_path: Path) -> None:
    target = tmp_path / "count-mismatch.sqlite"

    with pytest.raises(ValueError):
        build_retrieval_index_stream(
            target,
            _passages(),
            corpus_name="public-qa-fixture",
            corpus_version="fixture@1",
            expected_passage_count=5,
        )

    assert not target.exists()


def test_streaming_builder_rejects_duplicate_passage_ids_atomically(tmp_path: Path) -> None:
    target = tmp_path / "streamed.sqlite"
    duplicate = DocumentPassage(
        passage_id="passage-alpha",
        document_id="another-document",
        title="Duplicate",
        text="A distinct row cannot reuse the passage identity.",
    )

    with pytest.raises(ValueError):
        build_retrieval_index_stream(
            target,
            (*_passages(), duplicate),
            corpus_name="public-qa-fixture",
            corpus_version="fixture@1",
            insert_batch_size=2,
        )

    assert not target.exists()


def test_index_uses_sqlite_fts5_instead_of_a_scan_fallback(tmp_path: Path) -> None:
    index_path = tmp_path / "retrieval.sqlite"
    _build(index_path)

    connection = sqlite3.connect(index_path)
    definition = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'passage_fts'",
    ).fetchone()
    connection.close()

    assert definition is not None
    assert "fts5" in definition[0].lower()


def test_search_and_read_are_deterministic_and_use_stable_tie_breaking(tmp_path: Path) -> None:
    index_path = tmp_path / "retrieval.sqlite"
    manifest = _build(index_path)

    with RetrievalIndex.open(index_path) as index:
        assert index.manifest == manifest
        first = index.search("quasar bright", limit=10)
        second = index.search("quasar bright", limit=10)
        passage = index.read("passage-alpha")

    assert first == second
    assert [hit.passage_id for hit in first] == ["tie-a", "tie-b"]
    assert [hit.rank for hit in first] == [1, 2]
    assert passage == _passages()[0]


@pytest.mark.parametrize(
    "benchmark",
    [
        QABenchmark.HOTPOT_QA,
        QABenchmark.TRIVIA_QA,
        QABenchmark.NATURAL_QUESTIONS,
    ],
)
def test_environment_identity_covers_the_three_retrieval_benchmarks(
    tmp_path: Path,
    benchmark: QABenchmark,
) -> None:
    index_path = tmp_path / "retrieval.sqlite"
    _build(index_path)
    with RetrievalIndex.open(index_path) as index:
        environment = QARetrievalEnvironment(
            index=index,
            benchmark=benchmark,
            dataset_revision="fixture@1",
            task_family="qa/retrieval",
            resource_id="qa-retrieval",
        )

    assert benchmark.value in environment.environment_id


def test_index_is_opened_read_only(tmp_path: Path) -> None:
    index_path = tmp_path / "retrieval.sqlite"
    _build(index_path)

    with RetrievalIndex.open(index_path) as index:
        assert index._connection.execute("PRAGMA query_only").fetchone() == (1,)
        with pytest.raises(sqlite3.OperationalError):
            index._connection.execute("DELETE FROM passages")


def test_atomic_replace_failure_preserves_the_previous_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path = tmp_path / "retrieval.sqlite"
    old_manifest = _build(index_path)
    replacement = (
        DocumentPassage(
            passage_id="new-passage",
            document_id="new-document",
            title="Replacement",
            text="This content must never become visible.",
        ),
    )

    real_replace = os.replace

    def fail_target_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        if Path(target) == index_path:
            raise OSError("injected atomic replacement failure")
        real_replace(source, target)

    monkeypatch.setattr(retrieval_module.os, "replace", fail_target_replace)
    with pytest.raises(OSError):
        _build(index_path, replacement)

    with RetrievalIndex.open(index_path) as index:
        assert index.manifest == old_manifest
        assert index.read("passage-alpha") == _passages()[0]
        with pytest.raises(KeyError):
            index.read("new-passage")


def test_content_tampering_is_rejected_by_explicit_offline_audit(tmp_path: Path) -> None:
    index_path = tmp_path / "retrieval.sqlite"
    _build(index_path)
    connection = sqlite3.connect(index_path)
    connection.execute(
        "UPDATE passages SET text = ? WHERE passage_id = ?",
        ("tampered public text", "passage-alpha"),
    )
    connection.commit()
    connection.close()

    with pytest.raises(ValueError):
        audit_retrieval_index_content(index_path)


def test_runtime_open_does_not_rehash_or_integrity_scan_full_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path = tmp_path / "retrieval.sqlite"
    manifest = _build(index_path)

    def unexpected_full_hash(_rows: object) -> object:
        raise AssertionError("runtime open must not rescan the full corpus")

    monkeypatch.setattr(retrieval_module, "_hash_ordered_passages", unexpected_full_hash)
    with RetrievalIndex.open(index_path) as index:
        assert index.manifest == manifest


def test_manifest_tampering_is_rejected_when_index_is_opened(tmp_path: Path) -> None:
    index_path = tmp_path / "retrieval.sqlite"
    _build(index_path)
    connection = sqlite3.connect(index_path)
    connection.execute("UPDATE retrieval_manifest SET manifest_json = ?", ("{}",))
    connection.commit()
    connection.close()

    with pytest.raises((TypeError, ValueError)):
        RetrievalIndex.open(index_path)


def test_environment_search_read_and_skill_are_public_and_metered(tmp_path: Path) -> None:
    index_path = tmp_path / "retrieval.sqlite"
    _build(index_path)
    with RetrievalIndex.open(index_path) as index:
        environment = _environment(index)
        search = _execute(
            environment,
            _tool("search", {"limit": 2, "query": "orbit"}),
            step_index=1,
        )
        read = _execute(
            environment,
            _tool("read", {"passage_id": "passage-alpha"}),
            step_index=2,
        )
        skill = _execute(
            environment,
            _skill(),
            step_index=3,
        )

    assert search.budget_usage == BudgetVector(tool_calls=1)
    assert read.budget_usage == BudgetVector(tool_calls=1)
    assert skill.budget_usage == BudgetVector(tool_calls=1)
    assert search.observation_status == "success"
    assert read.observation_status == "success"
    assert skill.invoked_skill_ids == ("skill-public-search",)
    public_wire = canonical_json(
        {
            "read": read.to_value(),
            "search": search.to_value(),
            "skill": skill.to_value(),
        }
    )
    assert PRIVATE_ANSWER_CANARY not in public_wire
    assert "gold_answer" not in public_wire
    assert "native_payload" not in public_wire


@pytest.mark.parametrize(
    "submission",
    [
        None,
        {},
        {"answer": ""},
        {"answer": "   "},
        {"answer": 42},
        {"answer": "public prediction", "extra": True},
    ],
)
def test_environment_completion_shape_is_exact(
    tmp_path: Path,
    submission: JsonValue,
) -> None:
    index_path = tmp_path / "retrieval.sqlite"
    _build(index_path)
    with RetrievalIndex.open(index_path) as index:
        environment = _environment(index)
        assert not environment.validate_completion(submission)
        assert environment.validate_completion({"answer": "public prediction"})


@pytest.mark.parametrize(
    "action",
    [
        _tool("unknown", {}),
        _tool("search", {"query": "orbit"}, resource_id="wrong-resource"),
        _tool("search", {}),
        _tool("search", {"limit": 0, "query": "orbit"}),
        _tool("read", {"passage_id": "missing-passage"}),
        StructuredAction(kind=ActionKind.COMPLETE, name="complete", arguments={"answer": "x"}),
    ],
)
def test_environment_rejects_every_action_outside_search_read_and_skill(
    tmp_path: Path,
    action: StructuredAction,
) -> None:
    index_path = tmp_path / "retrieval.sqlite"
    _build(index_path)
    with RetrievalIndex.open(index_path) as index:
        environment = _environment(index)
        with pytest.raises(EnvironmentMethodFailedError) as captured:
            _execute(environment, action, step_index=1)

    assert captured.value.budget_usage == BudgetVector(tool_calls=1)
