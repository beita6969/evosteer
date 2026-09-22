from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from skillev_private.benchmarks.acquisition import (
    AcquiredArtifactReceipt,
    BenchmarkAcquisitionLock,
    BenchmarkAcquisitionReceipt,
    load_benchmark_acquisition_lock,
)
from skillev_private.benchmarks.archive_preparation import (
    LockedArchiveBatchReceipt,
    PreparedLockedArchive,
    locked_archive_preparations,
)
from skillev_private.benchmarks.non_process_preparation import (
    load_locked_non_process_preparation_receipt,
    locked_non_process_source_plans,
    prepare_locked_non_process_benchmarks,
    publish_locked_non_process_preparation_receipt,
)

from skillev.experiments import Benchmark


def _lock() -> BenchmarkAcquisitionLock:
    return load_benchmark_acquisition_lock(
        Path(__file__).parents[2] / "benchmark-acquisition-lock.json"
    )


def _write(path: Path, payload: bytes = b"fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _atlas_inputs(root: Path) -> None:
    index_root = root / "triviaqa/prepared/dataindex"
    index_root.mkdir(parents=True)
    for benchmark in ("NQ", "TQA"):
        for split in ("train", "dev", "test"):
            (index_root / f"{benchmark}.{split}.idx.json").write_text(
                "[0]\n",
                encoding="utf-8",
            )
    trivia_root = root / "triviaqa/prepared/triviaqa-unfiltered/triviaqa-unfiltered"
    trivia_root.mkdir(parents=True)
    trivia = {
        "Data": [
            {
                "Answer": {"Aliases": ["fixture answer"], "Value": "fixture answer"},
                "Question": "fixture question",
            }
        ]
    }
    for split in ("train", "dev"):
        (trivia_root / f"unfiltered-web-{split}.json").write_text(
            json.dumps(trivia),
            encoding="utf-8",
        )
    nq = b'{"answer":["fixture answer"],"question":"fixture question"}\n'
    _write(root / "nq-open/raw/NQ-open.train.jsonl", nq)
    _write(root / "nq-open/raw/NQ-open.dev.jsonl", nq)


def _medqa_inputs(root: Path) -> None:
    source_root = root / "medqa/prepared/data_clean/data_clean/questions/US/4_options"
    row = {
        "answer": "option a",
        "answer_idx": "A",
        "meta_info": "step1",
        "metamap_phrases": ["auxiliary annotation"],
        "options": {
            "A": "option a",
            "B": "option b",
            "C": "option c",
            "D": "option d",
        },
        "question": "fixture question",
    }
    for split in ("dev", "test", "train"):
        _write(
            source_root / f"phrases_no_exclude_{split}.jsonl",
            (json.dumps(row) + "\n").encode(),
        )


def _external_inputs(root: Path) -> None:
    for split in ("train", "dev"):
        split_root = root / f"bird-sql/prepared/{split}/{split}"
        split_root.mkdir(parents=True, exist_ok=True)
        database = root / f"_fixture-bird/{split}/fixture/fixture.sqlite"
        database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE items (value INTEGER)")
            connection.execute("INSERT INTO items VALUES (1)")
        database_directory = f"{split}_databases"
        with zipfile.ZipFile(split_root / f"{database_directory}.zip", "w") as archive:
            archive.write(
                database,
                f"{database_directory}/fixture/fixture.sqlite",
            )
            archive.writestr(
                f"__MACOSX/{database_directory}/fixture/._fixture.sqlite",
                b"metadata",
            )
        (split_root / f"{split}.json").write_text(
            json.dumps(
                [
                    {
                        "SQL": "SELECT value FROM items",
                        "db_id": "fixture",
                        "difficulty": "simple",
                        "evidence": "",
                        "question": "Return the fixture value.",
                    }
                ]
            ),
            encoding="utf-8",
        )
    for benchmark in ("mbpp-plus", "humaneval-plus"):
        path = root / benchmark / "data/test.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "canonical_solution": "def answer(): return 1",
                        "prompt": "Write a function that returns one.",
                        "task_id": f"{benchmark}-fixture",
                        "test": "assert answer() == 1",
                    }
                ]
            ),
            path,
        )
    _write(
        root / "tablebench/raw/TableBench.jsonl",
        (
            json.dumps(
                {
                    "answer": "1",
                    "id": "table-fixture",
                    "qsubtype": "Aggregation",
                    "qtype": "NumericalReasoning",
                    "question": "What is the value?",
                    "table": {"columns": ["value"], "data": [["1"]]},
                }
            )
            + "\n"
        ).encode(),
    )


def _receipt_chain(
    lock: BenchmarkAcquisitionLock,
    root: Path,
) -> tuple[BenchmarkAcquisitionReceipt, LockedArchiveBatchReceipt]:
    acquired = BenchmarkAcquisitionReceipt(
        acquisition_lock_hash=lock.content_hash,
        target_root=root,
        artifacts=tuple(
            AcquiredArtifactReceipt(
                benchmark=entry.benchmark,
                locator=artifact.locator,
                relative_path=artifact.relative_path,
                size_bytes=1,
                sha256="a" * 64,
            )
            for entry in lock.benchmarks
            for artifact in entry.source.artifacts
        ),
    )
    archive = LockedArchiveBatchReceipt(
        acquisition_lock_hash=lock.content_hash,
        acquisition_receipt_hash=acquired.content_hash,
        archives=tuple(
            PreparedLockedArchive(
                benchmark=plan.benchmark,
                artifact_relative_path=plan.artifact_relative_path,
                output_relative_path=plan.output_relative_path,
                preparation_content_hash=f"sha256:{'b' * 64}",
            )
            for plan in locked_archive_preparations(lock)
        ),
    )
    return acquired, archive


def _prepared_fixture(
    tmp_path: Path,
) -> tuple[
    BenchmarkAcquisitionLock,
    BenchmarkAcquisitionReceipt,
    LockedArchiveBatchReceipt,
    Path,
]:
    lock = _lock()
    root = (tmp_path / "datasets").resolve()
    root.mkdir()
    _atlas_inputs(root)
    _medqa_inputs(root)
    _external_inputs(root)
    for plan in locked_non_process_source_plans(lock):
        for relative_path in (*plan.relative_files, *plan.auxiliary_files):
            if relative_path.startswith("_derived/"):
                continue
            path = root / relative_path
            if not path.is_file():
                _write(path)
    acquired, archive = _receipt_chain(lock, root)
    return lock, acquired, archive, root


def test_declares_all_exact_non_process_loader_inputs() -> None:
    plans = locked_non_process_source_plans(_lock())

    assert tuple(plan.benchmark for plan in plans) == (
        Benchmark.HOTPOT_QA,
        Benchmark.TRIVIA_QA,
        Benchmark.AIME_2026,
        Benchmark.MED_QA,
        Benchmark.BIRD_SQL,
        Benchmark.MBPP_PLUS,
        Benchmark.MUSIQUE,
        Benchmark.NQ_OPEN,
        Benchmark.MATH_HARD,
        Benchmark.GPQA_DIAMOND,
        Benchmark.MIND2WEB,
        Benchmark.TABLEBENCH,
        Benchmark.HUMANEVAL_PLUS,
    )
    medqa = next(plan for plan in plans if plan.benchmark is Benchmark.MED_QA)
    assert medqa.relative_file_splits == ("dev", "test", "train")
    assert medqa.relative_files == (
        "_derived/medqa-us-four-option/dev.jsonl",
        "_derived/medqa-us-four-option/test.jsonl",
        "_derived/medqa-us-four-option/train.jsonl",
    )
    mind2web = next(plan for plan in plans if plan.benchmark is Benchmark.MIND2WEB)
    assert len(mind2web.relative_files) == 15
    assert mind2web.relative_file_splits == (
        *(("test_domain",) * 10),
        *(("test_task",) * 3),
        *(("test_website",) * 2),
    )
    assert mind2web.auxiliary_files == ("mind2web/raw/scores_all_data.pkl",)
    assert mind2web.freeze_spec().auxiliary_files == mind2web.auxiliary_files


def test_prepares_atlas_sources_and_publishes_complete_receipt(
    tmp_path: Path,
) -> None:
    lock, acquired, archive, root = _prepared_fixture(tmp_path)

    first = prepare_locked_non_process_benchmarks(
        lock=lock,
        acquisition_receipt=acquired,
        archive_batch_receipt=archive,
        target_root=root,
    )
    second = prepare_locked_non_process_benchmarks(
        lock=lock,
        acquisition_receipt=acquired,
        archive_batch_receipt=archive,
        target_root=root,
    )

    assert first == second
    assert len(first.freeze_specs()) == 13
    assert (root / "_derived/atlas-qa/triviaqa_data/dev.jsonl").is_file()
    assert (root / "_derived/atlas-qa/nq_data/dev.jsonl").is_file()
    assert (root / "bird-sql/prepared/train/_nested-databases/preparation.manifest.json").is_file()
    assert (root / "bird-sql/prepared/dev/_nested-databases/preparation.manifest.json").is_file()
    receipt_path = (tmp_path / "non-process-receipt.json").resolve()
    publish_locked_non_process_preparation_receipt(first, receipt_path)
    assert load_locked_non_process_preparation_receipt(receipt_path) == first
    original = receipt_path.read_bytes()
    with pytest.raises(FileExistsError):
        publish_locked_non_process_preparation_receipt(second, receipt_path)
    assert receipt_path.read_bytes() == original


def test_resume_rejects_changed_atlas_output(tmp_path: Path) -> None:
    lock, acquired, archive, root = _prepared_fixture(tmp_path)
    prepare_locked_non_process_benchmarks(
        lock=lock,
        acquisition_receipt=acquired,
        archive_batch_receipt=archive,
        target_root=root,
    )
    output = root / "_derived/atlas-qa/nq_data/dev.jsonl"
    output.write_bytes(output.read_bytes() + b'{"changed":true}\n')

    with pytest.raises(ValueError):
        prepare_locked_non_process_benchmarks(
            lock=lock,
            acquisition_receipt=acquired,
            archive_batch_receipt=archive,
            target_root=root,
        )


def test_requires_the_complete_acquisition_receipt_order(tmp_path: Path) -> None:
    lock, acquired, archive, root = _prepared_fixture(tmp_path)
    incomplete = BenchmarkAcquisitionReceipt(
        acquisition_lock_hash=acquired.acquisition_lock_hash,
        target_root=acquired.target_root,
        artifacts=acquired.artifacts[:-1],
    )
    incompatible_archive = LockedArchiveBatchReceipt(
        acquisition_lock_hash=archive.acquisition_lock_hash,
        acquisition_receipt_hash=incomplete.content_hash,
        archives=archive.archives,
    )

    with pytest.raises(ValueError):
        prepare_locked_non_process_benchmarks(
            lock=lock,
            acquisition_receipt=incomplete,
            archive_batch_receipt=incompatible_archive,
            target_root=root,
        )
