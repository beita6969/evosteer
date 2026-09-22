from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from skillev_private.benchmarks.external_materialization import (
    _sqlite_schema,
    materialize_bird_sql,
    materialize_evalplus,
    materialize_tablebench,
)

from skillev.contracts import canonical_json
from skillev.experiments import Benchmark


def test_bird_schema_projection_does_not_create_wal_sidecars(tmp_path: Path) -> None:
    database = tmp_path / "wal-source.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE records(value TEXT NOT NULL)")
        connection.commit()
    finally:
        connection.close()

    assert tuple(path.name for path in tmp_path.iterdir()) == (database.name,)

    assert _sqlite_schema(database) == ("CREATE TABLE records(value TEXT NOT NULL)",)
    assert tuple(path.name for path in tmp_path.iterdir()) == (database.name,)


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())


def _bird_split(root: Path, split: str, *, answer: str) -> tuple[Path, Path]:
    directory = root / split
    databases = directory / f"{split}_databases"
    database = databases / "school" / "school.sqlite"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE pupils(name TEXT NOT NULL, grade INTEGER NOT NULL)")
    source = directory / f"{split}.json"
    source.write_text(
        canonical_json(
            [
                {
                    "SQL": answer,
                    "db_id": "school",
                    "difficulty": "simple",
                    "evidence": "Grades are integers.",
                    "question": "How many pupils are recorded?",
                }
            ]
        ),
        encoding="utf-8",
    )
    return source, databases


def test_bird_materialization_keeps_gold_sql_out_of_public_manifest(tmp_path: Path) -> None:
    train_json, train_databases = _bird_split(
        tmp_path, "train", answer="SELECT COUNT(*) FROM pupils"
    )
    dev_json, dev_databases = _bird_split(tmp_path, "dev", answer="SELECT COUNT(name) FROM pupils")
    public = tmp_path / "derived" / "bird-public.jsonl"
    private = tmp_path / "private" / "bird-cases.jsonl"

    identity = materialize_bird_sql(
        train_json=train_json,
        train_databases=train_databases,
        dev_json=dev_json,
        dev_databases=dev_databases,
        dataset_revision="fixture-revision",
        public_manifest=public,
        private_manifest=private,
    )

    assert identity.benchmark is Benchmark.BIRD_SQL
    assert identity.task_count == 2
    public_wire = public.read_text(encoding="utf-8")
    assert "SELECT COUNT(*) FROM pupils" not in public_wire
    assert "SELECT COUNT(name) FROM pupils" not in public_wire
    assert "CREATE TABLE pupils" in public_wire
    assert {row["source_split"] for row in _jsonl(private)} == {"train", "dev"}


def test_evalplus_materialization_keeps_tests_and_solutions_private(tmp_path: Path) -> None:
    source = tmp_path / "mbpp.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": 1,
                    "prompt": "Write a function named add_one.",
                    "canonical_solution": "def add_one(x): return x + 1",
                    "test": "assert add_one(1) == 2",
                }
            ]
        ),
        source,
    )
    public = tmp_path / "derived" / "mbpp-public.jsonl"
    private = tmp_path / "private" / "mbpp-cases.jsonl"

    identity = materialize_evalplus(
        benchmark=Benchmark.MBPP_PLUS,
        source_parquet=source,
        dataset_revision="fixture-revision",
        public_manifest=public,
        private_manifest=private,
    )

    assert identity.task_count == 1
    public_wire = public.read_text(encoding="utf-8")
    assert "canonical_solution" not in public_wire
    assert "assert add_one" not in public_wire
    assert "canonical_solution" in private.read_text(encoding="utf-8")


def test_tablebench_materialization_keeps_answer_private(tmp_path: Path) -> None:
    source = tmp_path / "tablebench.jsonl"
    source.write_text(
        canonical_json(
            {
                "answer": "private-answer-canary",
                "id": "table-1",
                "qsubtype": "lookup",
                "qtype": "fact",
                "question": "Which city has population 2?",
                "table": {"columns": ["city", "population"], "rows": [["A", 2]]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    public = tmp_path / "derived" / "tablebench-public.jsonl"
    private = tmp_path / "private" / "tablebench-cases.jsonl"

    identity = materialize_tablebench(
        source_jsonl=source,
        dataset_revision="fixture-revision",
        public_manifest=public,
        private_manifest=private,
    )

    assert identity.task_count == 1
    assert "private-answer-canary" not in public.read_text(encoding="utf-8")
    assert "private-answer-canary" in private.read_text(encoding="utf-8")


def test_tablebench_materialization_disambiguates_repeated_official_ids(tmp_path: Path) -> None:
    source = tmp_path / "tablebench.jsonl"
    rows = (
        {
            "answer": "first-private-answer",
            "id": "repeated-id",
            "qsubtype": "lookup",
            "qtype": "fact",
            "question": "Which city has population 2?",
            "table": {"columns": ["city", "population"], "rows": [["A", 2]]},
        },
        {
            "answer": "second-private-answer",
            "id": "repeated-id",
            "qsubtype": "lookup",
            "qtype": "fact",
            "question": "Which city has population 3?",
            "table": {"columns": ["city", "population"], "rows": [["B", 3]]},
        },
    )
    source.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    public = tmp_path / "derived" / "tablebench-public.jsonl"
    private = tmp_path / "private" / "tablebench-cases.jsonl"

    identity = materialize_tablebench(
        source_jsonl=source,
        dataset_revision="fixture-revision",
        public_manifest=public,
        private_manifest=private,
    )

    task_ids = tuple(
        json.loads(line)["task"]["task_id"] for line in public.read_text().splitlines()
    )
    assert identity.task_count == 2
    assert len(set(task_ids)) == 2
