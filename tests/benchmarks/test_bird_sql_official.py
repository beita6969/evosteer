from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from skillev_private.benchmarks.bird_sql_official import (
    BIRD_SQL_VERIFIER_VERSION,
    BirdSqlEvaluationInfrastructureError,
    BirdSqlOfficialWorker,
    load_bird_sql_session_factory,
)
from skillev_private.benchmarks.external_materialization import materialize_bird_sql

from skillev.contracts import canonical_json
from skillev.rollout import RolloutTask


def _split(root: Path, split: str, *, gold_sql: str) -> tuple[Path, Path]:
    prepared = root / "bird-sql" / "prepared" / split
    source = prepared / "source" / f"{split}.json"
    databases = prepared / "databases"
    database = databases / "school" / "school.sqlite"
    source.parent.mkdir(parents=True)
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE pupils(name TEXT NOT NULL, grade INTEGER NOT NULL)")
        connection.executemany(
            "INSERT INTO pupils VALUES (?, ?)",
            (("Ada", 5), ("Ben", 4), ("Cara", 5)),
        )
    source.write_text(
        canonical_json(
            [
                {
                    "SQL": gold_sql,
                    "db_id": "school",
                    "difficulty": "simple",
                    "evidence": "Grades are integers.",
                    "question": "Which pupils are in grade five?",
                }
            ]
        ),
        encoding="utf-8",
    )
    return source, databases


def _materialized(root: Path) -> tuple[tuple[RolloutTask, ...], Path, str]:
    train_json, train_databases = _split(
        root, "train", gold_sql="SELECT name FROM pupils WHERE grade = 5 ORDER BY name"
    )
    dev_json, dev_databases = _split(
        root, "dev", gold_sql="SELECT COUNT(*) FROM pupils WHERE grade = 5"
    )
    public = root / "_derived" / "bird-sql" / "tasks.jsonl"
    private = root / "bird-sql" / "private" / "cases.jsonl"
    identity = materialize_bird_sql(
        train_json=train_json,
        train_databases=train_databases,
        dev_json=dev_json,
        dev_databases=dev_databases,
        dataset_revision="fixture-revision",
        public_manifest=public,
        private_manifest=private,
    )
    tasks = tuple(
        RolloutTask.from_value(json.loads(line)["task"])
        for line in public.read_text(encoding="utf-8").splitlines()
    )
    assert identity.private_manifest_hash
    return tasks, private, "sha256:" + hashlib.sha256(private.read_bytes()).hexdigest()


def test_bird_worker_uses_official_execution_accuracy_without_private_output(
    tmp_path: Path,
) -> None:
    tasks, manifest, manifest_hash = _materialized(tmp_path)
    factory = load_bird_sql_session_factory(
        tasks=tasks,
        private_manifest=manifest,
        expected_private_manifest_sha256=manifest_hash,
        dataset_root=tmp_path,
    )
    worker = factory.worker

    # BIRD execution accuracy compares result sets: order and duplicates do not
    # change the score.
    correct = asyncio.run(
        worker.evaluate(
            task_id=tasks[0].task_id,
            submission=(
                "SELECT name FROM pupils WHERE grade = 5 "
                "UNION ALL SELECT name FROM pupils WHERE name = 'Ada' ORDER BY name DESC"
            ),
        )
    )
    incorrect = asyncio.run(
        worker.evaluate(task_id=tasks[0].task_id, submission="SELECT name FROM pupils")
    )

    assert correct.reward_value == 1.0
    assert correct.success is True
    assert correct.native_metric_name == "execution_accuracy"
    assert correct.public_metrics == {"execution_accuracy": 1.0}
    assert correct.verifier_version == BIRD_SQL_VERIFIER_VERSION
    assert incorrect.reward_value == 0.0
    assert incorrect.success is False
    result_wire = canonical_json(correct.public_metrics)
    assert "SELECT name" not in result_wire
    assert "Ada" not in result_wire


def test_bird_no_submission_is_zero_without_executing_candidate_sql(tmp_path: Path) -> None:
    tasks, manifest, manifest_hash = _materialized(tmp_path)
    worker = load_bird_sql_session_factory(
        tasks=tasks,
        private_manifest=manifest,
        expected_private_manifest_sha256=manifest_hash,
        dataset_root=tmp_path,
    ).worker

    result = worker.no_submission_result(task_id=tasks[0].task_id)

    assert result.reward_value == 0.0
    assert result.success is False
    assert result.native_metric_name == "execution_accuracy"


def test_bird_worker_treats_invalid_or_mutating_model_sql_as_failed_execution(
    tmp_path: Path,
) -> None:
    tasks, manifest, manifest_hash = _materialized(tmp_path)
    factory = load_bird_sql_session_factory(
        tasks=tasks,
        private_manifest=manifest,
        expected_private_manifest_sha256=manifest_hash,
        dataset_root=tmp_path,
    )

    malformed = asyncio.run(
        factory.worker.evaluate(task_id=tasks[0].task_id, submission="not sqlite syntax")
    )
    mutation = asyncio.run(
        factory.worker.evaluate(task_id=tasks[0].task_id, submission="DELETE FROM pupils")
    )

    assert malformed.reward_value == 0.0
    assert mutation.reward_value == 0.0
    database = tmp_path / "bird-sql/prepared/train/databases/school/school.sqlite"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM pupils").fetchone() == (3,)


def test_bird_factory_rejects_private_manifest_identity_mismatch(tmp_path: Path) -> None:
    tasks, manifest, _ = _materialized(tmp_path)

    with pytest.raises(ValueError):
        load_bird_sql_session_factory(
            tasks=tasks,
            private_manifest=manifest,
            expected_private_manifest_sha256="sha256:" + "0" * 64,
            dataset_root=tmp_path,
        )


def test_bird_worker_reports_broken_gold_as_infrastructure_failure(tmp_path: Path) -> None:
    tasks, manifest, _ = _materialized(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["gold_sql"] = "SELECT missing_private_column FROM pupils"
    manifest.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    factory = load_bird_sql_session_factory(
        tasks=tasks,
        private_manifest=manifest,
        expected_private_manifest_sha256=(
            "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
        ),
        dataset_root=tmp_path,
    )
    assert isinstance(factory.worker, BirdSqlOfficialWorker)

    with pytest.raises(BirdSqlEvaluationInfrastructureError):
        asyncio.run(
            factory.worker.evaluate(
                task_id=tasks[0].task_id,
                submission="SELECT name FROM pupils",
            )
        )
