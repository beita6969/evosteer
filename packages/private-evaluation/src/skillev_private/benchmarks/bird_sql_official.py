"""Private BIRD-SQL execution-accuracy worker.

The public BIRD task contains the question, evidence, and database schema.  This
module is the sealed side of that boundary: it alone loads the materialized
gold SQL and opens the official database.  Evaluation follows BIRD's SQLite
execution-accuracy rule by comparing the two result sets without exposing
either result or the gold query to the rollout process.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import ClassVar, cast

from skillev.contracts import JsonValue, validate_sha256
from skillev.experiments import Benchmark
from skillev.rollout import RolloutTask

from .external_completion import (
    ExternalCompletionInfrastructureError,
    ExternalCompletionResult,
    ExternalCompletionSessionFactory,
)

BIRD_SQL_VERIFIER_VERSION = "bird-sql-execution-accuracy@1"


class BirdSqlEvaluationInfrastructureError(ExternalCompletionInfrastructureError):
    """The frozen private case or its official SQLite database is unusable."""


@dataclass(frozen=True, slots=True)
class BirdSqlEvaluationCase:
    """One private BIRD evaluator case loaded from the sealed manifest."""

    task_id: str
    source_split: str
    database_path: Path
    gold_sql: str

    def __post_init__(self) -> None:
        for field_name in ("task_id", "source_split", "gold_sql"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip() or "\x00" in value:
                raise ValueError(f"BIRD {field_name} must be non-empty text without NUL")
        if not isinstance(self.database_path, Path) or not self.database_path.is_absolute():
            raise ValueError("BIRD database_path must be absolute")
        if not self.database_path.is_file():
            raise FileNotFoundError(self.database_path)


def _manifest_object(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or set(value) != {
        "database_relative_path",
        "gold_sql",
        "source_split",
        "task_id",
    }:
        raise ValueError("BIRD private manifest record has an incompatible field set")
    if any(type(key) is not str for key in value):
        raise TypeError("BIRD private manifest keys must be text")
    return cast(dict[str, JsonValue], value)


def _manifest_text(value: JsonValue, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"BIRD private manifest {field} must be non-empty text without NUL")
    return value


def _database_path(root: Path, relative: str) -> Path:
    wire_path = PurePosixPath(relative)
    if wire_path.is_absolute() or ".." in wire_path.parts or wire_path.as_posix() != relative:
        raise ValueError("BIRD database_relative_path must be normalized relative POSIX text")
    resolved_root = root.resolve(strict=True)
    resolved = (resolved_root / Path(*wire_path.parts)).resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("BIRD database path escapes its split root") from error
    return resolved


def load_bird_sql_cases(
    *,
    private_manifest: Path,
    split_database_roots: tuple[tuple[str, Path], ...],
) -> tuple[BirdSqlEvaluationCase, ...]:
    """Load the answer-bearing manifest using explicit per-split database roots.

    ``materialize_bird_sql`` stores database paths relative to the prepared root
    for the corresponding source split.  Keeping the root mapping explicit
    avoids path discovery and lets deployment mount train and dev independently.
    """

    if not private_manifest.is_file():
        raise FileNotFoundError(private_manifest)
    roots = dict(split_database_roots)
    if len(roots) != len(split_database_roots) or not roots:
        raise ValueError("BIRD split database roots must be non-empty and unique")
    if any(type(split) is not str or not split.strip() for split in roots):
        raise ValueError("BIRD split database root names must be non-empty text")

    cases: list[BirdSqlEvaluationCase] = []
    with private_manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("BIRD private manifest contains an empty record")
            value = _manifest_object(json.loads(line))
            split = _manifest_text(value["source_split"], field="source_split")
            if split not in roots:
                raise ValueError("BIRD private manifest references an unmounted source split")
            relative = _manifest_text(
                value["database_relative_path"], field="database_relative_path"
            )
            cases.append(
                BirdSqlEvaluationCase(
                    task_id=_manifest_text(value["task_id"], field="task_id"),
                    source_split=split,
                    database_path=_database_path(roots[split], relative),
                    gold_sql=_manifest_text(value["gold_sql"], field="gold_sql"),
                )
            )
    if not cases or len({case.task_id for case in cases}) != len(cases):
        raise ValueError("BIRD private cases must be non-empty and task-unique")
    return tuple(cases)


def load_bird_sql_session_factory(
    *,
    tasks: tuple[RolloutTask, ...],
    private_manifest: Path,
    expected_private_manifest_sha256: str,
    dataset_root: Path,
    query_timeout_seconds: float = 30.0,
    verifier_version: str = BIRD_SQL_VERIFIER_VERSION,
) -> ExternalCompletionSessionFactory:
    """Construct the production session factory from frozen private inputs.

    ``dataset_root`` is the same exact root used by the production catalog.
    BIRD's materializer publishes database paths relative to the matching
    ``bird-sql/prepared/{source_split}`` root; each record carries that split
    without disclosing its SQL to any public task object.
    """

    validate_sha256(expected_private_manifest_sha256)
    manifest_bytes = private_manifest.read_bytes()
    manifest_sha256 = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != expected_private_manifest_sha256:
        raise ValueError("BIRD private manifest differs from its frozen identity")
    manifest_wire = manifest_bytes.decode("utf-8")
    if not isinstance(dataset_root, Path) or not dataset_root.is_absolute():
        raise ValueError("BIRD dataset_root must be an absolute Path")
    splits: set[str] = set()
    for line in manifest_wire.splitlines():
        if not line:
            raise ValueError("BIRD private manifest contains an empty record")
        value = _manifest_object(json.loads(line))
        splits.add(_manifest_text(value["source_split"], field="source_split"))
    split_database_roots = tuple(
        (split, dataset_root / Benchmark.BIRD_SQL.value / "prepared" / split)
        for split in sorted(splits)
    )
    worker = BirdSqlOfficialWorker.from_manifest(
        private_manifest=private_manifest,
        split_database_roots=split_database_roots,
        query_timeout_seconds=query_timeout_seconds,
        verifier_version=verifier_version,
    )
    private_task_ids = {case.task_id for case in worker.cases}
    if any(task.task_id not in private_task_ids for task in tasks):
        raise ValueError("BIRD public task has no frozen private evaluation case")
    return ExternalCompletionSessionFactory(tasks=tasks, worker=worker)


def _execute_sql(
    database_path: Path,
    sql: str,
    *,
    timeout_seconds: float,
) -> tuple[tuple[object, ...], ...]:
    deadline = time.monotonic() + timeout_seconds
    uri = database_path.as_uri() + "?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1_000)
        cursor = connection.execute(sql)
        return tuple(tuple(row) for row in cursor.fetchall())


def _official_results_equal(
    predicted: tuple[tuple[object, ...], ...],
    gold: tuple[tuple[object, ...], ...],
) -> bool:
    """Apply BIRD's official order- and duplicate-insensitive result comparison."""

    return set(predicted) == set(gold)


@dataclass(frozen=True, slots=True)
class BirdSqlOfficialWorker:
    """Evaluate raw model SQL against the sealed BIRD SQLite cases."""

    benchmark_id: ClassVar[str] = Benchmark.BIRD_SQL.value

    cases: tuple[BirdSqlEvaluationCase, ...]
    query_timeout_seconds: float = 30.0
    verifier_version: str = BIRD_SQL_VERIFIER_VERSION

    def __post_init__(self) -> None:
        if not self.cases or len({case.task_id for case in self.cases}) != len(self.cases):
            raise ValueError("BIRD worker cases must be non-empty and task-unique")
        if (
            not isinstance(self.query_timeout_seconds, float)
            or not math.isfinite(self.query_timeout_seconds)
            or self.query_timeout_seconds <= 0.0
        ):
            raise ValueError("BIRD query timeout must be a positive finite float")
        if type(self.verifier_version) is not str or not self.verifier_version.strip():
            raise ValueError("BIRD verifier_version must be non-empty text")

    @classmethod
    def from_manifest(
        cls,
        *,
        private_manifest: Path,
        split_database_roots: tuple[tuple[str, Path], ...],
        query_timeout_seconds: float = 30.0,
        verifier_version: str = BIRD_SQL_VERIFIER_VERSION,
    ) -> BirdSqlOfficialWorker:
        return cls(
            cases=load_bird_sql_cases(
                private_manifest=private_manifest,
                split_database_roots=split_database_roots,
            ),
            query_timeout_seconds=query_timeout_seconds,
            verifier_version=verifier_version,
        )

    async def evaluate(self, *, task_id: str, submission: str) -> ExternalCompletionResult:
        matches = tuple(case for case in self.cases if case.task_id == task_id)
        if len(matches) != 1:
            raise BirdSqlEvaluationInfrastructureError(
                "BIRD task is absent from the frozen private manifest"
            )
        # BIRD is a historical (non-Protocol-10) evaluator.  Its SQLite
        # progress handler already provides the bounded execution boundary;
        # keeping the call on the owning event-loop thread also avoids WSL
        # SQLite handles crossing the default executor boundary.
        return self._evaluate_case(matches[0], submission)

    def no_submission_result(self, *, task_id: str) -> ExternalCompletionResult:
        matches = tuple(case for case in self.cases if case.task_id == task_id)
        if len(matches) != 1:
            raise BirdSqlEvaluationInfrastructureError(
                "BIRD task is absent from the frozen private manifest"
            )
        return ExternalCompletionResult(
            reward_value=0.0,
            success=False,
            native_metric_name="execution_accuracy",
            public_metrics={
                "execution_accuracy": 0.0,
                "no_submission_reason": "horizon-exhausted",
            },
            verifier_version=self.verifier_version,
        )

    def _evaluate_case(
        self,
        case: BirdSqlEvaluationCase,
        submission: str,
    ) -> ExternalCompletionResult:
        try:
            gold = _execute_sql(
                case.database_path,
                case.gold_sql,
                timeout_seconds=self.query_timeout_seconds,
            )
        except sqlite3.Error as error:
            raise BirdSqlEvaluationInfrastructureError(
                "BIRD official database or gold query could not be evaluated"
            ) from error

        try:
            predicted = _execute_sql(
                case.database_path,
                submission,
                timeout_seconds=self.query_timeout_seconds,
            )
        except sqlite3.Error:
            execution_accuracy = 0.0
        else:
            execution_accuracy = float(_official_results_equal(predicted, gold))
        return ExternalCompletionResult(
            reward_value=execution_accuracy,
            success=execution_accuracy == 1.0,
            native_metric_name="execution_accuracy",
            public_metrics={"execution_accuracy": execution_accuracy},
            verifier_version=self.verifier_version,
        )


__all__ = [
    "BIRD_SQL_VERIFIER_VERSION",
    "BirdSqlEvaluationCase",
    "BirdSqlEvaluationInfrastructureError",
    "BirdSqlOfficialWorker",
    "load_bird_sql_cases",
    "load_bird_sql_session_factory",
]
