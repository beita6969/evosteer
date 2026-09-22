"""Deterministic preparation of BIRD, EvalPlus, and TableBench task manifests.

The public JSONL files contain only model-visible task projections.  Gold SQL,
tests, canonical solutions, and table answers are written to separate private
worker manifests and are never returned by this module's public receipt.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from skillev.contracts import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.experiments import Benchmark
from skillev.rollout import RolloutTask

EXTERNAL_MATERIALIZATION_FORMAT = "skillev-private-external-materialization@1"


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if type(value) is not str or "\x00" in value or (not allow_empty and not value.strip()):
        raise ValueError(f"{field} must be text without NUL")
    return value


def _source_identifier(value: object, *, field: str) -> str:
    """Normalize official identifiers without weakening free-text validation."""

    if type(value) is int:
        return str(value)
    return _text(value, field=field)


def _read_json_array(path: Path) -> tuple[dict[str, object], ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not list or not value or any(type(row) is not dict for row in value):
        raise ValueError("official JSON source must be a non-empty object array")
    return tuple(cast(dict[str, object], row) for row in value)


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("official JSONL source contains an empty record")
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError("official JSONL record must be an object")
            rows.append(cast(dict[str, object], value))
    if not rows:
        raise ValueError("official JSONL source must contain records")
    return tuple(rows)


def _read_parquet(path: Path) -> tuple[dict[str, object], ...]:
    import pyarrow.parquet as parquet  # type: ignore[import-untyped]

    rows = parquet.read_table(path).to_pylist()
    if not rows or any(type(row) is not dict for row in rows):
        raise ValueError("official EvalPlus source must contain object rows")
    return tuple(cast(dict[str, object], row) for row in rows)


def _sqlite_schema(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    # BIRD ships databases whose persistent journal mode is WAL.  A plain
    # read-only SQLite URI still creates ``-wal``/``-shm`` sidecars for those
    # inputs, mutating the archive tree that B0 has already admitted.  Treat
    # the pinned source database as immutable so schema projection is truly
    # read-only.
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type IN ('table','view') AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    schema = tuple(_text(row[0], field="BIRD schema SQL") for row in rows)
    if not schema:
        raise ValueError("BIRD database has no public schema")
    return schema


def _task_id(benchmark: Benchmark, source_id: str) -> str:
    digest = stable_hash({"benchmark": benchmark.value, "source_id": source_id}).removeprefix(
        "sha256:"
    )
    return f"{benchmark.value}/{digest}"


def _completion_task(
    *,
    benchmark: Benchmark,
    revision: str,
    split: str,
    source_id: str,
    query: str,
    task_family_suffix: str,
    public_context: dict[str, JsonValue],
) -> RolloutTask:
    task_id = _task_id(benchmark, source_id)
    return RolloutTask(
        task_id=task_id,
        environment_id=f"{benchmark.value}:{revision}:{split}",
        task_family=f"{benchmark.value}/{task_family_suffix}",
        context_id=f"{benchmark.value}:{split}:{source_id}",
        query=query,
        available_tools=(),
        public_context={
            "benchmark_id": benchmark.value,
            "dataset_revision": revision,
            "source_id": source_id,
            "split": split,
            **public_context,
        },
    )


def _write_once_or_verify(path: Path, records: Iterable[dict[str, JsonValue]]) -> str:
    payload = "".join(canonical_json(record) + "\n" for record in records).encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("existing prepared manifest differs from deterministic output")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    return stable_hash(payload.decode("utf-8"))


@dataclass(frozen=True, slots=True)
class MaterializedBenchmarkIdentity:
    benchmark: Benchmark
    task_count: int
    public_manifest_hash: str
    private_manifest_hash: str

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "private_manifest_hash": self.private_manifest_hash,
            "public_manifest_hash": self.public_manifest_hash,
            "task_count": self.task_count,
        }


def materialize_bird_sql(
    *,
    train_json: Path,
    train_databases: Path,
    dev_json: Path,
    dev_databases: Path,
    dataset_revision: str,
    public_manifest: Path,
    private_manifest: Path,
) -> MaterializedBenchmarkIdentity:
    public: list[dict[str, JsonValue]] = []
    private: list[dict[str, JsonValue]] = []
    for split, source, databases in (
        ("train", train_json, train_databases),
        ("dev", dev_json, dev_databases),
    ):
        for ordinal, row in enumerate(_read_json_array(source)):
            question = _text(row.get("question"), field="BIRD question")
            gold_sql = _text(row.get("SQL"), field="BIRD gold SQL")
            db_id = _text(row.get("db_id"), field="BIRD db_id")
            evidence = _text(row.get("evidence", ""), field="BIRD evidence", allow_empty=True)
            difficulty = _text(row.get("difficulty", "unspecified"), field="BIRD difficulty")
            database = databases / db_id / f"{db_id}.sqlite"
            schema = _sqlite_schema(database)
            source_id = f"{split}:{ordinal}:{db_id}"
            task = _completion_task(
                benchmark=Benchmark.BIRD_SQL,
                revision=dataset_revision,
                split=split,
                source_id=source_id,
                query=(
                    "Write one SQLite query that answers the question. Return only the SQL.\n\n"
                    f"Question: {question}\nEvidence: {evidence}\nSchema:\n" + "\n\n".join(schema)
                ),
                task_family_suffix=f"text-to-sql/{difficulty.lower()}",
                public_context={
                    "database_id": db_id,
                    "difficulty": difficulty,
                    "evidence": evidence,
                    "schema": list(schema),
                },
            )
            public.append({"source_split": "train+dev", "task": task.to_value()})
            private.append(
                {
                    "database_relative_path": database.relative_to(
                        train_json.parents[1] if split == "train" else dev_json.parents[1]
                    ).as_posix(),
                    "gold_sql": gold_sql,
                    "source_split": split,
                    "task_id": task.task_id,
                }
            )
    return MaterializedBenchmarkIdentity(
        Benchmark.BIRD_SQL,
        len(public),
        _write_once_or_verify(public_manifest, public),
        _write_once_or_verify(private_manifest, private),
    )


def materialize_evalplus(
    *,
    benchmark: Benchmark,
    source_parquet: Path,
    dataset_revision: str,
    public_manifest: Path,
    private_manifest: Path,
) -> MaterializedBenchmarkIdentity:
    if benchmark not in {Benchmark.MBPP_PLUS, Benchmark.HUMANEVAL_PLUS}:
        raise ValueError("EvalPlus materialization supports MBPP+ and HumanEval+ only")
    public: list[dict[str, JsonValue]] = []
    private: list[dict[str, JsonValue]] = []
    for ordinal, row in enumerate(_read_parquet(source_parquet)):
        source_id = _source_identifier(row.get("task_id", str(ordinal)), field="EvalPlus task_id")
        prompt = _text(row.get("prompt"), field="EvalPlus prompt")
        task = _completion_task(
            benchmark=benchmark,
            revision=dataset_revision,
            split="test",
            source_id=source_id,
            query=prompt,
            task_family_suffix="python/code-generation",
            public_context={"language": "python"},
        )
        public.append({"source_split": "test", "task": task.to_value()})
        private.append(
            {
                "official_row": cast(JsonValue, normalize_json(row)),
                "task_id": task.task_id,
            }
        )
    return MaterializedBenchmarkIdentity(
        benchmark,
        len(public),
        _write_once_or_verify(public_manifest, public),
        _write_once_or_verify(private_manifest, private),
    )


def materialize_tablebench(
    *,
    source_jsonl: Path,
    dataset_revision: str,
    public_manifest: Path,
    private_manifest: Path,
) -> MaterializedBenchmarkIdentity:
    public: list[dict[str, JsonValue]] = []
    private: list[dict[str, JsonValue]] = []
    rows = _read_jsonl(source_jsonl)
    official_ids = tuple(
        _text(row.get("id", str(ordinal)), field="TableBench id")
        for ordinal, row in enumerate(rows)
    )
    id_counts = Counter(official_ids)
    for ordinal, (row, official_id) in enumerate(zip(rows, official_ids, strict=True)):
        # TableBench contains a repeated official row ID. Preserve the official
        # identity while binding repeated rows to their stable source ordinal.
        source_id = official_id if id_counts[official_id] == 1 else f"{official_id}#row-{ordinal}"
        question = _text(row.get("question"), field="TableBench question")
        qtype = _text(row.get("qtype"), field="TableBench qtype")
        qsubtype = _text(row.get("qsubtype"), field="TableBench qsubtype")
        table = normalize_json(row.get("table"))
        if not isinstance(table, dict):
            raise ValueError("TableBench table must be an object")
        if qtype == "Visualization":
            chart_type = _text(row.get("chart_type"), field="TableBench chart_type")
            response_instruction = (
                "The table is available to your program as table.csv. Return exactly one "
                "Python code block that creates the requested chart with matplotlib."
            )
            chart_context: dict[str, JsonValue] = {"chart_type": chart_type}
        else:
            response_instruction = (
                "End the response with one line in the exact form `Final Answer: <answer>`."
            )
            chart_context = {}
        task = _completion_task(
            benchmark=Benchmark.TABLEBENCH,
            revision=dataset_revision,
            split="test",
            source_id=source_id,
            query=(
                "Answer the question using the supplied table. "
                f"{response_instruction}\n\nQuestion: {question}"
            ),
            task_family_suffix=f"{qtype.lower()}/{qsubtype.lower()}",
            public_context={
                "qsubtype": qsubtype,
                "qtype": qtype,
                "table": table,
                **chart_context,
            },
        )
        public.append({"source_split": "test", "task": task.to_value()})
        private.append(
            {
                "answer": cast(JsonValue, normalize_json(row.get("answer"))),
                "official_row": cast(JsonValue, normalize_json(row)),
                "task_id": task.task_id,
            }
        )
    return MaterializedBenchmarkIdentity(
        Benchmark.TABLEBENCH,
        len(public),
        _write_once_or_verify(public_manifest, public),
        _write_once_or_verify(private_manifest, private),
    )


__all__ = [
    "EXTERNAL_MATERIALIZATION_FORMAT",
    "MaterializedBenchmarkIdentity",
    "materialize_bird_sql",
    "materialize_evalplus",
    "materialize_tablebench",
]
