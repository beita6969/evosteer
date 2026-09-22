"""Strict private JSONL ingestion for completion-style benchmark snapshots.

The JSONL itself belongs in an ignored/private run directory.  This loader
returns split public/private case objects and a content hash; it never copies a
dataset row into the source tree or a public serialization format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skillev.benchmarks import BenchmarkPublicItem
from skillev.contracts import JsonValue, canonical_json, parse_canonical_json, stable_hash
from skillev.experiments import FIXED_SEED, DatasetSnapshotIdentity, fixed_subsample_rank

from .static import PrivateStaticBenchmarkCase, PrivateStaticTarget, StaticScoringRule

_SNAPSHOT_FORMAT = "skillev-private-static-jsonl@1"
_FIELDS = {
    "accepted_answers",
    "benchmark_id",
    "dataset_revision",
    "public_context",
    "query",
    "scoring_rule",
    "split",
    "task_family",
    "task_id",
}


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class PrivateStaticSnapshot:
    """Loaded private cases plus the only safe identity that may be committed."""

    cases: tuple[PrivateStaticBenchmarkCase, ...]
    identity: DatasetSnapshotIdentity

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("private static snapshot cannot be empty")
        if len({case.public.task_id for case in self.cases}) != len(self.cases):
            raise ValueError("private static snapshot task identities must be unique")

    def public_items(self) -> tuple[BenchmarkPublicItem, ...]:
        return tuple(case.public for case in self.cases)


def load_private_static_jsonl(
    path: str | Path,
    *,
    snapshot_name: str,
    snapshot_version: str,
) -> PrivateStaticSnapshot:
    """Load an exact canonical JSONL snapshot and split each private case."""

    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("private benchmark JSONL must contain non-empty canonical records")
    cases: list[PrivateStaticBenchmarkCase] = []
    line_hashes: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        value = parse_canonical_json(line)
        if not isinstance(value, dict) or set(value) != _FIELDS:
            raise ValueError(f"private benchmark JSONL line {line_number} has invalid fields")
        if canonical_json(value) != line:
            raise ValueError(f"private benchmark JSONL line {line_number} is not canonical")
        cases.append(_case_from_value(value, line_number=line_number))
        line_hashes.append(stable_hash(value))
    task_ids = tuple(case.public.task_id for case in cases)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("private benchmark JSONL contains duplicate task identities")
    snapshot_hash = stable_hash(
        {
            "format": _SNAPSHOT_FORMAT,
            "line_hashes": line_hashes,
            "record_count": len(cases),
        }
    )
    return PrivateStaticSnapshot(
        cases=tuple(cases),
        identity=DatasetSnapshotIdentity(
            name=_text(snapshot_name, field_name="snapshot_name"),
            version=_text(snapshot_version, field_name="snapshot_version"),
            snapshot_hash=snapshot_hash,
        ),
    )


def fixed_result_blind_subsample(
    snapshot: PrivateStaticSnapshot,
    sample_size: int,
) -> PrivateStaticSnapshot:
    """Select by task-identity hash before any evaluator result is observed."""

    if type(sample_size) is not int or not 0 < sample_size <= len(snapshot.cases):
        raise ValueError("sample_size must lie in [1, snapshot size]")
    benchmark_ids = {case.public.benchmark_id for case in snapshot.cases}
    if len(benchmark_ids) != 1:
        raise ValueError("a private snapshot must contain exactly one benchmark identity")
    benchmark_id = next(iter(benchmark_ids))
    ranked = tuple(
        sorted(
            snapshot.cases,
            key=lambda case: (
                fixed_subsample_rank(
                    benchmark_id,
                    case.public.task_id,
                ),
                case.public.task_id,
            ),
        )
    )
    selected = ranked[:sample_size]
    selection_hash = stable_hash(
        {
            "format": "skillev-private-static-subsample@1",
            "parent_snapshot_hash": snapshot.identity.snapshot_hash,
            "sample_size": sample_size,
            "seed": FIXED_SEED,
            "selected_task_ids": [case.public.task_id for case in selected],
        }
    )
    return PrivateStaticSnapshot(
        cases=selected,
        identity=DatasetSnapshotIdentity(
            name=snapshot.identity.name,
            version=f"{snapshot.identity.version}/fixed-{sample_size}",
            snapshot_hash=selection_hash,
        ),
    )


def _case_from_value(
    value: dict[str, JsonValue],
    *,
    line_number: int,
) -> PrivateStaticBenchmarkCase:
    answers = value["accepted_answers"]
    if type(answers) is not list:
        raise ValueError(f"private benchmark JSONL line {line_number} answers must be an array")
    accepted_answers = tuple(
        _text(answer, field_name=f"line {line_number} accepted answer") for answer in answers
    )
    public = BenchmarkPublicItem(
        benchmark_id=_text(value["benchmark_id"], field_name="benchmark_id"),
        dataset_revision=_text(value["dataset_revision"], field_name="dataset_revision"),
        split=_text(value["split"], field_name="split"),
        task_id=_text(value["task_id"], field_name="task_id"),
        task_family=_text(value["task_family"], field_name="task_family"),
        query=_text(value["query"], field_name="query"),
        public_context=value["public_context"],
    )
    scoring_rule = StaticScoringRule(_text(value["scoring_rule"], field_name="scoring_rule"))
    return PrivateStaticBenchmarkCase(
        public=public,
        target=PrivateStaticTarget(
            task_id=public.task_id,
            scoring_rule=scoring_rule,
            accepted_answers=accepted_answers,
        ),
    )


__all__ = [
    "PrivateStaticSnapshot",
    "fixed_result_blind_subsample",
    "load_private_static_jsonl",
]
