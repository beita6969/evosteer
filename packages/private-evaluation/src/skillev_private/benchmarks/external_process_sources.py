"""Materialize public task inventories for file-backed official process suites.

The source repositories remain private deployment assets.  This module reads
only model-visible task instructions and public metadata when it builds a
``RolloutTask`` inventory; solutions, tests, evaluator payloads, and outputs
never enter the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from skillev.contracts import JsonValue, canonical_json, stable_hash
from skillev.experiments import Benchmark
from skillev.rollout import RolloutTask

_BFCL_V3_OFFICIAL_CATEGORIES = (
    "exec_simple",
    "exec_parallel",
    "exec_multiple",
    "exec_parallel_multiple",
    "simple",
    "irrelevance",
    "parallel",
    "multiple",
    "parallel_multiple",
    "java",
    "javascript",
    "rest",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "multi_turn_composite",
)


@dataclass(frozen=True, slots=True)
class ExternalProcessSourceMaterialization:
    benchmark: Benchmark
    task_count: int
    task_manifest_relative_path: str
    asset_relative_files: tuple[str, ...]
    task_manifest_sha256: str


def _relative_source_path(value: object, *, field: str) -> PurePosixPath:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text without NUL")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{field} must be a normalized relative path")
    return path


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be non-empty text without NUL")
    return value


def _text_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise TypeError(f"{field} must be a text array")
    result = tuple(sorted({_text(item, field=field) for item in value}))
    return result


def _write_once_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError("existing external process manifest differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _repository_assets(
    *, repository_root: Path, target_root: Path, benchmark: Benchmark
) -> tuple[str, ...]:
    files = tuple(
        path
        for path in sorted(repository_root.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(repository_root).parts
    )
    if not files:
        raise ValueError("official process source repository contains no deployment assets")
    try:
        relative = tuple(path.relative_to(target_root).as_posix() for path in files)
    except ValueError as error:
        raise ValueError("official process repository must remain below target_root") from error
    prefix = f"{benchmark.value}/"
    if any(not item.startswith(prefix) for item in relative):
        raise ValueError("official process asset escaped its benchmark root")
    return relative


def _benchmark_assets(
    *, benchmark_root: Path, target_root: Path, benchmark: Benchmark
) -> tuple[str, ...]:
    files = tuple(
        path
        for path in sorted(benchmark_root.rglob("*"))
        if path.is_file()
        and ".git" not in path.relative_to(benchmark_root).parts
        and "prepared" not in path.relative_to(benchmark_root).parts
    )
    if not files:
        raise ValueError("official benchmark source contains no deployment assets")
    relative = tuple(path.relative_to(target_root).as_posix() for path in files)
    prefix = f"{benchmark.value}/"
    if any(not item.startswith(prefix) for item in relative):
        raise ValueError("official benchmark asset escaped its benchmark root")
    return relative


def materialize_skillflow_bench_source(
    *,
    repository_root: Path,
    target_root: Path,
    dataset_revision: str,
    task_manifest_relative_path: str = "skillflow-bench/prepared/tasks.jsonl",
) -> ExternalProcessSourceMaterialization:
    """Read all 166 official public instructions from the pinned snapshot."""

    if not repository_root.is_dir() or not target_root.is_dir():
        raise NotADirectoryError(repository_root if not repository_root.is_dir() else target_root)
    if len(dataset_revision) != 40 or any(
        character not in "0123456789abcdef" for character in dataset_revision
    ):
        raise ValueError("SkillFlow-Bench revision must be a full lowercase Git commit")
    manifest_path = _relative_source_path(
        task_manifest_relative_path, field="SkillFlow-Bench task manifest path"
    )
    if not manifest_path.as_posix().startswith("skillflow-bench/"):
        raise ValueError("SkillFlow-Bench task manifest must remain below its benchmark root")

    benchmark_manifest = json.loads(
        (repository_root / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(benchmark_manifest, dict)
        or type(benchmark_manifest.get("task_count")) is not int
    ):
        raise ValueError("SkillFlow-Bench benchmark manifest is incompatible")

    inventory_file = repository_root / "viewer" / "task_manifest.jsonl"
    rows: list[dict[str, object]] = []
    for line in inventory_file.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError("SkillFlow-Bench inventory record must be an object")
        rows.append(cast(dict[str, object], value))
    if len(rows) != benchmark_manifest["task_count"]:
        raise ValueError("SkillFlow-Bench inventory differs from its declared task count")

    tasks: list[RolloutTask] = []
    seen_paths: set[str] = set()
    for row in rows:
        task_path = _relative_source_path(row.get("task_path"), field="SkillFlow-Bench task_path")
        if task_path.as_posix() in seen_paths:
            raise ValueError("SkillFlow-Bench inventory repeats a task path")
        seen_paths.add(task_path.as_posix())
        family = _text(row.get("family"), field="SkillFlow-Bench family")
        task_name = _text(row.get("task_name"), field="SkillFlow-Bench task_name")
        if task_path.parts != (family, task_name):
            raise ValueError("SkillFlow-Bench task path differs from its public identity")
        task_root = repository_root / task_path
        query = _text(
            (task_root / "instruction.md").read_text(encoding="utf-8"),
            field="SkillFlow-Bench instruction",
        )
        environment = task_root / "environment"
        environment_files = tuple(
            path.relative_to(environment).as_posix()
            for path in sorted(environment.rglob("*"))
            if path.is_file()
        )
        if not environment_files:
            raise ValueError("SkillFlow-Bench task has no public environment files")
        source_id = task_path.as_posix()
        digest = stable_hash(
            {
                "benchmark": Benchmark.SKILLFLOW_BENCH.value,
                "dataset_revision": dataset_revision,
                "source_id": source_id,
            }
        ).removeprefix("sha256:")
        tasks.append(
            RolloutTask(
                task_id=f"{Benchmark.SKILLFLOW_BENCH.value}/{digest}",
                environment_id=(
                    f"{Benchmark.SKILLFLOW_BENCH.value}:{dataset_revision}:{source_id}"
                ),
                task_family=f"{Benchmark.SKILLFLOW_BENCH.value}/{family}",
                context_id=f"{Benchmark.SKILLFLOW_BENCH.value}:{family}:{task_name}",
                query=query,
                available_tools=("skillflow-bench.execute", "submit"),
                public_context={
                    "benchmark_id": Benchmark.SKILLFLOW_BENCH.value,
                    "category": _text(row.get("category"), field="SkillFlow-Bench category"),
                    "dataset_revision": dataset_revision,
                    "difficulty": _text(row.get("difficulty"), field="SkillFlow-Bench difficulty"),
                    "environment_files": list(environment_files),
                    "source_id": source_id,
                    "split": "train",
                    "tags": list(_text_tuple(row.get("tags"), field="SkillFlow-Bench tags")),
                    "tools": {
                        "skillflow-bench.execute": {"arguments": ["command"]},
                        "submit": {"arguments": ["submission"]},
                    },
                },
            )
        )
    tasks.sort(key=lambda task: task.task_id)
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("SkillFlow-Bench public task identities are not unique")
    payload = "".join(canonical_json(task.to_value()) + "\n" for task in tasks).encode()
    output = target_root / manifest_path
    _write_once_or_verify(output, payload)
    assets = _repository_assets(
        repository_root=repository_root,
        target_root=target_root,
        benchmark=Benchmark.SKILLFLOW_BENCH,
    )
    return ExternalProcessSourceMaterialization(
        benchmark=Benchmark.SKILLFLOW_BENCH,
        task_count=len(tasks),
        task_manifest_relative_path=manifest_path.as_posix(),
        asset_relative_files=assets,
        task_manifest_sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )


def materialize_tua_bench_source(
    *,
    repository_root: Path,
    target_root: Path,
    dataset_revision: str,
    task_manifest_relative_path: str = "tua-bench/prepared/tasks.jsonl",
) -> ExternalProcessSourceMaterialization:
    """Materialize every official TUA-Bench task without reading verifier data."""

    if not repository_root.is_dir() or not target_root.is_dir():
        raise NotADirectoryError(repository_root if not repository_root.is_dir() else target_root)
    if len(dataset_revision) != 40 or any(
        character not in "0123456789abcdef" for character in dataset_revision
    ):
        raise ValueError("TUA-Bench revision must be a full lowercase Git commit")
    manifest_path = _relative_source_path(
        task_manifest_relative_path, field="TUA-Bench task manifest path"
    )
    if not manifest_path.as_posix().startswith("tua-bench/"):
        raise ValueError("TUA-Bench task manifest must remain below its benchmark root")

    tasks_root = repository_root / "tasks"
    task_roots = tuple(
        path
        for path in sorted(tasks_root.iterdir())
        if path.is_dir() and (path / "instruction.md").is_file() and (path / "task.toml").is_file()
    )
    if not task_roots:
        raise ValueError("TUA-Bench repository contains no official tasks")
    tasks: list[RolloutTask] = []
    for task_root in task_roots:
        source_id = task_root.name
        metadata_wire = tomllib.loads((task_root / "task.toml").read_text(encoding="utf-8"))
        metadata = metadata_wire.get("metadata")
        task_metadata = metadata_wire.get("task")
        if not isinstance(metadata, dict) or not isinstance(task_metadata, dict):
            raise ValueError("TUA-Bench task metadata is missing")
        category = _text(metadata.get("category"), field="TUA-Bench category")
        difficulty = _text(metadata.get("difficulty"), field="TUA-Bench difficulty")
        task_name = _text(task_metadata.get("name"), field="TUA-Bench task name")
        if task_name != f"local/{source_id}":
            raise ValueError("TUA-Bench task name differs from its source directory")
        description = _text(task_metadata.get("description"), field="TUA-Bench task description")
        tags = _text_tuple(task_metadata.get("keywords"), field="TUA-Bench keywords")
        query = _text(
            (task_root / "instruction.md").read_text(encoding="utf-8"),
            field="TUA-Bench instruction",
        )
        public_files = tuple(
            path.relative_to(task_root).as_posix()
            for directory_name in ("environment", "input")
            for path in sorted((task_root / directory_name).rglob("*"))
            if path.is_file()
        )
        if not public_files:
            raise ValueError("TUA-Bench task has no public deployment files")
        digest = stable_hash(
            {
                "benchmark": Benchmark.TUA_BENCH.value,
                "dataset_revision": dataset_revision,
                "source_id": source_id,
            }
        ).removeprefix("sha256:")
        tasks.append(
            RolloutTask(
                task_id=f"{Benchmark.TUA_BENCH.value}/{digest}",
                environment_id=f"{Benchmark.TUA_BENCH.value}:{dataset_revision}:{source_id}",
                task_family=f"{Benchmark.TUA_BENCH.value}/{category}",
                context_id=f"{Benchmark.TUA_BENCH.value}:{source_id}",
                query=query,
                available_tools=("submit",),
                public_context={
                    "benchmark_id": Benchmark.TUA_BENCH.value,
                    "category": category,
                    "dataset_revision": dataset_revision,
                    "description": description,
                    "difficulty": difficulty,
                    "public_files": list(public_files),
                    "source_id": source_id,
                    "split": "test",
                    "tags": list(tags),
                    "tools": {"submit": {"arguments": ["submission"]}},
                },
            )
        )
    tasks.sort(key=lambda task: task.task_id)
    payload = "".join(canonical_json(task.to_value()) + "\n" for task in tasks).encode()
    _write_once_or_verify(target_root / manifest_path, payload)
    return ExternalProcessSourceMaterialization(
        benchmark=Benchmark.TUA_BENCH,
        task_count=len(tasks),
        task_manifest_relative_path=manifest_path.as_posix(),
        asset_relative_files=_repository_assets(
            repository_root=repository_root,
            target_root=target_root,
            benchmark=Benchmark.TUA_BENCH,
        ),
        task_manifest_sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )


def materialize_appworld_source(
    *,
    benchmark_root: Path,
    target_root: Path,
    dataset_revision: str,
    task_manifest_relative_path: str = "appworld/prepared/tasks.jsonl",
) -> ExternalProcessSourceMaterialization:
    """Materialize the official AppWorld train split from answer-free specs."""

    data_root = benchmark_root / "data"
    tasks_root = data_root / "tasks"
    split_file = data_root / "datasets" / "train.txt"
    if not tasks_root.is_dir() or not split_file.is_file() or not target_root.is_dir():
        raise ValueError("AppWorld official data bundle is not installed below benchmark_root")
    if len(dataset_revision) != 40 or any(
        character not in "0123456789abcdef" for character in dataset_revision
    ):
        raise ValueError("AppWorld revision must be a full lowercase Git commit")
    manifest_path = _relative_source_path(
        task_manifest_relative_path, field="AppWorld task manifest path"
    )
    if not manifest_path.as_posix().startswith("appworld/"):
        raise ValueError("AppWorld task manifest must remain below its benchmark root")
    source_ids = tuple(
        line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not source_ids or len(set(source_ids)) != len(source_ids):
        raise ValueError("AppWorld train split must contain unique task IDs")

    tasks: list[RolloutTask] = []
    for source_id in source_ids:
        _relative_source_path(source_id, field="AppWorld source task ID")
        specs = json.loads((tasks_root / source_id / "specs.json").read_text(encoding="utf-8"))
        if not isinstance(specs, dict):
            raise TypeError("AppWorld task specs must be an object")
        query = _text(specs.get("instruction"), field="AppWorld instruction")
        generator_id = source_id.rsplit("_", 1)[0]
        digest = stable_hash(
            {
                "benchmark": Benchmark.APPWORLD.value,
                "dataset_revision": dataset_revision,
                "source_id": source_id,
            }
        ).removeprefix("sha256:")
        tasks.append(
            RolloutTask(
                task_id=f"{Benchmark.APPWORLD.value}/{digest}",
                environment_id=f"{Benchmark.APPWORLD.value}:{dataset_revision}:{source_id}",
                task_family=f"{Benchmark.APPWORLD.value}/{generator_id}",
                context_id=f"{Benchmark.APPWORLD.value}:{generator_id}",
                query=query,
                available_tools=("appworld.execute", "submit"),
                public_context={
                    "benchmark_id": Benchmark.APPWORLD.value,
                    "dataset_revision": dataset_revision,
                    "generator_id": generator_id,
                    "source_id": source_id,
                    "split": "train",
                    "tools": {
                        "appworld.execute": {"arguments": ["app", "api", "arguments"]},
                        "submit": {"arguments": ["submission"]},
                    },
                },
            )
        )
    tasks.sort(key=lambda task: task.task_id)
    payload = "".join(canonical_json(task.to_value()) + "\n" for task in tasks).encode()
    _write_once_or_verify(target_root / manifest_path, payload)
    return ExternalProcessSourceMaterialization(
        benchmark=Benchmark.APPWORLD,
        task_count=len(tasks),
        task_manifest_relative_path=manifest_path.as_posix(),
        asset_relative_files=_benchmark_assets(
            benchmark_root=benchmark_root,
            target_root=target_root,
            benchmark=Benchmark.APPWORLD,
        ),
        task_manifest_sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )


def materialize_bfcl_v3_source(
    *,
    repository_root: Path,
    target_root: Path,
    dataset_revision: str,
    task_manifest_relative_path: str = "bfcl-v3/prepared/tasks.jsonl",
) -> ExternalProcessSourceMaterialization:
    """Materialize the official BFCL v3 ``all`` collection without answers."""

    data_root = repository_root / "berkeley-function-call-leaderboard" / "data"
    if not data_root.is_dir() or not target_root.is_dir():
        raise ValueError("BFCL v3 official data directory is not installed")
    if len(dataset_revision) != 40 or any(
        character not in "0123456789abcdef" for character in dataset_revision
    ):
        raise ValueError("BFCL v3 revision must be a full lowercase Git commit")
    manifest_path = _relative_source_path(
        task_manifest_relative_path, field="BFCL v3 task manifest path"
    )
    if not manifest_path.as_posix().startswith("bfcl-v3/"):
        raise ValueError("BFCL v3 task manifest must remain below its benchmark root")

    tasks: list[RolloutTask] = []
    source_ids: set[str] = set()
    for category in _BFCL_V3_OFFICIAL_CATEGORIES:
        source = data_root / f"BFCL_v3_{category}.json"
        if not source.is_file():
            raise FileNotFoundError(source)
        for line in source.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("BFCL v3 source row must be an object")
            source_id = _text(row.get("id"), field="BFCL v3 source ID")
            if source_id in source_ids:
                raise ValueError("BFCL v3 source task IDs must be unique")
            source_ids.add(source_id)
            question = row.get("question")
            functions = row.get("function")
            if not isinstance(question, list) or not question:
                raise TypeError("BFCL v3 question must be a non-empty turn array")
            if not isinstance(functions, list):
                raise TypeError("BFCL v3 functions must be an array")
            function_names: list[str] = []
            function_schemas: dict[str, JsonValue] = {}
            for function in functions:
                if not isinstance(function, dict):
                    raise TypeError("BFCL v3 function schema must be an object")
                function_name = _text(function.get("name"), field="BFCL v3 function name")
                function_names.append(function_name)
                function_schemas[function_name] = cast(JsonValue, function)
            if len(set(function_names)) != len(function_names):
                raise ValueError("BFCL v3 function names must be unique within a task")
            digest = stable_hash(
                {
                    "benchmark": Benchmark.BFCL_V3.value,
                    "dataset_revision": dataset_revision,
                    "source_id": source_id,
                }
            ).removeprefix("sha256:")
            tasks.append(
                RolloutTask(
                    task_id=f"{Benchmark.BFCL_V3.value}/{digest}",
                    environment_id=(f"{Benchmark.BFCL_V3.value}:{dataset_revision}:{source_id}"),
                    task_family=f"{Benchmark.BFCL_V3.value}/{category}",
                    context_id=f"{Benchmark.BFCL_V3.value}:{category}",
                    query=canonical_json(question),
                    available_tools=tuple(sorted(function_names)),
                    public_context={
                        "benchmark_id": Benchmark.BFCL_V3.value,
                        "category": category,
                        "dataset_revision": dataset_revision,
                        "function_schemas": cast(JsonValue, functions),
                        "source_id": source_id,
                        "split": "test",
                        "tools": function_schemas,
                    },
                )
            )
    tasks.sort(key=lambda task: task.task_id)
    payload = "".join(canonical_json(task.to_value()) + "\n" for task in tasks).encode()
    _write_once_or_verify(target_root / manifest_path, payload)
    return ExternalProcessSourceMaterialization(
        benchmark=Benchmark.BFCL_V3,
        task_count=len(tasks),
        task_manifest_relative_path=manifest_path.as_posix(),
        asset_relative_files=_repository_assets(
            repository_root=repository_root,
            target_root=target_root,
            benchmark=Benchmark.BFCL_V3,
        ),
        task_manifest_sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )


__all__ = [
    "ExternalProcessSourceMaterialization",
    "materialize_appworld_source",
    "materialize_bfcl_v3_source",
    "materialize_skillflow_bench_source",
    "materialize_tua_bench_source",
]
