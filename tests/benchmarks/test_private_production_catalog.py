from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from skillev_private.benchmarks.converters import GPQA_DIAMOND_CSV_HEADER
from skillev_private.benchmarks.production_catalog import (
    ProductionBenchmarkSource,
    ProductionCatalogConfig,
    ProductionCatalogDependencies,
    ProductionSourceFormat,
    PyArrowParquetRowReader,
    load_production_benchmark_catalog,
)
from skillev_private.benchmarks.snapshot import create_private_dataset_snapshot

from skillev.benchmarks import DocumentPassage, build_retrieval_index
from skillev.contracts import canonical_json
from skillev.experiments import Benchmark
from skillev.rollout import RolloutTask
from skillev.training import RolloutSessionBundle

PRIVATE_CANARY = "PRIVATE-PRODUCTION-CATALOG-CANARY"
REVISION = "synthetic-pinned-revision@1"


@dataclass(frozen=True, slots=True)
class _ParquetReader:
    rows_by_name: dict[str, tuple[dict[str, object], ...]]

    def read_rows(self, path: Path) -> tuple[dict[str, object], ...]:
        return self.rows_by_name[path.name]


@dataclass(frozen=True, slots=True)
class _ExternalFactory:
    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        del task
        raise AssertionError("external session construction is outside catalog loading")


@dataclass(frozen=True, slots=True)
class _ExternalFactoryBuilder:
    def build(self, tasks: tuple[RolloutTask, ...]) -> _ExternalFactory:
        assert len(tasks) == 1
        return _ExternalFactory()


def _hotpot_row() -> dict[str, object]:
    return {
        "id": "hotpot-fixture",
        "question": "Which public entity connects the documents?",
        "answer": PRIVATE_CANARY,
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 0]},
        "context": {
            "title": ["A", "B"],
            "sentences": [["Public A."], ["Public B."]],
        },
    }


def _medqa_row() -> dict[str, object]:
    return {
        "question": "Which public option is correct?",
        "answer": "Bravo",
        "options": {"A": "Alpha", "B": "Bravo", "C": "Charlie", "D": "Delta"},
        "meta_info": "USMLE",
        "answer_idx": "B",
    }


def _musique_row() -> dict[str, object]:
    return {
        "id": "2hop__1_2",
        "paragraphs": [
            {"idx": 0, "title": "A", "paragraph_text": "Public A.", "is_supporting": True},
            {"idx": 1, "title": "B", "paragraph_text": "Public B.", "is_supporting": True},
        ],
        "question": "What is the public multi-hop answer?",
        "question_decomposition": [
            {"id": 1, "question": "private q1", "answer": "private a1", "paragraph_support_idx": 0},
            {
                "id": 2,
                "question": "private q2",
                "answer": PRIVATE_CANARY,
                "paragraph_support_idx": 1,
            },
        ],
        "answer": PRIVATE_CANARY,
        "answer_aliases": ["private alias"],
        "answerable": True,
    }


def _gpqa_row() -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(GPQA_DIAMOND_CSV_HEADER, "")
    row.update(
        {
            "Question": "Which public scientific option follows?",
            "Correct Answer": "Choice one",
            "Incorrect Answer 1": "Choice two",
            "Incorrect Answer 2": "Choice three",
            "Incorrect Answer 3": "Choice four",
            "Explanation": f"private explanation {PRIVATE_CANARY}",
            "Record ID": "gpqa-fixture",
            "High-level domain": "Physics",
            "Subdomain": "Mechanics",
            "Canary String": PRIVATE_CANARY,
        }
    )
    return row


def _mind2web_row() -> dict[str, object]:
    return {
        "annotation_id": "mind-fixture",
        "website": "public.example",
        "domain": "shopping",
        "subdomain": "retail",
        "confirmed_task": "Click the requested public element.",
        "action_reprs": [f"private action {PRIVATE_CANARY}"],
        "actions": [
            {
                "action_uid": "action-1",
                "raw_html": "<button>public</button>",
                "cleaned_html": "<button>public</button>",
                "operation": {"op": "CLICK", "original_op": "CLICK", "value": ""},
                "pos_candidates": [
                    {
                        "tag": "button",
                        "is_original_target": True,
                        "is_top_level_target": True,
                        "backend_node_id": "node-positive",
                        "attributes": '{"name":"public target"}',
                    }
                ],
                "neg_candidates": [
                    {
                        "tag": "button",
                        "backend_node_id": "node-negative",
                        "attributes": '{"name":"public distractor"}',
                    }
                ],
            }
        ],
    }


def _write_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_mind2web_scores(path: Path, action_ids: tuple[str, ...]) -> None:
    scores = {
        action_id: {
            b"node-negative".decode(): 0.1,
            b"node-positive".decode(): 0.9,
        }
        for action_id in action_ids
    }
    ranks = {action_id: {"node-negative": 1, "node-positive": 0} for action_id in action_ids}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"scores": scores, "ranks": ranks}, protocol=4))


def _write_sources(root: Path) -> dict[str, tuple[dict[str, object], ...]]:
    parquet = {
        "hotpot.parquet": (_hotpot_row(),),
        "aime.parquet": ({"answer": 42, "problem": "A public problem.", "problem_idx": 1},),
        "math.parquet": (
            {
                "problem": "Compute the public expression.",
                "level": "Level 5",
                "type": "Algebra",
                "solution": f"private derivation {PRIVATE_CANARY} \\boxed{{7}}",
            },
        ),
    }
    for name in parquet:
        path = root / "columnar" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic parquet bytes for {name}".encode())

    _write_jsonl(
        root / "atlas" / "trivia.jsonl",
        {
            "question": "A public trivia question?",
            "answers": [PRIVATE_CANARY],
            "target": PRIVATE_CANARY,
        },
    )
    _write_jsonl(root / "medqa" / "test.jsonl", _medqa_row())
    _write_jsonl(root / "musique" / "dev.jsonl", _musique_row())
    _write_jsonl(
        root / "atlas" / "nq.jsonl",
        {"question": "A public natural question?", "answers": [PRIVATE_CANARY]},
    )

    gpqa = root / "gpqa" / "diamond.csv"
    gpqa.parent.mkdir(parents=True, exist_ok=True)
    with gpqa.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=GPQA_DIAMOND_CSV_HEADER)
        writer.writeheader()
        writer.writerow(_gpqa_row())

    external_splits = {
        Benchmark.BIRD_SQL: "train+dev",
        Benchmark.MBPP_PLUS: "test",
        Benchmark.TABLEBENCH: "test",
        Benchmark.HUMANEVAL_PLUS: "test",
    }
    for benchmark, source_split in external_splits.items():
        task = RolloutTask(
            task_id=f"{benchmark.value}-fixture",
            environment_id=f"environment:{benchmark.value}:fixture",
            task_family=f"{benchmark.value}/fixture",
            context_id=f"{benchmark.value}/fixture",
            query=f"Complete the public {benchmark.value} fixture.",
            available_tools=(),
            public_context={"benchmark_id": benchmark.value, "public": True},
        )
        _write_jsonl(
            root / "external" / f"{benchmark.value}.jsonl",
            {"source_split": source_split, "task": task.to_value()},
        )

    mind = root / "mind2web" / "test_task.json"
    mind.parent.mkdir(parents=True, exist_ok=True)
    mind.write_text(json.dumps([_mind2web_row()]), encoding="utf-8")
    _write_mind2web_scores(
        root / "mind2web" / "scores_all_data.pkl",
        ("mind-fixture_action-1", "unused-fixture_action-2"),
    )
    return parquet


def _source(
    root: Path,
    benchmark: Benchmark,
    relative_file: str,
    source_format: ProductionSourceFormat,
    split: str,
    auxiliary_files: tuple[str, ...] = (),
) -> ProductionBenchmarkSource:
    snapshot_files = tuple(sorted((relative_file, *auxiliary_files)))
    return ProductionBenchmarkSource(
        benchmark=benchmark,
        dataset_revision=REVISION,
        split=split,
        source_format=source_format,
        relative_files=(relative_file,),
        snapshot=create_private_dataset_snapshot(
            name=benchmark.value,
            version=REVISION,
            root=root,
            relative_files=snapshot_files,
        ),
        relative_file_splits=(split,),
        auxiliary_files=auxiliary_files,
    )


def _configuration(root: Path) -> tuple[ProductionCatalogConfig, _ParquetReader]:
    parquet = _write_sources(root)
    index_path = root / "retrieval" / "public.sqlite"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_manifest = build_retrieval_index(
        index_path,
        (
            DocumentPassage(
                passage_id="public-passage",
                document_id="public-document",
                title="Public title",
                text="Public retrieval passage.",
            ),
        ),
        corpus_name="atlas-wikipedia",
        corpus_version="fixture@1",
    )
    sources = (
        _source(
            root,
            Benchmark.HOTPOT_QA,
            "columnar/hotpot.parquet",
            ProductionSourceFormat.PARQUET,
            "validation",
        ),
        _source(
            root, Benchmark.TRIVIA_QA, "atlas/trivia.jsonl", ProductionSourceFormat.JSONL, "dev"
        ),
        _source(
            root,
            Benchmark.AIME_2026,
            "columnar/aime.parquet",
            ProductionSourceFormat.PARQUET,
            "test",
        ),
        _source(
            root,
            Benchmark.MED_QA,
            "medqa/test.jsonl",
            ProductionSourceFormat.JSONL,
            "test",
        ),
        _source(
            root,
            Benchmark.BIRD_SQL,
            "external/bird-sql.jsonl",
            ProductionSourceFormat.JSONL,
            "train+dev",
        ),
        _source(
            root,
            Benchmark.MBPP_PLUS,
            "external/mbpp-plus.jsonl",
            ProductionSourceFormat.JSONL,
            "test",
        ),
        _source(
            root, Benchmark.MUSIQUE, "musique/dev.jsonl", ProductionSourceFormat.JSONL, "validation"
        ),
        _source(root, Benchmark.NQ_OPEN, "atlas/nq.jsonl", ProductionSourceFormat.JSONL, "dev"),
        _source(
            root,
            Benchmark.MATH_HARD,
            "columnar/math.parquet",
            ProductionSourceFormat.PARQUET,
            "test",
        ),
        _source(
            root, Benchmark.GPQA_DIAMOND, "gpqa/diamond.csv", ProductionSourceFormat.CSV, "test"
        ),
        _source(
            root,
            Benchmark.MIND2WEB,
            "mind2web/test_task.json",
            ProductionSourceFormat.JSON_ARRAY,
            "test_task",
            ("mind2web/scores_all_data.pkl",),
        ),
        _source(
            root,
            Benchmark.TABLEBENCH,
            "external/tablebench.jsonl",
            ProductionSourceFormat.JSONL,
            "test",
        ),
        _source(
            root,
            Benchmark.HUMANEVAL_PLUS,
            "external/humaneval-plus.jsonl",
            ProductionSourceFormat.JSONL,
            "test",
        ),
    )
    return (
        ProductionCatalogConfig(
            dataset_root=root,
            sources=sources,
            retrieval_index_relative_path="retrieval/public.sqlite",
            retrieval_index_id=index_manifest.index_id,
        ),
        _ParquetReader(parquet),
    )


def _dependencies(reader: _ParquetReader) -> ProductionCatalogDependencies:
    return ProductionCatalogDependencies(
        parquet_reader=reader,
        external_session_factory_builders=tuple(
            (benchmark, _ExternalFactoryBuilder())
            for benchmark in (
                Benchmark.BIRD_SQL,
                Benchmark.MBPP_PLUS,
                Benchmark.TABLEBENCH,
                Benchmark.HUMANEVAL_PLUS,
            )
        ),
    )


def test_production_loader_builds_all_non_process_workloads_without_answer_leakage(
    tmp_path: Path,
) -> None:
    config, reader = _configuration(tmp_path)

    with load_production_benchmark_catalog(config, _dependencies(reader)) as loaded:
        assert tuple(workload.benchmark for workload in loaded.catalog.workloads) == tuple(
            source.benchmark for source in config.sources
        )
        assert all(len(workload.tasks) == 1 for workload in loaded.catalog.workloads)
        public_wire = canonical_json(
            [task.to_value() for workload in loaded.catalog.workloads for task in workload.tasks]
        )
        assert PRIVATE_CANARY not in public_wire
        assert REVISION in str(loaded.catalog.workload(Benchmark.HOTPOT_QA).tasks[0].to_value())
        assert (
            loaded.catalog.workload(Benchmark.BIRD_SQL).tasks[0].public_context["benchmark_id"]
            == Benchmark.BIRD_SQL.value
        )
        assert "retrieval_index" in public_wire


def test_production_catalog_config_has_a_canonical_round_trip(tmp_path: Path) -> None:
    config, _ = _configuration(tmp_path)
    mind_index = next(
        index
        for index, source in enumerate(config.sources)
        if source.benchmark is Benchmark.MIND2WEB
    )
    mind2web = replace(
        config.sources[mind_index],
        relative_file_splits=("test_task",),
    )
    sources = list(config.sources)
    sources[mind_index] = mind2web
    configured = replace(config, sources=tuple(sources))

    restored = ProductionCatalogConfig.from_value(json.loads(canonical_json(configured.to_value())))

    assert restored == configured
    assert restored.content_hash == configured.content_hash
    assert restored.sources[mind_index].file_splits == ("test_task",)


def test_production_loader_rejects_changed_or_missing_exact_source_bytes(tmp_path: Path) -> None:
    config, reader = _configuration(tmp_path)
    trivia = tmp_path / "atlas" / "trivia.jsonl"
    trivia.write_text(trivia.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_production_benchmark_catalog(config, _dependencies(reader))

    config, reader = _configuration(tmp_path / "fresh")
    (config.dataset_root / "atlas" / "trivia.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        load_production_benchmark_catalog(config, _dependencies(reader))


def test_production_loader_rejects_duplicate_json_keys_without_schema_coercion(
    tmp_path: Path,
) -> None:
    config, reader = _configuration(tmp_path)
    trivia = tmp_path / "atlas" / "trivia.jsonl"
    trivia.write_text(
        '{"answers":["private"],"question":"public","question":"changed","target":"private"}\n',
        encoding="utf-8",
    )
    replacement = replace(
        config.sources[1],
        snapshot=create_private_dataset_snapshot(
            name=Benchmark.TRIVIA_QA.value,
            version=REVISION,
            root=tmp_path,
            relative_files=("atlas/trivia.jsonl",),
        ),
    )
    changed = replace(config, sources=(config.sources[0], replacement, *config.sources[2:]))

    with pytest.raises(ValueError):
        load_production_benchmark_catalog(changed, _dependencies(reader))


def test_production_loader_streams_shards_from_all_three_official_mind2web_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, reader = _configuration(tmp_path)
    mind_index = next(
        index
        for index, source in enumerate(config.sources)
        if source.benchmark is Benchmark.MIND2WEB
    )
    relative_files = (
        "mind2web/test_domain/test_domain_0.json",
        "mind2web/test_task/test_task_0.json",
        "mind2web/test_task/test_task_1.json",
        "mind2web/test_website/test_website_0.json",
    )
    file_splits = ("test_domain", "test_task", "test_task", "test_website")
    for ordinal, relative_file in enumerate(relative_files, start=1):
        row = _mind2web_row()
        row["annotation_id"] = f"mind-fixture-{ordinal}"
        actions = cast(list[dict[str, object]], row["actions"])
        actions[0]["action_uid"] = f"action-{ordinal}"
        path = tmp_path / relative_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([row], ensure_ascii=False),
            encoding="utf-8",
        )
    _write_mind2web_scores(
        tmp_path / "mind2web" / "scores_all_data.pkl",
        tuple(
            f"mind-fixture-{ordinal}_action-{ordinal}"
            for ordinal in range(1, len(relative_files) + 1)
        ),
    )

    source = replace(
        config.sources[mind_index],
        split="official-evaluation",
        relative_files=relative_files,
        relative_file_splits=file_splits,
        snapshot=create_private_dataset_snapshot(
            name=Benchmark.MIND2WEB.value,
            version=REVISION,
            root=tmp_path,
            relative_files=tuple(sorted((*relative_files, "mind2web/scores_all_data.pkl"))),
        ),
    )
    sources = list(config.sources)
    sources[mind_index] = source
    changed = replace(config, sources=tuple(sources))
    original_read_text = Path.read_text

    def reject_whole_file_read(path: Path, *args: object, **kwargs: object) -> str:
        if "mind2web" in path.parts:
            raise AssertionError("Mind2Web JSON arrays must be streamed")
        return original_read_text(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", reject_whole_file_read)
    with load_production_benchmark_catalog(changed, _dependencies(reader)) as loaded:
        workload = loaded.catalog.workload(Benchmark.MIND2WEB)
        assert tuple(task.public_context["split"] for task in workload.tasks) == file_splits


def test_streamed_mind2web_json_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    config, reader = _configuration(tmp_path)
    mind_index = next(
        index
        for index, source in enumerate(config.sources)
        if source.benchmark is Benchmark.MIND2WEB
    )
    source_path = tmp_path / "mind2web" / "test_task.json"
    encoded = json.dumps([_mind2web_row()], ensure_ascii=False)
    encoded = encoded.replace(
        '"annotation_id": "mind-fixture"',
        '"annotation_id": "first", "annotation_id": "second"',
        1,
    )
    source_path.write_text(encoded, encoding="utf-8")
    source = replace(
        config.sources[mind_index],
        snapshot=create_private_dataset_snapshot(
            name=Benchmark.MIND2WEB.value,
            version=REVISION,
            root=tmp_path,
            relative_files=(
                "mind2web/scores_all_data.pkl",
                "mind2web/test_task.json",
            ),
        ),
    )
    sources = list(config.sources)
    sources[mind_index] = source
    changed = replace(config, sources=tuple(sources))

    with pytest.raises(ValueError):
        load_production_benchmark_catalog(changed, _dependencies(reader))


def test_production_config_rejects_discovery_like_or_wrong_format_specs(tmp_path: Path) -> None:
    config, _ = _configuration(tmp_path)
    with pytest.raises(ValueError):
        replace(config.sources[0], relative_files=("columnar/*.parquet",))
    with pytest.raises(ValueError):
        replace(config.sources[0], source_format=ProductionSourceFormat.JSONL)
    with pytest.raises(ValueError):
        replace(config, sources=tuple(reversed(config.sources)))
    hotpot = replace(config.sources[0], relative_file_splits=("validation",))
    assert hotpot.file_splits == ("validation",)
    with pytest.raises(ValueError):
        replace(
            next(source for source in config.sources if source.benchmark is Benchmark.MIND2WEB),
            relative_file_splits=("test_task", "test_domain"),
        )


def test_pyarrow_reader_preserves_exact_row_order_and_nested_values(tmp_path: Path) -> None:
    source = (tmp_path / "official.parquet").resolve()
    expected = [
        {"count": 3, "identity": "first", "nested": {"values": ["a", "b"]}},
        {"count": 7, "identity": "second", "nested": {"values": []}},
    ]
    pq.write_table(pa.Table.from_pylist(expected), source)

    rows = PyArrowParquetRowReader().read_rows(source)

    assert rows == tuple(expected)
    assert type(rows[0]["count"]) is int
    assert type(rows[0]["nested"]) is dict


def test_pyarrow_reader_rejects_empty_missing_and_nonabsolute_sources(tmp_path: Path) -> None:
    source = (tmp_path / "empty.parquet").resolve()
    pq.write_table(pa.table({"identity": pa.array([], type=pa.string())}), source)
    reader = PyArrowParquetRowReader()

    with pytest.raises(ValueError):
        reader.read_rows(source)
    with pytest.raises(FileNotFoundError):
        reader.read_rows((tmp_path / "missing.parquet").resolve())
    with pytest.raises(ValueError):
        reader.read_rows(Path("relative.parquet"))
