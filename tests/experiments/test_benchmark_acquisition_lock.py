from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from skillev.experiments import BENCHMARK_SPECS, FIXED_SEED


def _lock_value() -> dict[str, object]:
    path = Path(__file__).parents[2] / "benchmark-acquisition-lock.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return cast(dict[str, object], value)


def test_acquisition_lock_matches_the_result_blind_active_benchmark_protocol() -> None:
    value = _lock_value()
    assert value["format"] == "skillev-benchmark-acquisition-lock@1"
    assert value["fixed_seed"] == FIXED_SEED
    assert value["preservation"] == "preserve-every-source-and-archive"
    entries = cast(list[dict[str, object]], value["benchmarks"])
    assert tuple(entry["benchmark"] for entry in entries) == tuple(
        spec.benchmark.value for spec in BENCHMARK_SPECS
    )

    for entry, spec in zip(entries, BENCHMARK_SPECS, strict=True):
        assert entry["role"] == spec.role.value
        assert entry["training_use"] == spec.training_use.value
        assert entry["sampling"] == spec.evaluation_sampling.to_value()


def test_acquisition_lock_uses_immutable_revisions_and_explicit_artifacts() -> None:
    entries = cast(list[dict[str, object]], _lock_value()["benchmarks"])
    logical_artifacts: set[tuple[str, str]] = set()
    for entry in entries:
        source = cast(dict[str, object], entry["source"])
        revision = source["revision"]
        assert isinstance(revision, str)
        assert len(revision) == 40
        int(revision, 16)
        assert str(source["repository"]).startswith("https://")

        artifacts = cast(list[dict[str, object]], source["artifacts"])
        prepared_files = cast(list[str], source["prepared_files"])
        assert artifacts
        assert prepared_files
        for artifact in artifacts:
            locator = artifact["locator"]
            relative_path = artifact["relative_path"]
            expected_size = artifact["expected_size"]
            expected_sha256 = artifact["expected_sha256"]
            assert isinstance(locator, str)
            assert locator
            assert isinstance(relative_path, str)
            assert relative_path
            logical_identity = (cast(str, entry["benchmark"]), relative_path)
            assert logical_identity not in logical_artifacts
            logical_artifacts.add(logical_identity)
            assert expected_size is None or (
                type(expected_size) is int and cast(int, expected_size) > 0
            )
            if expected_sha256 is not None:
                assert isinstance(expected_sha256, str)
                assert len(expected_sha256) == 64
                int(expected_sha256, 16)


def test_acquisition_lock_contains_only_metadata_not_local_or_result_state() -> None:
    value = _lock_value()
    infrastructure = cast(dict[str, dict[str, object]], value["shared_infrastructure"])
    retrieval = infrastructure["atlas_wikipedia_fts5"]
    assert retrieval["retrieval_backend"] == "sqlite-fts5-lexical"
    assert retrieval["corpus_source"] == "official-dpr-wikipedia-passages"
    assert cast(int, retrieval["passage_count"]) > 20_000_000
    encoded = json.dumps(value, ensure_ascii=False).lower()
    assert "/mnt/" not in encoded
    assert "e:\\\\" not in encoded
    assert "reward" not in encoded
    assert "result" not in encoded
    assert "answer" not in encoded
