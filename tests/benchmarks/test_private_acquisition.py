from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

import pytest
from skillev_private.benchmarks.acquisition import (
    BenchmarkAcquisitionLock,
    CurlLockedArtifactFetcher,
    LockedAcquisitionArtifact,
    LockedBenchmarkSource,
    acquire_locked_benchmark_artifacts,
    inspect_acquisition_progress,
    load_benchmark_acquisition_lock,
    load_benchmark_acquisition_receipt,
    locked_artifact_url,
    publish_benchmark_acquisition_receipt,
)

from skillev.contracts import stable_hash
from skillev.experiments import BENCHMARK_SPECS, Benchmark


def _lock_path() -> Path:
    return Path(__file__).parents[2] / "benchmark-acquisition-lock.json"


def _lock_value() -> dict[str, object]:
    value = json.loads(_lock_path().read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def test_loads_the_complete_result_blind_acquisition_authority() -> None:
    lock = load_benchmark_acquisition_lock(_lock_path())

    assert tuple(item.benchmark for item in lock.benchmarks) == tuple(
        spec.benchmark for spec in BENCHMARK_SPECS
    )
    artifact_count = sum(len(item.source.artifacts) for item in lock.benchmarks)
    assert artifact_count == 28
    assert load_benchmark_acquisition_lock(_lock_path()).content_hash == lock.content_hash
    assert lock.to_value() == _lock_value()
    assert lock.content_hash == stable_hash(_lock_value())
    mind2web = next(item for item in lock.benchmarks if item.benchmark is Benchmark.MIND2WEB)
    assert tuple(artifact.relative_path for artifact in mind2web.source.artifacts) == (
        "raw/test.zip",
        "raw/scores_all_data.pkl",
    )


def test_lock_parser_rejects_protocol_or_path_drift() -> None:
    wrong_seed = _lock_value()
    wrong_seed["fixed_seed"] = 1
    with pytest.raises(ValueError):
        BenchmarkAcquisitionLock.from_value(wrong_seed)

    wrong_path = copy.deepcopy(_lock_value())
    entries = cast(list[dict[str, object]], wrong_path["benchmarks"])
    source = cast(dict[str, object], entries[0]["source"])
    artifacts = cast(list[dict[str, object]], source["artifacts"])
    artifacts[0]["relative_path"] = "../outside"
    with pytest.raises(ValueError):
        BenchmarkAcquisitionLock.from_value(wrong_path)


def test_progress_inspects_only_locked_paths_and_streams_exact_digests(
    tmp_path: Path,
) -> None:
    payload = b"fixed-aime-fixture\n"
    value = copy.deepcopy(_lock_value())
    entries = cast(list[dict[str, object]], value["benchmarks"])
    aime = next(entry for entry in entries if entry["benchmark"] == Benchmark.AIME_2026.value)
    source = cast(dict[str, object], aime["source"])
    artifact = cast(list[dict[str, object]], source["artifacts"])[0]
    artifact["expected_size"] = len(payload)
    artifact["expected_sha256"] = hashlib.sha256(payload).hexdigest()
    lock = BenchmarkAcquisitionLock.from_value(value)

    target = (tmp_path / "private-datasets").resolve()
    aime_path = target / Benchmark.AIME_2026.value / cast(str, artifact["relative_path"])
    aime_path.parent.mkdir(parents=True)
    aime_path.write_bytes(payload)
    musique_entry = next(entry for entry in lock.benchmarks if entry.benchmark is Benchmark.MUSIQUE)
    musique = musique_entry.source.artifacts[0]
    musique_path = target / Benchmark.MUSIQUE.value / musique.relative_path
    musique_path.parent.mkdir(parents=True)
    musique_path.write_bytes(b"PK\x03\x04unpinned-official-source\n")
    unrelated = target / "not-in-lock.bin"
    unrelated.write_bytes(b"ignored\n")

    fast = inspect_acquisition_progress(lock, target_root=target, verify_digests=False)
    verified = inspect_acquisition_progress(lock, target_root=target, verify_digests=True)

    assert fast.artifact_count == verified.artifact_count == 28
    assert fast.complete_count == 0
    assert verified.complete_count == 2
    assert verified.invalid_count == 0
    assert verified.present_bytes == aime_path.stat().st_size + musique_path.stat().st_size
    assert verified.known_expected_bytes > 0
    assert unrelated not in {item.local_path for item in verified.artifacts}

    aime_path.write_bytes(b"x" * len(payload))
    damaged = inspect_acquisition_progress(lock, target_root=target, verify_digests=True)
    assert damaged.complete_count == 1
    assert damaged.invalid_count == 1


def _git_blob_sha1(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode())
    digest.update(payload)
    return digest.hexdigest()


def _fully_pinned_fixture_lock() -> tuple[BenchmarkAcquisitionLock, dict[str, bytes]]:
    value = copy.deepcopy(_lock_value())
    payloads: dict[str, bytes] = {}
    entries = cast(list[dict[str, object]], value["benchmarks"])
    for entry in entries:
        source = cast(dict[str, object], entry["source"])
        artifacts = cast(list[dict[str, object]], source["artifacts"])
        for artifact in artifacts:
            key = f"{entry['benchmark']}/{artifact['relative_path']}"
            locator = str(artifact["locator"])
            relative_path = str(artifact["relative_path"])
            if locator.startswith("gdrive:") and relative_path.endswith(".zip"):
                payload = b"PK\x03\x04" + key.encode()
            elif locator.startswith("gdrive:") and relative_path.endswith(".json"):
                payload = json.dumps({"fixture": key}).encode()
            else:
                payload = (key + "\n").encode()
            payloads[key] = payload
            artifact["expected_size"] = len(payload)
            artifact["expected_sha256"] = hashlib.sha256(payload).hexdigest()
            if str(artifact["locator"]).startswith("git-blob:"):
                artifact["locator"] = f"git-blob:{_git_blob_sha1(payload)}"
    return BenchmarkAcquisitionLock.from_value(value), payloads


@dataclass
class _FixtureFetcher:
    payloads: dict[str, bytes]
    calls: list[str] = field(default_factory=list)

    def fetch(
        self,
        *,
        benchmark: Benchmark,
        source: LockedBenchmarkSource,
        artifact: LockedAcquisitionArtifact,
        destination: Path,
    ) -> None:
        del source
        key = f"{benchmark.value}/{artifact.relative_path}"
        self.calls.append(key)
        destination.write_bytes(self.payloads[key])


def test_acquires_each_locked_artifact_once_and_preserves_existing_files(
    tmp_path: Path,
) -> None:
    lock, payloads = _fully_pinned_fixture_lock()
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()
    first_fetcher = _FixtureFetcher(payloads)

    first = acquire_locked_benchmark_artifacts(
        lock,
        target_root=target,
        fetcher=first_fetcher,
    )
    second_fetcher = _FixtureFetcher(payloads)
    second = acquire_locked_benchmark_artifacts(
        lock,
        target_root=target,
        fetcher=second_fetcher,
    )

    assert len(first.artifacts) == len(second.artifacts) == 28
    assert len(first_fetcher.calls) == 28
    assert second_fetcher.calls == []
    assert first.to_value() == second.to_value()
    output = (tmp_path / "acquisition-receipt.json").resolve()
    publish_benchmark_acquisition_receipt(first, output)
    original = output.read_bytes()
    assert load_benchmark_acquisition_receipt(output) == first
    with pytest.raises(FileExistsError):
        publish_benchmark_acquisition_receipt(second, output)
    assert output.read_bytes() == original


def test_failed_identity_check_preserves_partial_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    lock, payloads = _fully_pinned_fixture_lock()
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()
    entry = lock.benchmarks[0]
    first = entry.source.artifacts[0]
    payloads[f"{entry.benchmark.value}/{first.relative_path}"] = b"wrong\n"

    with pytest.raises(ValueError):
        acquire_locked_benchmark_artifacts(
            lock,
            target_root=target,
            fetcher=_FixtureFetcher(payloads),
        )

    destination = target / entry.benchmark.value / first.relative_path
    assert not destination.exists()
    assert destination.with_name(destination.name + ".partial").read_bytes() == b"wrong\n"


def test_google_drive_html_response_is_not_admitted_as_an_unpinned_source(
    tmp_path: Path,
) -> None:
    lock, payloads = _fully_pinned_fixture_lock()
    entry = next(item for item in lock.benchmarks if item.benchmark is Benchmark.MED_QA)
    artifact = entry.source.artifacts[0]
    html = b"<html>download confirmation failed</html>\n"
    changed_artifact = replace(
        artifact,
        expected_size=len(html),
        expected_sha256=hashlib.sha256(html).hexdigest(),
    )
    changed_source = replace(
        entry.source,
        artifacts=(changed_artifact, *entry.source.artifacts[1:]),
    )
    changed_lock = replace(
        lock,
        benchmarks=tuple(
            replace(item, source=changed_source) if item is entry else item
            for item in lock.benchmarks
        ),
    )
    payloads[f"{entry.benchmark.value}/{artifact.relative_path}"] = html
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()

    with pytest.raises(ValueError):
        acquire_locked_benchmark_artifacts(
            changed_lock,
            target_root=target,
            fetcher=_FixtureFetcher(payloads),
        )

    destination = target / entry.benchmark.value / artifact.relative_path
    assert not destination.exists()
    assert destination.with_name(destination.name + ".partial").read_bytes() == html


def test_locator_translation_is_pinned_and_huggingface_token_stays_off_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = load_benchmark_acquisition_lock(_lock_path())
    by_benchmark = {entry.benchmark: entry for entry in lock.benchmarks}
    hotpot = by_benchmark[Benchmark.HOTPOT_QA]
    webshop = by_benchmark[Benchmark.WEBSHOP]
    medqa = by_benchmark[Benchmark.MED_QA]
    science = by_benchmark[Benchmark.SCIENCE_WORLD]
    direct = by_benchmark[Benchmark.ALFWORLD]
    assert "/datasets/hotpotqa/hotpot_qa/resolve/" in locked_artifact_url(
        hotpot.source,
        hotpot.source.artifacts[0],
    )
    assert "/datasets/" in locked_artifact_url(
        webshop.source,
        webshop.source.artifacts[0],
    )
    assert "drive.usercontent.google.com/download?" in locked_artifact_url(
        medqa.source,
        medqa.source.artifacts[0],
    )
    assert science.source.revision in locked_artifact_url(
        science.source,
        science.source.artifacts[0],
    )
    assert locked_artifact_url(direct.source, direct.source.artifacts[0]) == (
        direct.source.artifacts[0].locator
    )

    captured: dict[str, object] = {}

    def fake_run(
        args: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        captured.update(args=args, **kwargs)
        output = Path(args[args.index("--output") + 1])
        output.write_bytes(b"downloaded\n")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setenv("HF_TOKEN", "fixture-secret-canary")
    monkeypatch.setattr(subprocess, "run", fake_run)
    destination = tmp_path / "download.partial"
    CurlLockedArtifactFetcher().fetch(
        benchmark=Benchmark.HOTPOT_QA,
        source=hotpot.source,
        artifact=hotpot.source.artifacts[0],
        destination=destination,
    )

    assert destination.read_bytes() == b"downloaded\n"
    assert "fixture-secret-canary" not in repr(captured["args"])
    assert "Authorization: Bearer fixture-secret-canary" in cast(str, captured["input"])
