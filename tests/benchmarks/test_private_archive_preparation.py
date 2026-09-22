from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
from skillev_private.benchmarks.acquisition import (
    AcquiredArtifactReceipt,
    BenchmarkAcquisitionLock,
    BenchmarkAcquisitionReceipt,
)
from skillev_private.benchmarks.archive_preparation import (
    ArchiveKind,
    load_locked_archive_batch_receipt,
    locked_archive_preparations,
    prepare_archive,
    prepare_locked_archives,
    publish_locked_archive_batch_receipt,
)

from skillev.experiments import Benchmark

_ENCRYPTED_ZIP = base64.b64decode(
    "UEsDBAoACQAAAECW91yImxodJgAAABoAAAAKABwAbWVtYmVyLnR4dFVUCQAD1/FhatfxYWp1"
    "eAsAAQToAwAABOgDAACvdUsxB/MD674ED8XaeQAUAZbPaf1InDk8eY89ztxB3b2chUsyu1BL"
    "BwiImxodJgAAABoAAABQSwECHgMKAAkAAABAlvdciJsaHSYAAAAaAAAACgAYAAAAAAABAAAA/4"
    "EAAAAAbWVtYmVyLnR4dFVUBQAD1/FhanV4CwABBOgDAAAE6AMAAFBLBQYAAAAAAQABAFAAAAB6"
    "AAAAAAA="
)
_FIXTURE_PASSWORD = b"fixture-secret"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            information = tarfile.TarInfo(name)
            information.size = len(payload)
            archive.addfile(information, io.BytesIO(payload))


def _archive_payload(kind: ArchiveKind, label: str) -> bytes:
    buffer = io.BytesIO()
    member_name = f"official/{label.replace('/', '_')}.txt"
    payload = (label + "\n").encode()
    if kind is ArchiveKind.ZIP:
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(member_name, payload)
    else:
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            information = tarfile.TarInfo(member_name)
            information.size = len(payload)
            archive.addfile(information, io.BytesIO(payload))
    return buffer.getvalue()


def _locked_archive_fixture(
    tmp_path: Path,
) -> tuple[BenchmarkAcquisitionLock, BenchmarkAcquisitionReceipt, Path]:
    lock_path = Path(__file__).parents[2] / "benchmark-acquisition-lock.json"
    value = copy.deepcopy(json.loads(lock_path.read_text(encoding="utf-8")))
    preliminary = BenchmarkAcquisitionLock.from_value(value)
    plans = {
        (plan.benchmark, plan.artifact_relative_path): plan
        for plan in locked_archive_preparations(preliminary)
    }
    target = (tmp_path / "private-datasets").resolve()
    target.mkdir()
    payloads: dict[tuple[Benchmark, str], bytes] = {}
    raw_entries = value["benchmarks"]
    assert isinstance(raw_entries, list)
    for raw_entry, entry in zip(raw_entries, preliminary.benchmarks, strict=True):
        assert isinstance(raw_entry, dict)
        raw_source = raw_entry["source"]
        assert isinstance(raw_source, dict)
        raw_artifacts = raw_source["artifacts"]
        assert isinstance(raw_artifacts, list)
        for raw_artifact, artifact in zip(
            raw_artifacts,
            entry.source.artifacts,
            strict=True,
        ):
            assert isinstance(raw_artifact, dict)
            identity = (entry.benchmark, artifact.relative_path)
            plan = plans.get(identity)
            payload = (
                (
                    _ENCRYPTED_ZIP
                    if plan.password_required
                    else _archive_payload(
                        plan.archive_kind,
                        f"{entry.benchmark.value}/{artifact.relative_path}",
                    )
                )
                if plan is not None
                else f"{entry.benchmark.value}/{artifact.relative_path}\n".encode()
            )
            payloads[identity] = payload
            raw_artifact["expected_size"] = len(payload)
            raw_artifact["expected_sha256"] = hashlib.sha256(payload).hexdigest()
    lock = BenchmarkAcquisitionLock.from_value(value)
    receipts: list[AcquiredArtifactReceipt] = []
    for entry in lock.benchmarks:
        for artifact in entry.source.artifacts:
            payload = payloads[(entry.benchmark, artifact.relative_path)]
            path = target / entry.benchmark.value / artifact.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            receipts.append(
                AcquiredArtifactReceipt(
                    benchmark=entry.benchmark,
                    locator=artifact.locator,
                    relative_path=artifact.relative_path,
                    size_bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
    receipt = BenchmarkAcquisitionReceipt(
        acquisition_lock_hash=lock.content_hash,
        target_root=target,
        artifacts=tuple(receipts),
    )
    return lock, receipt, target


@pytest.mark.parametrize(
    ("kind", "builder", "suffix"),
    [
        (ArchiveKind.ZIP, _zip, ".zip"),
        (ArchiveKind.TAR, _tar, ".tar.gz"),
    ],
)
def test_extracts_exact_archive_without_modifying_source(
    tmp_path: Path,
    kind: ArchiveKind,
    builder: Callable[[Path, dict[str, bytes]], None],
    suffix: str,
) -> None:
    archive = (tmp_path / f"source{suffix}").resolve()
    members = {
        "official/data/a.json": b'{"a":1}\n',
        "official/data/b.txt": b"pinned\n",
    }
    builder(archive, members)
    source_before = archive.read_bytes()
    output = (tmp_path / "prepared").resolve()

    receipt = prepare_archive(
        archive_path=archive,
        archive_kind=kind,
        expected_size=archive.stat().st_size,
        expected_sha256=_sha256(archive),
        output_root=output,
    )

    assert archive.read_bytes() == source_before
    assert (output / "official/data/a.json").read_bytes() == members["official/data/a.json"]
    assert (output / "official/data/b.txt").read_bytes() == members["official/data/b.txt"]
    assert receipt.extracted_file_count == 2
    assert receipt.extracted_size_bytes == sum(map(len, members.values()))
    member_lines = (output / "extracted-members.jsonl").read_bytes()
    assert hashlib.sha256(member_lines).hexdigest() == receipt.members_jsonl_sha256
    manifest = json.loads((output / "preparation.manifest.json").read_text())
    assert manifest["content_hash"] == receipt.content_hash
    with pytest.raises(FileExistsError):
        prepare_archive(
            archive_path=archive,
            archive_kind=kind,
            expected_size=archive.stat().st_size,
            expected_sha256=_sha256(archive),
            output_root=output,
        )


def test_rejects_wrong_source_identity_before_creating_output(tmp_path: Path) -> None:
    archive = (tmp_path / "source.zip").resolve()
    _zip(archive, {"data/a.txt": b"content\n"})
    output = (tmp_path / "prepared").resolve()

    with pytest.raises(ValueError):
        prepare_archive(
            archive_path=archive,
            archive_kind=ArchiveKind.ZIP,
            expected_size=archive.stat().st_size,
            expected_sha256="0" * 64,
            output_root=output,
        )

    assert not output.exists()
    assert archive.is_file()


@pytest.mark.parametrize("member_name", ["../outside.txt", "/absolute.txt"])
def test_rejects_archive_members_outside_the_new_tree(
    tmp_path: Path,
    member_name: str,
) -> None:
    archive = (tmp_path / "source.zip").resolve()
    _zip(archive, {member_name: b"escaped\n"})
    output = (tmp_path / "prepared").resolve()

    with pytest.raises(ValueError):
        prepare_archive(
            archive_path=archive,
            archive_kind=ArchiveKind.ZIP,
            expected_size=archive.stat().st_size,
            expected_sha256=_sha256(archive),
            output_root=output,
        )

    assert not output.exists()
    assert not (tmp_path / "outside.txt").exists()


def test_rejects_tar_links_instead_of_following_them(tmp_path: Path) -> None:
    archive = (tmp_path / "source.tar.gz").resolve()
    with tarfile.open(archive, "w:gz") as stream:
        information = tarfile.TarInfo("data/link")
        information.type = tarfile.SYMTYPE
        information.linkname = "../../outside"
        stream.addfile(information)
    output = (tmp_path / "prepared").resolve()

    with pytest.raises(ValueError):
        prepare_archive(
            archive_path=archive,
            archive_kind=ArchiveKind.TAR,
            expected_size=archive.stat().st_size,
            expected_sha256=_sha256(archive),
            output_root=output,
        )

    assert not output.exists()


def test_encrypted_zip_requires_the_exact_ephemeral_password(tmp_path: Path) -> None:
    archive = (tmp_path / "source.zip").resolve()
    archive.write_bytes(_ENCRYPTED_ZIP)
    source_before = archive.read_bytes()
    expected = {
        "archive_path": archive,
        "archive_kind": ArchiveKind.ZIP,
        "expected_size": archive.stat().st_size,
        "expected_sha256": _sha256(archive),
    }

    with pytest.raises(ValueError):
        prepare_archive(output_root=(tmp_path / "missing").resolve(), **expected)
    with pytest.raises(ValueError):
        prepare_archive(
            output_root=(tmp_path / "wrong").resolve(),
            zip_password=b"wrong",
            **expected,
        )
    output = (tmp_path / "prepared").resolve()
    receipt = prepare_archive(
        output_root=output,
        zip_password=_FIXTURE_PASSWORD,
        **expected,
    )

    assert archive.read_bytes() == source_before
    assert (output / "member.txt").read_bytes() == b"encrypted fixture payload\n"
    assert receipt.extracted_file_count == 1
    persisted = (output / "preparation.manifest.json").read_bytes() + (
        output / "extracted-members.jsonl"
    ).read_bytes()
    assert _FIXTURE_PASSWORD not in persisted


def test_prepares_the_explicit_locked_archive_set_and_resumes_by_verification(
    tmp_path: Path,
) -> None:
    lock, acquisition_receipt, target = _locked_archive_fixture(tmp_path)

    first = prepare_locked_archives(
        lock=lock,
        acquisition_receipt=acquisition_receipt,
        target_root=target,
        zip_passwords={Benchmark.MIND2WEB: _FIXTURE_PASSWORD},
    )
    second = prepare_locked_archives(
        lock=lock,
        acquisition_receipt=acquisition_receipt,
        target_root=target,
        zip_passwords={},
    )

    assert len(first.archives) == len(second.archives) == 11
    assert first == second
    for archive in first.archives:
        output = target / archive.benchmark.value / archive.output_relative_path
        assert (output / "preparation.manifest.json").is_file()
        assert (output / "extracted-members.jsonl").is_file()
    receipt_path = (tmp_path / "archive-batch-receipt.json").resolve()
    publish_locked_archive_batch_receipt(first, receipt_path)
    assert load_locked_archive_batch_receipt(receipt_path) == first
    original = receipt_path.read_bytes()
    with pytest.raises(FileExistsError):
        publish_locked_archive_batch_receipt(second, receipt_path)
    assert receipt_path.read_bytes() == original


def test_locked_archive_resume_rejects_changed_extracted_bytes(tmp_path: Path) -> None:
    lock, acquisition_receipt, target = _locked_archive_fixture(tmp_path)
    prepared = prepare_locked_archives(
        lock=lock,
        acquisition_receipt=acquisition_receipt,
        target_root=target,
        zip_passwords={Benchmark.MIND2WEB: _FIXTURE_PASSWORD},
    )
    first = prepared.archives[0]
    output = target / first.benchmark.value / first.output_relative_path
    first_member = json.loads(
        (output / "extracted-members.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    member_path = output / first_member["relative_path"]
    member_path.write_bytes(member_path.read_bytes() + b"changed\n")

    with pytest.raises(ValueError):
        prepare_locked_archives(
            lock=lock,
            acquisition_receipt=acquisition_receipt,
            target_root=target,
            zip_passwords={Benchmark.MIND2WEB: _FIXTURE_PASSWORD},
        )
