"""Synthetic private ledger bytes; no rubric or candidate from a dataset."""

import json

import pytest

from skillev.training import evidence_mirror
from tests.training.test_run_observer import observer


def attach(source, args, path, *, positions=(1, 2)):
    reference = {
        "format": "healthbench-criterion-ledger@1",
        "path": str(path),
        "status": "completed",
        "request_count": 1,
    }
    for position in positions:
        source["payload"]["records"][position - 1]["reward"]["native_payload"][
            "criterion_ledger"
        ] = reference
        artifact_path = args["inflight"] / "step-00000001" / f"trajectory-{position:06d}.json"
        value = json.loads(artifact_path.read_text())
        value["artifact"]["record"] = source["payload"]["records"][position - 1]
        artifact_path.write_text(json.dumps(value))
    rows = [json.loads(line) for line in args["events"].read_text().splitlines()]
    args["events"].write_text(
        "".join(
            json.dumps(source if row["event_id"] == source["event_id"] else row) + "\n"
            for row in rows
        )
    )


def test_exact_referenced_ledgers_only_original_bytes_and_relocation(tmp_path):
    obs, source, args = observer(tmp_path)
    directory = tmp_path / "healthbench-criteria"
    directory.mkdir()
    ledger = directory / "one.json"
    raw = b'{ "status": "completed", "requests": [{"response":"SYNTHETIC_PRIVATE"}]}\n'
    ledger.write_bytes(raw)
    (directory / "unrelated.json").write_text("must not copy")
    attach(source, args, ledger)
    before = args["events"].read_bytes()
    artifact = args["inflight"] / "step-00000001" / "trajectory-000001.json"
    before_artifact = artifact.read_bytes()
    with obs:
        obs.observe_commits(1)
        assert obs.drain()["mirror_counts"]["mirrored"] == 1
    final = tmp_path / "persistent" / "step-00000001"
    manifest = json.loads((final / "mirror.json").read_text())
    relocations = manifest["criterion_ledger_relocations"]
    assert len(relocations) == 1  # Same explicit file referenced twice, copied once.
    assert (final / relocations[str(ledger)]).read_bytes() == raw
    assert len(list((final / "criterion-ledgers").iterdir())) == 1
    assert (final / "inflight" / "step-00000001" / artifact.name).read_bytes() == before_artifact
    assert args["events"].read_bytes() == before
    assert ledger.read_bytes() == raw
    ledger.unlink()  # A complete published backup does not depend on old storage.
    index = tmp_path / "evidence" / "step-00000001.json"
    assert evidence_mirror.mirror_step(index, tmp_path / "persistent") == manifest
    (final / relocations[str(ledger)]).unlink()
    with pytest.raises(FileNotFoundError):
        evidence_mirror.mirror_step(index, tmp_path / "persistent")


@pytest.mark.parametrize("fault", ["missing", "copy-failure"])
def test_missing_or_failed_referenced_ledger_is_backup_failure(tmp_path, monkeypatch, fault):
    obs, source, args = observer(tmp_path)
    ledger = tmp_path / "selected-ledger.json"
    if fault == "copy-failure":
        ledger.write_text('{"status":"completed"}')
        original = evidence_mirror.copy_file

        def copy(source, target):
            if source == ledger:
                raise OSError("synthetic ledger I/O failure")
            original(source, target)

        monkeypatch.setattr(evidence_mirror, "copy_file", copy)
    attach(source, args, ledger)
    with obs:
        obs.observe_commits(1)
        status = obs.drain()
        assert status["mirror_counts"]["failed"] == 1
        assert status["mirror_counts"]["mirrored"] == 0
        assert status["pause_required"]
    assert not (tmp_path / "persistent" / "step-00000001").exists()
    assert list((tmp_path / "persistent").glob(".step-*.partial-*"))


def test_legacy_mirror_without_references_stays_readable(tmp_path):
    obs, _, _ = observer(tmp_path)
    with obs:
        obs.observe_commits(1)
        assert obs.drain()["mirror_counts"]["mirrored"] == 1
    final = tmp_path / "persistent" / "step-00000001"
    metadata = json.loads((final / "mirror.json").read_text())
    metadata.pop("criterion_ledger_relocations")
    metadata["format"] = "committed-evidence-mirror@1"
    (final / "mirror.json").write_text(json.dumps(metadata))
    assert (
        evidence_mirror.mirror_step(
            tmp_path / "evidence" / "step-00000001.json", tmp_path / "persistent"
        )
        == metadata
    )
