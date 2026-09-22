"""Write-once formal seven-arm admission and terminal regression coverage."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
import skillev_private.experiments.attempt_worker as private_attempt_worker
from skillev_private.experiments.attempt_supervisor import PrivateBenchmarkAttemptSupervisor
from skillev_private.experiments.attempt_worker import (
    run_private_benchmark_attempt_worker,
)

from skillev.contracts import stable_hash
from skillev.experiments import FormalRunClaimSecret
from skillev.experiments.formal_run_manifest import (
    FormalRunLedger,
    FormalRunManifest,
    FormalRunSlot,
    FormalRunTerminalStatus,
)
from skillev.experiments.protocol import FREEZE_FORMAT, ProtocolFreeze
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptProcessFailedError,
    AttemptPublisher,
    AttemptRequest,
    AttemptSourceLogDigest,
    AttemptSourceLogKind,
    FinalTrainingArtifact,
    PreparedSuccessfulAttempt,
    PublishedSuccessfulAttemptBundle,
    QuarantinedFailedAttemptBundle,
    UnpublishedAttemptBundle,
)


def _freeze() -> ProtocolFreeze:
    protocol_hash = stable_hash("formal-run-protocol")
    return ProtocolFreeze(
        protocol_hash=protocol_hash,
        freeze_id=stable_hash(
            {
                "format": FREEZE_FORMAT,
                "protocol_hash": protocol_hash,
                "result_ingestion_count": 0,
            }
        ),
    )


def _slots() -> tuple[FormalRunSlot, ...]:
    return tuple(
        FormalRunSlot(
            builder_kind=kind,
            attempt_id=f"formal-{kind.value}",
            exact_input_sha256=stable_hash({"input": kind.value}),
            public_identity_content_hash=stable_hash({"identity": kind.value}),
        )
        for kind in AttemptBuilderKind
    )


def _manifest() -> FormalRunManifest:
    return FormalRunManifest.create(protocol_freeze=_freeze(), slots=_slots())


def _ledger(tmp_path: Path, manifest: FormalRunManifest) -> FormalRunLedger:
    return FormalRunLedger.create(
        directory=tmp_path / "ledger",
        manifest=manifest,
    )


def _request(tmp_path: Path, slot: FormalRunSlot) -> AttemptRequest:
    exact_input = tmp_path / "inputs" / f"{slot.builder_kind.value}.json"
    exact_input.parent.mkdir(exist_ok=True)
    exact_input.write_text("{}\n", encoding="utf-8")
    return AttemptRequest(
        run_id="formal-run-test",
        attempt_id=slot.attempt_id,
        builder_kind=slot.builder_kind,
        exact_input_path=exact_input,
        exact_input_sha256=slot.exact_input_sha256,
        private_bundle_directory=tmp_path / "private" / slot.builder_kind.value,
    )


def _prepared(
    tmp_path: Path,
    request: AttemptRequest,
    slot: FormalRunSlot,
    *,
    run_group_id: str,
) -> PreparedSuccessfulAttempt:
    return PreparedSuccessfulAttempt(
        private=UnpublishedAttemptBundle(
            directory=(tmp_path / "prepared" / slot.builder_kind.value).resolve()
        ),
        request=request,
        public_identity_sha256=stable_hash({"identity-file": slot.builder_kind.value}),
        public_identity_content_hash=slot.public_identity_content_hash,
        source_logs=(
            AttemptSourceLogDigest(
                name="events.jsonl",
                kind=AttemptSourceLogKind.FULL_METHOD,
                sha256=stable_hash({"events": slot.builder_kind.value}),
            ),
        )
        if slot.builder_kind is not AttemptBuilderKind.NO_BAYESIAN
        else (
            AttemptSourceLogDigest(
                name="arm-events.jsonl",
                kind=AttemptSourceLogKind.FLOW_ONLY,
                sha256=stable_hash({"arm-events": slot.builder_kind.value}),
            ),
            AttemptSourceLogDigest(
                name="events.jsonl",
                kind=AttemptSourceLogKind.OPERATIONAL,
                sha256=stable_hash({"events": slot.builder_kind.value}),
            ),
        ),
        final_training_artifact=FinalTrainingArtifact(
            artifact_sha256=stable_hash({"artifact": slot.builder_kind.value}),
            runtime_state_sha256=stable_hash({"runtime": slot.builder_kind.value}),
            policy_snapshot_id=f"snapshot-{slot.builder_kind.value}",
            library_version=stable_hash({"library": slot.builder_kind.value}),
            optimizer_step=1,
        ),
        outcome_sha256=stable_hash({"outcome": slot.builder_kind.value}),
        formal_run_group_id=run_group_id,
    )


def _terminal_bundle(
    tmp_path: Path,
    ledger: FormalRunLedger,
    slot: FormalRunSlot,
) -> PublishedSuccessfulAttemptBundle:
    request = _request(tmp_path, slot)
    claim = ledger.claim(request)
    ledger.consume_private_launch(ledger.launch_for(request=request, claim_secret=claim))
    prepared = _prepared(
        tmp_path,
        request,
        slot,
        run_group_id=ledger.manifest.run_group_id,
    )
    admission = ledger.record_success(request=request, prepared=prepared)
    return PublishedSuccessfulAttemptBundle(
        directory=tmp_path / "published" / slot.builder_kind.value,
        run_id=request.run_id,
        attempt_id=slot.attempt_id,
        builder_kind=slot.builder_kind,
        exact_input_sha256=slot.exact_input_sha256,
        public_identity_sha256=prepared.public_identity_sha256,
        public_identity_content_hash=slot.public_identity_content_hash,
        source_logs=prepared.source_logs,
        final_training_artifact=prepared.final_training_artifact,
        outcome_sha256=prepared.outcome_sha256,
        formal_publication=admission,
    )


def test_formal_manifest_is_derived_from_one_freeze_and_closed_seven_arm_slots() -> None:
    manifest = _manifest()

    assert FormalRunManifest.from_value(manifest.to_value()) == manifest
    assert tuple(slot.builder_kind for slot in manifest.slots) == tuple(AttemptBuilderKind)
    with pytest.raises(ValueError):
        FormalRunManifest.create(protocol_freeze=_freeze(), slots=manifest.slots[:-1])


def test_formal_ledger_is_bound_directly_to_its_manifest(tmp_path: Path) -> None:
    manifest = _manifest()
    ledger = FormalRunLedger.create(
        directory=tmp_path / "ledger",
        manifest=manifest,
    )
    assert ledger.manifest == manifest
    assert FormalRunLedger.open(directory=ledger.directory).manifest == manifest


def test_formal_ledger_claims_each_slot_once_and_failed_slots_cannot_be_replaced(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    ledger = _ledger(tmp_path, manifest)
    slot = manifest.slots[0]
    request = _request(tmp_path, slot)

    ledger.claim(request)
    outcome_path = tmp_path / "failed-outcome.json"
    outcome_path.write_text('{"outcome_type":"failed"}\n', encoding="utf-8")
    terminal = ledger.record_failure(request=request, outcome_path=outcome_path)

    assert terminal.status is FormalRunTerminalStatus.FAILED
    assert terminal.outcome_sha256 is not None
    with pytest.raises(ValueError):
        ledger.claim(request)
    with pytest.raises(ValueError):
        ledger.require_complete_successes(())
    assert FormalRunLedger.open_exact(directory=tmp_path / "ledger", manifest=manifest) == ledger
    with pytest.raises(FileExistsError):
        FormalRunLedger.create(
            directory=tmp_path / "ledger",
            manifest=manifest,
        )


def test_formal_worker_requires_and_consumes_one_claim(tmp_path: Path) -> None:
    manifest = _manifest()
    ledger = _ledger(tmp_path, manifest)
    request = _request(tmp_path, manifest.slots[0])

    with pytest.raises(TypeError):
        asyncio.run(run_private_benchmark_attempt_worker(request))  # type: ignore[arg-type]

    claim = ledger.claim(request)
    launch = ledger.launch_for(request=request, claim_secret=claim)
    ledger.consume_private_launch(launch)
    with pytest.raises(FileExistsError):
        ledger.consume_private_launch(launch)


def test_formal_worker_sets_frozen_backend_flags_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    manifest = _manifest()
    ledger = _ledger(tmp_path, manifest)
    request = _request(tmp_path, manifest.slots[0])
    claim = ledger.claim(request)
    launch = ledger.launch_for(request=request, claim_secret=claim)
    observed: dict[str, bool] = {}

    async def _execute(
        child_request: AttemptRequest,
        *,
        build: object,
        formal_run_group_id: str | None = None,
    ) -> None:
        assert child_request == request
        assert build is private_attempt_worker.build_private_benchmark_attempt
        assert formal_run_group_id == manifest.run_group_id
        observed.update(
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        )

    monkeypatch.setattr(private_attempt_worker, "execute_attempt_request", _execute)
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
    )
    try:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        asyncio.run(private_attempt_worker.run_private_benchmark_attempt_worker(launch))
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cudnn.deterministic = previous[1]
        torch.backends.cudnn.benchmark = previous[2]
        torch.backends.cuda.matmul.allow_tf32 = previous[3]

    assert observed == {
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_allow_tf32": False,
    }


def test_formal_aggregate_accepts_only_the_manifest_selected_terminal_successes(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    ledger = _ledger(tmp_path, manifest)
    bundles = tuple(_terminal_bundle(tmp_path, ledger, slot) for slot in manifest.slots)

    accepted = ledger.require_complete_successes(tuple(reversed(bundles)))
    assert tuple(bundle.builder_kind for bundle in accepted) == tuple(AttemptBuilderKind)

    mismatched = replace(bundles[0], exact_input_sha256=stable_hash("different-input"))
    with pytest.raises(ValueError):
        ledger.require_complete_successes((mismatched, *bundles[1:]))


def test_private_supervisor_records_terminal_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    ledger = _ledger(tmp_path, manifest)
    slot = manifest.slots[0]
    request = _request(tmp_path, slot)
    publisher = AttemptPublisher(
        published_root=tmp_path / "published-root",
        quarantine_root=tmp_path / "quarantine-root",
    )

    def _run_prepared(
        self: PrivateBenchmarkAttemptSupervisor,
        observed: AttemptRequest,
        claim: FormalRunClaimSecret,
    ) -> PreparedSuccessfulAttempt:
        assert observed == request
        assert claim is not None
        ledger.consume_private_launch(ledger.launch_for(request=observed, claim_secret=claim))
        return _prepared(
            tmp_path,
            request,
            slot,
            run_group_id=ledger.manifest.run_group_id,
        )

    def _publish(
        self: AttemptPublisher,
        prepared: PreparedSuccessfulAttempt,
        *,
        formal_publication: object,
    ) -> PublishedSuccessfulAttemptBundle:
        assert self is publisher
        assert ledger.terminals()[0].status is FormalRunTerminalStatus.SUCCEEDED
        assert formal_publication is not None
        return PublishedSuccessfulAttemptBundle(
            directory=tmp_path / "published" / slot.builder_kind.value,
            run_id=request.run_id,
            attempt_id=slot.attempt_id,
            builder_kind=slot.builder_kind,
            exact_input_sha256=slot.exact_input_sha256,
            public_identity_sha256=prepared.public_identity_sha256,
            public_identity_content_hash=prepared.public_identity_content_hash,
            source_logs=prepared.source_logs,
            final_training_artifact=prepared.final_training_artifact,
            outcome_sha256=prepared.outcome_sha256,
            formal_publication=formal_publication,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(PrivateBenchmarkAttemptSupervisor, "_run_claimed", _run_prepared)
    monkeypatch.setattr(AttemptPublisher, "publish_prepared", _publish)
    supervisor = PrivateBenchmarkAttemptSupervisor(
        publisher=publisher,
        formal_run_ledger=ledger,
    )

    assert supervisor.run(request).attempt_id == request.attempt_id
    assert ledger.terminals()[0].status is FormalRunTerminalStatus.SUCCEEDED
    with pytest.raises(ValueError):
        supervisor.run(request)


def test_private_supervisor_records_a_failed_slot_without_permitting_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    ledger = _ledger(tmp_path, manifest)
    slot = manifest.slots[0]
    request = _request(tmp_path, slot)
    quarantine = tmp_path / "quarantine" / slot.builder_kind.value
    quarantine.mkdir(parents=True)
    (quarantine / "outcome.json").write_text('{"outcome_type":"failed"}\n', encoding="utf-8")

    def _failed(
        self: PrivateBenchmarkAttemptSupervisor,
        observed: AttemptRequest,
        claim: object,
    ) -> PreparedSuccessfulAttempt:
        assert observed == request
        assert claim is not None
        raise AttemptProcessFailedError(QuarantinedFailedAttemptBundle(quarantine))

    monkeypatch.setattr(PrivateBenchmarkAttemptSupervisor, "_run_claimed", _failed)
    supervisor = PrivateBenchmarkAttemptSupervisor(
        publisher=AttemptPublisher(
            published_root=tmp_path / "published-root",
            quarantine_root=tmp_path / "quarantine-root",
        ),
        formal_run_ledger=ledger,
    )

    with pytest.raises(AttemptProcessFailedError):
        supervisor.run(request)
    assert ledger.terminals()[0].status is FormalRunTerminalStatus.FAILED
    with pytest.raises(ValueError):
        supervisor.run(request)
