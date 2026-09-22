from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillev.audit.source_reducer import AuditSourceReducer
from skillev.contracts import canonical_json
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptFailed,
    AttemptFailureCode,
    AttemptFailureStage,
    AttemptPublisher,
    AttemptRequest,
    AttemptSourceLogKind,
    AttemptSucceeded,
    FinalTrainingArtifact,
    FlowOnlyAttemptSummary,
    FullAttemptSummary,
    PublishedSuccessfulAttemptBundle,
    QuarantinedFailedAttemptBundle,
    UnpublishedAttemptBundle,
)
from skillev.runtime.attempt_publication import sha256_bytes


def _request(
    private: UnpublishedAttemptBundle,
    *,
    attempt_id: str,
    builder_kind: AttemptBuilderKind,
) -> AttemptRequest:
    return AttemptRequest(
        run_id="publication-test-run",
        attempt_id=attempt_id,
        builder_kind=builder_kind,
        exact_input_path=private.directory / "exact-input.json",
        exact_input_sha256=sha256_bytes(b"publication-test-exact-input"),
        private_bundle_directory=private.directory,
    )


def _summary(builder_kind: AttemptBuilderKind) -> FullAttemptSummary | FlowOnlyAttemptSummary:
    if builder_kind is AttemptBuilderKind.NO_BAYESIAN:
        return FlowOnlyAttemptSummary(
            reports=(),
            planned_training_steps_this_attempt=0,
            completed_training_steps_this_attempt=0,
            actions_committed_this_attempt=0,
            cycles_committed_this_attempt=0,
            cycles_committed_in_run=0,
            initial_optimizer_step=0,
            final_optimizer_step=0,
            final_library_version="library-test",
            final_policy_snapshot_id="snapshot-test",
        )
    return FullAttemptSummary(
        reports=(),
        planned_training_steps_this_attempt=0,
        completed_training_steps_this_attempt=0,
        actions_committed_this_attempt=0,
        cycles_committed_this_attempt=0,
        cycles_committed_in_run=0,
        initial_optimizer_step=0,
        final_optimizer_step=0,
        final_library_version="library-test",
        final_policy_snapshot_id="snapshot-test",
    )


def _final_training_artifact() -> FinalTrainingArtifact:
    """A path-free final checkpoint descriptor for publication-only tests."""

    return FinalTrainingArtifact(
        artifact_sha256=sha256_bytes(b"publication-test-final-artifact"),
        runtime_state_sha256=sha256_bytes(b"publication-test-runtime-state"),
        policy_snapshot_id="snapshot-test",
        library_version="library-test",
        optimizer_step=0,
    )


def _published_success(
    tmp_path: Path,
    *,
    builder_kind: AttemptBuilderKind,
) -> PublishedSuccessfulAttemptBundle:
    attempt_id = f"published-{builder_kind.value}"
    private_root = tmp_path / "private"
    private_root.mkdir()
    private = UnpublishedAttemptBundle.create(private_root / attempt_id)
    request = _request(private, attempt_id=attempt_id, builder_kind=builder_kind)
    private.event_log_path.write_text('{"source":"operational"}\n', encoding="utf-8")
    if builder_kind is AttemptBuilderKind.NO_BAYESIAN:
        private.arm_event_log_path.write_text('{"source":"flow-only"}\n', encoding="utf-8")
    identity_sha256, identity_content_hash = private.write_public_identity_once(
        {"builder_kind": builder_kind.value, "format": "publication-test-identity@1"}
    )
    source_logs = private.source_log_digests(builder_kind)
    private.write_outcome_once(
        AttemptSucceeded(
            attempt_id=attempt_id,
            builder_kind=builder_kind,
            exact_input_sha256=request.exact_input_sha256,
            public_identity_sha256=identity_sha256,
            public_identity_content_hash=identity_content_hash,
            source_logs=source_logs,
            summary=_summary(builder_kind),
            final_training_artifact=_final_training_artifact(),
        ).to_value()
    )
    publisher = AttemptPublisher(
        published_root=tmp_path / "published",
        quarantine_root=tmp_path / "quarantine",
    )
    published = publisher.finalize(private, request=request)
    assert isinstance(published, PublishedSuccessfulAttemptBundle)
    return published


@pytest.mark.parametrize(
    ("fault_stage", "failure_code"),
    [
        ("projection-commit", AttemptFailureCode.INTERNAL_ATTEMPT_FAILURE),
        ("phase-source-append", AttemptFailureCode.EVENT_APPEND_FAILED),
        ("authoring", AttemptFailureCode.AUTHORING_FAILED),
        ("library-apply", AttemptFailureCode.LIBRARY_APPLY_FAILED),
        ("partition-reset", AttemptFailureCode.PARTITION_RESET_FAILED),
        ("cycle-source-append", AttemptFailureCode.EVENT_APPEND_FAILED),
    ],
)
def test_every_failed_attempt_stage_is_quarantined_and_never_published(
    tmp_path: Path,
    fault_stage: str,
    failure_code: AttemptFailureCode,
) -> None:
    attempt_id = f"failed-{fault_stage}"
    private_root = tmp_path / "private"
    private_root.mkdir()
    private = UnpublishedAttemptBundle.create(private_root / attempt_id)
    request = _request(
        private,
        attempt_id=attempt_id,
        builder_kind=AttemptBuilderKind.FULL,
    )
    private.event_log_path.write_text(
        '{"attempt-private-partial-source":"not-publishable"}\n',
        encoding="utf-8",
    )
    private.write_outcome_once(
        AttemptFailed(
            attempt_id=attempt_id,
            builder_kind=request.builder_kind,
            exact_input_sha256=request.exact_input_sha256,
            code=failure_code,
            stage=AttemptFailureStage.EXECUTION,
            exception_type="builtins.RuntimeError",
        ).to_value()
    )
    publisher = AttemptPublisher(
        published_root=tmp_path / "published",
        quarantine_root=tmp_path / "quarantine",
    )

    finalized = publisher.finalize(
        private,
        request=request,
    )

    assert isinstance(finalized, QuarantinedFailedAttemptBundle)
    assert tuple((tmp_path / "published").iterdir()) == ()
    assert tuple((tmp_path / "quarantine").iterdir()) == (finalized.directory,)
    with pytest.raises(TypeError):
        AuditSourceReducer().consume_published_bundle(finalized)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("builder_kind", "target"),
    [
        (AttemptBuilderKind.FULL, "outcome"),
        (AttemptBuilderKind.FULL, "public-identity"),
        (AttemptBuilderKind.FULL, "events"),
        (AttemptBuilderKind.FULL, "manifest-builder-kind"),
        (AttemptBuilderKind.FULL, "manifest-exact-input"),
        (AttemptBuilderKind.FULL, "manifest-extra-log"),
        (AttemptBuilderKind.NO_BAYESIAN, "arm-events"),
    ],
)
def test_open_exact_rejects_any_bound_publication_component_tamper(
    tmp_path: Path,
    builder_kind: AttemptBuilderKind,
    target: str,
) -> None:
    bundle = _published_success(tmp_path, builder_kind=builder_kind)

    if target == "outcome":
        bundle.outcome_path.write_text(
            bundle.outcome_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
    elif target == "public-identity":
        bundle.public_identity_path.write_text(
            bundle.public_identity_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
    elif target == "events":
        bundle.event_log_path.write_text(
            bundle.event_log_path.read_text(encoding="utf-8") + '{"tampered":true}\n',
            encoding="utf-8",
        )
    elif target == "arm-events":
        arm_log = bundle.source_log_path(bundle.source_log(AttemptSourceLogKind.FLOW_ONLY))
        arm_log.write_text(
            arm_log.read_text(encoding="utf-8") + '{"tampered":true}\n',
            encoding="utf-8",
        )
    else:
        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        if target == "manifest-builder-kind":
            manifest["builder_kind"] = AttemptBuilderKind.NO_BAYESIAN.value
        elif target == "manifest-exact-input":
            manifest["exact_input_sha256"] = "sha256:" + "f" * 64
        elif target == "manifest-extra-log":
            manifest["source_logs"].append(
                {
                    "kind": "operational-events",
                    "name": "unused-events.jsonl",
                    "sha256": "sha256:" + "0" * 64,
                }
            )
        else:
            raise AssertionError(f"unknown publication tamper target: {target}")
        bundle.manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        PublishedSuccessfulAttemptBundle.open_exact(bundle.directory)
