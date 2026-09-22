from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import skillev.runtime as runtime
from skillev.contracts import stable_hash
from skillev.runtime import (
    EventEnvelope,
    EventType,
    FullRetrievedSkillContext,
    LiveAttemptEventLog,
    RetrievalInclusionReason,
    RuntimeEventEmitter,
    SkillApplicability,
    SkillCatalog,
    SkillDocument,
    SkillManifest,
    SkillRequirement,
)
from skillev.runtime.event_log_reader import read_event_history


def _document() -> SkillDocument:
    applicability = SkillApplicability(
        task_families=("statistics",),
        contexts=("statistics",),
        required_tools=(),
        excluded_contexts=(),
    )
    requirements = (
        SkillRequirement(
            requirement_id="preserve-outlier-check",
            text="Inspect outliers before reporting the estimate.",
        ),
    )
    content = {
        "applicability": applicability.to_value(),
        "instructions": "Inspect outliers, then compute and report a robust center.",
        "requirements": [item.to_value() for item in requirements],
        "summary": "Estimate a center robustly.",
        "title": "Robust center estimation",
    }
    return SkillDocument(
        manifest=SkillManifest(
            skill_id="robust-estimate",
            version="1",
            content_hash=stable_hash(content),
            input_schema_id="samples",
            output_schema_id="estimate",
            license_id="unit-test-license-v2",
            provenance_hash=stable_hash({"source": "unit-test"}),
        ),
        title="Robust center estimation",
        summary="Estimate a center robustly.",
        instructions="Inspect outliers, then compute and report a robust center.",
        applicability=applicability,
        requirements=requirements,
    )


def test_live_writer_is_fresh_and_offline_reader_is_separate(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = LiveAttemptEventLog(path, run_id="run", attempt_id="attempt")
    emitter = RuntimeEventEmitter(
        log=log,
        producer_id="runtime",
        clock=lambda: "2026-01-01T00:00:00Z",
    )
    emitted = emitter.emit(EventType.AGENT_TURN_STARTED, {"turn": 1})
    assert read_event_history(path) == (emitted,)
    with pytest.raises(FileExistsError):
        LiveAttemptEventLog(path, run_id="run", attempt_id="attempt")

    wrong_attempt = EventEnvelope.create(
        event_type=EventType.AGENT_TURN_STARTED,
        run_id="run",
        attempt_id="other",
        producer_id="runtime",
        producer_seq=2,
        occurred_at="2026-01-01T00:00:01Z",
        payload={"turn": 2},
    )
    with pytest.raises(ValueError):
        log.append(wrong_attempt)


def test_live_writer_resumes_only_through_explicit_recovery(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first_log = LiveAttemptEventLog(path, run_id="run", attempt_id="attempt")
    first_emitter = RuntimeEventEmitter(
        log=first_log,
        producer_id="runtime",
        clock=lambda: "2026-01-01T00:00:00Z",
    )
    first = first_emitter.emit(EventType.AGENT_TURN_STARTED, {"turn": 1})

    resumed_log = LiveAttemptEventLog.resume(path, run_id="run", attempt_id="attempt")
    resumed_emitter = RuntimeEventEmitter(
        log=resumed_log,
        producer_id="runtime",
        clock=lambda: "2026-01-01T00:00:01Z",
    )
    second = resumed_emitter.emit(EventType.AGENT_TURN_STARTED, {"turn": 2})

    assert second.producer_seq == 2
    assert read_event_history(path) == (first, second)
    assert resumed_log.append_idempotent(second) is False
    with pytest.raises(ValueError):
        LiveAttemptEventLog.resume(path, run_id="run", attempt_id="other")


def test_prepared_event_sequence_replays_exactly_after_crash(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = LiveAttemptEventLog(path, run_id="run", attempt_id="attempt")
    emitter = RuntimeEventEmitter(
        log=log,
        producer_id="runtime",
        clock=lambda: "2026-01-01T00:00:00Z",
    )
    prepared = emitter.prepare_many(
        (
            (EventType.TRAINING_STEP_COMMITTED, {"step": 1}),
            (EventType.EVOLUTION_PHASE_OPENED, {"phase": 1}),
        )
    )
    emitter.publish_prepared(prepared[0])

    resumed_log = LiveAttemptEventLog.resume(path, run_id="run", attempt_id="attempt")
    resumed = RuntimeEventEmitter(
        log=resumed_log,
        producer_id="runtime",
        clock=lambda: "2099-01-01T00:00:00Z",
    )
    resumed.publish_prepared(prepared[1])

    assert read_event_history(path) == prepared


def test_prepared_reconciliation_verifies_prefix_and_appends_suffix(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = LiveAttemptEventLog(path, run_id="run", attempt_id="attempt")
    emitter = RuntimeEventEmitter(log=log, producer_id="runtime")
    prepared = emitter.prepare_many(
        (
            (EventType.TRAINING_STEP_COMMITTED, {"step": 1}),
            (EventType.EVOLUTION_PHASE_OPENED, {"phase": 1}),
        )
    )
    emitter.publish_prepared(prepared[0])

    resumed = RuntimeEventEmitter(
        log=LiveAttemptEventLog.resume(path, run_id="run", attempt_id="attempt"),
        producer_id="runtime",
    )
    resumed.reconcile_prepared(prepared)

    assert read_event_history(path) == prepared


def test_active_skill_context_always_contains_full_instructions() -> None:
    document = _document()
    catalog = SkillCatalog((document,))
    context = FullRetrievedSkillContext(
        metadata=catalog.metadata(document.manifest.skill_id),
        content=document.instructions,
        inclusion_reason=RetrievalInclusionReason.APPLICABILITY_MATCH,
    )
    assert FullRetrievedSkillContext.from_value(context.to_value()) == context
    assert context.content == document.instructions
    assert catalog.document(document.manifest.skill_id) is document


def test_progressive_disclosure_surface_is_absent_from_active_runtime() -> None:
    for retired in (
        "DisclosureLevel",
        "RetrievedSkillContext",
        "SkillActivation",
        "SkillDisclosure",
    ):
        assert retired not in runtime.__all__
        assert not hasattr(runtime, retired)
    source = inspect.getsource(SkillCatalog)
    assert "activate" not in source
    assert "discover" not in source


def test_skill_document_rejects_content_that_does_not_match_manifest() -> None:
    document = _document()
    with pytest.raises(ValueError):
        SkillDocument(
            manifest=SkillManifest(
                skill_id=document.manifest.skill_id,
                version=document.manifest.version,
                content_hash=stable_hash({"different": "content"}),
                input_schema_id=document.manifest.input_schema_id,
                output_schema_id=document.manifest.output_schema_id,
                license_id=document.manifest.license_id,
                provenance_hash=document.manifest.provenance_hash,
            ),
            title=document.title,
            summary=document.summary,
            instructions=document.instructions,
            applicability=document.applicability,
            requirements=document.requirements,
        )
