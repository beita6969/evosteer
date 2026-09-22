from __future__ import annotations

from dataclasses import replace

import pytest

from skillev.contracts import (
    SkillEvolutionAction,
    SkillEvolutionRecord,
    SkillVersionRef,
    stable_hash,
)
from skillev.orchestration import (
    InMemorySkillLibraryStore,
    SkillLibraryPointerEvent,
    SkillLibrarySnapshot,
)
from skillev.runtime import SkillManifest


def _manifest(skill_id: str, version: int) -> SkillManifest:
    return SkillManifest(
        skill_id=skill_id,
        version=str(version),
        content_hash=stable_hash({"instructions": f"Instructions for {skill_id} v{version}"}),
        input_schema_id=f"{skill_id}-input",
        output_schema_id=f"{skill_id}-output",
        license_id="unit-test-license-v2",
        provenance_hash=stable_hash({"skill": skill_id, "version": version}),
    )


def _ref(manifest: SkillManifest) -> SkillVersionRef:
    return SkillVersionRef(
        manifest.skill_id,
        int(manifest.version),
        manifest.content_hash,
    )


def _record(
    sequence_number: int,
    action: SkillEvolutionAction,
    inputs: tuple[SkillVersionRef, ...],
    outputs: tuple[SkillVersionRef, ...],
    *,
    previous: SkillEvolutionRecord | None = None,
) -> SkillEvolutionRecord:
    return SkillEvolutionRecord(
        lineage_id="main-library",
        sequence_number=sequence_number,
        action=action,
        input_versions=inputs,
        output_versions=outputs,
        previous_record_hash=None if previous is None else previous.record_hash,
        evidence_hash=stable_hash({"batch": sequence_number}),
        reason="flow-directed update",
    )


def _two_snapshot_store() -> tuple[
    InMemorySkillLibraryStore,
    SkillLibrarySnapshot,
    SkillLibrarySnapshot,
]:
    alpha_v1 = _manifest("alpha", 1)
    beta_v1 = _manifest("beta", 1)
    initial = _record(
        1,
        SkillEvolutionAction.GENERATE,
        (),
        (_ref(alpha_v1), _ref(beta_v1)),
    )
    store = InMemorySkillLibraryStore("main-library")
    first = store.add_snapshot(initial, (alpha_v1, beta_v1))

    alpha_v2 = _manifest("alpha", 2)
    refined = _record(
        2,
        SkillEvolutionAction.REFINE,
        (_ref(alpha_v1),),
        (_ref(alpha_v2),),
        previous=initial,
    )
    second = store.add_snapshot(
        refined,
        (alpha_v2, beta_v1),
        parent_snapshot_hash=first.snapshot_hash,
    )
    return store, first, second


def test_child_snapshot_inherits_unchanged_manifests_and_binds_both_links() -> None:
    store, first, second = _two_snapshot_store()

    assert second.parent_snapshot_hash == first.snapshot_hash
    assert second.evolution_record_hash == store.record_for(second.snapshot_hash).record_hash
    assert second.manifests[1] is first.manifests[1]
    assert SkillLibrarySnapshot.from_value(second.to_value()) == second


def test_promotion_appends_pointer_history_and_selects_the_active_snapshot() -> None:
    store, first, second = _two_snapshot_store()

    initial_event = store.promote(first.snapshot_hash, reason="initial publication")
    history_before = store.pointer_events
    next_event = store.promote(second.snapshot_hash, reason="publish refinement")

    assert store.pointer_events[:1] == history_before
    assert next_event.follows(initial_event)
    assert store.active_snapshot == second
    assert SkillLibraryPointerEvent.from_value(next_event.to_value()) == next_event


def test_rollback_is_a_new_event_and_preserves_snapshots_and_promotions() -> None:
    store, first, second = _two_snapshot_store()
    store.promote(first.snapshot_hash, reason="initial publication")
    promotion = store.promote(second.snapshot_hash, reason="publish refinement")
    snapshots_before = store.snapshots

    rollback = store.rollback(first.snapshot_hash, reason="revert active library")

    assert rollback.previous_event_hash == promotion.event_hash
    assert store.active_snapshot == first
    assert store.snapshots == snapshots_before
    assert len(store.pointer_events) == 3
    assert InMemorySkillLibraryStore.from_value(store.to_value()).to_value() == store.to_value()


def test_snapshot_append_rejects_disconnected_parent_or_record_links() -> None:
    store, first, _ = _two_snapshot_store()
    alpha_v2 = _manifest("alpha", 2)
    alpha_v3 = _manifest("alpha", 3)
    parent_record = store.record_for(first.snapshot_hash)
    disconnected = _record(
        2,
        SkillEvolutionAction.REFINE,
        (_ref(alpha_v2),),
        (_ref(alpha_v3),),
        previous=parent_record,
    )
    snapshot = SkillLibrarySnapshot.for_record(
        disconnected,
        (alpha_v3, _manifest("beta", 1)),
        parent_snapshot=first,
    )

    with pytest.raises(ValueError):
        InMemorySkillLibraryStore("main-library").append_snapshot(snapshot, disconnected)

    wrong_record_link = replace(
        disconnected,
        previous_record_hash=stable_hash({"different": "record"}),
    )
    with pytest.raises(ValueError):
        store.append_snapshot(
            replace(snapshot, evolution_record_hash=wrong_record_link.record_hash),
            wrong_record_link,
        )


def test_pointer_replay_rejects_a_broken_event_chain() -> None:
    store, first, second = _two_snapshot_store()
    store.promote(first.snapshot_hash, reason="initial publication")
    promoted = store.promote(second.snapshot_hash, reason="publish refinement")
    broken = replace(
        promoted,
        sequence_number=promoted.sequence_number + 1,
        previous_event_hash=stable_hash({"different": "event"}),
    )

    with pytest.raises(ValueError):
        store.append_pointer_event(broken)
