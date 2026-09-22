from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from skillev.contracts import (
    EntityKind,
    Seed,
    artifact_hash,
    attempt_id,
    deterministic_id,
    event_id,
    stable_hash,
    utc_timestamp,
    validate_identifier,
    validate_sha256,
)


def test_deterministic_ids_follow_the_canonical_field_meaning() -> None:
    first = deterministic_id(EntityKind.TRAJECTORY, {"task": "t", "step": 2})
    reordered = deterministic_id("trajectory", {"step": 2, "task": "t"})
    changed = deterministic_id(EntityKind.TRAJECTORY, {"task": "t", "step": 3})

    assert first == reordered
    assert first != changed


def test_retry_and_event_identifiers_change_with_their_order() -> None:
    first_attempt = attempt_id(run_id="run-a", attempt_no=1, lease_epoch=1)
    retry = attempt_id(run_id="run-a", attempt_no=2, lease_epoch=2)
    payload_hash = stable_hash({"observation": "ok"})

    assert first_attempt != retry
    assert event_id(producer_id="runtime", producer_seq=1, payload_hash=payload_hash) != event_id(
        producer_id="runtime",
        producer_seq=2,
        payload_hash=payload_hash,
    )


def test_raw_byte_hash_does_not_normalize_artifact_content() -> None:
    semantic_hash = stable_hash({"alpha": 1})
    noncanonical_bytes_hash = artifact_hash(b'{ "alpha": 1 }')

    assert semantic_hash != noncanonical_bytes_hash
    assert validate_sha256(noncanonical_bytes_hash) == noncanonical_bytes_hash


def test_seed_and_identifier_boundaries_are_checked() -> None:
    assert Seed(0).value == 0
    assert Seed(2**64 - 1).value == 2**64 - 1
    assert validate_identifier("skill-version_2") == "skill-version_2"

    with pytest.raises(ValueError):
        Seed(-1)
    with pytest.raises(ValueError):
        Seed(2**64)
    with pytest.raises(ValueError):
        validate_identifier("Skill Version")


def test_timestamp_is_normalized_to_utc_and_requires_a_timezone() -> None:
    local = datetime(2026, 7, 17, 20, 0, tzinfo=timezone(timedelta(hours=8)))

    assert utc_timestamp(local) == utc_timestamp(datetime(2026, 7, 17, 12, 0, tzinfo=UTC))
    with pytest.raises(ValueError):
        utc_timestamp(datetime(2026, 7, 17, 12, 0))
