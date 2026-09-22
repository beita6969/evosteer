from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from skillev.evidence import EvidenceRecord, EvidenceSplit, EvidenceStore, SplitEvidenceStores


def _record(event_id: str, *, split: EvidenceSplit = EvidenceSplit.TRAIN) -> EvidenceRecord:
    return EvidenceRecord(
        event_id=event_id,
        run_id="run-1",
        split=split,
        seed=0,
        train_step=1,
        question_id="question-id-only",
        task_type="qa",
        trajectory_id="trajectory-1",
        skill_uid="skill-search",
        skill_version_id="skill-search-v1",
        action_index=0,
        verified_outcome=1.0,
        verifier_name="terminal-reward",
        verifier_passed=True,
        failure_mode=None,
        reward=1.0,
        r_tilde=1.1,
        log_i=0.2,
        state_log_flow=0.3,
        skill_marginal_log_flow=0.4,
        ttb_residual=-0.1,
        token_count=32,
        turn_count=2,
        latency_ms=10.0,
        context_features={
            "context": "qa",
            "failure_mode": "none",
            "horizon_bucket": "short",
            "token_bucket": "small",
        },
        evidence_weight=0.8,
        created_at="2026-08-12T00:00:00+00:00",
    )


def _store(root: Path, name: str) -> EvidenceStore:
    return EvidenceStore(
        database_path=root / f"{name}.sqlite",
        event_log_path=root / f"{name}.jsonl",
    )


def test_append_is_idempotent_and_conflicting_event_identity_fails(tmp_path: Path) -> None:
    with _store(tmp_path, "train") as store:
        assert store.append(_record("event-1")) is True
        assert store.append(_record("event-1")) is False
        with pytest.raises(ValueError):
            store.append(replace(_record("event-1"), evidence_weight=0.5))

    lines = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event_id"] == "event-1"


def test_only_verified_training_evidence_can_update_the_posterior(tmp_path: Path) -> None:
    with _store(tmp_path, "train") as store:
        store.append(_record("train-good"))
        store.append(
            replace(
                _record("train-rejected"),
                event_id="train-rejected",
                verifier_passed=False,
            )
        )
        store.append(
            replace(
                _record("validation"),
                event_id="validation",
                split=EvidenceSplit.VALIDATION,
            )
        )

        assert [record.event_id for record in store.posterior_evidence()] == ["train-good"]


def test_split_router_keeps_eval_evidence_out_of_train_storage(tmp_path: Path) -> None:
    with _store(tmp_path, "train") as train, _store(tmp_path, "eval") as evaluation:
        stores = SplitEvidenceStores(train=train, evaluation=evaluation)
        stores.append(_record("train"))
        stores.append(replace(_record("test"), event_id="test", split=EvidenceSplit.TEST))

        assert train.get("test") is None
        assert evaluation.get("train") is None
        assert [record.event_id for record in stores.posterior_evidence()] == ["train"]


def test_snapshot_and_read_only_store_preserve_evidence_without_writes(tmp_path: Path) -> None:
    with _store(tmp_path, "train") as store:
        store.append(_record("event"))
        snapshot = store.snapshot_to(tmp_path / "snapshot.sqlite")

    with EvidenceStore(
        database_path=snapshot,
        event_log_path=tmp_path / "unused.jsonl",
        read_only=True,
    ) as restored:
        assert restored.get("event") == _record("event")
        with pytest.raises(PermissionError):
            restored.append(_record("other"))
