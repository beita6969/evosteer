import asyncio
import json
from dataclasses import asdict, replace

import pytest

from skillev.training.quality_monitor import (
    QualityCheckpointStop,
    QualityProbeCoordinator,
    load_quality_policy,
)
from tests.training.test_quality_gate import policy, probe


def test_quality_pause_persists_missing_evidence_and_new_probe_does_not_erase_it(tmp_path):
    monitor = QualityCheckpointStop(tmp_path, policy())
    assert monitor.check(policy_step=0, policy_snapshot_id="policy-0")
    (tmp_path / "probe-00000000.json").write_text(json.dumps(asdict(probe(0, 1.0))))
    recovered = QualityCheckpointStop(tmp_path, load_quality_policy(tmp_path / "policy.json"))
    assert not recovered.check(policy_step=0, policy_snapshot_id="policy-0")
    assert not recovered.check(policy_step=0, policy_snapshot_id="policy-0")
    history = json.loads((tmp_path / "decisions-00000000.json").read_text())
    assert [item["status"] for item in history] == ["metrics-missing", "verified"]
    with pytest.raises(ValueError):
        QualityCheckpointStop(tmp_path, replace(policy(), cadence=5))


def test_fixed_regression_requests_stop_without_mutating_probe(tmp_path):
    monitor = QualityCheckpointStop(tmp_path, policy())
    for step, score in ((0, 1.0), (2, 0.6), (4, 0.5)):
        (tmp_path / f"probe-{step:08d}.json").write_text(json.dumps(asdict(probe(step, score))))
    original = (tmp_path / "probe-00000004.json").read_text()
    assert monitor.check(policy_step=4, policy_snapshot_id="policy-4")
    assert (tmp_path / "probe-00000004.json").read_text() == original


def test_async_probe_collects_only_due_versions_and_reuses_persisted_result(tmp_path):
    monitor = QualityCheckpointStop(tmp_path, policy())
    calls = []

    async def collect(root, step, snapshot):
        calls.append((step, snapshot))
        await asyncio.sleep(0)
        result = probe(step, 1.0)
        root.mkdir(parents=True)
        (root / f"probe-{step:08d}.json").write_text(json.dumps(asdict(result)))
        return result

    coordinator = QualityProbeCoordinator(monitor, collect)
    assert not asyncio.run(coordinator.check(policy_step=0, policy_snapshot_id="policy-0"))
    assert not asyncio.run(coordinator.check(policy_step=1, policy_snapshot_id="policy-1"))
    assert calls == [(0, "policy-0")]
    # Crash after collector persistence but before the outer installation.
    (tmp_path / "probe-00000000.json").unlink()
    resumed = QualityProbeCoordinator(monitor, collect)
    assert not asyncio.run(resumed.check(policy_step=0, policy_snapshot_id="policy-0"))
    assert calls == [(0, "policy-0")]


def test_probe_infrastructure_failure_pauses_and_does_not_resample(tmp_path):
    calls = []

    async def collect(root, step, snapshot):
        calls.append(step)
        raise TimeoutError("No authoritative native verdict")

    monitor = QualityCheckpointStop(tmp_path, policy())
    coordinator = QualityProbeCoordinator(monitor, collect)
    for _ in range(2):
        assert asyncio.run(coordinator.check(policy_step=0, policy_snapshot_id="policy-0"))
    assert calls == [0]
    assert not (tmp_path / "probe-00000000.json").exists()
    decisions = json.loads((tmp_path / "decisions-00000000.json").read_text())
    assert decisions[-1]["status"] == "metrics-missing"


def test_probe_identity_or_programming_failure_is_not_masked_as_availability(tmp_path):
    calls = []

    async def collect(root, step, snapshot):
        calls.append(step)
        raise ValueError("a different scientific identity")

    coordinator = QualityProbeCoordinator(QualityCheckpointStop(tmp_path, policy()), collect)
    with pytest.raises(ValueError):
        asyncio.run(coordinator.check(policy_step=0, policy_snapshot_id="policy-0"))
    assert not (tmp_path / "probe-00000000.json").exists()
    assert asyncio.run(coordinator.check(policy_step=0, policy_snapshot_id="policy-0"))
    assert calls == [0]


def test_legacy_policy_defaults_and_new_initial_floor_survive_restart(tmp_path):
    old = asdict(policy())
    for rule in old["rules"]:
        rule.pop("baseline_minimum")
    path = tmp_path / "policy.json"
    original = json.dumps(old)
    path.write_text(original)
    monitor = QualityCheckpointStop(tmp_path, policy())
    assert monitor.policy.rules[0].baseline_minimum is None
    assert path.read_text() == original
    stronger = replace(policy(), rules=(replace(policy().rules[0], baseline_minimum=0.95),))
    with pytest.raises(ValueError):
        QualityCheckpointStop(tmp_path, stronger)
    new_root = tmp_path / "new-condition"
    QualityCheckpointStop(new_root, stronger)
    recovered = QualityCheckpointStop(new_root, load_quality_policy(new_root / "policy.json"))
    assert recovered.policy == stronger
    assert recovered.check(policy_step=0, policy_snapshot_id="policy-0")


def test_condition_branch_collects_baseline_at_restored_step_without_relabeling_t0(tmp_path):
    rules = replace(policy(), cadence=5, baseline_step=2, condition_id="new-reasoning")
    monitor = QualityCheckpointStop(tmp_path, rules)
    calls = []

    async def collect(root, step, snapshot):
        calls.append(step)
        return replace(probe(step, 1.0 if step == 2 else 0.5), condition_id=rules.condition_id)

    coordinator = QualityProbeCoordinator(monitor, collect)
    assert not asyncio.run(coordinator.check(policy_step=2, policy_snapshot_id="policy-2"))
    assert not (tmp_path / "probe-00000000.json").exists()
    assert not asyncio.run(coordinator.check(policy_step=3, policy_snapshot_id="policy-3"))
    assert not asyncio.run(coordinator.check(policy_step=5, policy_snapshot_id="policy-5"))
    assert asyncio.run(coordinator.check(policy_step=10, policy_snapshot_id="policy-10"))
    assert calls == [2, 5, 10]
    resumed = QualityCheckpointStop(tmp_path, load_quality_policy(tmp_path / "policy.json"))
    assert resumed.check(policy_step=10, policy_snapshot_id="policy-10")
    with pytest.raises(ValueError):
        resumed.check(policy_step=0, policy_snapshot_id="policy-0")
