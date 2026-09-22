"""Only a complete controller boundary can advance exported training progress."""

import json
from types import SimpleNamespace

from skillev_private.experiments.training_controller import TrainingController


class Evidence:
    def __init__(self):
        self.step = 0
        self.pause = False
        self.calls = []

    def record_timings(self, timings):
        self.calls.append(("timings", timings))

    def observe_commits(self, committed_through):
        self.calls.append(("commit", committed_through))
        self.step = committed_through

    def status(self):
        return {
            "state": "RAW_MISSING" if self.pause else "pending" if self.step else "empty",
            "source_commit_id": f"commit-{self.step}" if self.step else None,
            "committed_at": "2026-09-13T00:00:00Z" if self.step else None,
            "pause_required": self.pause,
        }

    def drain(self):
        return self.status()


def test_optimizer_apply_alone_cannot_advance_controller_commit(tmp_path):
    evidence = Evidence()
    controller = TrainingController(
        tmp_path, resource_roles={"test": "synthetic"}, gpu_uuids=("GPU-test",), evidence=evidence
    )
    loop = SimpleNamespace(
        optimizer_step=0,
        execution_progress={"transaction_stage": "collecting"},
        rollout_progress={"trajectories": []},
        gradient_progress=None,
        finalized_timings=(),
    )
    controller.attach(loop)
    controller.set_state(None)
    assert controller.publish()["controller"]["state"] == "collecting"
    loop.optimizer_step = 1  # optimizer applied, wider transaction not committed
    loop.execution_progress = {"transaction_stage": "adapter-publication"}
    value = controller.publish()
    assert value["controller"]["last_committed_step"] == 0
    assert evidence.step == 0
    assert evidence.calls == []
    assert value["controller"]["state"] == "updating"
    assert not controller.checkpoint_boundary()  # now caller confirms full transaction
    value = json.loads((tmp_path / "controller-status.json").read_text())
    assert value["last_committed_step"] == evidence.step == 1
    assert value["source_commit_id"] == "commit-1"
    assert evidence.calls == [("timings", loop.finalized_timings), ("commit", 1)]
    controller.set_state("validating")
    assert controller.publish()["controller"]["state"] == "validating"
    controller.set_state("paused")
    assert controller.publish()["controller"]["state"] == "paused"


def test_evidence_failure_pauses_at_boundary_and_resource_scopes_do_not_add(tmp_path):
    evidence = Evidence()
    controller = TrainingController(
        tmp_path,
        resource_roles={"test": "synthetic"},
        gpu_uuids=("GPU-one", "GPU-one", "GPU-two"),
        evidence=evidence,
    )
    controller.attach(
        SimpleNamespace(
            optimizer_step=0,
            execution_progress={"transaction_stage": "idle"},
            rollout_progress={},
            gradient_progress=None,
            finalized_timings=(),
        )
    )
    evidence.pause = True
    assert controller.checkpoint_boundary()
    value = controller.publish()
    assert value["controller"]["evidence_status"] == "RAW_MISSING"
    assert value["resources"]["reserved_gpu_count"] == 2
    assert value["resources"]["process_reserved_gpu_hours"] >= 0
    assert "never-add" in value["resources"]["accounting"]


def test_mirror_failure_is_not_hidden_by_optimizer_completion(tmp_path):
    evidence = Evidence()
    controller = TrainingController(tmp_path, resource_roles={}, gpu_uuids=(), evidence=evidence)
    evidence.pause = True
    result = controller.finish_evidence()
    assert result["pause_required"]
    assert result["state"] == "RAW_MISSING"
    assert controller.publish()["controller"]["last_committed_step"] == 0


def test_resource_time_includes_drain_after_terminal_training_state(tmp_path, monkeypatch):
    from skillev_private.experiments import training_controller as module

    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    controller = TrainingController(tmp_path, resource_roles={}, gpu_uuids=("GPU-one",))
    now[0] = 110.0
    controller.set_state("paused")
    now[0] = 120.0  # closing I/O and gradient workers still belongs to this process
    controller.close_resources()
    now[0] = 140.0
    resource = controller.publish()["resources"]
    assert resource["controller_scope_closed"]
    assert resource["process_elapsed_seconds"] == 20
