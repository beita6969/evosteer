"""CPU-only synthetic wiring. These tests are not real Qwen training evidence."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from skillev_private.experiments import local_training_diagnostic as diagnostic

from skillev.runtime.formal_sglang_runtime import BoundFormalSGLangRuntime
from skillev.runtime.service_topology import ServiceTopology
from skillev.training.performance_config import TrainingPerformanceConfig
from skillev.training.stopping import StopAfterCheckpoint
from tests.runtime.test_formal_sglang_runtime import _binding

CONFIG = Path("configs/training/bayesianimprove_development_handoff.yaml")


def bindings_file(tmp_path, **changes):
    value = {
        "format": diagnostic.FORMAT,
        **{
            key: str(tmp_path / key)
            for key in (
                "preparation",
                "dataset",
                "deployments",
                "evalplus_python",
                "evalplus_source_root",
                "evidence_mirror_root",
            )
        },
        "endpoint": "http://127.0.0.1:12345",
        "base_model": "synthetic-Qwen",
        "adapter_namespace": "synthetic-local-diagnostic",
        "training_gpu_uuids": ["GPU-gradient"],
        "serving_gpu_uuid": "GPU-actor",
        "data_condition": {"purpose": "synthetic-training-only"},
        **changes,
    }
    path = tmp_path / "bindings.json"
    path.write_text(json.dumps(value))
    return path


def local_runtime(tmp_path):
    return BoundFormalSGLangRuntime.build_local_diagnostic(
        binding=replace(_binding(tmp_path), performance=TrainingPerformanceConfig())
    )


@pytest.mark.parametrize("steps", [3, 5])
def test_schedule_only_override_retains_new_candidate_and_full_method(steps):
    original = diagnostic.load_fresh_config(CONFIG)
    candidate = diagnostic.diagnostic_config(original, total_steps=steps)
    changed = {k for k, v in original.to_value().items() if candidate.to_value().get(k) != v}
    assert changed == {"steps", "checkpoint_every"}
    assert candidate.run_plan.phase_search_steps == steps - 1
    assert candidate.run_plan.closure_steps == 1
    assert candidate.run_plan.maximum_cycles == 2
    app = candidate.application_config("synthetic-diagnostic")
    assert candidate.steps == steps
    assert app.trainer.execution.batch_size == 28
    assert app.trainer.checkpoint.every_n_steps == 1
    assert app.evolution.cold_start == original.cold_start
    assert app.evolution.cold_start.min_batches == 2
    assert app.diagnostics.window_size == 50
    assert app.trainer.method.temperature_beta == 1
    assert (app.calibration.alpha_0, app.calibration.beta_0) == (1, 1)
    assert candidate.reasoning_modes == original.reasoning_modes
    assert candidate.domain_task_budgets == original.domain_task_budgets
    assert (
        candidate.sampling_config.task_semantic_guidance
        == original.sampling_config.task_semantic_guidance
    )
    assert candidate.sampling_config.skill_exposure == "catalog-then-read@1"


@pytest.mark.parametrize(
    "changes",
    [
        {"training_gpu_uuids": []},
        {"training_gpu_uuids": ["GPU-a", "GPU-b"]},
        {"training_gpu_uuids": ["GPU-actor"]},
        {"data_condition": None},
        {"format": "formal"},
        {"resume": "old46"},
        {"iid_baselines": "A0"},
    ],
)
def test_private_bindings_reject_inferred_or_wrong_topology_and_source_role(tmp_path, changes):
    with pytest.raises(ValueError):
        diagnostic.load_bindings(bindings_file(tmp_path, **changes))


def test_local_shared_factories_do_not_relax_formal_dependencies(tmp_path, monkeypatch):
    runtime = local_runtime(tmp_path)
    assert runtime.gradient_preparer is None
    shared = runtime.shared_dependencies()
    assert shared.workflow_resources is runtime.resources
    with pytest.raises(TypeError):
        runtime.dependencies()
    with pytest.raises(TypeError):
        BoundFormalSGLangRuntime.build(binding=runtime.binding, gradient_preparer=None)
    # Formal placement still requires two gradient owners.
    with pytest.raises(ValueError):
        ServiceTopology((), (), (), (), ("GPU-one",))
    captured = {}
    sentinel = object()

    def build(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(diagnostic.SKILLEVApplication, "build", build)
    storage = diagnostic.PrivateCheckpointStorageBinding(directory=str(tmp_path / "checkpoints"))
    assert (
        diagnostic.build_application(runtime=runtime, storage=storage, clock=lambda: "fixed")
        is sentinel
    )
    assert captured["gradient_preparer"] is None
    assert captured["workflow_resources"] is runtime.resources
    assert captured["generator_factory"] is not None
    assert captured["skill_author_factory"] is not None
    assert captured["step_adapter_publisher_factory"] is not None
    assert (
        captured["step_transaction_journal"].directory == tmp_path / "checkpoints/step-transactions"
    )


def test_handoff_requires_actual_same_owner_and_never_reuses_old_release(tmp_path):
    ready, release = tmp_path / "ready.json", tmp_path / "release.json"
    identity = {"owner_pid": 321, "gradient_gpu_uuid": "GPU-synthetic", "cuda_initialized": True}
    stop = StopAfterCheckpoint(tmp_path / "stop")

    async def handoff():
        async def release_once():
            while not ready.exists():
                await asyncio.sleep(0)
            assert json.loads(ready.read_text())["state"] == "cuda-ready-before-backbone"
            release.write_text(json.dumps(identity))

        task = asyncio.create_task(release_once())
        await diagnostic.owner_handoff(identity, ready, release, stop)
        await task

    asyncio.run(handoff())
    with pytest.raises(ValueError):
        asyncio.run(diagnostic.owner_handoff(identity, ready, release, stop))


def test_handoff_stop_and_timeout_prevent_model_loading(tmp_path, monkeypatch):
    stop = StopAfterCheckpoint(tmp_path / "stop")
    stop.request()
    with pytest.raises(InterruptedError):
        asyncio.run(
            diagnostic.owner_handoff(
                {"owner_pid": 1}, tmp_path / "ready", tmp_path / "release", stop
            )
        )
    stop = StopAfterCheckpoint(tmp_path / "new-stop")
    times = iter([0, 601])
    # Replace only the diagnostic's module reference, not asyncio's real clock.
    monkeypatch.setattr(diagnostic, "time", SimpleNamespace(monotonic=lambda: next(times)))
    with pytest.raises(TimeoutError):
        asyncio.run(
            diagnostic.owner_handoff(
                {"owner_pid": 1}, tmp_path / "ready2", tmp_path / "release2", stop
            )
        )


def test_cli_has_no_arbitrary_step_expansion_or_iid_admission(monkeypatch, tmp_path):
    for extra in (["--steps", "4"], ["--total-steps", "46"], ["--iid-baselines", "A0"]):
        monkeypatch.setattr(
            "sys.argv",
            [
                "diagnostic",
                "--config",
                str(CONFIG),
                "--bindings",
                str(tmp_path / "b"),
                "--runroot",
                str(tmp_path / "r"),
                *extra,
            ],
        )
        with pytest.raises(SystemExit) as error:
            diagnostic.main()
        assert error.value.code == 2


@pytest.mark.parametrize("pause", [False, True, "owner-failure", "append", "fresh-five"])
def test_complete_run_assembly_carries_candidate_identity_sources_and_drain(
    tmp_path, monkeypatch, pause
):
    """Production run orchestration, mocked CPU transport/model only; no real scores."""
    from collections import Counter
    from threading import Event

    from skillev_private.benchmarks.protocol_v13_training import build_protocol13_training_records

    from tests.benchmarks.test_protocol_v13_training import _sources

    bindings_path = bindings_file(tmp_path)
    bindings = diagnostic.load_bindings(bindings_path)
    rows = build_protocol13_training_records(_sources())
    bindings.dataset.write_text("\n".join(json.dumps(r.to_value()) for r in rows) + "\n")
    root = tmp_path / "new-diagnostic"
    seen = []
    appended = pause == "append"
    target_steps = 10 if appended else 5 if pause == "fresh-five" else 3
    config = diagnostic.diagnostic_config(
        diagnostic.load_fresh_config(CONFIG), total_steps=target_steps
    )
    resume = root / "checkpoints/cadence-step-00000003" if appended else None
    run_kwargs = {"resume": resume, "total_steps": target_steps}
    if appended:
        root.mkdir()
        (root / "events.jsonl").touch()
        for name in ("summary.json", "resolved-run-plan.json", "final-method-state.json"):
            (root / name).write_text("original Step3 declaration")
        source_config = diagnostic.diagnostic_config(diagnostic.load_fresh_config(CONFIG))
        append = SimpleNamespace(
            source_application=source_config.application_config(root.name),
            source_plan=source_config.run_plan,
            plan=source_config.run_plan.append(phase_search_steps=6, closure_steps=1),
            entries=(),
            snapshot=SimpleNamespace(
                optimizer_step=3,
                execution_state=SimpleNamespace(task_cursor=SimpleNamespace(cursor=84)),
            ),
            identity=lambda source, candidate, ids: (candidate, "synthetic-transition"),
        )
        monkeypatch.setattr(diagnostic, "require_append", lambda **_: append)
    backbone = SimpleNamespace(
        device="cuda",
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0,
        torch_dtype=config.base_dtype,
        base_model_path="synthetic-base",
        tokenizer_path=None,
    )
    checkpoint = SimpleNamespace(directory=str(tmp_path / "new-preparation"))
    monkeypatch.setattr(diagnostic, "_read_preparation", lambda *_: (backbone, checkpoint))
    monkeypatch.setattr(diagnostic, "initialization_condition", lambda _: {})

    def require_candidate(preparation, *, sampling):
        assert preparation == bindings.preparation
        assert sampling == config.sampling_config.to_value()
        seen.append("initialization-candidate")

    monkeypatch.setattr(diagnostic, "require_initialization_candidate", require_candidate)
    monkeypatch.setattr(diagnostic, "_validated_worker_interpreter", lambda p: p)
    monkeypatch.setattr(
        diagnostic, "resolve_mbpp_profile", lambda _: diagnostic.MBPPScorerProfile()
    )
    monkeypatch.setattr(
        diagnostic, "require_training_service", lambda *a, **k: seen.append("service-profile")
    )

    def initialize(uuid):
        if pause == "owner-failure":
            raise ValueError("synthetic mismatched physical device")
        return {"gradient_gpu_uuid": uuid}

    monkeypatch.setattr(diagnostic, "initialize_cuda_owner", initialize)
    runtime = local_runtime(tmp_path)
    monkeypatch.setattr(type(runtime.gateway), "read_serving_profile", lambda self: {})
    monkeypatch.setattr(type(runtime.gateway), "bind_serving_profile", lambda *a: None)
    monkeypatch.setattr(
        diagnostic.BoundFormalSGLangRuntime, "build_local_diagnostic", lambda **_: runtime
    )
    assembly = {}

    async def sessions(selected, **kwargs):
        assert len(selected) == target_steps * 28
        for offset in range(0, len(selected), 28):
            assert (
                sorted(
                    Counter(
                        r.episode.benchmark.value for r in selected[offset : offset + 28]
                    ).values()
                )
                == [4] * 7
            )
        assert kwargs["domain_rollout_budgets"] == config.domain_task_budgets
        assert kwargs["static_rollout_budget"].max_turns == 8
        assert kwargs["rollout_budget"].max_turns == 25
        seen.append("sessions")
        return tuple(r.input for r in selected), SimpleNamespace(
            alfworld_goal_binding="reset-public-goal@2"
        )

    monkeypatch.setattr(diagnostic, "build_protocol13_training_sessions", sessions)

    def public_identity(**kwargs):
        assembly["identity-inputs"] = kwargs
        return SimpleNamespace(
            application_config=kwargs["application"], run_plan=kwargs["run_plan"]
        )

    monkeypatch.setattr(diagnostic, "_public_identity", public_identity)
    loop = SimpleNamespace(
        optimizer_step=3 if appended else 0,
        finalized_timings=(),
        ledger=SimpleNamespace(assert_fully_settled=lambda: seen.append("settled")),
        configure_inflight=lambda *a, **k: seen.append("inflight"),
        enable_update_observation=lambda: seen.append("optimizer-observation"),
    )

    async def evolve(plan, stop_requested):
        assert ("fresh" in seen) is not appended
        assert "monitor" in seen
        assert plan == (append.plan if appended else config.run_plan)
        assert not await stop_requested()
        for step in range(4 if appended else 1, target_steps + 1):
            # This fixture supplies transaction boundaries, not real Adam evidence.
            loop.optimizer_step = step
            checkpoint_path = root / "checkpoints" / f"step-{step:08d}"
            checkpoint_path.mkdir(parents=True)
            (checkpoint_path / "COMPLETE").touch()
            if pause is True and step == 1:
                (root / "STOP_AFTER_CHECKPOINT").touch()
            if await stop_requested():
                raise diagnostic.TrainingPausedError(step, checkpoint_path)
        return SimpleNamespace(
            final_optimizer_step=target_steps,
            to_value=lambda: {"final_optimizer_step": target_steps},
        )

    app = SimpleNamespace(
        training_loop=loop, evolution_loop=SimpleNamespace(run=evolve), generator=object()
    )

    def build(**kwargs):
        seen.append("build")
        assembly.update(kwargs)
        assert kwargs["public_identity"].application_config == config.application_config(root.name)
        assert kwargs["public_identity"].run_plan == (append.plan if appended else config.run_plan)
        if appended:
            assert "seed_documents" not in kwargs
            assert "task_provider" not in kwargs
            assert len(kwargs["task_provider_factory"].tasks) == 280
            assert kwargs["plan_continuation"] == "synthetic-transition"
            assert kwargs["resume"] == resume
        else:
            assert kwargs["seed_documents"] == diagnostic.planned_seed_documents()
        return app

    monkeypatch.setattr(diagnostic, "build_application", build)

    def fresh_check(*args, **kwargs):
        from skillev.training.fresh_state import FreshNamespaces, _namespace_observations

        # Exercise the real bounded namespace check even though model state is
        # synthetic in this orchestration fixture. A premature mirror.lock fails.
        namespaces = FreshNamespaces((), (kwargs["root"] / "evidence",))
        assert all(row["empty"] for row in _namespace_observations(namespaces))
        assert "observer-created" not in seen
        seen.append("fresh")

    monkeypatch.setattr(diagnostic, "require_clean_initial_application", fresh_check)
    monkeypatch.setattr(
        diagnostic, "resolved_method_state", lambda _: {"optimizer_step": loop.optimizer_step}
    )
    real_observer = diagnostic.CommittedRunObserver

    def observer(*args, **kwargs):
        assert ("fresh" in seen) is not appended
        value = real_observer(*args, **kwargs)
        seen.append("observer-created")
        assert (root / "evidence/mirror.lock").is_file()
        close = value.close

        def close_observer():
            seen.append("mirror-close")
            return close()

        value.close = close_observer
        return value

    monkeypatch.setattr(diagnostic, "CommittedRunObserver", observer)
    controller = SimpleNamespace(
        attach=lambda _: None,
        checkpoint_boundary=lambda: False,
        publish=dict,
        set_state=lambda state: seen.append(state),
        finish_evidence=lambda: {"pause_required": False},
        close_resources=lambda: seen.append("closed"),
    )
    monkeypatch.setattr(diagnostic, "TrainingController", lambda *a, **k: controller)
    monitor_stop = Event()

    def monitor(*a, **kwargs):
        seen.append("monitor")
        assert kwargs["total_steps"] == target_steps
        assert kwargs["metrics_condition_id"] == config.condition
        assert kwargs["performance_path"] == root / "performance.jsonl"
        return monitor_stop, SimpleNamespace(join=lambda **k: None)

    monkeypatch.setattr(diagnostic, "_start_progress_monitor", monitor)
    if pause == "owner-failure":
        with pytest.raises(ValueError):
            asyncio.run(
                diagnostic.run(
                    config_path=CONFIG, bindings_path=bindings_path, root=root, **run_kwargs
                )
            )
        assert "build" not in seen
        assert "monitor" not in seen
        assert "observer-created" not in seen
        assert "mirror-close" not in seen
        assert "closed" in seen
        assert "failed" in seen
        assert not (root / "summary.json").exists()
        failure = json.loads((root / "failure-private.json").read_text())
        assert failure["optimizer_step"] is None
        return
    asyncio.run(
        diagnostic.run(config_path=CONFIG, bindings_path=bindings_path, root=root, **run_kwargs)
    )
    if not appended:
        assert seen.index("build") < seen.index("fresh") < seen.index("observer-created")
    assert seen.index("observer-created") < seen.index(None) < seen.index("monitor")
    assert controller.evidence is not None
    assert monitor_stop.is_set()
    assert "settled" in seen
    assert "mirror-close" in seen
    assert "closed" in seen
    expected = 1 if pause is True else target_steps
    output = next(root.glob("continuation-step3-to10-*")) if appended else root
    if appended:
        for name in ("summary.json", "resolved-run-plan.json", "final-method-state.json"):
            assert (root / name).read_text() == "original Step3 declaration"
    result = json.loads((output / ("paused.json" if pause is True else "summary.json")).read_text())
    assert result["status"] == ("paused-after-complete-checkpoint" if pause is True else "finished")
    assert loop.optimizer_step == expected
    assert len(list((root / "checkpoints").glob("step-*/COMPLETE"))) == (
        7 if appended else expected
    )
    assert (output / "final-evidence-status.json").is_file()
    assert assembly["identity-inputs"]["application"].trainer.checkpoint.every_n_steps == 1
    assert assembly["identity-inputs"]["application"].evolution.cold_start == config.cold_start
    assert assembly["identity-inputs"]["alfworld_goal_binding"] == "reset-public-goal@2"


@pytest.mark.parametrize("display", ["nvml", "torch", "uuid-object", "cuda-uuid-object"])
def test_cuda_owner_uuid_display_forms_match_without_cuda_calls(monkeypatch, display):
    from uuid import UUID

    import torch

    plain = "00112233-4455-6677-8899-aabbccddeeff"

    class SyntheticCudaUUID:
        def __str__(self):
            return plain

    values = {
        "nvml": "GPU-" + plain,
        "torch": plain,
        "uuid-object": UUID(plain),
        "cuda-uuid-object": SyntheticCudaUUID(),
    }
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("WORLD_SIZE", "1")
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: calls.append(index))
    monkeypatch.setattr(torch.cuda, "init", lambda: calls.append("mock-init"))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid=values[display])
    )
    expected = "GPU-" + plain
    identity = diagnostic.initialize_cuda_owner(expected)
    assert identity["gradient_gpu_uuid"] == expected
    assert identity["cuda_initialized"] is True
    assert calls == [0, "mock-init"]
    assert diagnostic._gpu_uuid(plain) == diagnostic._gpu_uuid(expected)
    with pytest.raises(ValueError):
        diagnostic.initialize_cuda_owner("GPU-ffffffff-ffff-ffff-ffff-ffffffffffff")


def test_malformed_gpu_identity_is_not_an_alias_for_a_device():
    with pytest.raises(ValueError):
        diagnostic._gpu_uuid("5")


def test_real_observer_lock_is_not_exempted_from_fresh_namespace(tmp_path):
    from skillev.training.fresh_state import FreshNamespaces, _namespace_observations
    from skillev.training.run_observer import CommittedRunObserver

    root = tmp_path / "new-run"
    root.mkdir()
    namespaces = FreshNamespaces((), (root / "evidence",))
    assert all(row["empty"] for row in _namespace_observations(namespaces))
    observer = CommittedRunObserver(
        root, run_id=root.name, condition_id="synthetic", mirror_root=tmp_path / "mirror"
    )
    try:
        assert (root / "evidence/mirror.lock").is_file()
        assert any(not row["empty"] for row in _namespace_observations(namespaces))
    finally:
        observer.close()
