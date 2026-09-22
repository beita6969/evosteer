"""Formal B28 input, assembly and budget contracts; no GPU or benchmark claims."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from skillev_private.benchmarks import protocol_v13_training_sessions as sessions_module
from skillev_private.benchmarks.protocol_v13_seven_training import (
    seven_domain_training_trajectories,
)
from skillev_private.benchmarks.protocol_v13_training import build_protocol13_training_records
from skillev_private.experiments.bayesian_improve_training import (
    FormalTrainingBindings,
    _bind_run_directory,
    require_formal_execution,
)
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig

from skillev.rollout import UnskilledRolloutSessionBundle
from skillev.runtime import BudgetVector
from skillev.runtime.attempt_run_plan import RunSlotKind
from skillev.training import FixedAttemptBudgetPlan, RolloutWorkflowResources
from skillev.training.performance_config import TrainingPerformanceConfig
from tests.benchmarks.test_protocol_v10_session_deployments import _value
from tests.benchmarks.test_protocol_v13_training import _sources
from tests.rollout.engine_fakes import (
    FakeTerminalEvaluator,
    ScriptedEnvironment,
    make_harness,
)

CONFIG = Path("configs/training/bayesianimprove_250.yaml")


def test_declared_250_b28_full_method_and_retained_cadence():
    config = BayesianFormalConfig.load(CONFIG)
    app = config.application_config("formal-condition-test")
    rollout = app.trainer.rollout
    assert config.batch_size == app.trainer.execution.batch_size == 28
    assert config.schedule_summary()["question_occurrences"] == 1750
    assert config.schedule_summary()["trajectories"] == 7000
    assert config.steps == 250
    assert config.seed == 0
    assert config.lora_rank == 4
    assert config.lora_alpha == 8
    assert rollout.max_turns == config.task_budget.max_turns == 20
    assert config.static_task_budget.max_turns == config.static_max_turns == 8
    assert rollout.max_reasoning_tokens == config.task_budget.max_reasoning_tokens == 1024
    assert rollout.max_action_tokens == config.task_budget.max_action_tokens == 2048
    assert rollout.reasoning_native_thinking
    assert app.maximum_h0_tokens == config.max_input_tokens == 65_536
    assert rollout.input_window.max_tokens == config.max_input_tokens
    assert type(app).from_value(app.to_value()) == app
    assert rollout.per_rollout_maximum.input_tokens == 40 * 65_536
    assert app.trainer.method.epsilon_min == 0.1
    assert app.trainer.method.temperature_beta == 1
    assert (
        app.trainer.optimizer.adapter_learning_rate == app.trainer.optimizer.z_learning_rate == 1e-4
    )
    assert app.trainer.optimizer.weight_decay == 0
    assert not config.gradient_clipping
    assert config.extra_kl == 0
    assert (app.calibration.alpha_0, app.calibration.beta_0, app.calibration.default_k) == (1, 1, 1)
    assert app.evolution.k == 1
    assert app.diagnostics.stagnation_rho == 0.05
    assert app.diagnostics.window_size == app.evolution.entropy_window == 50
    assert config.run_plan.maximum_cycles == 2
    assert config.run_plan.slot_kind(249) is RunSlotKind.PHASE_SEARCH
    assert config.run_plan.slot_kind(250) is RunSlotKind.CLOSURE
    assert config.run_plan.phase_search_steps == 249
    assert list(range(app.trainer.checkpoint.every_n_steps, 251, 10)) == list(range(10, 251, 10))
    budget = FixedAttemptBudgetPlan.from_trainer_and_run_plan(
        trainer=app.trainer,
        run_plan=config.run_plan,
        phi_per_cycle_maximum=BudgetVector(),
    ).required()
    assert budget.agent_turns == 7000 * 20
    assert budget.model_calls == 7000 * 40
    service = yaml.safe_load(Path("configs/serving/qwen35_9b_bayesian_250.yaml").read_text())
    assert service["context_length"] >= config.max_input_tokens + config.max_action_tokens
    assert service["fp32_mamba_checkpoints"]
    assert service["page_size"] == 64


def test_formal_resume_cannot_silently_change_the_frozen_condition(tmp_path):
    config = BayesianFormalConfig.load(CONFIG)
    root = tmp_path / "formal-run"
    _bind_run_directory(root, config, None)
    _bind_run_directory(root, config, root / "checkpoints/step-00000010")
    before = (root / "formal-config.json").read_text()
    with pytest.raises(ValueError):
        _bind_run_directory(root, replace(config, max_reasoning_tokens=512), root / "checkpoint")
    with pytest.raises(ValueError):
        _bind_run_directory(root, replace(config, static_max_turns=50), root / "checkpoint")
    assert (root / "formal-config.json").read_text() == before
    with pytest.raises(FileExistsError):
        _bind_run_directory(root, config, None)


def test_formal_execution_requires_two_real_gradient_owners_and_a_separate_service(tmp_path):
    profile = TrainingPerformanceConfig.load(Path("configs/training/protocol13_turn_latency.yaml"))
    require_formal_execution(profile)
    for candidate in (
        replace(profile, coordinator_participates=False),
        replace(profile, pipeline_mode="sealed-batch"),
    ):
        with pytest.raises(ValueError):
            require_formal_execution(candidate)
    bindings = FormalTrainingBindings(
        *(tmp_path for _ in range(5)),
        "http://localhost:1234",
        "Qwen3.5-9B",
        "formal-policy",
        ("GPU-gradient-a", "GPU-gradient-b"),
        "GPU-serving",
    )
    runtime = bindings.runtime(tmp_path, profile)
    assert runtime.rollout.request_timeout_seconds == 1800
    assert runtime.gateway.request_timeout_seconds == 600
    bindings.require_device_mapping("GPU-gradient-a,GPU-gradient-b", 2)
    for candidate, visible, ranks in (
        (bindings, "", 2),
        (bindings, "GPU-gradient-a,GPU-gradient-b", 1),
        (replace(bindings, serving_gpu_uuid="GPU-gradient-a"), "GPU-gradient-a,GPU-gradient-b", 2),
    ):
        with pytest.raises(ValueError):
            candidate.require_device_mapping(visible, ranks)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("domain_budgets", [False, True])
def test_hydration_really_overrides_each_domain_budget_but_not_labels(
    tmp_path, monkeypatch, lazy, domain_budgets
):
    config = BayesianFormalConfig.load(CONFIG)
    if domain_budgets:
        config = replace(
            config,
            format="skillev-bayesian-formal-training@6",
            reasoning_tokens_by_domain=(("aime-2026", 16384), ("alfworld", 2048)),
        )
    records = seven_domain_training_trajectories(
        build_protocol13_training_records(_sources(9)), steps=1
    )
    records = tuple(
        replace(
            record,
            output=replace(
                record.output,
                target={**record.output.target, "environment_route": {"max_steps": 50}},
            ),
        )
        if record.episode.benchmark.value == "alfworld"
        else record
        for record in records
    )
    value = _value(tmp_path)
    path = tmp_path / "deployments.json"
    path.write_text(json.dumps(value))
    source = tmp_path / "evalplus"
    source.mkdir()
    (source / "config.py").write_text(
        "DEFAULT_MIN_TIME_LIMIT=4.0\nDEFAULT_GT_TIME_LIMIT_FACTOR=4.0\n"
    )
    alf_created = []
    hydrated_ids = []

    async def alf_route(record, _deployments):
        hydrated_ids.append(record.input.task_id)
        surface, budget = sessions_module._action_contract(record.episode.benchmark, 50)
        task = replace(record.input, action_surface=surface, budget_profile=budget)

        def create():
            env = ScriptedEnvironment(
                [], environment_id=task.environment_id, task_family=task.task_family
            )
            alf_created.append(env)
            return UnskilledRolloutSessionBundle(env, FakeTerminalEvaluator(value=0.0))

        return task, create

    monkeypatch.setattr(sessions_module, "_alfworld_route", alf_route)
    profile = TrainingPerformanceConfig.load(Path(config.performance_profile))
    tasks, factory = asyncio.run(
        sessions_module.build_protocol13_training_sessions(
            records,
            deployments_path=path,
            endpoint_base="http://localhost:1234",
            base_model=value["healthbench"]["model_revision"],
            resources=RolloutWorkflowResources(profile.workflow()),
            mbpp_interpreter=Path(value["healthbench"]["interpreter"]),
            mbpp_source_root=tmp_path,
            mbpp_profile=sessions_module.MBPPScorerProfile(),
            hotpot_deliberation=True,
            rollout_budget=config.task_budget,
            static_rollout_budget=config.static_task_budget,
            domain_rollout_budgets=config.domain_task_budgets,
            lazy_environments=lazy,
        )
    )
    assert len(tasks) == 28
    assert [t.task_id for t in tasks] == [r.input.task_id for r in records]
    for task, record in zip(tasks, records, strict=True):
        expected = (
            config.task_budget
            if record.episode.benchmark.value == "alfworld"
            else config.static_task_budget
        )
        expected = config.domain_task_budgets.get(record.episode.benchmark.value, expected)
        assert task.budget_profile == expected
        from types import SimpleNamespace

        from skillev.policy.interface import THINKING_ROLLOUT_PROMPT_ENCODER_VERSION
        from skillev.training.runtime_components import RolloutBatchCollector

        collector = SimpleNamespace(_config=config.application_config("thinking-test").trainer)
        decoding = RolloutBatchCollector._decoding_for_task(collector, task)
        assert (decoding.prompt_encoder_version == THINKING_ROLLOUT_PROMPT_ENCODER_VERSION) == (
            record.episode.benchmark.value in {"healthbench", "aime-2026"}
        )
    assert all(t.query == r.input.query for t, r in zip(tasks, records, strict=True))
    if lazy:
        assert hydrated_ids == []
        with pytest.raises(RuntimeError):
            factory.create(tasks[4])
        ready = asyncio.run(factory.prepare_tasks((tasks[4],)))
        assert hydrated_ids == [tasks[4].task_id]
        assert asyncio.run(factory.prepare_tasks((tasks[4],))) == ready
        assert len(hydrated_ids) == 1
        alf = ready[0]
    else:
        alf = tasks[4]
    assert factory.create(alf).environment is not factory.create(alf).environment
    assert len(alf_created) == 2


def test_legacy_undeclared_input_overflow_remains_an_error_not_a_negative_label(tmp_path):
    harness = make_harness(tmp_path, scripts=[])
    harness.engine._reasoning_call_maximum = BudgetVector(
        input_tokens=1, output_tokens=128, model_calls=1
    )
    with pytest.raises(ValueError):
        asyncio.run(harness.engine.run(harness.request))
    assert not harness.generator.requests
    assert not harness.evaluator.requests


@pytest.mark.parametrize("mirror_enabled", [False])
@pytest.mark.parametrize("data_enabled", [False, True])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("serving_ready", [False, True])
@pytest.mark.parametrize("import_quality", [False, True])
def test_formal_entry_delegates_build_or_full_resume_then_runs_all_250_slots(
    tmp_path,
    monkeypatch,
    resume,
    serving_ready,
    data_enabled,
    import_quality,
    mirror_enabled,
):
    from types import SimpleNamespace

    from skillev_private.experiments import bayesian_improve_training as entry

    from tests.rollout.engine_fakes import default_request

    config = BayesianFormalConfig.load(CONFIG)
    root = tmp_path / "formal-entry"
    if resume:
        entry._bind_run_directory(root, config, None)
    quality_arguments = {}
    if import_quality:
        from dataclasses import asdict

        from skillev_private.experiments.quality_panel import FixedQualityPanel, PanelSlot

        from skillev.training.quality_gate import ProtocolProbe, QualityGatePolicy, QualityRule

        panel = FixedQualityPanel(
            "fixed-panel",
            config.condition,
            (PanelSlot("occurrence", "synthetic", "pool", "source"),),
        )
        policy = QualityGatePolicy(
            "t0",
            panel.panel_id,
            config.condition,
            25,
            1,
            2,
            (QualityRule("success", 0.25, 0.2, baseline_minimum=0.5),),
        )
        probe = ProtocolProbe(
            "original",
            panel.panel_id,
            config.condition,
            "initial-policy",
            0,
            1,
            {"success": 0.75},
        )
        policy_path, probe_path = tmp_path / "policy.json", tmp_path / "original.json"
        policy_path.write_text(json.dumps(asdict(policy)))
        probe_path.write_text(json.dumps(asdict(probe), indent=4) + "\n")
        quality_arguments = {
            "quality_policy": policy_path,
            "quality_panel": tmp_path / "binding.json",
            "initial_quality_probe": probe_path,
        }
        monkeypatch.setattr(
            entry.QualityCollectionBinding,
            "load",
            lambda *a, **kw: SimpleNamespace(panel=panel, freeze=lambda root: None),
        )

        async def forbidden_collect(*args):
            pytest.fail("formal imported T0 must not resample")

        monkeypatch.setattr(entry, "FormalQualityCollector", lambda *args: forbidden_collect)
        if resume:
            entry.import_initial_quality_probe(
                probe_path,
                monitor=entry.QualityCheckpointStop(root / "quality", policy),
                panel=panel,
                optimizer_step=0,
                policy_snapshot_id="initial-policy",
            )
    model = SimpleNamespace(
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        torch_dtype="bfloat16",
        base_model_path="/models/qwen",
        tokenizer_path=None,
        tokenizer_id="fixture-tokenizer",
        revision="fixture-revision",
        base_model_artifact=None,
    )
    bindings = FormalTrainingBindings(
        *(tmp_path for _ in range(5)),
        "http://localhost:1234",
        "Qwen3.5-9B",
        "formal-policy",
        ("GPU-a", "GPU-b"),
        "GPU-c",
    )
    if mirror_enabled:
        from skillev.training import run_observer

        bindings = replace(bindings, evidence_mirror_root=tmp_path / "mirror")
        real_observer = run_observer.CommittedRunObserver

        def observer(*args, **kwargs):
            assert "build" in captures or "resume" in captures
            value = real_observer(*args, **kwargs)
            assert (root / "evidence/mirror.lock").is_file()
            # No real commits in this assembly fixture. Exercise actual namespace
            # creation, not fake posterior/evidence completeness.
            value.observe_commits = lambda step: None
            value.status = lambda: {"pause_required": False}
            value.drain = lambda: {"pause_required": False}
            return value

        monkeypatch.setattr(run_observer, "CommittedRunObserver", observer)
    from tests.training.test_training_data_condition import DECLARATION, effective, selected_records

    selected = selected_records()
    if data_enabled:
        bindings = replace(bindings, data_condition=DECLARATION)
        if resume:
            entry._write(root / "effective-condition-process-previous.json", effective().to_value())
    profile = TrainingPerformanceConfig.load(Path(config.performance_profile))
    task = default_request().task
    captures = {}
    check_data = entry.require_same_run_data_condition

    def checked_data(root, current, **kwargs):
        captures["data_resume_checked"] = True
        check_data(root, current, **kwargs)

    monkeypatch.setattr(entry, "require_same_run_data_condition", checked_data)

    def register(*_a, **kw):
        captures["serving"] = kw
        if not serving_ready:
            raise ValueError("actual service differs")

    monkeypatch.setattr(entry, "register_serving", register)
    checkpoint = SimpleNamespace(trainable_state=SimpleNamespace(to_value=lambda: {"step": 0}))
    monkeypatch.setattr(entry, "_read_preparation", lambda _: (model, checkpoint))
    monkeypatch.setattr(entry, "_validated_worker_interpreter", lambda path: path)
    monkeypatch.setattr(
        entry, "resolve_mbpp_profile", lambda _: sessions_module.MBPPScorerProfile()
    )
    monkeypatch.setattr(entry, "load_seven_domain_training_sources", lambda _: ())
    monkeypatch.setattr(entry, "training_schedule", lambda *_, **kw: selected)

    async def hydrate(*_args, **kwargs):
        assert kwargs["rollout_budget"] == config.task_budget
        assert kwargs["static_rollout_budget"] == config.static_task_budget
        assert kwargs["hotpot_deliberation"]
        assert kwargs["healthbench_judge"] == config.healthbench_judge
        return (task,), sessions_module.Protocol13TrainingSessionFactory(
            {}, kwargs["mbpp_profile"], healthbench_judge=kwargs["healthbench_judge"]
        )

    monkeypatch.setattr(entry, "build_protocol13_training_sessions", hydrate)

    def public_identity(**kwargs):
        captures["identity"] = kwargs
        return SimpleNamespace(snapshot_identity=SimpleNamespace(to_value=dict))

    monkeypatch.setattr(entry, "_public_identity", public_identity)
    coordinator = SimpleNamespace(close=lambda: captures.setdefault("closed", True))
    monkeypatch.setattr(entry, "DistributedTTBGradientCoordinator", lambda *_a, **_kw: coordinator)
    monkeypatch.setattr(
        entry.BoundFormalSGLangRuntime,
        "build",
        lambda **_kw: SimpleNamespace(
            gateway=object(), resources=object(), dependencies=lambda: "formal-runtime"
        ),
    )
    monkeypatch.setattr(entry, "resolved_method_state", lambda _app: {"observed": "fixture"})
    monkeypatch.setattr(entry, "_start_progress_monitor", lambda *_a, **_kw: (None, None))
    monkeypatch.setattr(entry.LiveAttemptEventLog, "resume", lambda *_a, **_kw: object())

    async def run(plan, *, stop_requested):
        if import_quality:
            assert (root / "quality/probe-00000000.json").read_bytes() == probe_path.read_bytes()
        assert not await stop_requested()
        captures["run_plan"] = plan
        return SimpleNamespace(
            final_optimizer_step=250, to_value=lambda: {"final_optimizer_step": 250}
        )

    app = SimpleNamespace(
        training_loop=SimpleNamespace(
            optimizer_step=10 if resume else 0,
            policy_snapshot_id="restored-policy" if resume else "initial-policy",
            execution_deadline=None,
            finalized_timings=(),
            configure_inflight=lambda *a, **kw: None,
            ledger=SimpleNamespace(assert_fully_settled=lambda: None),
            rollout_progress={"trajectories": []},
            execution_progress={"transaction_stage": "committed"},
            gradient_progress=None,
        ),
        evolution_loop=SimpleNamespace(run=run),
    )

    def build(**kwargs):
        assert "data_resume_checked" not in captures
        captures["build"] = kwargs
        if mirror_enabled:
            from skillev.training.fresh_state import FreshNamespaces, _namespace_observations

            assert all(
                row["empty"]
                for row in _namespace_observations(FreshNamespaces((), (root / "evidence",)))
            )
        return app

    def restore(**kwargs):
        assert captures["data_resume_checked"]
        captures["resume"] = kwargs
        return app

    monkeypatch.setattr(entry.SKILLEVApplication, "build_formal", build)
    monkeypatch.setattr(entry.SKILLEVApplication, "resume_formal", restore)
    snapshot = root / "checkpoints/step-00000010" if resume else None
    coroutine = entry.run_coordinator(
        config=config,
        bindings=bindings,
        profile=profile,
        root=root,
        resume=snapshot,
        topology=object(),
        **quality_arguments,
    )
    if not serving_ready:
        with pytest.raises(ValueError):
            asyncio.run(coroutine)
        assert captures["closed"]
        assert "run_plan" not in captures
        assert "build" not in captures
        assert "resume" not in captures
        return
    asyncio.run(coroutine)
    assert captures["serving"]["minimum_context"] == 67584
    assert captures["run_plan"] == config.run_plan
    assert captures["identity"]["entry_kind"] == "formal"
    assert captures["identity"]["sampling_condition"] == config.condition
    assert captures["closed"]
    assert app.training_loop.execution_deadline is None
    assert ("resume" in captures) is resume
    assert ("build" in captures) is not resume
    kwargs = captures["resume" if resume else "build"]
    assert kwargs["runtime"] == "formal-runtime"
    if resume:
        assert kwargs["snapshot_directory"] == snapshot
        assert kwargs["task_provider_factory"].curriculum_id == config.condition
    else:
        assert kwargs["task_provider"].curriculum_id == config.condition
    plan = json.loads((root / "resolved-run-plan.json").read_text())
    assert plan["cadence_steps"] == list(range(10, 251, 10))
    assert json.loads((root / "summary.json").read_text())["schedule"]["trajectories"] == len(
        selected
    )
    import os

    effective = json.loads((root / f"effective-condition-process-{os.getpid()}.json").read_text())
    assert effective["scientific"]["batch_size"] == 28
    assert effective["scientific"]["tokenizer_id"] == model.tokenizer_id
    assert effective["execution"]["checkpoint_every"] == 10
    assert "performance_profile" not in effective["scientific"]["formal"]
    if data_enabled:
        data = effective["scientific"]["data_condition"]
        assert data["declaration"] == DECLARATION
        assert [r["occurrence_id"] for r in data["ordered_selected_sources"]] == [
            r.episode.episode_id for r in selected
        ]
    else:
        assert "data_condition" not in effective["scientific"]


def test_resume_monitor_preserves_cold_step_and_ignores_torn_advisory_line(tmp_path):
    import time
    from types import SimpleNamespace

    from skillev_private.experiments.bayesian_training_setup import _start_progress_monitor

    from skillev.training.step_timing import StepTiming

    path = tmp_path / "performance.jsonl"
    prior = [
        {"committed": True, "optimizer_step": i, "step_wall_seconds": 20, "warmup": True}
        for i in (1, 2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in prior) + '{"committed":')
    timing = StepTiming(
        "batch-three",
        3,
        0,
        10,
        2,
        14,
        15,
        None,
        process_instance_id="new-process",
        process_step_index=1,
    )
    app = SimpleNamespace(
        training_loop=SimpleNamespace(
            optimizer_step=2,
            finalized_timings=(timing,),
            rollout_progress={},
            gradient_progress=None,
        ),
        generator=SimpleNamespace(physical_usage={}),
    )
    stop, thread = _start_progress_monitor(
        app, total_steps=250, performance_path=path, run_started=time.monotonic()
    )
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    row = json.loads(path.read_text().splitlines()[-1])
    assert row["warmup"]
    assert row["post_warmup_steps_per_hour"] is None
    assert row["incomplete_monitor_records"] == 1
    assert row["prior_committed_wall_seconds"] == 40
    assert row["step_wall_seconds"] == 15
    assert row["forecast"]["remaining_steps"] == 247


def test_mixed_thinking_is_frozen_in_condition_and_snapshot():
    config = BayesianFormalConfig.load(CONFIG)
    assert {d for d, on in config.reasoning_modes.items() if on} == {"healthbench", "aime-2026"}
    assert {d for d, on in config.reasoning_modes.items() if not on} == {
        "humaneval",
        "hotpotqa",
        "triviaqa",
        "mbpp-plus",
        "alfworld",
    }
    rollout = config.application_config("mixed-thinking").trainer.rollout
    assert dict(rollout.reasoning_by_domain) == config.reasoning_modes
    assert type(rollout).from_value(rollout.to_value()) == rollout
    assert replace(config, thinking_off_domains=()).condition != config.condition
    with pytest.raises(ValueError):
        replace(config, thinking_off_domains=("webshop",))
    with pytest.raises(ValueError):
        replace(config, thinking_off_domains=("humaneval", "humaneval"))
    legacy = replace(config, format="skillev-bayesian-formal-training@2", thinking_off_domains=())
    assert "thinking_off_domains" not in legacy.to_value()
    assert not legacy.application_config("legacy-thinking").trainer.rollout.reasoning_by_domain


def test_reasoning_resume_is_explicit_and_does_not_change_thinking_or_horizon(tmp_path):
    from skillev_private.experiments.bayesian_condition_transition import sampling_condition

    config = replace(
        BayesianFormalConfig.load(CONFIG),
        format="skillev-bayesian-formal-training@6",
        phase_context=True,
        action_wire="native-single-tool-call@2",
    )
    root = tmp_path / "formal-run"
    _bind_run_directory(root, config, None)
    target = replace(
        config, reasoning_tool_catalog=True, reasoning_tokens_by_domain=(("healthbench", 4096),)
    )
    assert (
        _bind_run_directory(root, target, root / "checkpoint", allow_new_reasoning=True) == config
    )
    for changed in (
        replace(target, max_turns=25),
        replace(target, max_action_tokens=1024),
        replace(target, thinking_off_domains=()),
    ):
        with pytest.raises(ValueError):
            _bind_run_directory(root, changed, root / "checkpoint", allow_new_reasoning=True)
    (root / "branch-source.json").write_text(json.dumps({"sampling_condition": config.condition}))
    assert sampling_condition(root, target.condition) == config.condition
    assert json.loads((root / "formal-config.json").read_text()) == config.to_value()


def test_formal_observer_is_created_after_initial_application_namespace_check(
    tmp_path, monkeypatch
):
    test_formal_entry_delegates_build_or_full_resume_then_runs_all_250_slots(
        tmp_path,
        monkeypatch,
        resume=False,
        serving_ready=True,
        data_enabled=False,
        import_quality=False,
        mirror_enabled=True,
    )
    assert (tmp_path / "formal-entry/evidence/mirror.lock").is_file()
