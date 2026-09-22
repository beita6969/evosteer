"""Synthetic CPU wiring of explicit five-step, two-gradient-owner observation."""

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from skillev_private.experiments import bayesian_improve_training as entry
from skillev_private.experiments import fresh_restart
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig
from skillev_private.experiments.training_observation import (
    observation_condition,
    observation_config,
    require_observation_sources,
)

from skillev.training.distributed_ttb import DistributedTTBTopology
from tests.benchmarks.test_protocol_v13_training import _sources

CONFIG = Path("configs/training/bayesianimprove_development_handoff.yaml")


@pytest.fixture
def selection(tmp_path):
    from skillev_private.benchmarks.protocol_v13_training import build_protocol13_training_records

    records = build_protocol13_training_records(_sources())
    selected = entry.seven_domain_training_trajectories(records, steps=5)
    value = {
        "training": sorted({(r.episode.benchmark.value, r.episode.source_id) for r in selected}),
        "excluded_sources": {
            "iid": [["healthbench", "held-out-iid"]],
            "development": [["alfworld", "public-development"]],
            "quality": [["triviaqa", "held-out-quality"]],
        },
        "source_aliases": {},
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(value, indent=2))
    return selected, path


def test_five_step_plan_changes_only_authorized_schedule_and_preserves_long_default(selection):
    _, sources = selection
    config = BayesianFormalConfig.load(CONFIG)
    assert observation_config(config, steps=None, resume=None, sources=None) is config
    assert observation_condition(None) == {}
    short = observation_config(config, steps=5, resume=None, sources=sources)
    assert (
        replace(
            short,
            steps=config.steps,
            closure_steps=config.closure_steps,
            checkpoint_every=config.checkpoint_every,
        )
        == config
    )
    assert short.sampling_config == config.sampling_config
    assert short.batch_size == 28
    assert short.run_plan.phase_search_steps == 4
    assert short.run_plan.closure_steps == 1
    assert short.run_plan.maximum_cycles == 2
    assert short.application_config("synthetic").trainer.checkpoint.every_n_steps == 1
    assert observation_condition(5)["training_observation"]["a0_admission"] == "not-claimed"
    for steps, resume, continuation in (
        (3, None, False),
        (10, None, False),
        (True, None, False),
        (5, sources, False),
        (5, None, True),
    ):
        with pytest.raises(ValueError):
            observation_config(
                config, steps=steps, resume=resume, sources=sources, continuation=continuation
            )


def test_real_source_schedule_and_alias_exclusions_are_required(selection):
    selected, path = selection
    value = require_observation_sources(path, selected)
    assert len(selected) == 140
    with pytest.raises(ValueError):
        require_observation_sources(path, selected[:-1])
    first = selected[0].episode
    value["source_aliases"] = {first.benchmark.value: {"prior-alias": first.source_id}}
    value["excluded_sources"]["development"] = [[first.benchmark.value, "prior-alias"]]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        require_observation_sources(path, selected)
    del value["excluded_sources"]["development"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        require_observation_sources(path, selected)


@pytest.mark.parametrize("mode", ["observation", "unqualified"])
def test_real_coordinator_uses_formal_runtime_fresh_check_and_all_five_boundaries(
    tmp_path, monkeypatch, selection, mode
):
    from skillev_private.benchmarks.protocol_v13_training_sessions import (
        Protocol13TrainingSessionFactory,
    )

    from skillev.training import run_observer
    from skillev.training.fresh_state import FreshNamespaces, _namespace_observations
    from tests.rollout.engine_fakes import default_request

    selected, sources = selection
    config = BayesianFormalConfig.load(CONFIG)
    profile = entry.TrainingPerformanceConfig.load(Path(config.performance_profile))
    target = 5 if mode == "observation" else 250
    if mode == "unqualified":
        from skillev_private.benchmarks.protocol_v13_training import (
            build_protocol13_training_records,
        )

        selected = entry.seven_domain_training_trajectories(
            build_protocol13_training_records(_sources()), steps=250
        )
        declaration = json.loads(sources.read_text())
        declaration.update(
            format="owner-authorized-unqualified-training@1", authorization="owner-explicit"
        )
        sources.write_text(json.dumps(declaration))
    root = tmp_path / "new-observation"
    preparation = tmp_path / "preparation.json"
    preparation.write_text(json.dumps({"format": "skillev-private-training-preparation@1"}))
    bindings = entry.FormalTrainingBindings(
        preparation,
        *(tmp_path for _ in range(4)),
        "http://localhost:1234",
        "Qwen3.5-9B",
        "synthetic-policy",
        ("GPU-a", "GPU-b"),
        "GPU-c",
        data_condition={"source": "frozen-synthetic-training"},
        evidence_mirror_root=tmp_path / "mirror",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    captures = {}
    monkeypatch.setattr(
        fresh_restart,
        "require_run_iid_baselines",
        lambda **kw: pytest.fail("observation must not claim A0"),
    )
    model = SimpleNamespace(
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        torch_dtype="bfloat16",
        base_model_path="/synthetic-model",
        tokenizer_path=None,
        tokenizer_id="synthetic",
        revision="synthetic",
        base_model_artifact=None,
    )
    checkpoint = SimpleNamespace(
        directory=str(tmp_path / "policy"),
        trainable_state=SimpleNamespace(to_value=lambda: {"step": 0}),
    )
    monkeypatch.setattr(entry, "_read_preparation", lambda _: (model, checkpoint))
    monkeypatch.setattr(entry, "_validated_worker_interpreter", lambda path: path)
    from skillev_private.benchmarks.mbpp_scoring import MBPPScorerProfile

    monkeypatch.setattr(entry, "resolve_mbpp_profile", lambda _: MBPPScorerProfile())
    monkeypatch.setattr(entry, "load_seven_domain_training_sources", lambda _: selected)
    # The actual deterministic schedule is separately exercised above; here keep
    # these exact complete 140 trajectory coordinates through the assembly.
    monkeypatch.setattr(entry, "training_schedule", lambda *a, **kw: selected)
    monkeypatch.setattr(entry, "register_serving", lambda *a, **kw: {})
    tasks = tuple(replace(default_request().task, task_id=r.episode.episode_id) for r in selected)

    async def hydrate(records, **kwargs):
        assert records == selected
        assert kwargs["domain_rollout_budgets"] == config.domain_task_budgets
        return tasks, Protocol13TrainingSessionFactory(
            {},
            kwargs["mbpp_profile"],
            healthbench_judge=kwargs["healthbench_judge"],
            alfworld_goal_binding="reset-public-goal@2",
        )

    monkeypatch.setattr(entry, "build_protocol13_training_sessions", hydrate)

    def identity(**kwargs):
        captures["identity"] = kwargs
        return SimpleNamespace(snapshot_identity=SimpleNamespace(to_value=dict))

    monkeypatch.setattr(entry, "_public_identity", identity)
    from skillev.training.run_condition import EffectiveRunCondition

    monkeypatch.setattr(fresh_restart, "resolved_input_profiles", lambda tasks: {})
    monkeypatch.setattr(
        fresh_restart,
        "resolve_effective_run_condition",
        lambda **kw: EffectiveRunCondition.create(
            condition_id=kw["condition_id"],
            scientific={"data_condition": kw["data_condition"]},
            execution=kw["execution"],
        ),
    )
    coordinator = SimpleNamespace(close=lambda: captures.setdefault("closed", True))

    def gradient(topology, **kwargs):
        assert topology.world_size == 2
        assert kwargs["coordinator_participates"] is True
        captures["gradient"] = True
        return coordinator

    monkeypatch.setattr(entry, "DistributedTTBGradientCoordinator", gradient)

    def runtime(**kwargs):
        assert kwargs["gradient_preparer"] is coordinator
        return SimpleNamespace(
            gateway=object(), resources=object(), dependencies=lambda: "real-formal-binding"
        )

    monkeypatch.setattr(entry.BoundFormalSGLangRuntime, "build", runtime)
    monkeypatch.setattr(entry, "resolved_method_state", lambda app: {"fixture": True})
    monkeypatch.setattr(entry, "_start_progress_monitor", lambda *a, **kw: (None, None))
    loop = SimpleNamespace(
        optimizer_step=0,
        policy_snapshot_id="initial",
        execution_deadline=None,
        finalized_timings=(),
        ledger=SimpleNamespace(assert_fully_settled=lambda: None),
        rollout_progress={},
        execution_progress={"transaction_stage": "committed"},
        gradient_progress=None,
        configure_inflight=lambda *a, **kw: captures.setdefault("inflight", kw),
        enable_update_observation=lambda: captures.setdefault("update_observation", True),
    )

    async def run(plan, *, stop_requested):
        assert captures["fresh"]
        assert not await stop_requested()
        for step in range(1, target + 1):
            loop.optimizer_step = step
            assert not await stop_requested()
        return SimpleNamespace(
            final_optimizer_step=target, to_value=lambda: {"final_optimizer_step": target}
        )

    app = SimpleNamespace(training_loop=loop, evolution_loop=SimpleNamespace(run=run))

    def build(**kwargs):
        assert kwargs["runtime"] == "real-formal-binding"
        assert kwargs["initial_checkpoint"] is checkpoint
        captures["build"] = True
        return app

    monkeypatch.setattr(entry.SKILLEVApplication, "build_formal", build)

    def fresh(application, **kwargs):
        assert application is app
        assert captures["build"]
        assert all(
            row["empty"]
            for row in _namespace_observations(FreshNamespaces((), (root / "evidence",)))
        )
        captures["fresh"] = True

    monkeypatch.setattr(fresh_restart, "require_clean_initial_application", fresh)
    real_observer = run_observer.CommittedRunObserver

    def observer(*a, **kw):
        assert captures["fresh"]
        value = real_observer(*a, **kw)
        assert (root / "evidence/mirror.lock").exists()
        captures["boundaries"] = []
        value.observe_commits = lambda step: captures["boundaries"].append(step)
        value.status = lambda: {"pause_required": False}
        value.drain = lambda: {"pause_required": False}
        return value

    monkeypatch.setattr(run_observer, "CommittedRunObserver", observer)
    asyncio.run(
        entry.run_coordinator(
            config=config,
            bindings=bindings,
            profile=profile,
            root=root,
            resume=None,
            topology=DistributedTTBTopology(0, 2, 0, "synthetic-cpu"),
            **(
                {"observation_steps": 5, "observation_sources": sources}
                if mode == "observation"
                else {"unqualified_training_sources": sources}
            ),
        )
    )
    assert set(captures["boundaries"]) == set(range(target + 1))
    assert captures["closed"]
    cadence = 1 if mode == "observation" else 10
    assert captures["identity"]["application"].trainer.checkpoint.every_n_steps == cadence
    assert captures["identity"]["run_plan"].total_training_steps == target
    assert captures["identity"]["alfworld_goal_binding"] == "reset-public-goal@2"
    assert len(captures["identity"]["task_ids"]) == target * 28
    result = json.loads((root / "summary.json").read_text())
    assert result["summary"]["final_optimizer_step"] == target
    assert json.loads((root / "resolved-run-plan.json").read_text())["cadence_steps"] == list(
        range(cadence, target + 1, cadence)
    )
    if mode == "observation":
        assert (root / "observation-sources-private.json").read_bytes() == sources.read_bytes()
        assert (
            json.loads((root / "observation-candidate-config.json").read_text())["steps"]
            == config.steps
        )
        assert result["training_observation"]["accepted_250"] is False
    else:
        assert result["unqualified_training"]["cold_start_failed"]
        assert result["unqualified_training"]["a0_admission"] == "not-claimed"
        assert (
            root / "unqualified-training-sources-private.json"
        ).read_bytes() == sources.read_bytes()
        assert not (root / "iid-baselines-private.json").exists()
        effective = next(root.glob("effective-condition-process-*.json"))
        assert (
            json.loads(effective.read_text())["scientific"]["unqualified_training"]
            == result["unqualified_training"]
        )


def test_default_cli_still_refuses_no_a0_before_distributed_initialization(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training",
            "--config",
            str(CONFIG),
            "--bindings",
            str(tmp_path / "missing"),
            "--run-root",
            str(tmp_path / "run"),
        ],
    )
    monkeypatch.setattr(
        entry, "initialize_distributed_ttb", lambda **kw: pytest.fail("must not initialize")
    )
    with pytest.raises(SystemExit):
        entry.main()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("mode", ["observation", "unqualified"])
@pytest.mark.parametrize("failed_rank", [False, True])
def test_cli_dispatches_observation_with_actual_three_role_validation(
    tmp_path, monkeypatch, selection, rank, mode, failed_rank
):
    selected, sources = selection
    if mode == "unqualified":
        declaration = json.loads(sources.read_text())
        declaration.update(
            format="owner-authorized-unqualified-training@1", authorization="owner-explicit"
        )
        sources.write_text(json.dumps(declaration))
    config = BayesianFormalConfig.load(CONFIG)
    preparation = tmp_path / "preparation.json"
    preparation.write_text(json.dumps({"format": "legacy-original-preparation"}))
    bindings = entry.FormalTrainingBindings(
        preparation,
        *(tmp_path for _ in range(4)),
        "http://localhost:1234",
        "Qwen3.5-9B",
        "synthetic-policy",
        ("GPU-a", "GPU-b"),
        "GPU-c",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(entry.FormalTrainingBindings, "load", lambda path: bindings)
    monkeypatch.setattr(entry, "load_seven_domain_training_sources", lambda path: selected)
    profile = SimpleNamespace(
        coordinator_participates=True, pipeline_mode="within-step", configure_process=lambda: None
    )
    monkeypatch.setattr(entry.TrainingPerformanceConfig, "load", lambda path: profile)
    captures = {}
    topology = DistributedTTBTopology(rank, 2, rank, "synthetic-cpu")
    monkeypatch.setattr(entry, "initialize_distributed_ttb", lambda **kw: topology)
    monkeypatch.setattr(entry.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        entry.dist, "destroy_process_group", lambda: captures.setdefault("destroyed", True)
    )

    class RankFailureError(RuntimeError):
        pass

    async def coordinator(**kwargs):
        captures.update(kwargs)
        if failed_rank:
            raise RankFailureError

    monkeypatch.setattr(entry, "run_coordinator", coordinator)
    checkpoint = SimpleNamespace(directory="synthetic-policy", trainable_state="synthetic-initial")
    monkeypatch.setattr(entry, "_read_preparation", lambda path: (None, checkpoint))
    monkeypatch.setattr(BayesianFormalConfig, "require_backbone", lambda *args: None)
    backbone = SimpleNamespace(
        load_checkpoint=lambda path: captures.setdefault("loaded", path),
        bind_initial_trainable_state=lambda state: captures.setdefault("bound", state),
    )
    monkeypatch.setattr(entry, "build_qwen_policy_backbone", lambda *a, **kw: backbone)

    def worker(**kwargs):
        captures.update(kwargs)
        if failed_rank:
            raise RankFailureError

    monkeypatch.setattr(entry, "serve_distributed_ttb_worker", worker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training",
            "--config",
            str(CONFIG),
            "--bindings",
            str(tmp_path / "bindings"),
            "--run-root",
            str(tmp_path / "new-root"),
            *(
                ["--observation-steps", "5", "--observation-sources", str(sources)]
                if mode == "observation"
                else ["--unqualified-training-sources", str(sources)]
            ),
        ],
    )
    if failed_rank:
        with pytest.raises(RankFailureError):
            entry.main()
        assert not captures.get("destroyed")
    else:
        entry.main()
        assert captures["destroyed"]
    if rank == 0:
        assert captures["config"] == config  # coordinator alone declares the short schedule
        assert captures["observation_steps"] == (5 if mode == "observation" else None)
        assert captures["unqualified_training_sources"] == (
            sources if mode == "unqualified" else None
        )
        assert captures["resume"] is None
        assert captures["iid_baselines"] is None
    else:
        assert captures["backbone"] is backbone
        assert captures["loaded"] == checkpoint.directory
        assert captures["bound"] == checkpoint.trainable_state
