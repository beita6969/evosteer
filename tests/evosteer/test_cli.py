"""Actual offline HF/PEFT entrypoint runs, restart and private-evaluator boundaries."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from skillev.contracts.evosteer import EvoTrajectory
from skillev.evosteer_application import SessionRequest
from skillev.evosteer_cli import (
    _completion_bindings,
    _configuration,
    _factory_bindings,
    _read_completion_tasks,
    _run,
    main,
)


def command(*arguments):
    root = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    return subprocess.run(  # noqa: S603 — fixed interpreter/module and trusted fixture arguments
        [sys.executable, "-m", "skillev.evosteer_cli", *map(str, arguments)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    output = tmp_path_factory.mktemp("cli") / "smoke"
    result = command("smoke", "--output", output, "--steps", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def test_real_offline_smoke_updates_actor_and_saves_complete_histories(smoke_run):
    import torch

    from skillev.evosteer_demo import build_tiny_policy

    summary = json.loads((smoke_run / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["synthetic"] is True
    assert summary["batch_index"] == summary["actor_version"] == 2
    assert not (smoke_run / "failure.json").exists()
    checkpoint = Path(summary["checkpoint"])
    stored = torch.load(checkpoint / "trainable.pt", weights_only=True, map_location="cpu")
    initial = build_tiny_policy().adapter_state()
    assert set(stored["actor"]) == set(initial)
    assert any(not torch.equal(stored["actor"][name], tensor) for name, tensor in initial.items())
    assert all(torch.isfinite(tensor).all() for tensor in stored["actor"].values())
    metrics = [json.loads(line) for line in (smoke_run / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 2
    for batch in metrics:
        assert batch["synthetic"] is True
        assert batch["source_counts"] == {
            "current": 1,
            "natural_reference": 1,
            "paired_control": 1,
            "paired_treatment": 1,
        }
        histories = [
            EvoTrajectory.from_value(json.loads(line))
            for line in (smoke_run / "trajectories" / f"{batch['batch_id']}.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(histories) == 4
        assert all(
            step.prompt_ids and step.action_token_ids
            for item in histories
            for step in item.decisions
        )
        assert len({row["session_id"] for row in batch["reset_receipts"].values()}) == 4
    assert "not a paper benchmark" in json.loads((smoke_run / "run.json").read_text())["purpose"]


def test_resume_uses_saved_batch_index_and_requires_explicit_new_output(smoke_run, tmp_path):
    checkpoint = smoke_run / "checkpoints" / "batch-000002"
    before = (checkpoint / "state.json").read_bytes()
    output = tmp_path / "resumed"
    result = command("smoke", "--output", output, "--steps", 3, "--resume", checkpoint)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["batch_index"] == 3
    assert summary["completed_batches"] == 1
    assert len(list((output / "checkpoints").iterdir())) == 1
    assert (checkpoint / "state.json").read_bytes() == before


def test_resume_rejects_changed_task_source(smoke_run, tmp_path):
    checkpoint = tmp_path / "changed-checkpoint"
    shutil.copytree(smoke_run / "checkpoints" / "batch-000002", checkpoint)
    context_file = checkpoint / "cli-context.json"
    context = json.loads(context_file.read_text())
    context["task_source_id"] = "changed-data@1"
    context_file.write_text(json.dumps(context))
    output = tmp_path / "rejected-resume"
    result = command("smoke", "--output", output, "--steps", 3, "--resume", checkpoint)
    assert result.returncode == 1
    assert "task source" in json.loads((output / "failure.json").read_text())["message"]
    assert not (output / "summary.json").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["smoke", "--steps", "0"],
        ["smoke", "--steps", "-1"],
        ["train", "--config", "missing.json"],
        ["train", "--config", "missing.json", "--tasks", "a.jsonl", "--task-factory", "x:y"],
    ],
)
def test_invalid_arguments_do_not_create_output(arguments, tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        main([*arguments, "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_existing_output_is_never_modified(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    assert main(["smoke", "--output", str(output)]) == 2
    assert [item.name for item in output.iterdir()] == ["keep.txt"]
    assert (output / "keep.txt").read_text() == "keep"


def test_config_failure_is_recorded_before_loading_weights(tmp_path, monkeypatch):
    from skillev.policy.evosteer import CausalLMOrchestrator

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid config must be rejected before model allocation")

    monkeypatch.setattr(CausalLMOrchestrator, "from_pretrained", forbidden)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"format": "bad"}))
    output = tmp_path / "failed"
    assert (
        main(["train", "--config", str(config), "--tasks", "missing", "--output", str(output)]) == 1
    )
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["error_type"] == "ValueError"
    assert not (output / "summary.json").exists()


def test_completion_answers_stay_in_evaluator_and_resets_are_fresh(tmp_path):
    path = tmp_path / "tasks.jsonl"
    secret = "EVALUATOR_ONLY_SENTINEL_734"
    path.write_text(
        json.dumps(
            {
                "task_id": "public-id",
                "family": "qa",
                "prompt": "Public question.",
                "expected_answers": [secret],
            }
        )
        + "\n"
    )
    rows, source_id = _read_completion_tasks(path)
    (binding,) = _completion_bindings(
        rows, SimpleNamespace(reference_id="local@1", configuration_id="local-config@1"), 0.3
    )
    assert source_id.startswith("sha256:")
    assert secret not in json.dumps(asdict(binding.task))
    request = SessionRequest(binding.task, 19, "test-pair")
    first, second = binding.session_factory(request), binding.session_factory(request)
    assert first.evaluator(secret) == 1.0
    assert first.evaluator("other") == 0.0
    assert first.reset_receipt.initial_state_id == second.reset_receipt.initial_state_id
    assert first.reset_receipt.session_id != second.reset_receipt.session_id
    first.reset_receipt.validate(request)
    second.reset_receipt.validate(request)


def test_explicit_task_factory_receives_policy_and_config(tmp_path, monkeypatch):
    from skillev.evosteer_demo import build_smoke_application

    app, bindings = build_smoke_application()
    module = ModuleType("evosteer_test_task_factory")

    def factory(*, policy, config):
        assert policy is app.policy
        assert config is app.config
        return bindings

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    actual, identity = _factory_bindings(f"{module.__name__}:factory", app.policy, app.config)
    assert actual == bindings
    assert isinstance(identity, str)
    assert identity


def test_real_train_from_local_weights_keeps_expected_answers_out_of_artifacts(tmp_path):
    from skillev.evosteer_demo import build_tiny_policy

    policy = build_tiny_policy()
    model_path = tmp_path / "local-model"
    policy.model.unload().save_pretrained(model_path)
    policy.tokenizer.save_pretrained(model_path)
    config = {
        "format": "evosteer-local-training@2",
        "method_mode": "debug",
        "model": {
            "model_path": "local-model",
            "reference_id": "local-test@1",
            "target_modules": ["c_proj"],
            "lora_rank": 2,
            "lora_alpha": 4,
            "context_window": 768,
            "context_mode": "debug_head_tail",
            "max_action_tokens": 256,
        },
        "application": {
            "task_families": ["qa"],
            "max_nodes": 1,
            "max_actions": 4,
            "current_rollouts": 1,
            "reference_rollouts": 1,
            "roles": [
                {
                    "role_id": role,
                    "instruction": "Answer the question.",
                    "model_maximum": {
                        "input_tokens": 768,
                        "output_tokens": 4,
                        "model_calls": 1,
                        "agent_turns": 1,
                        "wall_time_milliseconds": 120000,
                    },
                }
                for role in ("solver", "verifier")
            ],
        },
        "steps": 1,
        "batch_size": 1,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    private_answer = "PRIVATE_EXPECTED_ANSWER_NOT_MODEL_INPUT_923"
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "task_id": "q1",
                "family": "qa",
                "prompt": "A public question.",
                "expected_answers": [private_answer],
            }
        )
        + "\n"
    )
    output = tmp_path / "trained"
    result = command("train", "--config", config_path, "--tasks", tasks, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["batch_index"] == 1
    assert summary["synthetic"] is False
    for artifact in output.rglob("*"):
        if artifact.suffix in {".json", ".jsonl"}:
            assert private_answer not in artifact.read_text()


def test_source_schedule_is_applied_and_restored_with_optimizer_checkpoint(tmp_path):
    from tests.evosteer.test_application import binding
    from tests.evosteer.test_method_conformance import method_application

    bindings = tuple(binding(task_id) for task_id in ("a1", "a2", "b1", "b2"))
    config = {
        "batch_size": 2,
        "method_mode": "full",
        "sampling": {
            "task_sources": {
                "a1": "dataset-a",
                "a2": "dataset-a",
                "b1": "dataset-b",
                "b2": "dataset-b",
            },
            "curriculum": [{"through_step": 12, "source_ids": ["dataset-a", "dataset-b"]}],
        },
    }
    output = tmp_path / "scheduled"
    output.mkdir()
    asyncio.run(
        _run(
            method_application(),
            bindings,
            output=output,
            steps=1,
            batch_size=2,
            source_id="synthetic-source-manifest@1",
            synthetic=True,
            resume=None,
            run_config=config,
        )
    )
    first = json.loads((output / "batches" / "batch-000001.json").read_text())
    assert first["task_sampling"]["source_counts"] == {"dataset-a": 1, "dataset-b": 1}
    checkpoint = output / "checkpoints" / "batch-000001"
    context = json.loads((checkpoint / "cli-context.json").read_text())
    assert context["sampling_state"]["committed_steps"] == 1
    resumed = tmp_path / "resumed-schedule"
    resumed.mkdir()
    asyncio.run(
        _run(
            method_application(),
            bindings,
            output=resumed,
            steps=2,
            batch_size=2,
            source_id="synthetic-source-manifest@1",
            synthetic=True,
            resume=checkpoint,
            run_config=config,
        )
    )
    second = json.loads((resumed / "batches" / "batch-000002.json").read_text())
    assert second["task_sampling"]["step_index"] == 2
    assert second["task_sampling"]["source_counts"] == {"dataset-a": 1, "dataset-b": 1}
    restored_context = json.loads(
        (resumed / "checkpoints" / "batch-000002" / "cli-context.json").read_text()
    )
    assert restored_context["sampling_state"]["committed_steps"] == 2


def test_full_method_configuration_rejects_silent_ablations(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs" / "evosteer" / "with-author.example.json").read_text())
    model = tmp_path / "placeholder-model"
    model.mkdir()
    config["model"]["model_path"] = str(model)
    config["author"]["model"]["model_path"] = str(model)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    parsed, application = _configuration(path)
    assert parsed["method_mode"] == "full"
    assert application.max_nodes is application.max_actions is None
    assert application.value_refresh_interval == 1
    assert application.resource_cap.model_calls == application.total_token_cap
    for changed in (
        {**config, "author": None},
        {**config, "model": {**config["model"], "context_mode": "debug_head_tail"}},
        {**config, "application": {**config["application"], "max_nodes": 8}},
        {**config, "application": {**config["application"], "reference_rollouts": 1}},
    ):
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            _configuration(path)
    del config["author"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="independent frozen skill author"):
        _configuration(path)


def test_pipelined_schedule_prefetches_with_replica_and_resumes(tmp_path, monkeypatch):
    from tests.evosteer.test_application import _replica, binding
    from tests.evosteer.test_method_conformance import method_application

    monkeypatch.setenv("EVOSTEER_PIPELINE", "1")
    bindings = tuple(binding(task_id) for task_id in ("a1", "a2", "b1", "b2"))
    config = {
        "batch_size": 2,
        "method_mode": "full",
        "sampling": {
            "task_sources": {
                "a1": "dataset-a",
                "a2": "dataset-a",
                "b1": "dataset-b",
                "b2": "dataset-b",
            },
            "curriculum": [{"through_step": 12, "source_ids": ["dataset-a", "dataset-b"]}],
        },
    }

    def run(app, output, steps, resume=None):
        output.mkdir()
        return asyncio.run(
            _run(
                app,
                bindings,
                output=output,
                steps=steps,
                batch_size=2,
                source_id="synthetic-source-manifest@1",
                synthetic=True,
                resume=resume,
                run_config=config,
            )
        )

    with pytest.raises(ValueError, match="rollout replica"):
        run(method_application(), tmp_path / "no-replica", 1)
    app = method_application()
    app.set_rollout_policy(_replica(app.policy))
    output = tmp_path / "pipelined"
    summary = run(app, output, 3)
    assert summary["batch_index"] == 3
    for step in (1, 2, 3):
        batch_id = f"batch-{step:06d}"
        metrics = json.loads((output / "batches" / f"{batch_id}.json").read_text())
        assert metrics["task_sampling"]["step_index"] == step
        context = json.loads((output / "checkpoints" / batch_id / "cli-context.json").read_text())
        assert context["sampling_state"]["committed_steps"] == step
    assert app.rollout_policy.version == app.policy.version == 3
    resumed = method_application()
    resumed.set_rollout_policy(_replica(resumed.policy))
    run(resumed, tmp_path / "resumed", 3, resume=output / "checkpoints" / "batch-000002")
    third = json.loads((tmp_path / "resumed" / "batches" / "batch-000003.json").read_text())
    assert third["task_sampling"]["step_index"] == 3
    assert resumed.rollout_policy.version == resumed.policy.version == 3
