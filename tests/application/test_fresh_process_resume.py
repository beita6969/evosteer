"""New process/model restoration; synthetic policy/author, not natural evolution."""

import asyncio
import json
import multiprocessing
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

from skillev.application_reporting import resolved_method_state
from skillev.experiments import PublishedAttemptIdentity
from skillev.policy import QwenBackboneConfig, TrainableStateIdentity
from skillev.rollout import RolloutTask
from skillev.runtime import LiveAttemptEventLog, StepTransactionJournal
from skillev.training.step_math import named_ttb_parameters
from tests.application.test_bayesian_recovery import restore_fixture
from tests.application.test_full_vertical_loop import build_application_fixture


def _fresh_process(payload: dict) -> None:
    root = Path(payload["root"])
    identity = PublishedAttemptIdentity.from_value(payload["identity"])
    fixture = SimpleNamespace(
        tasks=tuple(RolloutTask.from_value(task) for task in payload["tasks"]),
        rewards=tuple(payload["rewards"]),
        initial_policy_directory=root / "initial-policy",
        initial_trainable_state=TrainableStateIdentity.from_value(
            payload["initial_trainable_state"]
        ),
        attempt_budget=identity.attempt_budget,
        authority=identity.authoring_authority,
        phi_maximum=identity.phi_per_cycle_maximum,
        public_identity=identity,
        event_log=LiveAttemptEventLog.resume(
            root / "events.jsonl", run_id="v3-application", attempt_id="attempt-1"
        ),
        journal=StepTransactionJournal(root / "snapshots/step-transactions"),
    )
    app, _ = restore_fixture(
        fixture,
        root / "snapshots/step-00000002",
        QwenBackboneConfig.from_value(payload["backbone"]),
    )
    asyncio.run(app.evolution_loop.run(identity.run_plan))
    torch.save(
        {
            key: parameter.detach().clone()
            for key, parameter in named_ttb_parameters(app.backbone.parameter_groups()).items()
        },
        root / "next-parameters.pt",
    )
    (root / "next-state.json").write_text(json.dumps(resolved_method_state(app)))


def test_fresh_process_restores_evolved_library_and_identical_next_update(
    tmp_path, training_backbone_config
):
    fixture = build_application_fixture(tmp_path / "original", training_backbone_config)
    app = fixture.application
    asyncio.run(app.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=2))
    copied = tmp_path / "fresh-process"
    shutil.copytree(tmp_path / "original", copied)
    payload = {
        "root": str(copied),
        "identity": fixture.public_identity.to_value(),
        "initial_trainable_state": fixture.initial_trainable_state.to_value(),
        "backbone": training_backbone_config.to_value(),
        "tasks": [task.to_value() for task in fixture.tasks],
        "rewards": fixture.rewards,
    }
    process = multiprocessing.get_context("spawn").Process(target=_fresh_process, args=(payload,))
    process.start()
    try:
        process.join(timeout=90)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
    asyncio.run(app.evolution_loop.run(fixture.run_plan))
    assert json.loads((copied / "next-state.json").read_text()) == resolved_method_state(app)
    actual = torch.load(copied / "next-parameters.pt", weights_only=True)
    for name, parameter in named_ttb_parameters(app.backbone.parameter_groups()).items():
        torch.testing.assert_close(actual[name], parameter.detach(), rtol=0, atol=0)
