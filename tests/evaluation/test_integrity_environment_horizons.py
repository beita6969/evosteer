"""Declared horizons reach both the actor budget and the native simulator."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from skillev_private.evaluation import integrity_environments
from skillev_private.evaluation.integrity_runtime import PrivateIntegrityRuntime

from skillev.evaluation.input_metric_contracts import PublicTaskView


def test_frozen_action_budget_can_override_historical_manifest_without_mutating_it():
    instance = object.__new__(PrivateIntegrityRuntime)
    instance.config = {
        "context_length": 98304,
        "budgets": {"alfworld": {"total_model_calls": 160, "total_output_tokens": 163840}},
        "environment_horizons": {"alfworld": 100},
    }
    instance.source = SimpleNamespace(interactive={"case": {"case": {"max_steps": 20}}})
    entry = PublicTaskView.from_record("case", "alfworld", {"task": "Synthetic household task."})
    limits = instance._budgets(entry)
    assert limits["environment_steps"] == 100
    assert limits["total_model_calls"] == 160
    assert limits["total_output_tokens"] == 163840
    assert instance.source.interactive["case"]["case"]["max_steps"] == 20
    instance.config.pop("environment_horizons")
    assert instance._budgets(entry)["environment_steps"] == 20


@pytest.mark.parametrize("value", [True, 0, -1, "100", 10.5])
def test_invalid_native_action_budget_is_not_silently_coerced(value):
    instance = object.__new__(PrivateIntegrityRuntime)
    instance.config = {
        "context_length": 98304,
        "budgets": {"webshop": {"total_model_calls": 4, "total_output_tokens": 1024}},
        "environment_horizons": {"webshop": value},
    }
    instance.source = SimpleNamespace(interactive={"case": {"case": {"max_steps": 10}}})
    entry = PublicTaskView.from_record("case", "webshop", {"task": "Synthetic shopping task."})
    with pytest.raises(ValueError):
        instance._budgets(entry)


@pytest.mark.parametrize(
    ("outer", "simulator", "expected"), [(None, 20, 20), (100, 20, 100), (100, 200, 200)]
)
def test_native_simulator_never_retains_a_shorter_historical_horizon(
    monkeypatch, outer, simulator, expected
):
    specification = {
        "case": {
            "task_id": "case",
            "benchmark": "alfworld",
            "task": "Synthetic task.",
            "max_steps": 20,
            "deployment": "fictional",
            "payload": {"environment_id": "fictional", "game_id": "fictional-game"},
        },
        "manifest": {
            "deployments": {
                "fictional": {
                    "runtime": "fixture",
                    "config_path": "/fictional/config.yaml",
                    "simulator_max_steps": simulator,
                    "games": {
                        "fictional-game": {
                            "data_directory": "/fictional/game",
                            "train_eval": "eval_out_of_distribution",
                            "instruction_text": "Synthetic task.",
                        }
                    },
                }
            },
            "runtimes": {
                "fixture": {
                    "interpreter_path": "/fictional/python",
                    "source_root": "/fictional/source",
                    "source_revision": "fixture",
                }
            },
        },
    }
    original = deepcopy(specification)
    constructed = []
    monkeypatch.setattr(integrity_environments, "PinnedOfficialProcess", lambda *a, **kw: None)
    monkeypatch.setattr(integrity_environments, "ALFWorldGameDeployment", lambda *a: None)

    class Factory:
        def __init__(self, *args):
            self.native_limit = args[-1]

        def create(self, task):
            constructed.append((task.max_steps, self.native_limit))
            return None

    monkeypatch.setattr(integrity_environments, "OfficialALFWorldProcessFactory", Factory)
    monkeypatch.setattr(integrity_environments, "DirectALFWorldEnvironment", lambda *a: None)
    integrity_environments.create_native_environment(specification, seed=0, maximum_steps=outer)
    assert constructed == [(20 if outer is None else outer, expected)]
    assert specification == original
