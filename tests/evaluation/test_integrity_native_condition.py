"""Named decoding changes reach the actual profile without reducing budgets."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from skillev_private.evaluation.integrity_runtime import PrivateIntegrityRuntime

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.step0_integrity import InferenceArm
from tests.evaluation.test_step0_architecture import _profile


@pytest.mark.parametrize("benchmark", ["webshop", "alfworld"])
def test_native_condition_preserves_existing_allowances_and_thinking_policy(benchmark):
    settings = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "configs/evaluation/step0_native_interaction_repairs.yaml"
        ).read_text()
    )
    original = asdict(_profile(max_tokens=8192))
    instance = object.__new__(PrivateIntegrityRuntime)
    instance.thinking_policy = None
    instance.config = {"decoding": {benchmark: original}, **settings}
    entry = PublicTaskView.from_record("case", benchmark, {"task": "Synthetic task."})
    arm = InferenceArm(settings["condition_id"])
    observed = asdict(instance._profile(entry, arm))
    assert (
        observed["presence_penalty"]
        == settings["decoding_overrides"][benchmark]["presence_penalty"]
    )
    assert not observed["enable_thinking"]
    for key, value in original.items():
        if key not in ("presence_penalty", "enable_thinking"):
            assert observed[key] == value
    assert instance.config["decoding"][benchmark] == original


@pytest.mark.parametrize("benchmark", ["webshop", "alfworld"])
def test_native_condition_uses_shared_calls_without_a_smaller_local_limit(benchmark):
    settings = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "configs/evaluation/step0_native_interaction_repairs.yaml"
        ).read_text()
    )
    original = {
        "calls_per_turn": 8,
        "total_model_calls": 160,
        "total_output_tokens": 163840,
        "history_input_tokens": 32768,
        "skill_instruction_tokens": 2048,
    }
    instance = object.__new__(PrivateIntegrityRuntime)
    instance.source = SimpleNamespace(interactive={"case": {"case": {"max_steps": 50}}})
    instance.config = {"budgets": {benchmark: original}, "context_length": 98304, **settings}
    entry = PublicTaskView.from_record("case", benchmark, {"task": "Synthetic task."})
    observed = instance._budgets(entry)
    assert observed["calls_per_turn"] == observed["total_model_calls"] == 160
    for name, value in original.items():
        if name != "calls_per_turn":
            assert observed[name] == value
    assert instance.config["budgets"][benchmark] == original


def test_current_thinking_overlay_overrides_legacy_whole_episode_action_allowance():
    root = Path(__file__).resolve().parents[2] / "configs/evaluation"
    old = yaml.safe_load((root / "step0_native_interaction_repairs.yaml").read_text())
    current = yaml.safe_load((root / "step0_seven_iid32_integrity_repair.yaml").read_text())
    overrides = {**old["budget_overrides"]["alfworld"], **current["budget_overrides"]["alfworld"]}
    instance = object.__new__(PrivateIntegrityRuntime)
    instance.source = SimpleNamespace(interactive={"case": {"case": {"max_steps": 100}}})
    instance.config = {
        "budgets": {
            "alfworld": {
                "calls_per_turn": 8,
                "total_model_calls": 160,
                "total_output_tokens": 163840,
            }
        },
        "budget_overrides": {"alfworld": overrides},
        "context_length": 98304,
    }
    limits = instance._budgets(PublicTaskView.from_record("case", "alfworld", {"task": "Public"}))
    assert limits["calls_per_turn"] == 8
    assert limits["total_model_calls"] == 160
    assert limits["total_output_tokens"] == 163840
    assert limits["environment_steps"] == 100
    assert limits["calls_per_turn"] * limits["native_chunk_tokens"] < limits["context_length"]
