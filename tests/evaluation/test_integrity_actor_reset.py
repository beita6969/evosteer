"""The isolated actor uses the live public task, not a stale crowd annotation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import pytest

from skillev.evaluation import integrity_actor
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.step0_integrity import InferenceArm
from skillev.rollout import GenerationPhase
from skillev.rollout.evaluation_sglang import EvaluationPolicyDescriptor
from tests.evaluation.test_step0_architecture import _Generator, _profile
from tests.evaluation.test_step0_integrity_runtime import CleanTokenizer


class ScriptedGenerator(_Generator):
    descriptor = EvaluationPolicyDescriptor("synthetic-reset-policy", "synthetic-model", "fixture")

    def snapshot(self):
        return self.descriptor


def initial_task(benchmark, task):
    return {
        "public_task": asdict(PublicTaskView.from_record("synthetic", benchmark, {"task": task})),
        "arm": InferenceArm(
            "synthetic-reset",
        ).to_value(),
        "run_id": "synthetic-reset",
        "policy": asdict(ScriptedGenerator.descriptor),
        "decoding": asdict(_profile()),
        "population_id": "synthetic-only",
        "panel_position": 7,
        "budgets": {
            "skill_instruction_tokens": 0,
            "calls_per_turn": 4,
            "history_input_tokens": 10000,
            "total_model_calls": 8,
            "total_output_tokens": 2048,
            "context_length": 32768,
            "environment_steps": 2,
        },
    }


@pytest.mark.parametrize("benchmark", ["alfworld", "webshop"])
def test_actor_owner_and_later_turn_receive_the_public_reset_task(monkeypatch, benchmark):
    manifest = "Find the fictional indigo marker on a nonexistent pedestal."
    native_task = "Examine the fictional amber marker." if benchmark == "alfworld" else manifest
    actions = (
        ("look", "inventory") if benchmark == "alfworld" else ("search[marker]", "click[item]")
    )
    script = [(GenerationPhase.ACTION, f"Action: {actions[0]}")]
    script.append((GenerationPhase.ACTION, f"Action: {actions[1]}"))
    tokenizer = CleanTokenizer()
    generator = ScriptedGenerator(script, tokenizer=tokenizer)
    monkeypatch.setattr(integrity_actor, "BrokerGenerator", lambda descriptor: generator)
    executed, traced_inputs = [], []

    def broker(operation, **arguments):
        if operation == "environment-reset":
            return {
                "observation_text": f"Synthetic room.\nYour task is to: {native_task}\n",
                "available_actions": [actions[0]],
            }
        if operation == "environment-step":
            executed.append(arguments["action"])
            return {
                "observation": "Synthetic public transition.",
                "terminal": len(executed) == 2,
                "action_valid": True,
                "available_actions": [actions[1]],
            }
        assert operation == "trace"
        if arguments["stage"] == "input":
            traced_inputs.append(arguments["payload"]["messages"])

    monkeypatch.setattr(integrity_actor, "rpc", broker)
    candidate = asyncio.run(integrity_actor.run_episode(initial_task(benchmark, manifest)))

    assert json.loads(candidate.text) == list(actions) == executed
    assert candidate.intervention_counts["model_calls"] == 2
    assert candidate.intervention_counts["peer_model_calls"] == 0
    for messages, thinking in tokenizer.messages:
        assert thinking is False
        assert any(item["role"] == "user" and item["content"] == native_task for item in messages)
        if benchmark == "alfworld":
            assert manifest not in repr(messages)
    assert all(messages[0]["content"] == native_task for messages in traced_inputs)


def test_actor_does_not_fall_back_to_an_unrelated_manifest_if_reset_task_is_absent(monkeypatch):
    generator = ScriptedGenerator([], tokenizer=CleanTokenizer())
    monkeypatch.setattr(integrity_actor, "BrokerGenerator", lambda descriptor: generator)
    monkeypatch.setattr(
        integrity_actor,
        "rpc",
        lambda operation, **arguments: {
            "observation_text": "Synthetic reset without its task instruction.",
            "available_actions": ["look"],
        },
    )
    with pytest.raises(RuntimeError):
        asyncio.run(integrity_actor.run_episode(initial_task("alfworld", "Unrelated task")))
    assert not generator.tokenizer.messages
