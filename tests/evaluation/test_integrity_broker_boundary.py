"""Real isolated actor -> trusted broker -> synthetic serving/environment boundaries."""

import asyncio
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from skillev_private.evaluation import integrity_runtime
from skillev_private.evaluation.integrity_replicas import ReplicaLoadBalancer

from skillev.evaluation.actor_sandbox import ActorSandbox
from skillev.evaluation.direct_baseline.interactive_tasks import (
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativePublicState,
)
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.integrity_communication_report import communication_summary
from skillev.evaluation.sealed_candidates import CandidateJournal, EventOrigin
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from skillev.rollout import RolloutGenerationResult
from skillev.rollout.evaluation_sglang import EvaluationPolicyDescriptor
from skillev.runtime import BudgetVector
from tests.evaluation.test_step0_architecture import _profile
from tests.evaluation.test_step0_integrity_runtime import CleanTokenizer


class ServingFixture:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.profiles = []

    async def generate_evaluation(self, request, **kwargs):
        self.profiles.append(kwargs["profile"])
        text = self.outputs.pop(0)
        return RolloutGenerationResult(
            content_token_ids=tuple(ord(char) for char in text),
            stop_token_ids=(),
            finish_reason="stop",
            policy_snapshot_id="fixture-policy",
            backend_id="fixture",
            usage=BudgetVector(
                input_tokens=len(request.input_ids), output_tokens=len(text), model_calls=1
            ),
        )


def runtime(tmp_path, entry, outputs, *, calls=4):
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("real isolation requires bubblewrap on the approved CPU test host")
    instance = object.__new__(integrity_runtime.PrivateIntegrityRuntime)
    instance.directory = tmp_path
    instance.thinking_policy = None
    instance.config = {
        "budgets": {
            entry.benchmark: {
                "calls_per_turn": 4,
                "total_model_calls": calls,
                "total_output_tokens": 4096,
                "history_input_tokens": 16000,
                "skill_instruction_tokens": 0,
            }
        },
        "decoding": {entry.benchmark: asdict(_profile(max_tokens=512))},
        "context_length": 20000,
        "episode_timeout_seconds": 30,
    }
    instance.source = SimpleNamespace(
        interactive={},
        provenance={},
        public_input_receipts={},
        panel=SimpleNamespace(exposure="synthetic-development"),
    )
    instance.indices = {entry.task_id: 11}
    instance.journal = CandidateJournal(tmp_path / "candidate.sqlite")
    instance.policy = EvaluationPolicyDescriptor("fixture-policy", "Qwen3.5-9B", "fixture")
    instance.tokenizer = CleanTokenizer()
    instance.generators = [ServingFixture(outputs)]
    instance.trained_policies = {}
    instance.trained_generators = {}
    instance.replicas = ReplicaLoadBalancer(1)
    instance.sandbox = replace(
        ActorSandbox.current(Path(__file__).resolve().parents[2] / "src"), bubblewrap=Path(bwrap)
    )
    return instance


def test_actual_actor_rejects_consultation_and_keeps_one_trusted_final_owner(tmp_path):
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic arithmetic."})
    instance = runtime(tmp_path, entry, ["Message to solver:\nPlease help.", "Final answer: 42"])
    final = asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
    assert final.text == r"\boxed{42}"
    assert final.submission["raw_response"] == "Final answer: 42"
    scope = ("synthetic", "A2", "case")
    assert instance.journal.get(*scope) == final
    report = communication_summary(
        instance.journal, (scope,), benchmark_by_scope={scope: "aime-2026"}
    )
    assert report["authoritative_model_requests"] == 2
    assert not report["peer_requests"]
    assert not report["peer_replies"]
    assert all(row.participant == "owner" for row in instance.journal.model_outputs(scope))
    instance.journal.close()


def test_actor_trace_cannot_impersonate_native_outcome(tmp_path):
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic task."})
    instance = runtime(tmp_path, entry, [])

    class ForgingActor:
        async def run(self, initial, handle, **kwargs):
            return await handle(
                {"operation": "trace", "stage": "native-outcome", "payload": {"success": True}}
            )

    instance.sandbox = ForgingActor()
    with pytest.raises(ValueError):
        asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
    assert not instance.journal.traces(("synthetic", "A2", "case"), "native-outcome")
    instance.journal.close()


def test_actor_cannot_choose_a_seed_outside_the_declared_call_stream(tmp_path):
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic arithmetic."})
    instance = runtime(tmp_path, entry, ["Final answer: 42"])
    original = instance.sandbox

    class SeedRewritingActor:
        async def run(self, initial, handle, **kwargs):
            async def changed(message):
                if message.get("operation") == "generate":
                    message["profile"]["seed"] += 1
                    message["request"]["seed"] = message["profile"]["seed"]
                return await handle(message)

            return await original.run(initial, changed, **kwargs)

    instance.sandbox = SeedRewritingActor()
    with pytest.raises(ValueError):
        asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
    assert not instance.generators[0].profiles
    assert not instance.journal.model_outputs(("synthetic", "A2", "case"))
    instance.journal.close()


@pytest.mark.parametrize(
    "response",
    [
        "Draft calculation: \\boxed{12}\nFinal Answer: 17\n\n\\boxed{017}",
        "Final answer: 17\nDiscussion continues.\nFinal answer: 017\n\\boxed{17}",
        "Final answer:\n\\boxed{17}\nDiscussion continues.\nFinal answer: 17",
    ],
)
def test_same_integer_repeated_in_owner_final_crosses_broker_without_a_repair(tmp_path, response):
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic arithmetic."})
    instance = runtime(tmp_path, entry, [response], calls=2)
    final = asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
    assert final.text == r"\boxed{17}"
    assert final.submission["raw_response"] == response
    assert final.intervention_counts["model_calls"] == 1
    assert final.intervention_counts["communication_repairs"] == 0
    assert final.intervention_counts["peer_model_calls"] == 0
    instance.journal.close()


def test_conflicting_repeated_finals_need_a_fresh_owner_submission(tmp_path):
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic arithmetic."})
    response = "Final answer: 17\nFinal answer: 17\nFinal answer: 18"
    instance = runtime(tmp_path, entry, [response, "Final answer: 19"], calls=2)
    final = asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
    assert final.text == r"\boxed{19}"
    assert final.submission["raw_response"] == "Final answer: 19"
    assert final.intervention_counts["model_calls"] == 2
    assert final.intervention_counts["communication_repairs"] == 1
    assert final.intervention_counts["peer_model_calls"] == 0
    instance.journal.close()


def test_unbalanced_code_fences_request_owner_submission_within_the_shared_budget(tmp_path):
    entry = PublicTaskView.from_record(
        "case", "humaneval", {"prompt": "def f():\n    # Synthetic public function request\n"}
    )
    malformed = '```python\ndef f():\n    """Draft\n```python\ndef f():\n    return 2\n```'
    source = "def f():\n    return 3"
    instance = runtime(tmp_path, entry, [malformed, "Final code:\n" + source], calls=2)
    final = asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
    assert final.text == source  # The later owner submission, not either discussed program.
    assert final.submission["raw_response"] == "Final code:\n" + source
    assert final.intervention_counts["model_calls"] == 2
    assert final.intervention_counts["communication_repairs"] == 1
    scope = ("synthetic", "A2", "case")
    requests = instance.journal.traces(
        scope, "rendered-request", origin=EventOrigin.ACTOR_DIAGNOSTIC
    )
    assert requests[-1]["purpose"] == "interface-repair"
    assert "not submitted" in "\n".join(row["content"] for row in requests[-1]["messages"])
    report = communication_summary(instance.journal, (scope,))
    assert report["authoritative_model_requests"] == 2
    assert report["status"] == "complete-with-repairs"
    instance.journal.close()


@pytest.mark.parametrize(
    ("decision", "command", "tool_call_mode"),
    [("Action: look", "look", ToolCallMode.PLAIN_TEXT)]
    + [
        (
            f"<tool_call>\n<function={name}>\n</function>\n</tool_call>",
            name,
            ToolCallMode.QWEN_XML,
        )
        for name in ("look", "inventory", "help")
    ]
    + [
        (
            "<tool_call>\n<function=go to shelf 2>\n</function>\n</tool_call>",
            "go to shelf 2",
            ToolCallMode.QWEN_XML,
        ),
        (
            "<tool_call>\n<function=go to shelf 2\n</parameter>\n</function>\n</tool_call>",
            "go to shelf 2",
            ToolCallMode.QWEN_XML,
        ),
    ],
)
def test_native_action_survives_later_model_budget_exhaustion_and_closes(
    monkeypatch, tmp_path, decision, command, tool_call_mode
):
    entry = PublicTaskView.from_record(
        "case", "alfworld", {"task": "Examine the fictional marker."}
    )
    instance = runtime(tmp_path, entry, [decision], calls=1)
    instance.source.interactive["case"] = {"case": {"max_steps": 2}}
    executed, closed = [], []

    class Environment:
        async def reset(self):
            return NativePublicState(
                "Your task is to: Examine the fictional marker.\nSynthetic room.", (command,)
            )

        async def step(self, action):
            executed.append(action)
            return NativeEnvironmentStep(
                "Observed the synthetic room.",
                False,
                0.0,
                False,
                True,
                available_actions=(command,),
            )

        async def outcome(self):
            return NativeEnvironmentOutcome(0.0, False, False)

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(
        integrity_runtime, "create_native_environment", lambda *args, **kwargs: Environment()
    )
    final = asyncio.run(
        instance.generate(entry, InferenceArm("A2", tool_call_mode=tool_call_mode), "synthetic")
    )
    assert json.loads(final.text) == executed == [command]
    assert final.intervention_counts["model_calls"] == 1
    assert final.intervention_counts["communication_repairs"] == 0
    assert closed == [True]
    scope = ("synthetic", "A2", "case")
    closures = instance.journal.traces(
        scope, "public-episode-close", origin=EventOrigin.ACTOR_DIAGNOSTIC
    )
    assert closures[0]["reason"] == "model-budget"
    assert instance.journal.traces(scope, "native-outcome", origin=EventOrigin.ENVIRONMENT)
    instance.journal.close()


@pytest.mark.parametrize("shared_turn_limit", [False, True])
def test_native_turn_limit_does_not_masquerade_as_whole_episode_exhaustion(
    monkeypatch, tmp_path, shared_turn_limit
):
    entry = PublicTaskView.from_record("case", "alfworld", {"task": "Synthetic task."})
    instance = runtime(
        tmp_path, entry, ["Discussion without an action."] * 4 + ["Action: look"], calls=8
    )
    instance.source.interactive["case"] = {"case": {"max_steps": 3}}
    if shared_turn_limit:
        instance.config["budget_overrides"] = {"alfworld": {"calls_per_turn": 8}}
    executed = []

    class Environment:
        async def reset(self):
            return NativePublicState("Your task is to: Synthetic task.", ("look",))

        async def step(self, action):
            executed.append(action)
            return NativeEnvironmentStep("Native terminal observation.", True, 1.0, True, True)

        async def outcome(self):
            return NativeEnvironmentOutcome(float(bool(executed)), bool(executed), bool(executed))

        async def close(self):
            pass

    monkeypatch.setattr(
        integrity_runtime, "create_native_environment", lambda *a, **kw: Environment()
    )
    final = asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
    scope = ("synthetic", "A2", "case")
    close = instance.journal.traces(
        scope, "public-episode-close", origin=EventOrigin.ACTOR_DIAGNOSTIC
    )[0]
    if shared_turn_limit:
        assert executed == ["look"]
        assert final.intervention_counts["model_calls"] == 5
        assert close["reason"] == "terminal"
    else:
        assert not executed
        assert final.intervention_counts["model_calls"] == 4
        assert close["reason"] == "turn-call-budget"
    assert final.intervention_counts["model_calls"] < 8
    instance.journal.close()


@pytest.mark.parametrize(
    "decision",
    [
        "点击[Red Mug]",
        'click("Red Mug")',
        'Action: click[target]\nArguments: {"target":"Red Mug"}',
        "I will inspect the visible item.\n\nclick[Red Mug]",
        'I will inspect the visible item.\n{"kind":"tool","resource_id":"webshop",'
        '"name":"click","arguments":{"target":"Red Mug"}}',
        "<tool_call>\n<function=click>\n<parameter=target>\nRed Mug\n</parameter>\n"
        "</function>\n</tool_call>",
        "<tool_call>\n<function=click>\n<parameter=target>\nclick[Red Mug]\n</parameter>\n"
        "</function>\n</tool_call>",
        "<tool_call>\n<function=click[Red Mug]>\n</function>\n</tool_call>",
        "<tool_call>\n<function=click[Red Mug]\n</function>\n</tool_call>",
        "<tool_call>\n<function=click[Red Mug]\n</parameter>\n</function>\n</tool_call>",
    ],
)
def test_literal_click_crosses_actual_broker_and_owner_receives_native_feedback(
    monkeypatch, tmp_path, decision
):
    entry = PublicTaskView.from_record("case", "webshop", {"task": "Choose the fictional mug."})
    instance = runtime(
        tmp_path,
        entry,
        [
            decision,
            "click[Buy Now]",
        ],
    )
    instance.source.interactive["case"] = {"case": {"max_steps": 2}}
    actions = []

    class Environment:
        async def reset(self):
            return NativePublicState("Synthetic catalogue.", ("click[Red Mug]",))

        async def step(self, action):
            actions.append(action)
            terminal = len(actions) == 2
            return NativeEnvironmentStep(
                "Purchase complete." if terminal else "Public mug page, displayed price 7.",
                terminal,
                float(terminal),
                terminal,
                None,
                available_actions=() if terminal else ("click[Buy Now]",),
            )

        async def outcome(self):
            return NativeEnvironmentOutcome(1.0, True, True)

        async def close(self):
            pass

    monkeypatch.setattr(
        integrity_runtime, "create_native_environment", lambda *args, **kwargs: Environment()
    )
    final = asyncio.run(
        instance.generate(
            entry,
            InferenceArm(
                "A2",
                tool_call_mode=ToolCallMode.QWEN_XML
                if "<tool_call>" in decision
                else ToolCallMode.PLAIN_TEXT,
            ),
            "synthetic",
        )
    )
    scope = ("synthetic", "A2", "case")
    assert json.loads(final.text) == actions == ["click[Red Mug]", "click[Buy Now]"]
    responses = instance.journal.traces(
        scope, "model-response", origin=EventOrigin.ACTOR_DIAGNOSTIC
    )
    assert responses[0]["text"] == decision
    requests = instance.journal.traces(
        scope, "rendered-request", origin=EventOrigin.ACTOR_DIAGNOSTIC
    )
    assert all(row["participant"] == "owner" for row in requests)
    if "<tool_call>" in decision:
        owner_system = requests[0]["messages"][0]["content"]
        assert "Action:" not in owner_system
        assert '"kind":"tool"' not in owner_system
        assert any(tool["function"]["name"] == "click" for tool in requests[0]["tools"])
    next_text = "\n".join(row["content"] for row in requests[1]["messages"])
    assert "Public mug page, displayed price 7." in next_text
    assert "click[Buy Now]" in next_text
    transition = instance.journal.traces(
        scope, "public-transition", origin=EventOrigin.ACTOR_DIAGNOSTIC
    )[0]
    assert transition["execution_status"] == "acknowledged"
    owner_after_action = next(row for row in requests[1:] if row["participant"] == "owner")
    owner_text = "\n".join(row["content"] for row in owner_after_action["messages"])
    assert "acknowledged" in owner_text
    assert "separate action-validity flag" in owner_text
    report = communication_summary(instance.journal, (scope,))
    assert report["acknowledged_decisions"] == report["executed_decisions"] == 2
    assert report["authoritative_model_requests"] == 2
    assert final.intervention_counts["communication_repairs"] == 0
    assert report["feedback_mismatches"] == 0
    assert report["status"] == "complete"
    instance.journal.close()


def test_repair_context_crosses_broker_without_replaying_superseded_unexecuted_drafts(
    monkeypatch, tmp_path
):
    entry = PublicTaskView.from_record("case", "alfworld", {"task": "Inspect the synthetic room."})
    first, repeated = "A first unexecuted draft.", "A later unexecuted draft."
    instance = runtime(tmp_path, entry, [first, repeated, repeated, "Action: look"], calls=4)
    instance.source.interactive["case"] = {"case": {"max_steps": 2}}
    actions = []

    class Environment:
        async def reset(self):
            return NativePublicState(
                "Your task is to: Inspect the synthetic room.\nThe fictional marker is on desk 1.",
                ("look",),
            )

        async def step(self, action):
            actions.append(action)
            return NativeEnvironmentStep("Native terminal result.", True, 1.0, True, True)

        async def outcome(self):
            return NativeEnvironmentOutcome(float(bool(actions)), bool(actions), bool(actions))

        async def close(self):
            pass

    monkeypatch.setattr(
        integrity_runtime, "create_native_environment", lambda *a, **kw: Environment()
    )
    final = asyncio.run(instance.generate(entry, InferenceArm("A2"), "synthetic"))
    scope = ("synthetic", "A2", "case")
    requests = instance.journal.traces(
        scope, "rendered-request", origin=EventOrigin.ACTOR_DIAGNOSTIC
    )
    messages = requests[-1]["messages"]
    assert all(row["participant"] == "owner" for row in requests)
    assert first not in [row["content"] for row in messages]
    assert sum(row["content"] == repeated for row in messages) == 1
    assert any(row["role"] == "user" and "no state change" in row["content"] for row in messages)
    assert "The fictional marker is on desk 1." in "\n".join(row["content"] for row in messages)
    assert requests[-1]["archived_message_count"] >= 4
    assert json.loads(final.text) == actions == ["look"]
    assert final.intervention_counts["model_calls"] == 4
    assert final.intervention_counts["communication_repairs"] == 3
    assert len(instance.journal.model_outputs(scope)) == 4
    seed = instance.config["decoding"][entry.benchmark]["seed"]
    assert [profile.seed for profile in instance.generators[0].profiles] == [
        (seed + index) % (1 << 64) for index in range(4)
    ]
    instance.journal.close()
