"""Explicit stopping and cooperative deadlines on the real isolated owner path."""

import asyncio
import json
import time

import pytest
from skillev_private.evaluation import integrity_runtime
from skillev_private.evaluation.interactive_lifecycle import InteractiveLifecycle

from skillev.evaluation.capability_registry import native_tool_definitions
from skillev.evaluation.direct_baseline import DirectGenerationRequest
from skillev.evaluation.direct_baseline.interactive_tasks import (
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativePublicState,
)
from skillev.evaluation.integrity_generation import EvaluationDeadlineExceeded
from skillev.evaluation.interactive_termination import OWNER_FINISH_PROFILE, owner_finish_requested
from skillev.evaluation.sealed_candidates import EventOrigin
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from tests.evaluation.test_integrity_broker_boundary import ServingFixture, runtime
from tests.evaluation.test_step0_architecture import _Generator, _profile
from tests.evaluation.test_step0_integrity_runtime import CleanTokenizer, clean_client

FINISH = "<tool_call>\n<function=finish>\n</function>\n</tool_call>"
ACT = "<tool_call>\n<function=act>\n<parameter=command>look</parameter>\n</function>\n</tool_call>"
ARM = InferenceArm("finish", tool_call_mode=ToolCallMode.QWEN_XML)
SCOPE = ("synthetic", ARM.arm_id, "case")


class Environment:
    def __init__(self):
        self.actions = []
        self.closed = False
        self.outcome_calls = 0

    async def reset(self):
        return NativePublicState("Synthetic room and marker.", ("act",))

    async def step(self, action):
        self.actions.append(action)
        return NativeEnvironmentStep(
            "You see the marker.", False, 0.75, False, True, available_actions=("act",)
        )

    async def outcome(self):
        self.outcome_calls += 1
        return NativeEnvironmentOutcome(0.75, False, False)

    async def close(self):
        self.closed = True


def science_runtime(monkeypatch, tmp_path, outputs):
    from skillev.evaluation.input_metric_contracts import PublicTaskView

    entry = PublicTaskView.from_record("case", "scienceworld", {"task": "Examine the marker."})
    instance = runtime(tmp_path, entry, outputs)
    instance.config.update(
        scienceworld_termination_profile=OWNER_FINISH_PROFILE, request_timeout_seconds=4
    )
    instance.source.interactive["case"] = {"case": {"max_steps": 200}}
    environment = Environment()
    monkeypatch.setattr(
        integrity_runtime, "create_native_environment", lambda *args, **kwargs: environment
    )
    return instance, entry, environment


@pytest.mark.parametrize("text", [FINISH, "Finish", "finish().", "I decide to stop.\n" + FINISH])
def test_only_explicit_finish_carrier(text):
    assert owner_finish_requested(text)


@pytest.mark.parametrize(
    "text",
    [
        "The task is complete.",
        "I will finish after looking.",
        FINISH + ACT,
        "Action: look\n" + FINISH,
        "For example:\n" + FINISH,
        "```\n" + FINISH + "\n```",
        FINISH.replace("</function>", "<parameter=score>100</parameter></function>"),
    ],
)
def test_never_infer_completion_or_choose_between_submissions(text):
    assert not owner_finish_requested(text)


def test_finish_is_an_opt_in_scienceworld_tool():
    def names(definitions):
        return {row["function"]["name"] for row in definitions}

    assert "finish" not in names(native_tool_definitions("scienceworld"))
    assert "finish" in names(native_tool_definitions("scienceworld", owner_finish=True))
    with pytest.raises(ValueError):
        native_tool_definitions("completion", owner_finish=True)


@pytest.mark.parametrize("outputs", [[FINISH], ["The task is complete.", FINISH], [ACT, FINISH]])
def test_owner_stops_at_current_partial_score_without_an_extra_native_action(
    monkeypatch, tmp_path, outputs
):
    instance, entry, environment = science_runtime(monkeypatch, tmp_path, outputs)
    try:
        final = asyncio.run(instance.generate(entry, ARM, "synthetic"))
        assert json.loads(final.text) == environment.actions == (["look"] if ACT in outputs else [])
        assert final.intervention_counts["model_calls"] == len(outputs)
        assert final.intervention_counts["peer_model_calls"] == 0
        assert final.intervention_counts["communication_repairs"] == int(
            len(outputs) == 2 and ACT not in outputs
        )
        assert (
            len(instance.journal.traces(SCOPE, "owner-finish", origin=EventOrigin.MODEL_TRANSPORT))
            == 1
        )
        outcome = instance.journal.traces(SCOPE, "native-outcome", origin=EventOrigin.ENVIRONMENT)
        assert len(outcome) == 1
        assert outcome[0]["reward"] == 0.75
        assert not outcome[0]["success"]
        assert not outcome[0]["terminal_reached"]
        assert environment.closed
        assert environment.outcome_calls == 1
        instance.validate_candidate(instance.journal, entry, ARM, "synthetic")
    finally:
        instance.journal.close()


def test_legacy_interface_does_not_silently_gain_finish(monkeypatch, tmp_path):
    instance, entry, environment = science_runtime(monkeypatch, tmp_path, [FINISH])
    instance.config.pop("scienceworld_termination_profile")
    instance.config["budgets"]["scienceworld"]["total_model_calls"] = 1
    try:
        asyncio.run(instance.generate(entry, ARM, "synthetic"))
        assert not instance.journal.traces(SCOPE, "owner-finish")
        assert not environment.actions
    finally:
        instance.journal.close()


def test_admitted_response_drains_but_cannot_act_after_deadline(monkeypatch, tmp_path):
    instance, entry, environment = science_runtime(monkeypatch, tmp_path, [])
    instance.config["episode_timeout_seconds"] = 2

    class SlowSecondResponse(ServingFixture):
        async def generate_evaluation(self, request, **kwargs):
            if self.profiles:
                await asyncio.sleep(2.05)
            return await super().generate_evaluation(request, **kwargs)

    instance.generators = [SlowSecondResponse([ACT, ACT])]
    try:
        final = asyncio.run(instance.generate(entry, ARM, "synthetic"))
        assert json.loads(final.text) == environment.actions == ["look"]
        outputs = instance.journal.model_outputs(SCOPE)
        assert len(outputs) == final.intervention_counts["model_calls"] == 2
        assert final.completion_tokens == sum(
            row.result["usage"]["output_tokens"] for row in outputs
        )
        assert not instance.journal.traces(SCOPE, "model-transport-failure")
        assert environment.closed
        assert environment.outcome_calls == 1
        closes = instance.journal.traces(
            SCOPE, "public-episode-close", origin=EventOrigin.ACTOR_DIAGNOSTIC
        )
        assert closes[0]["reason"] == "wall-clock"
        instance.validate_candidate(instance.journal, entry, ARM, "synthetic")
    finally:
        instance.journal.close()


@pytest.mark.parametrize("failed_outcome", [False, True])
def test_real_transport_failure_keeps_known_outcome_without_fabricating_submission(
    monkeypatch, tmp_path, failed_outcome
):
    instance, entry, environment = science_runtime(monkeypatch, tmp_path, [])

    class FailedGenerator:
        async def generate_evaluation(self, request, **kwargs):
            raise TimeoutError("synthetic serving timeout")

    async def unknown_outcome():
        environment.outcome_calls += 1
        raise ConnectionError("synthetic environment disconnected")

    if failed_outcome:
        environment.outcome = unknown_outcome
    instance.generators = [FailedGenerator()]
    try:
        with pytest.raises(TimeoutError):
            asyncio.run(instance.generate(entry, ARM, "synthetic"))
        with pytest.raises(KeyError):
            instance.journal.get(*SCOPE)
        assert len(instance.journal.traces(SCOPE, "native-outcome")) == int(not failed_outcome)
        assert len(instance.journal.traces(SCOPE, "native-outcome-incomplete")) == int(
            failed_outcome
        )
        assert environment.closed
        assert environment.outcome_calls == 1
        failures = instance.journal.traces(SCOPE, "model-transport-failure")
        assert len(failures) == 1
        assert "unknown" in failures[0]["usage"]
    finally:
        instance.journal.close()


def test_expired_before_call_spends_no_model_budget():
    generator = _Generator([], tokenizer=CleanTokenizer())
    client = clean_client(generator, InferenceArm("deadline"))
    client.generation.deadline_monotonic = time.monotonic() - 1
    with pytest.raises(EvaluationDeadlineExceeded):
        asyncio.run(
            client.generate(
                DirectGenerationRequest(
                    "case", ({"role": "user", "content": "Synthetic calculation."},), _profile()
                )
            )
        )
    assert client.counts.model_calls == 0
    assert not client.generation.calls
    assert not generator.profiles


def test_deadline_crossing_during_ipc_is_a_nonadmitted_call_not_unknown_usage():
    class RefusedGenerator(_Generator):
        async def generate_evaluation(self, request, **kwargs):
            raise EvaluationDeadlineExceeded("broker declined admission")

    generator = RefusedGenerator([], tokenizer=CleanTokenizer())
    client = clean_client(generator, InferenceArm("deadline"))
    with pytest.raises(EvaluationDeadlineExceeded):
        asyncio.run(
            client.generate(
                DirectGenerationRequest(
                    "case", ({"role": "user", "content": "Synthetic calculation."},), _profile()
                )
            )
        )
    assert client.counts.model_calls == client.generation.output_tokens == 0
    assert client.generation.calls[0].transport_status == "not-admitted"


def test_broker_rejects_post_deadline_action_before_any_native_intent(monkeypatch, tmp_path):
    instance, entry, environment = science_runtime(monkeypatch, tmp_path, [ACT])
    original = InteractiveLifecycle.admit

    def close_action_admission(self, operation):
        if operation == "environment-step":
            self.deadline = time.monotonic() - 1
        return original(self, operation)

    monkeypatch.setattr(InteractiveLifecycle, "admit", close_action_admission)
    try:
        final = asyncio.run(instance.generate(entry, ARM, "synthetic"))
        assert json.loads(final.text) == environment.actions == []
        assert final.intervention_counts["model_calls"] == 1
        assert not instance.journal.traces(SCOPE, "environment-result")
        assert instance.journal.traces(SCOPE, "deadline-admission")
        assert environment.closed
    finally:
        instance.journal.close()


def test_broker_declines_late_model_call_and_seals_prior_actions(monkeypatch, tmp_path):
    instance, entry, environment = science_runtime(monkeypatch, tmp_path, [ACT])
    original = InteractiveLifecycle.admit

    def close_next_admission(self, operation):
        if operation == "generate" and environment.actions:
            self.deadline = time.monotonic() - 1
        return original(self, operation)

    monkeypatch.setattr(InteractiveLifecycle, "admit", close_next_admission)
    try:
        final = asyncio.run(instance.generate(entry, ARM, "synthetic"))
        assert json.loads(final.text) == environment.actions == ["look"]
        assert final.intervention_counts["model_calls"] == 1
        assert len(instance.journal.model_outputs(SCOPE)) == 1
        assert not instance.journal.traces(SCOPE, "model-transport-failure")
        accounting = instance.journal.traces(
            SCOPE, "call-accounting", origin=EventOrigin.ACTOR_DIAGNOSTIC
        )
        assert accounting[-1]["transport_status"] == "not-admitted"
        assert environment.closed
        instance.validate_candidate(instance.journal, entry, ARM, "synthetic")
    finally:
        instance.journal.close()
