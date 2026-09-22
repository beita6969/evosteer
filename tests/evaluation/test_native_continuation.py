"""One native trajectory, actual chunk provenance and non-discardable draft state."""

import asyncio
from dataclasses import replace

import pytest

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.integrity_context import EpisodeContext
from skillev.evaluation.model_output_provenance import CandidateStatus, served_token_stream
from skillev.evaluation.native_channels import ChannelStatus
from skillev.evaluation.native_continuation import native_call_allowance
from skillev.evaluation.step0_integrity import InferenceArm
from skillev.rollout import RolloutGenerationResult
from skillev.runtime import BudgetVector
from tests.evaluation.test_integrity_broker_boundary import runtime


class ChunkServing:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.requests = []

    async def generate_evaluation(self, request, **kwargs):
        self.requests.append(request)
        text, reason = next(self.chunks)
        assert len(text) <= request.max_new_tokens
        return RolloutGenerationResult(
            content_token_ids=tuple(map(ord, text)),
            stop_token_ids=(),
            finish_reason=reason,
            policy_snapshot_id="fixture-policy",
            backend_id="fixture",
            usage=BudgetVector(
                input_tokens=len(request.input_ids), output_tokens=len(text), model_calls=1
            ),
        )


def setup_runtime(tmp_path, chunks, *, total=512):
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic arithmetic."})
    instance = runtime(tmp_path, entry, [])
    instance.config["budgets"][entry.benchmark].update(
        total_output_tokens=total,
        native_chunk_tokens=128,
        native_final_reserve_tokens=64,
    )
    serving = ChunkServing(chunks)
    instance.generators = [serving]
    return instance, entry, serving, InferenceArm("A2", native_thinking=True)


@pytest.mark.parametrize("first", ["r" * 128, "r" * 123 + "</thi", "r" * 120 + "</think>"])
def test_real_actor_continues_exact_tokens_and_only_owner_final_is_submitted(tmp_path, first):
    tail = "nk>" if first.endswith("</thi") else "" if first.endswith("</think>") else "</think>"
    second = tail + "Final answer: 42\nExplanation."
    instance, entry, serving, arm = setup_runtime(tmp_path, [(first, "length"), (second, "stop")])
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        rows = instance.journal.model_outputs(("synthetic", "A2", "case"))
        assert final.text == r"\boxed{42}"
        assert final.completion_tokens == len(first + second)
        assert final.intervention_counts["model_calls"] == 2
        assert final.intervention_counts["communication_repairs"] == 0
        assert serving.requests[1].input_ids == serving.requests[0].input_ids + tuple(
            map(ord, first)
        )
        assert rows[1].continuation_of_call_id == rows[0].call_id
        assert served_token_stream(rows) == tuple(map(ord, first + second))
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
        with pytest.raises(ValueError):
            served_token_stream((rows[0], replace(rows[1], continuation_of_call_id="unrelated")))
    finally:
        instance.journal.close()


def test_unclosed_reasoning_stays_a_failure_after_all_declared_budget(tmp_path):
    chunks = [("r" * 128, "length"), ("r" * 128, "length"), ("r" * 64, "length")]
    instance, entry, serving, arm = setup_runtime(tmp_path, chunks, total=320)
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        assert not final.text
        assert final.terminal_status is CandidateStatus.BUDGET_EXHAUSTED
        assert final.completion_tokens == 320
        assert [request.max_new_tokens for request in serving.requests] == [128, 128, 64]
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    finally:
        instance.journal.close()


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Draft</think>Final answer: 1", "23"),
        ("Draft</think>\\boxed{12", "3}"),
        (
            "Draft</think><tool_call><function=submit_answer><parameter=answer>1",
            "23</parameter></function></tool_call>",
        ),
    ],
)
def test_length_pause_in_final_continues_instead_of_submitting_a_numeric_prefix(
    tmp_path, first, second
):
    instance, entry, serving, arm = setup_runtime(tmp_path, [(first, "length"), (second, "stop")])
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        rows = instance.journal.model_outputs(("synthetic", "A2", "case"))
        assert final.text == r"\boxed{123}"
        assert rows[0].channel_status is ChannelStatus.FINAL_UNFINISHED
        assert rows[1].channel_status is ChannelStatus.COMPLETE
        assert rows[1].continuation_of_call_id == rows[0].call_id
        assert serving.requests[1].input_ids == serving.requests[0].input_ids + tuple(
            map(ord, first)
        )
        assert served_token_stream(rows) == tuple(map(ord, first + second))
        assert final.intervention_counts["communication_repairs"] == 0
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    finally:
        instance.journal.close()


def test_exhausted_final_prefix_is_kept_in_evidence_but_is_not_an_answer(tmp_path):
    # Even a parseable integer is not a complete final when serving stopped for
    # length. Neither the actor nor trusted broker may submit this prefix.
    first = "r" * 104 + "</think>Final answer: 12"
    assert len(first) == 128
    instance, entry, serving, arm = setup_runtime(tmp_path, [(first, "length")], total=192)
    instance.config["budgets"][entry.benchmark].update(total_model_calls=1, calls_per_turn=1)
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        rows = instance.journal.model_outputs(("synthetic", "A2", "case"))
        assert not final.text
        assert final.terminal_status is CandidateStatus.BUDGET_EXHAUSTED
        assert rows[0].channel_status is ChannelStatus.FINAL_UNFINISHED
        assert rows[0].final_text == "Final answer: 12"
        assert len(serving.requests) == 1
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    finally:
        instance.journal.close()


def test_native_runtime_notice_does_not_replace_the_last_user_problem(tmp_path):
    instance, entry, _, arm = setup_runtime(tmp_path, [("Done</think>42", "stop")])
    try:
        asyncio.run(instance.generate(entry, arm, "synthetic"))
        requests = instance.journal.traces(("synthetic", "A2", "case"), "rendered-request")
        messages = requests[0]["messages"]
        assert "Native thinking uses one continuous token stream" in messages[0]["content"]
        users = [message["content"] for message in messages if message["role"] == "user"]
        assert len(users) == 1
        assert "Synthetic arithmetic." in users[0]
    finally:
        instance.journal.close()


def test_reasoning_is_verbatim_required_context_for_an_interface_repair():
    context = EpisodeContext(({"role": "user", "content": "Original task"},), "Original task")
    draft = "Work already done. " * 200
    context.remember_reasoning(draft, 0)
    context.interface_feedback("Please submit a final answer.", 0)
    packed = context.prompt(maximum_input_tokens=32).pack(
        lambda messages: list(repr(messages).encode()),
        context_length=10000,
        maximum_output_tokens=200,
    )
    assert any(
        draft in message["content"] and message["role"] == "user" for message in packed.messages
    )
    assert context.latest_reasoning_id in packed.visible_event_ids


def test_episode_budget_is_uniform_and_reserve_is_not_extra_computation():
    remaining = 81920
    allowances = []
    while remaining:
        allowance = native_call_allowance(remaining, 32768, 4096)
        allowances.append(allowance)
        remaining -= allowance
    assert allowances == [32768, 32768, 12288, 4096]


def test_interactive_episode_budget_is_not_one_response_context_reservation(tmp_path, monkeypatch):
    from skillev_private.evaluation import integrity_runtime

    from skillev.evaluation.direct_baseline.interactive_tasks import (
        NativeEnvironmentOutcome,
        NativeEnvironmentStep,
        NativePublicState,
    )

    entry = PublicTaskView.from_record("case", "alfworld", {"task": "Observe the marker."})
    instance = runtime(tmp_path, entry, [], calls=20)
    instance.config["budgets"][entry.benchmark].update(
        total_output_tokens=100000,
        native_chunk_tokens=128,
        native_final_reserve_tokens=64,
    )
    instance.source.interactive[entry.task_id] = {"case": {"max_steps": 2}}
    first = "r" * 128
    serving = ChunkServing(
        [
            (first, "length"),
            ("</think>Action: look", "stop"),
            ("Done</think>Action: inventory", "stop"),
        ]
    )
    instance.generators = [serving]
    actions = []

    class Environment:
        async def reset(self):
            return NativePublicState("Your task is to: Observe the marker.", ("look", "inventory"))

        async def step(self, action):
            actions.append(action)
            return NativeEnvironmentStep(
                "Public observation after " + action,
                len(actions) == 2,
                0.0,
                len(actions) == 2,
                True,
                available_actions=("look", "inventory"),
            )

        async def outcome(self):
            return NativeEnvironmentOutcome(1.0, True, True)

        async def close(self):
            pass

    monkeypatch.setattr(
        integrity_runtime, "create_native_environment", lambda *a, **k: Environment()
    )
    arm = InferenceArm("A2", native_thinking=True)
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        assert actions == ["look", "inventory"]
        assert final.intervention_counts["model_calls"] == 3
        assert serving.requests[1].input_ids == serving.requests[0].input_ids + tuple(
            map(ord, first)
        )
        assert "after look" in instance.tokenizer.decode(serving.requests[2].input_ids)
        rows = instance.journal.model_outputs(("synthetic", "A2", "case"))
        assert all(row.budgets["total_output_tokens"] == 100000 for row in rows)
        assert rows[2].continuation_of_call_id is None
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    finally:
        instance.journal.close()
