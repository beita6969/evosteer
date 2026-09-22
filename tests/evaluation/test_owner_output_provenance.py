"""Actual isolated-owner output survives restart; another answer cannot take its identity."""

import asyncio
import json
from dataclasses import replace

import pytest

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.integrity_final_validation import validate_persisted_owner_source
from skillev.evaluation.model_output_provenance import CandidateStatus
from skillev.evaluation.native_channels import ChannelStatus
from skillev.evaluation.sealed_candidates import CandidateReader
from skillev.evaluation.step0_integrity import InferenceArm
from tests.evaluation.test_integrity_broker_boundary import runtime


def entry():
    return PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic arithmetic."})


def test_actual_output_tokens_and_owner_call_survive_a_process_restart(tmp_path):
    arm = InferenceArm("A2")
    instance = runtime(tmp_path, entry(), ["Final answer: 17"])
    final = asyncio.run(instance.generate(entry(), arm, "synthetic"))
    assert final.terminal_status is CandidateStatus.SUBMITTED
    assert final.attempt_id
    assert final.owner_call_id
    instance.journal.close()
    reader = CandidateReader(tmp_path / "candidate.sqlite")
    try:
        source = validate_persisted_owner_source(
            reader, final, benchmark="aime-2026", native_thinking=False, adapter_name=None
        )
        assert source.final_text == "Final answer: 17"
        assert source.result["content_token_ids"] == list(map(ord, source.final_text))
        assert source.call_id == final.owner_call_id
        instance.validate_candidate(reader, entry(), arm, "synthetic")
        assert reader.get("synthetic", "A2", "case") == final
        assert instance.generators[0].outputs == []  # No new serving call during recovery.
    finally:
        reader.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "another-run"},
        {"arm_id": "other-arm"},
        {"episode_id": "other-episode"},
        {"attempt_id": "different-attempt"},
        {"owner_call_id": "another-call"},
        {"policy_id": "different-checkpoint"},
        {"text": ""},
        {"terminal_status": CandidateStatus.INFRASTRUCTURE_UNRESOLVED},
        {"terminal_status": CandidateStatus.LEGACY_UNVERIFIED},
    ],
)
def test_persisted_output_cannot_be_relabelled_or_discarded(tmp_path, changes):
    instance = runtime(tmp_path, entry(), ["Final answer: 17"])
    final = asyncio.run(instance.generate(entry(), InferenceArm("A2"), "synthetic"))
    try:
        with pytest.raises(ValueError):
            validate_persisted_owner_source(
                instance.journal,
                replace(final, **changes),
                benchmark="aime-2026",
                native_thinking=False,
                adapter_name=None,
            )
    finally:
        instance.journal.close()


@pytest.mark.parametrize("intervention", ["empty", "another-call"])
def test_broker_prevents_discarding_or_regenerating_after_valid_owner_final(tmp_path, intervention):
    instance = runtime(tmp_path, entry(), ["Final answer: 17", "Final answer: 18"])
    delegate = instance.sandbox

    class TamperingActor:
        async def run(self, initial, handle, **kwargs):
            request = None

            async def recording(message):
                nonlocal request
                if message["operation"] == "generate":
                    request = message
                return await handle(message)

            value = await delegate.run(initial, recording, **kwargs)
            if intervention == "empty":
                return {**value, "text": "", "submission": None}
            return await handle(request)

    instance.sandbox = TamperingActor()
    try:
        with pytest.raises(ValueError):
            asyncio.run(instance.generate(entry(), InferenceArm("A2"), "synthetic"))
        assert len(instance.journal.model_outputs(("synthetic", "A2", "case"))) == 1
        assert instance.generators[0].outputs == ["Final answer: 18"]
        with pytest.raises(KeyError):
            instance.journal.get("synthetic", "A2", "case")
    finally:
        instance.journal.close()


def test_a_final_cannot_be_regenerated_or_falsely_bound_to_another_served_token_sequence(tmp_path):
    instance = runtime(tmp_path, entry(), ["Final answer: 17"])
    arm = InferenceArm("A2")
    asyncio.run(instance.generate(entry(), arm, "synthetic"))
    with pytest.raises(ValueError):
        asyncio.run(instance.generate(entry(), arm, "synthetic"))
    with instance.journal.connection:
        row = instance.journal.connection.execute("SELECT payload FROM model_outputs").fetchone()
        value = json.loads(row[0])
        value["result"]["content_token_ids"] = list(map(ord, "Final answer: 18"))
        instance.journal.connection.execute(
            "UPDATE model_outputs SET payload=?", (json.dumps(value),)
        )
    with pytest.raises(ValueError):
        instance.validate_candidate(instance.journal, entry(), arm, "synthetic")
    instance.journal.close()


@pytest.mark.parametrize("complete", [False, True])
def test_real_actor_and_broker_use_only_the_completed_native_final_channel(tmp_path, complete):
    output = "<think>Draft: \\boxed{18}"
    if complete:
        output += "</think>Final answer: 17"
    instance = runtime(tmp_path, entry(), [output], calls=1)
    arm = InferenceArm("A2", native_thinking=True)
    final = asyncio.run(instance.generate(entry(), arm, "synthetic"))
    source = instance.journal.model_outputs(("synthetic", "A2", "case"))[0]
    if complete:
        assert final.text == r"\boxed{17}"
        assert source.final_text == final.submission["raw_response"] == "Final answer: 17"
        assert source.channel_status is ChannelStatus.COMPLETE
    else:
        assert final.text == ""
        assert source.final_text == ""
        assert source.channel_status is ChannelStatus.REASONING_UNFINISHED
        assert final.terminal_status is CandidateStatus.BUDGET_EXHAUSTED
    instance.validate_candidate(instance.journal, entry(), arm, "synthetic")
    instance.journal.close()
