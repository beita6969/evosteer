"""Fixed-budget channel finalization, never selection of a reasoning draft."""

import asyncio
from dataclasses import replace

import pytest

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.native_channels import ChannelStatus
from skillev.evaluation.native_continuation import thinking_boundary_at_reserve
from skillev.evaluation.step0_integrity import InferenceArm
from skillev.rollout import GenerationPhase, RolloutGenerationRequest, RolloutGenerationResult
from skillev.rollout.evaluation_sglang import EvaluationGenerationProfile
from skillev.runtime import BudgetVector
from tests.evaluation.test_integrity_broker_boundary import runtime
from tests.evaluation.test_step0_integrity_runtime import CleanTokenizer
from tests.rollout.test_evaluation_sglang import _generator, _snapshot


class MarkerTokenizer(CleanTokenizer):
    def encode(self, text):
        return list(
            map(
                ord,
                text.replace("</think>", "\u2600")
                .replace("<think>", "\u2601")
                .replace("\n\n", "\u2602"),
            )
        )

    def decode(self, token_ids):
        return (
            "".join(map(chr, token_ids))
            .replace("\u2600", "</think>")
            .replace("\u2601", "<think>")
            .replace("\u2602", "\n\n")
        )


def test_closure_is_only_for_an_open_channel_at_the_frozen_reserve():
    arguments = {
        "enabled": True,
        "continuing": True,
        "remaining": 64,
        "reserve": 64,
        "channel_status": ChannelStatus.REASONING_UNFINISHED,
    }
    assert thinking_boundary_at_reserve(**arguments) == "\n"
    assert thinking_boundary_at_reserve(**arguments, previous_boundary="\n") == "</think>"
    assert thinking_boundary_at_reserve(**arguments, previous_boundary="</think>") == "\n\n"
    assert thinking_boundary_at_reserve(**arguments, previous_boundary="\n\n") is None
    for change in (
        {"enabled": False},
        {"continuing": False},
        {"remaining": 65},
        {"remaining": 0},
        {"channel_status": ChannelStatus.FINAL_UNFINISHED},
        {"channel_status": ChannelStatus.COMPLETE},
    ):
        assert thinking_boundary_at_reserve(**{**arguments, **change}) is None


def test_serving_uses_actual_delimiter_id_and_counts_it(monkeypatch):
    generator, transport = _generator()
    generator.tokenizer = MarkerTokenizer()
    delimiter = generator.tokenizer.encode("</think>")[0]

    async def request(self, **kwargs):
        transport.calls.append(kwargs["payload"])
        return 200, {
            "output_ids": [delimiter],
            "meta_info": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "finish_reason": {"type": "length"},
            },
        }

    monkeypatch.setattr(type(transport), "request", request)
    profile = EvaluationGenerationProfile(
        "bounded-fixture",
        True,
        0.7,
        0.8,
        20,
        0.0,
        1.5,
        1.0,
        1,
        0,
        thinking_boundary="</think>",
    )
    req = RolloutGenerationRequest(
        GenerationPhase.ACTION, (1, 2, 3), 1, 0, profile.profile_id, _snapshot().snapshot_id
    )
    result = asyncio.run(generator.generate_evaluation(req, profile=profile))
    assert result.usage.output_tokens == 1
    assert generator.tokenizer.decode(result.content_token_ids) == "</think>"
    assert transport.calls[0]["sampling_params"]["logit_bias"] == {str(delimiter): 1_000_000.0}
    with pytest.raises(ValueError):
        replace(profile, max_new_tokens=2)
    with pytest.raises(ValueError):
        replace(profile, enable_thinking=False)


def test_real_actor_preserves_draft_and_submits_only_the_new_owner_final(tmp_path):
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic arithmetic."})
    instance = runtime(tmp_path, entry, [], calls=8)
    tokenizer = instance.tokenizer = MarkerTokenizer()
    instance.config["budgets"][entry.benchmark].update(
        total_output_tokens=192,
        native_chunk_tokens=128,
        native_final_reserve_tokens=64,
        native_close_at_reserve=1,
        calls_per_turn=8,
    )

    class Serving:
        def __init__(self):
            self.requests = []

        async def generate_evaluation(self, request, *, profile, **kwargs):
            self.requests.append(request)
            index = len(self.requests) - 1
            text, reason = [
                ("Draft answer 999. ".ljust(128, "r"), "length"),
                ("\n", "length"),
                ("</think>", "length"),
                ("\n\n", "length"),
                ("Final answer: 42", "stop"),
            ][index]
            assert profile.thinking_boundary == (text if 1 <= index <= 3 else None)
            tokens = tuple(tokenizer.encode(text))
            assert len(tokens) <= request.max_new_tokens
            return RolloutGenerationResult(
                tokens,
                (),
                reason,
                "fixture-policy",
                "fixture",
                BudgetVector(
                    input_tokens=len(request.input_ids), output_tokens=len(tokens), model_calls=1
                ),
            )

    serving = Serving()
    instance.generators = [serving]
    try:
        final = asyncio.run(
            instance.generate(entry, InferenceArm("A2", native_thinking=True), "synthetic")
        )
        assert final.text == r"\boxed{42}"
        assert final.completion_tokens == 131 + len("Final answer: 42")
        assert [r.max_new_tokens for r in serving.requests] == [128, 1, 1, 1, 61]
        assert serving.requests[1].input_ids == serving.requests[0].input_ids + tuple(
            tokenizer.encode("Draft answer 999. ".ljust(128, "r"))
        )
        assert serving.requests[2].input_ids == serving.requests[1].input_ids + tuple(
            tokenizer.encode("\n")
        )
        assert serving.requests[4].input_ids == serving.requests[1].input_ids + tuple(
            tokenizer.encode("\n</think>\n\n")
        )
        instance.validate_candidate(
            instance.journal, entry, InferenceArm("A2", native_thinking=True), "synthetic"
        )
    finally:
        instance.journal.close()
