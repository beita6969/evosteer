import pytest

from skillev.evaluation.native_channels import ChannelStatus, split_native_channels
from tests.evaluation.test_step0_integrity_runtime import CleanTokenizer


@pytest.mark.parametrize(
    ("text", "status", "final"),
    [
        ("Draft 17", ChannelStatus.REASONING_UNFINISHED, ""),
        ("<think>Draft 17", ChannelStatus.REASONING_UNFINISHED, ""),
        ("Draft 18</think>Final answer: 17", ChannelStatus.COMPLETE, "Final answer: 17"),
        ("<think>Draft</think>Final answer: 17", ChannelStatus.COMPLETE, "Final answer: 17"),
        ("Draft</think>", ChannelStatus.FINAL_EMPTY, ""),
        ("Draft</think>   ", ChannelStatus.FINAL_EMPTY, "   "),
        ("Draft</think>17</think>18", ChannelStatus.MALFORMED, ""),
        ("Draft<think>again</think>17", ChannelStatus.MALFORMED, ""),
        ("Draft</think><think>17", ChannelStatus.MALFORMED, ""),
    ],
)
def test_native_token_boundaries_never_scan_reasoning_for_a_fallback(text, status, final):
    tokenizer = CleanTokenizer()
    value = split_native_channels(tuple(tokenizer.encode(text)), enabled=True, tokenizer=tokenizer)
    assert value.status is status
    assert tokenizer.decode(value.final_token_ids) == final


def test_thinking_off_text_is_not_reinterpreted_as_a_native_thinking_condition():
    tokenizer = CleanTokenizer()
    tokens = tuple(tokenizer.encode("A literal </think> in an ordinary response."))
    assert (
        split_native_channels(tokens, enabled=False, tokenizer=tokenizer).final_token_ids == tokens
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_length_status_keeps_partial_final_evidence_without_changing_thinking_off(enabled):
    tokenizer = CleanTokenizer()
    text = "Draft</think>Final answer: 1" if enabled else "Final answer: 1"
    value = split_native_channels(
        tuple(tokenizer.encode(text)), enabled=enabled, tokenizer=tokenizer, finish_reason="length"
    )
    assert tokenizer.decode(value.final_token_ids) == "Final answer: 1"
    assert value.status is (ChannelStatus.FINAL_UNFINISHED if enabled else ChannelStatus.COMPLETE)
