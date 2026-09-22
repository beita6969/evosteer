import asyncio
from pathlib import Path

import pytest
import yaml

from skillev.evaluation.input_metric_contracts import IID_BENCHMARKS
from skillev.evaluation.model_output_provenance import CandidateStatus
from skillev.evaluation.native_continuation import native_call_allowance
from skillev.evaluation.step0_integrity import InferenceArm
from skillev.evaluation.thinking_policy import ThinkingPolicy
from tests.evaluation.test_integrity_broker_boundary import runtime
from tests.evaluation.test_owner_output_provenance import entry


def policy():
    return ThinkingPolicy.from_value(
        yaml.safe_load(
            (
                Path(__file__).resolve().parents[2]
                / "configs/evaluation/step0_thinking_by_benchmark.yaml"
            ).read_text()
        )
    )


def test_current_thinking_default_keeps_explicit_aime_and_health_exceptions():
    condition = policy()
    assert {name for name, _ in condition.rows} == set(IID_BENCHMARKS)
    assert {name for name, enabled in condition.rows if enabled} == {"aime-2026", "healthbench"}
    assert condition.resolve("aime-2026", InferenceArm("A2", native_thinking=False)).native_thinking
    values = condition.to_value()
    values["thinking_by_benchmark"]["aime-2026"] = False
    explicit = ThinkingPolicy.from_value(values)
    assert not explicit.resolve("aime-2026", InferenceArm("explicit")).native_thinking


def test_historical_thinking_map_is_read_without_silently_dropping_its_domain():
    values = policy().to_value()
    values["thinking_by_benchmark"]["webshop"] = False
    values["thinking_by_benchmark"]["humaneval"] = True
    historical = ThinkingPolicy.from_value(values)
    assert historical.to_value() == values
    assert not historical.resolve("webshop", InferenceArm("historical")).native_thinking


def test_historical_seven_domain_launcher_preserves_its_thinking_and_continuation():
    settings = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "configs/evaluation/step0_seven_iid32_integrity_repair.yaml"
        ).read_text()
    )
    condition = ThinkingPolicy.from_value(settings["thinking_policy"])
    assert condition != policy()
    for benchmark in IID_BENCHMARKS:
        assert settings["evaluation_sample_counts"][benchmark] == (
            30 if benchmark == "aime-2026" else 32
        )
        assert condition.resolve(benchmark, InferenceArm("A2")).native_thinking
        limits = settings["budget_overrides"][benchmark]
        chunk = limits["native_chunk_tokens"]
        reserve = limits["native_final_reserve_tokens"]
        assert 0 < reserve < chunk
        assert native_call_allowance(chunk + reserve, chunk, reserve) == chunk
        assert native_call_allowance(reserve, chunk, reserve) == reserve
        decoding = settings["decoding_overrides"][benchmark]
        assert decoding["enable_thinking"]
        assert decoding["top_p"] == 0.95
        coding = benchmark in {"mbpp-plus", "humaneval"}
        assert decoding["temperature"] == (0.6 if coding else 1.0)
        assert decoding["presence_penalty"] == (0.0 if coding else 1.5)


@pytest.mark.parametrize("missing", ["mbpp-plus", "hotpotqa"])
def test_current_thinking_map_still_requires_each_active_domain(missing):
    values = policy().to_value()
    del values["thinking_by_benchmark"][missing]
    with pytest.raises(ValueError):
        ThinkingPolicy.from_value(values)


def test_resolved_thinking_reaches_owner_template_broker_and_stored_final(tmp_path):
    outputs = [
        "Check. </think>Message to solver:\nPlease help.",
        "Check. </think>Final answer: 17",
    ]
    instance = runtime(tmp_path, entry(), outputs)
    instance.thinking_policy = ThinkingPolicy(
        "explicit-thinking-on", tuple((name, True) for name in IID_BENCHMARKS)
    )
    root_arm = InferenceArm("A2", native_thinking=False)
    final = asyncio.run(instance.generate(entry(), root_arm, "synthetic"))
    assert final.text == r"\boxed{17}"
    assert final.terminal_status is CandidateStatus.SUBMITTED
    records = instance.journal.model_outputs(("synthetic", "A2", "case"))
    assert [record.participant for record in records] == ["owner", "owner"]
    assert all(record.native_thinking for record in records)
    assert all(enabled for _, enabled in instance.tokenizer.messages)
    assert final.intervention_counts["model_calls"] == 2
    assert final.intervention_counts["peer_model_calls"] == 0
    instance.validate_candidate(instance.journal, entry(), root_arm, "synthetic")
    instance.thinking_policy = None
    with pytest.raises(ValueError):
        instance.validate_candidate(instance.journal, entry(), root_arm, "synthetic")
    instance.journal.close()
