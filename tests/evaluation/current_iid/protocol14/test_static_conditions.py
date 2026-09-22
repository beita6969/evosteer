from pathlib import Path

import yaml

from skillev.evaluation.current_iid.protocol14.catalog import Protocol14Benchmark
from skillev.evaluation.current_iid.protocol14.config import load_execution_contracts_v4
from skillev.experiments.protocol_v14 import load_protocol_v14

ROOT = Path(__file__).parents[4]


def _executions():
    protocol = load_protocol_v14(
        ROOT / "configs/evaluation/protocol_v14.yaml",
        ROOT / "configs/evaluation/protocol_v14_sources.yaml",
    )
    return load_execution_contracts_v4(
        ROOT / "configs/evaluation/protocol_v14_conditions.yaml", protocol=protocol
    )


def test_static_conditions_bind_source_prompts_parsers_and_denominators() -> None:
    executions = _executions()
    assert executions[Protocol14Benchmark.HOTPOT_QA].prompt_profile == (
        "hotpotqa-raw-ten-passage@2"
    )
    assert executions[Protocol14Benchmark.TRIVIA_QA].scorer_profile == (
        "triviaqa-official-alias-em-f1@2"
    )
    aime = executions[Protocol14Benchmark.AIME_2026]
    assert aime.expected_count == 30
    assert aime.prompt_profile == "integer-cot@1"
    assert aime.parser_profile == "aime-integer-final@3"
    assert aime.thinking_mode.value == "enabled"
    assert aime.decoding.temperature == 1.0
    assert aime.decoding.top_p == 0.95
    assert aime.decoding.top_k == 20
    assert aime.decoding.presence_penalty == 1.5
    assert aime.decoding.max_new_tokens == 81920


def test_aime_candidate_evidence_cannot_self_declare_formal_eligibility() -> None:
    root = yaml.safe_load(
        (ROOT / "configs/evaluation/aime_protocol_candidates.yaml").read_text(encoding="utf-8")
    )
    assert root["final_population"] == "aime-2026-all-30-v13"
    assert len(root["profiles"]) == 1
    profile = root["profiles"][0]
    assert profile["formal_reference_eligible"] is False
    assert profile["evidence"]["status"] == "pre-final-calibration"
