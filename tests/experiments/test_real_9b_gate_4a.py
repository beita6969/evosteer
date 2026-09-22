from __future__ import annotations

import sys

import pytest

from scripts import real_9b_gate_4a as gate
from skillev.evolution import SplitAuthoringRequest, render_authoring_prompt
from skillev.policy import GenerationResult
from skillev.runtime import model_visible_skill_content
from tests.v3_helpers import CharacterTokenizer


def test_gate_4a_cli_has_one_fixed_seed_and_no_search_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "real_9b_gate_4a.py",
            "--model-path",
            "model",
            "--revision",
            "revision",
            "--eos-token-id",
            "1",
            "--source-package",
            "source.tar.gz",
            "--work-directory",
            "work",
        ],
    )

    args = gate._args()

    assert args.model_path == "model"
    assert gate.GATE_4A_SEED == 20_260_730
    assert gate.GATE_4A_SYNTHETIC_REWARD_LOG_TERM != 0.0
    assert not hasattr(args, "seed")
    assert not hasattr(args, "candidate_count")


@pytest.mark.parametrize(
    ("text", "classification"),
    [
        (
            '{"arguments":{},"kind":"tool","name":"debug.tool","resource_id":"debug.tool","skill_id":null}',
            "tool",
        ),
        (
            '{"arguments":{},"kind":"complete","name":"finish","resource_id":null,"skill_id":null}',
            "complete",
        ),
        ("not-json", "parse_error"),
    ],
)
def test_gate_4a_records_policy_action_class_without_exactness(
    text: str,
    classification: str,
) -> None:
    assert gate.classify_generated_action(text) == classification


def test_gate_4a_exercises_all_four_structured_authoring_requests() -> None:
    requests = gate.authoring_requests()

    assert gate.GATE_4A_FORMAT == "skillev-real-9b-gate-4a@7"
    assert tuple(type(item).__name__ for item in requests) == (
        "RetainAuthoringRequest",
        "RefineAuthoringRequest",
        "SplitAuthoringRequest",
        "GenerateAuthoringRequest",
    )
    source_ids = tuple(
        item.source.manifest.skill_id for item in requests if hasattr(item, "source")
    )
    assert len(source_ids) == len(set(source_ids)) == 3
    for request in requests:
        assert render_authoring_prompt(request, tokenizer=CharacterTokenizer())

    split = requests[2]
    assert isinstance(split, SplitAuthoringRequest)
    assert split.source.applicability.task_families == ("gate-4a-high", "gate-4a-low")
    assert {item.task_family for item in split.edge_exemplars} == {
        split.modality.low_mode.task_family,
        split.modality.high_mode.task_family,
    }

    retain = requests[0]
    visible_retain = model_visible_skill_content(retain.source)
    assert len(visible_retain.split()) >= 700


def test_gate_4a_accepts_many_to_one_sampled_action_span() -> None:
    sampled_ids = (41, 42)
    action_text = "not-json"

    class ManyToOneTokenizer(CharacterTokenizer):
        def encode(self, text: str) -> list[int]:
            if text == action_text:
                return [99]
            return [ord(character) for character in text]

        def decode(self, token_ids: tuple[int, ...]) -> str:
            if token_ids in {sampled_ids, (99,)}:
                return action_text
            return "".join(chr(token_id) for token_id in token_ids)

    generated = GenerationResult(
        content_token_ids=sampled_ids,
        stop_token_ids=(),
        finish_reason="length",
    )

    segment, evidence = gate.admit_action_generation(ManyToOneTokenizer(), generated)

    assert segment.token_ids is generated.content_token_ids
    assert evidence["sampled_span_preserved"] is True
    assert evidence["decoded_text_nonempty"] is True
    assert "token_roundtrip_exact" not in evidence
