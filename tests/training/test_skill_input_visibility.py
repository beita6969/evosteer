"""Body returned is not body visible; exact admitted token ranges decide."""

import json
from dataclasses import replace

import pytest

from skillev.contracts import canonical_json
from skillev.diagnostics.skill_visibility import CatalogVisibility
from skillev.policy.interface import EncodedPolicyPrompt, ModelInputWindow, encode_policy_prompt
from skillev.training.evidence_context import TrajectoryEvidenceContext
from tests.rollout.engine_fakes import ByteTokenizer
from tests.training.test_skill_discovery_chain import catalog_record


def returned_step():
    artifact = catalog_record()
    response = {
        "status": "skill-read",
        "skill_id": "skill-alpha",
        "version": "1",
        "library_version": "library-v1",
        "content": 'Public method. 中文\nQuoted "step" and \\ path.',
    }
    return replace(artifact.record.steps[0], observation_text=canonical_json(response))


def evidence(prompt, step, tokenizer=None):
    return CatalogVisibility(()).evidence(
        prompt,
        tokenizer or ByteTokenizer(),
        previous_steps=(step,),
        library_version="library-v1",
        step_index=2,
        phase="action",
    )


@pytest.mark.parametrize("typed", [False, True])
def test_complete_returned_body_not_id_mention_and_not_cross_window_join(typed):
    tokenizer, step = ByteTokenizer(), returned_step()
    text = canonical_json(step.observation_text) if typed else step.observation_text
    ids = tuple(tokenizer.encode(text))
    observed = evidence(EncodedPolicyPrompt(ids, len(ids)), step)
    assert observed["visible_skill_body_refs"][0]["read_step_index"] == 1
    cut = len(ids) // 2
    # Joined text deceptively equals the whole body, but the original input had
    # removed tokens between these ranges: it cannot establish a whole envelope.
    interrupted = EncodedPolicyPrompt(ids, len(ids) + 100, (cut, len(ids) - cut))
    assert evidence(interrupted, step)["visible_skill_body_refs"] == []
    assert len(evidence(interrupted, step)["not_fully_visible_skill_body_refs"]) == 1
    mention = tuple(tokenizer.encode("skill-alpha Public method"))
    assert (
        evidence(EncodedPolicyPrompt(mention, len(mention)), step)["visible_skill_body_refs"] == []
    )
    assert (
        evidence(EncodedPolicyPrompt(ids, len(ids) + 100), step)["visible_skill_body_refs"] is None
    )
    assert "Public method" not in json.dumps(observed)


def test_actual_window_returns_original_ids_and_body_visibility_changes_without_rewriting():
    tokenizer, step = ByteTokenizer(), returned_step()
    initial = "H0 public catalog\n"
    text = initial + step.observation_text + "\ncurrent reasoning " * 100
    full = encode_policy_prompt(tokenizer, text, initial_text=initial, window=None)
    window = ModelInputWindow(120)
    clipped = encode_policy_prompt(tokenizer, text, initial_text=initial, window=window)
    assert clipped.ids == window.apply(list(full.ids), tokenizer.encode_rollout_prompt(initial))
    assert sum(clipped.retained_segment_lengths) == len(clipped.ids)
    assert evidence(full, step)["visible_skill_body_refs"]
    assert evidence(clipped, step)["visible_skill_body_refs"] == []


def test_old_receipt_without_input_sidecar_stays_unknown_and_repeat_reads_have_distinct_refs():
    item = catalog_record()
    step = returned_step()
    item = replace(item, record=replace(item.record, steps=(step,)))
    assert TrajectoryEvidenceContext.from_artifact(item).invocation_links[0].body_returned is True
    assert TrajectoryEvidenceContext.from_artifact(item).body_visible_skill_ids is None
    tokenizer = ByteTokenizer()
    repeated = replace(step, index=2)
    ids = tuple(tokenizer.encode(step.observation_text + "\n" + repeated.observation_text))
    result = CatalogVisibility(()).evidence(
        EncodedPolicyPrompt(ids, len(ids)),
        tokenizer,
        previous_steps=(step, repeated),
        library_version="library-v1",
        step_index=3,
        phase="action",
    )
    assert [r["read_step_index"] for r in result["visible_skill_body_refs"]] == [1, 2]


@pytest.mark.parametrize("fault", ["library", "no-credit", "error"])
def test_unadmitted_or_stale_body_does_not_become_visible_evidence(fault):
    step = returned_step()
    if fault == "library":
        value = json.loads(step.observation_text)
        value["library_version"] = "obsolete-library"
        step = replace(step, observation_text=canonical_json(value))
    elif fault == "no-credit":
        step = replace(step, invoked_skill_ids=())
    else:
        step = replace(step, observation_status="schema_invalid")
    ids = tuple(ByteTokenizer().encode(step.observation_text))
    assert evidence(EncodedPolicyPrompt(ids, len(ids)), step)["visible_skill_body_refs"] == []


def test_real_qwen_typed_history_preserves_body_evidence_and_window_removes_it():
    import os

    if not os.environ.get("SKILLEV_PRIVATE_TOKENIZER_DIR"):
        pytest.skip("private deployed tokenizer not supplied")
    from transformers import AutoTokenizer

    from skillev.policy.interface import TYPED_PUBLIC_HISTORY, PhaseContextSpec
    from skillev.policy.phase_context import phase_chat_messages
    from skillev.scoring import render_forward_prefix_from_parts
    from tests.rollout.test_native_tool_wire import contract

    actual = AutoTokenizer.from_pretrained(
        os.environ["SKILLEV_PRIVATE_TOKENIZER_DIR"], local_files_only=True, trust_remote_code=False
    )

    class Tokenizer:
        def encode(self, text):
            return actual.encode(text, add_special_tokens=False)

        def decode(self, ids):
            return actual.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        def encode_rollout_prompt(self, text):
            messages, tools = phase_chat_messages(text)
            return actual.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )

    tokenizer, step = Tokenizer(), returned_step()
    initial = PhaseContextSpec(
        action_wire="native-single-tool-call@3",
        tools_json=canonical_json(list(contract().to_native_tools())),
        action_contract_json=canonical_json(contract().to_scoring_metadata()),
        history_format=TYPED_PUBLIC_HISTORY,
    ).wrap("Synthetic public task.\n")
    prefix = render_forward_prefix_from_parts(initial, (step,), 2, "Choose a public completion.")
    full = encode_policy_prompt(tokenizer, prefix.text, initial_text=initial, window=None)
    assert evidence(full, step, tokenizer)["visible_skill_body_refs"]
    long = render_forward_prefix_from_parts(initial, (step,), 2, "Public reasoning filler. " * 1000)
    clipped = encode_policy_prompt(
        tokenizer, long.text, initial_text=initial, window=ModelInputWindow(128)
    )
    assert evidence(clipped, step, tokenizer)["visible_skill_body_refs"] == []
