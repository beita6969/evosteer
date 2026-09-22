"""Optional CPU integration with the privately deployed Qwen tokenizer assets."""

import json
import os

import pytest

from skillev.contracts import canonical_json
from skillev.policy.phase_context import PhaseContextSpec, phase_chat_messages
from skillev.rollout.native_wire import NativeToolWire
from skillev.scoring.rendering import (
    render_forward_prefix_from_parts,
    render_hindsight_prefix_from_parts,
)
from tests.rollout.test_native_tool_wire import call, contract


@pytest.mark.skipif(
    not os.environ.get("SKILLEV_PRIVATE_TOKENIZER_DIR"),
    reason="private deployed tokenizer not supplied",
)
@pytest.mark.parametrize(
    "wire_version",
    ["native-single-tool-call@1", "native-single-tool-call@2", "native-single-tool-call@3"],
)
@pytest.mark.parametrize("reasoning_tool_catalog", [False, True])
def test_real_qwen_template_native_xml_and_complete_token_spans(
    wire_version, reasoning_tool_catalog
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["SKILLEV_PRIVATE_TOKENIZER_DIR"], local_files_only=True, trust_remote_code=False
    )
    action_contract = contract()
    wire = NativeToolWire(action_contract, format_version=wire_version)
    source = (
        "def long_function(value):\n"
        + '    # 中文 "quoted" \\ escaped\n' * 100
        + "    return value\n"
    )
    action = call("execute", code=source)
    if not wire_version.endswith("@1"):
        action = "I will execute the module.\n" + action
    original_ids = tuple(tokenizer.encode(action, add_special_tokens=False))
    decoded = tokenizer.decode(
        original_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    assert wire.parse(decoded).action.arguments == {"code": source}
    spec = PhaseContextSpec(
        action_wire=wire.format_version,
        tools_json=canonical_json(list(action_contract.to_native_tools())),
        reasoning_tool_catalog=reasoning_tool_catalog,
    )
    initial = spec.wrap("Synthetic public coding task\n")
    assert phase_chat_messages(initial + "Reasoning:\n")[1] is None
    reasoning_messages, reasoning_tools = phase_chat_messages(initial + "Reasoning:\n")
    displayed_reasoning = tokenizer.apply_chat_template(
        reasoning_messages,
        tools=reasoning_tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    assert (spec.tools_json in displayed_reasoning) is reasoning_tool_catalog
    assert displayed_reasoning.endswith("<think>\n")

    def encode(text):
        messages, tools = phase_chat_messages(text)
        return tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )

    forward = render_forward_prefix_from_parts(initial, (), 1, "reason one")
    changed = render_forward_prefix_from_parts(initial, (), 1, "reason two")
    assert encode(forward.text) != encode(changed.text)
    backward = render_hindsight_prefix_from_parts(initial, (), 1, "public observation")
    assert "reason one" not in tokenizer.decode(encode(backward.text))
    messages, tools = phase_chat_messages(forward.text)
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert "<function=example_function_name>" in prompt
    assert "submit_answer" in prompt
    assert "raw-full-vocabulary" not in prompt
    assert json.loads(spec.tools_json)


@pytest.mark.skipif(
    not os.environ.get("SKILLEV_PRIVATE_TOKENIZER_DIR"), reason="private tokenizer unavailable"
)
def test_real_qwen_template_preserves_seven_domain_public_meanings():
    from transformers import AutoTokenizer

    from skillev.benchmarks.protocol_v10_action import COMPLETION_WIRE_INSTRUCTION
    from tests.rollout.test_public_action_semantics import PUBLIC_MEANINGS, public_native_context

    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["SKILLEV_PRIVATE_TOKENIZER_DIR"], local_files_only=True, trust_remote_code=False
    )
    for domain, meaning in PUBLIC_MEANINGS:
        context, _ = public_native_context(domain)
        for phase in ("Reasoning:\n", "Action:\n"):
            messages, tools = phase_chat_messages(context.text + phase)
            ids = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            displayed = tokenizer.decode(ids, skip_special_tokens=False)
            assert meaning in displayed
            assert COMPLETION_WIRE_INSTRUCTION not in displayed
            assert "SKILLEV_PHASE_CONTEXT_V1" not in displayed


@pytest.mark.skipif(
    not os.environ.get("SKILLEV_PRIVATE_TOKENIZER_DIR"), reason="private tokenizer unavailable"
)
def test_real_qwen_template_preserves_shared_training_and_evaluation_semantics():
    from transformers import AutoTokenizer

    from skillev.task_semantic_guidance import (
        TRAINING_PUBLIC_INPUT,
        public_task_semantics,
    )
    from tests.evaluation.test_shared_task_semantics import shared_context
    from tests.rollout.test_public_action_semantics import PUBLIC_MEANINGS

    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["SKILLEV_PRIVATE_TOKENIZER_DIR"], local_files_only=True, trust_remote_code=False
    )
    for domain, _ in PUBLIC_MEANINGS:
        context, _, _ = shared_context(domain)
        meaning = public_task_semantics(
            domain.value, input_profile=TRAINING_PUBLIC_INPUT, hotpot_deliberation=True
        )
        for phase in ("Reasoning:\n", "Action:\n"):
            messages, tools = phase_chat_messages(context.text + phase)
            ids = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            assert meaning in tokenizer.decode(ids, skip_special_tokens=False)


@pytest.mark.skipif(
    not os.environ.get("SKILLEV_PRIVATE_TOKENIZER_DIR"), reason="private tokenizer unavailable"
)
@pytest.mark.parametrize("thinking", [False, True])
def test_real_template_keeps_draft_separate_from_final_environment_state(thinking):
    from transformers import AutoTokenizer

    from tests.rollout.test_typed_alfworld_history import DRAFT, typed_context

    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["SKILLEV_PRIVATE_TOKENIZER_DIR"], local_files_only=True, trust_remote_code=False
    )
    forward = render_forward_prefix_from_parts(typed_context().text, (), 1, DRAFT)
    messages, tools = phase_chat_messages(forward.text)
    ids = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=True, enable_thinking=thinking
    )
    displayed = tokenizer.decode(ids, skip_special_tokens=False)
    assert "Maybe open the box." in displayed
    assert "Imagined success" in displayed
    assert "Box is closed." in displayed.rsplit("<|im_start|>user", 1)[1]
    assert "Imagined success" not in displayed.rsplit("<|im_start|>user", 1)[1]
