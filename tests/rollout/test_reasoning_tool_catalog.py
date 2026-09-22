import json
from dataclasses import replace

import pytest

from skillev.contracts import canonical_json
from skillev.policy.phase_context import TYPED_PUBLIC_HISTORY, PhaseContextSpec, phase_chat_messages
from skillev.scoring.rendering import (
    render_forward_prefix_from_parts,
    render_hindsight_prefix_from_parts,
    render_reasoning_prefix,
)
from tests.rollout.test_native_tool_wire import contract


@pytest.mark.parametrize("typed", [False, True])
def test_planning_sees_exact_action_tools_but_does_not_enable_tool_dispatch(typed):
    tools = list(contract().to_native_tools())
    old = PhaseContextSpec(
        action_wire="native-single-tool-call@2",
        tools_json=canonical_json(tools),
        history_format=TYPED_PUBLIC_HISTORY if typed else None,
        max_turns=8 if typed else None,
    )
    candidate = replace(old, reasoning_tool_catalog=True)
    old_initial = old.wrap("Synthetic public task\n")
    initial = candidate.wrap("Synthetic public task\n")
    restored, _ = PhaseContextSpec.split(initial)
    assert restored == candidate
    assert PhaseContextSpec.split(old_initial)[0] == old
    messages, dispatched_tools = phase_chat_messages(render_reasoning_prefix(initial, (), 1).text)
    assert dispatched_tools is None
    # The names, descriptions and parameter schemas are the actual A catalog,
    # not a second handwritten list which can drift out of sync.
    assert canonical_json(tools) in messages[0]["content"]
    previous_messages, _ = phase_chat_messages(render_reasoning_prefix(old_initial, (), 1).text)
    assert canonical_json(tools) not in previous_messages[0]["content"]
    for renderer, current in (
        (render_forward_prefix_from_parts, "private current reasoning"),
        (render_hindsight_prefix_from_parts, "public execution observation"),
    ):
        old_chat = phase_chat_messages(renderer(old_initial, (), 1, current).text)
        new_chat = phase_chat_messages(renderer(initial, (), 1, current).text)
        assert new_chat == old_chat
        assert new_chat[1] == tools
    backward = phase_chat_messages(
        render_hindsight_prefix_from_parts(initial, (), 1, "public execution observation").text
    )
    assert "private current reasoning" not in json.dumps(backward)


def test_planning_catalog_does_not_silently_enable_a_nonexistent_native_interface():
    with pytest.raises(ValueError):
        PhaseContextSpec(reasoning_tool_catalog=True)
