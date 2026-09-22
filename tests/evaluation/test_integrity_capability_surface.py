"""Capability visibility is an API contract, not an expert strategy."""

from skillev.evaluation.capability_registry import (
    AdviceQuery,
    Capability,
    CapabilityEffect,
    CapabilityKind,
    CapabilityRegistry,
    runtime_capabilities,
)
from skillev.evaluation.integrity_context import EpisodeContext
from skillev.evaluation.native_tool_instructions import (
    native_tool_instructions,
    native_tool_reference,
)
from skillev.evaluation.public_context import PublicView


def test_native_api_reference_is_shared_with_owner_and_advice_only_peer():
    reference = native_tool_reference("alfworld")
    for native_functions in (False, True):
        assert reference in native_tool_instructions("alfworld", native_functions=native_functions)
    context = EpisodeContext(
        ({"role": "user", "content": "Synthetic public task."},),
        "Synthetic public task.",
        peer_reference=reference,
    )
    prompt = context.peer_prompt(
        PublicView(1, context.root_task, "Synthetic room.", ("look",)),
        referenced_ids=(),
        maximum_input_tokens=None,
    )
    public_api = [block for block in prompt.blocks if reference in block.content]
    assert len(public_api) == 1
    assert public_api[0].required
    assert public_api[0].role == "tool"


def test_released_rc_lane_does_not_advertise_a_nonexistent_search_service():
    registry = runtime_capabilities("completion", peers=("solver",))
    ids = {entry.capability_id for entry in registry.visible("owner")}
    assert "history" in ids
    assert "solver" in ids
    assert "search" not in ids
    assert registry.visible("solver") == ()


def test_tool_and_peer_effects_are_not_confused():
    registry = runtime_capabilities("webshop", peers=("researcher",))
    entries = {entry.capability_id: entry for entry in registry.entries}
    assert entries["click"].effect is CapabilityEffect.ENVIRONMENT_WRITE
    assert entries["history"].effect is CapabilityEffect.READ_ONLY
    assert entries["researcher"].effect is CapabilityEffect.ADVICE


def test_advice_abstains_on_common_words_and_uses_only_the_actual_query():
    registry = CapabilityRegistry(
        (
            Capability(
                "synthetic",
                CapabilityKind.TEXT_SKILL,
                "calculate numbers",
                "text",
                "synthetic",
                "advice",
            ),
        )
    )
    assert not registry.text_advice(AdviceQuery("the task is complete").text(), retrieve=True)
    assert registry.text_advice(AdviceQuery("calculate this").text(), retrieve=True)
    assert (
        AdviceQuery("framework must not be substituted", "actual request").text()
        == "actual request"
    )
