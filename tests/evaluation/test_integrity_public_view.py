"""Whole-template packing and lossless, episode-local public history."""

from dataclasses import FrozenInstanceError

import pytest

from skillev.evaluation.integrity_context import EpisodeContext
from skillev.evaluation.public_context import (
    ArchiveKind,
    ContextCapacityIncompleteError,
    PromptBlock,
    PublicArchive,
    PublicPrompt,
    PublicView,
    require_current_view,
)


def encode(messages):
    return list(range(len(repr(messages))))


def test_full_template_keeps_required_fields_and_archives_only_whole_messages():
    context = EpisodeContext(({"role": "user", "content": "Unchanged root task"},), "task")
    for index in range(6):
        context.remember(
            "assistant", f"Discussion {index}: " + "x" * 150, index, ArchiveKind.DISCUSSION
        )
    context.remember("tool", "latest observation: Object A-17", 7, ArchiveKind.ENVIRONMENT)
    prompt = context.prompt(maximum_input_tokens=1500).with_instruction("system rules" * 10)
    prompt = prompt.append(PromptBlock("tool", "Complete menu: Object A-17, Object B-88"))
    packed = prompt.pack(encode, context_length=2000, maximum_output_tokens=500)
    rendered = repr(packed.messages)
    assert len(packed.input_ids) + 500 <= 2000
    assert packed.omitted_messages > 0
    assert "Unchanged root task" in rendered
    assert "latest observation: Object A-17" in rendered
    assert "Complete menu: Object A-17, Object B-88" in rendered
    for entry in context.archive.entries:
        assert entry.content in rendered or entry.event_id not in packed.visible_event_ids
    assert len(context.archive.entries) == 7


def test_required_task_or_menu_is_never_shortened_to_make_a_request_fit():
    prompt = PublicPrompt.required(({"role": "user", "content": "Root task " * 100},))
    with pytest.raises(ContextCapacityIncompleteError):
        prompt.with_instruction("tool contract " * 100).pack(
            encode, context_length=500, maximum_output_tokens=100
        )


def test_conversation_history_is_verbatim_paginated_and_separate_from_execution():
    messages = (
        {"role": "system", "content": "Public instructions."},
        {"role": "user", "content": "Earlier context: café, 日常.\nSecond line."},
        {"role": "assistant", "content": "Which part?"},
        {"role": "user", "content": "The situation described earlier."},
    )
    context = EpisodeContext(messages, "Synthetic dialogue")
    context.remember("assistant", "New owner message.", 0, ArchiveKind.DISCUSSION)
    before = context.prompt(maximum_input_tokens=None)
    first = context.history_page(ArchiveKind.CONVERSATION, limit=2)
    second = context.history_page(ArchiveKind.CONVERSATION, cursor=first["next_cursor"], limit=2)
    rows = first["entries"] + second["entries"]
    assert tuple({"role": row["role"], "content": row["content"]} for row in rows) == messages
    assert first["total_entries"] == len(messages)
    assert second["next_cursor"] is None
    assert context.prompt(maximum_input_tokens=None) == before
    assert len(context.archive.entries) == 1
    assert context.history_page(ArchiveKind.DISCUSSION)["entries"][0]["content"] == (
        "New owner message."
    )
    empty = context.history_page(ArchiveKind.ENVIRONMENT)
    assert empty["entries"] == []
    assert empty["total_entries"] == 0
    assert "external records" in empty["scope"]
    assert "conversation" in empty["scope"]


def test_large_archive_does_not_require_one_tokenizer_rpc_per_old_message():
    context = EpisodeContext(({"role": "user", "content": "Root task"},), "Root task")
    for revision in range(128):
        context.remember(
            "assistant", "A public discussion. " * 10, revision, ArchiveKind.DISCUSSION
        )
    calls = []

    def counted(messages):
        calls.append(1)
        return encode(messages)

    packed = context.prompt(maximum_input_tokens=2400).pack(
        counted, context_length=3000, maximum_output_tokens=400
    )
    assert len(packed.input_ids) <= 2400
    assert len(calls) <= 12


def test_oversized_owner_draft_is_archived_but_current_feedback_is_preserved():
    context = EpisodeContext(({"role": "user", "content": "Original task"},), "Original task")
    draft = "Long model discussion. " * 1000
    context.remember("assistant", draft, 0, ArchiveKind.DISCUSSION)
    context.remember("tool", "Please state one explicit final answer.", 0, ArchiveKind.TOOL_RESULT)
    packed = context.prompt(maximum_input_tokens=1000).pack(
        encode, context_length=2000, maximum_output_tokens=1000
    )
    rendered = repr(packed.messages)
    assert "Original task" in rendered
    assert "Please state one explicit final answer." in rendered
    assert draft not in rendered
    assert context.archive.page(ArchiveKind.DISCUSSION)["entries"][0]["content"] == draft


def test_unexecuted_repair_history_does_not_displace_real_environment_observations():
    context = EpisodeContext(({"role": "user", "content": "Find the fictional marker."},), "task")
    context.remember("tool", "Marker was observed in cabinet 2.", 1, ArchiveKind.ENVIRONMENT)
    context.remember("tool", "Current location is desk 1.", 2, ArchiveKind.ENVIRONMENT)
    drafts = [f"Unexecuted draft {index}: " + "model discussion " * 100 for index in range(20)]
    for index, draft in enumerate(drafts):
        context.remember("assistant", draft, 2, ArchiveKind.DISCUSSION)
        context.interface_feedback(f"Attempt {index} was not executed.", 2)
    prompt = context.prompt(maximum_input_tokens=4000).with_instruction("Public tools.")
    packed = prompt.pack(encode, context_length=5000, maximum_output_tokens=1000)
    bodies = [row["content"] for row in packed.messages]
    assert "Marker was observed in cabinet 2." in bodies
    assert "Current location is desk 1." in bodies
    assert drafts[-1] in bodies
    assert all(draft not in bodies for draft in drafts[:-1])
    assert any(row["role"] == "user" and "Attempt 19" in row["content"] for row in packed.messages)
    assert packed.omitted_messages >= 38
    assert len(packed.input_ids) <= 4000
    assert [
        entry.content for entry in context.archive.entries if entry.kind is ArchiveKind.DISCUSSION
    ] == drafts
    first_page = context.archive.page(ArchiveKind.DISCUSSION, limit=8)
    assert first_page["entries"][0]["content"] == drafts[0]


def test_resolved_interface_feedback_leaves_accepted_owner_action_and_native_result_visible():
    context = EpisodeContext(({"role": "user", "content": "Synthetic task."},), "task")
    context.remember("tool", "Original native observation.", 1, ArchiveKind.ENVIRONMENT)
    context.remember("assistant", "A rejected draft.", 1, ArchiveKind.DISCUSSION)
    context.interface_feedback("No action was executed.", 1)
    context.remember("assistant", "Action: look", 1, ArchiveKind.DISCUSSION)
    context.resolve_interface_feedback()
    context.remember("tool", "Native look result.", 2, ArchiveKind.ENVIRONMENT)
    packed = context.prompt(maximum_input_tokens=None).pack(
        encode, context_length=None, maximum_output_tokens=1000
    )
    bodies = [row["content"] for row in packed.messages]
    assert "A rejected draft." not in bodies
    assert "No action was executed." not in bodies
    assert "Action: look" in bodies
    assert "Original native observation." in bodies
    assert "Native look result." in bodies
    assert (
        context.archive.page(ArchiveKind.TOOL_RESULT)["entries"][0]["content"]
        == "No action was executed."
    )


def test_runtime_feedback_survives_when_its_unexecuted_draft_exceeds_context_capacity():
    context = EpisodeContext(({"role": "user", "content": "Original task."},), "task")
    context.remember("tool", "Current observation.", 1, ArchiveKind.ENVIRONMENT)
    draft = "Unexecuted long draft. " * 1000
    context.remember("assistant", draft, 1, ArchiveKind.DISCUSSION)
    context.interface_feedback("The draft contained no executable call.", 1)
    packed = context.prompt(maximum_input_tokens=1000).pack(
        encode, context_length=2000, maximum_output_tokens=1000
    )
    assert any(
        row["role"] == "user" and "no executable call" in row["content"] for row in packed.messages
    )
    assert "Current observation." in [row["content"] for row in packed.messages]
    assert draft not in [row["content"] for row in packed.messages]
    assert context.archive.page(ArchiveKind.DISCUSSION)["entries"][0]["content"] == draft


@pytest.mark.parametrize("kind", list(ArchiveKind))
def test_archive_pages_keep_text_and_expose_next_cursor(kind):
    archive = PublicArchive()
    bodies = ["中文 \\boxed{17}\n    code('literal')\n", "next\n", "last\n"]
    for revision, body in enumerate(bodies):
        archive.add(kind, revision, "tool", body)
    first = archive.page(kind, limit=2)
    assert [entry["content"] for entry in first["entries"]] == bodies[:2]
    second = archive.page(kind, cursor=first["next_cursor"], limit=2)
    assert [entry["content"] for entry in second["entries"]] == bodies[2:]
    assert second["next_cursor"] is None
    with pytest.raises(ValueError):
        archive.resolve(("environment:999",))


def test_view_is_immutable_and_old_decisions_stay_old():
    before = PublicView(1, "task", "before", ("look",))
    after = PublicView(2, "task", "after", ("inventory",))
    with pytest.raises(FrozenInstanceError):
        before.revision = 2
    with pytest.raises(ValueError):
        require_current_view(before, after)


def test_peer_receives_current_public_state_and_only_requested_discussion():
    context = EpisodeContext(
        (
            {"role": "system", "content": "OWNER TOOL PERMISSIONS"},
            {"role": "user", "content": "TASK"},
        ),
        "TASK",
        owner_only_system=True,
    )
    context.remember("assistant", "Unrequested owner scratchpad", 1, ArchiveKind.DISCUSSION)
    requested = context.archive.add(ArchiveKind.TOOL_RESULT, 2, "tool", "Verbatim page\n")
    view = PublicView(2, "TASK", "Current observation", ("click[Exact Target]",))
    packed = context.peer_prompt(
        view, referenced_ids=(requested.event_id,), maximum_input_tokens=2000
    ).pack(encode, context_length=3000, maximum_output_tokens=200)
    rendered = repr(packed.messages)
    assert "TASK" in rendered
    assert "Verbatim page" in rendered
    assert "Current observation" in rendered
    assert "click[Exact Target]" in rendered
    assert "OWNER TOOL PERMISSIONS" not in rendered
    assert "Unrequested owner scratchpad" not in rendered
