"""Content-based public-input diagnostics; no hashes, private labels or new model calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .input_metric_contracts import PublicTaskView


@dataclass(frozen=True, slots=True)
class PublicInputReceipt:
    task_id: str
    input_profile: str
    source_paragraph_count: int | None
    paragraph_ids: tuple[str, ...]
    exported_source_complete: bool | None
    required_input_received: bool
    paragraph_order_received: bool | None
    actual_input_tokens: int
    omitted_history_messages: int


def describe_public_input(
    entry: PublicTaskView,
    decoded_input: str,
    *,
    input_tokens: int,
    omitted_history_messages: int,
    source_paragraphs: Sequence[Mapping[str, object]] | None = None,
) -> PublicInputReceipt:
    # Training bridges and role-structured conversations have their own renderer.
    # Their exact packed-message/token equivalence is verified by the transport.
    fields = dict(entry.fields)
    required = [value for _, value in entry.fields]
    if entry.input_profile == "training-public-source-bridge@1" or entry.benchmark == "healthbench":
        required = (
            [entry.render()]
            if entry.benchmark != "healthbench"
            else [message["content"] for message in entry.conversation()]
        )
    exported_complete = ordered = None
    ids: tuple[str, ...] = ()
    if source_paragraphs is not None:
        blocks = [f"[Paragraph {p['id']} | {p['title']}]\n{p['text']}" for p in source_paragraphs]
        ids = tuple(str(p["id"]) for p in source_paragraphs)
        exported_complete = fields.get("context") == "\n\n".join(blocks)
        cursor, ordered = 0, True
        for block in blocks:
            position = decoded_input.find(block, cursor)
            if position < 0:
                ordered = False
                break
            cursor = position + len(block)
    return PublicInputReceipt(
        entry.task_id,
        entry.input_profile,
        len(source_paragraphs) if source_paragraphs is not None else None,
        ids,
        exported_complete,
        # Qwen's chat template trims message-boundary whitespace. A trailing
        # source space is not a lost option; interior text must still be exact.
        all(value.strip() in decoded_input for value in required),
        ordered,
        input_tokens,
        omitted_history_messages,
    )
