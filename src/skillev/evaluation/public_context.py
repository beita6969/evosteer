"""Lossless public archives and whole-template context packing.

Only complete historical messages may leave the prompt. The source task,
current observation, current action surface and current interface feedback
remain intact; archived text is accessible through a paged public capability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from enum import StrEnum


class ContextCapacityIncompleteError(RuntimeError):
    """The declared context cannot hold the required public input and output."""


class ArchiveKind(StrEnum):
    CONVERSATION = "conversation"
    ENVIRONMENT = "environment"
    DISCUSSION = "discussion"
    TOOL_RESULT = "tool_result"


@dataclass(frozen=True, slots=True)
class PublicView:
    revision: int
    root_task: str
    observation: str
    actions: tuple[str, ...]
    visible_event_ids: tuple[str, ...] = ()


def require_current_view(decided_on: PublicView, current: PublicView) -> None:
    if decided_on.revision != current.revision:
        raise ValueError("decision was made on an older public state")


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    event_id: str
    kind: ArchiveKind
    revision: int
    role: str
    content: str


class PublicArchive:
    """Episode-local public text, never an evaluator or filesystem search API."""

    def __init__(self) -> None:
        self.entries: list[ArchiveEntry] = []

    def add(self, kind: ArchiveKind, revision: int, role: str, content: str) -> ArchiveEntry:
        entry = ArchiveEntry(f"{kind.value}:{len(self.entries) + 1}", kind, revision, role, content)
        self.entries.append(entry)
        return entry

    def resolve(self, event_ids: tuple[str, ...]) -> tuple[ArchiveEntry, ...]:
        by_id = {entry.event_id: entry for entry in self.entries}
        if any(event_id not in by_id for event_id in event_ids):
            raise ValueError("referenced public event is unavailable in this episode")
        return tuple(by_id[event_id] for event_id in event_ids)

    def page(self, kind: ArchiveKind, *, cursor: int = 0, limit: int = 4) -> dict[str, object]:
        if type(cursor) is not int or cursor < 0 or type(limit) is not int or not 1 <= limit <= 8:
            raise ValueError("archive cursor or page size is invalid")
        entries = [entry for entry in self.entries if entry.kind is kind]
        if cursor > len(entries):
            raise ValueError("archive cursor is outside this public archive")
        page = entries[cursor : cursor + limit]
        next_cursor = cursor + len(page)
        return {
            "kind": kind.value,
            "entries": [asdict(entry) for entry in page],
            "next_cursor": next_cursor if next_cursor < len(entries) else None,
        }


@dataclass(frozen=True, slots=True)
class PromptBlock:
    role: str
    content: str
    required: bool = True
    event_id: str | None = None
    source_revision: int | None = None
    event_kind: str | None = None
    skill_id: str | None = None
    runtime_metadata: bool = False

    def message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    def messages(self) -> tuple[dict[str, str], ...]:
        if self.source_revision is None:
            return (self.message(),)
        description = (
            f"Public archive {self.event_id}; source revision {self.source_revision}; "
            f"kind {self.event_kind}. "
            + (
                "Model discussion, not environment facts. "
                if self.event_kind == "discussion"
                else ""
            )
            + "The following message body is verbatim."
        )
        return ({"role": "tool", "content": description}, self.message())


@dataclass(frozen=True, slots=True)
class PackedPrompt:
    messages: tuple[dict[str, str], ...]
    input_ids: tuple[int, ...]
    visible_event_ids: tuple[str, ...]
    omitted_messages: int
    skill_blocks: tuple[PromptBlock, ...] = ()


@dataclass(frozen=True, slots=True)
class PublicPrompt:
    blocks: tuple[PromptBlock, ...]
    maximum_input_tokens: int | None = None
    archived_messages: int = 0

    @classmethod
    def required(cls, messages: tuple[dict[str, str], ...]) -> PublicPrompt:
        return cls(tuple(PromptBlock(item["role"], item["content"]) for item in messages))

    def with_instruction(self, instruction: str) -> PublicPrompt:
        blocks = self.blocks
        if blocks and blocks[0].role == "system":
            instruction = blocks[0].content + "\n\n" + instruction
            blocks = blocks[1:]
        return replace(self, blocks=(PromptBlock("system", instruction), *blocks))

    def append(self, block: PromptBlock) -> PublicPrompt:
        return replace(self, blocks=(*self.blocks, block))

    def with_runtime_metadata(self, instruction: str) -> PublicPrompt:
        """Update an existing runtime surface without invalidating its history prefix.

        Interactive prompts already contain a current capability/budget surface.
        Do not invent a tool response or user turn for counters. Conversational
        tasks without that surface retain their opening-system controls.
        """
        for index, block in enumerate(self.blocks):
            if block.runtime_metadata:
                updated = replace(block, content=block.content + "\n\n" + instruction)
                return replace(
                    self, blocks=(*self.blocks[:index], updated, *self.blocks[index + 1 :])
                )
        return self.with_instruction(instruction)

    def remember(self, entry: ArchiveEntry) -> PublicPrompt:
        return self.append(
            PromptBlock(
                entry.role,
                entry.content,
                required=False,
                event_id=entry.event_id,
                source_revision=entry.revision,
                event_kind=entry.kind.value,
            )
        )

    def pack(
        self,
        encode: Callable[[tuple[dict[str, str], ...]], list[int]],
        *,
        context_length: int | None,
        maximum_output_tokens: int,
    ) -> PackedPrompt:
        cap = self.maximum_input_tokens
        if context_length is not None:
            available = context_length - maximum_output_tokens
            cap = available if cap is None else min(cap, available)
        blocks = self.blocks
        recent = [
            i
            for i, block in enumerate(blocks)
            if not block.required and block.event_id is not None and block.role == "tool"
        ]
        # Current feedback or a newly delivered peer reply must remain visible.
        # A previous owner draft is archival discussion, not required input: a
        # long draft must not make an otherwise valid repair prompt impossible.
        protected = set(recent[-1:])
        removable = [
            i for i, block in enumerate(blocks) if not block.required and i not in protected
        ]
        packed_by_drop: dict[int, PackedPrompt] = {}

        def render(drop: int) -> PackedPrompt:
            if drop not in packed_by_drop:
                omitted = set(removable[:drop])
                selected = [block for i, block in enumerate(blocks) if i not in omitted]
                messages = tuple(message for block in selected for message in block.messages())
                packed_by_drop[drop] = PackedPrompt(
                    messages,
                    tuple(encode(messages)),
                    tuple(block.event_id for block in selected if block.event_id is not None),
                    self.archived_messages + drop,
                    tuple(block for block in selected if block.skill_id is not None),
                )
            return packed_by_drop[drop]

        complete = render(0)
        if cap is None or len(complete.input_ids) <= cap:
            return complete
        required = render(len(removable))
        if len(required.input_ids) > cap:
            raise ContextCapacityIncompleteError(
                "context-capacity-incomplete: required public input and output exceed the budget"
            )
        # Find a fitting suffix in logarithmic template encodes, not one full
        # tokenizer RPC per removed event. Every returned prompt is measured;
        # no text-length estimate substitutes for the actual chat template.
        low, high = 1, len(removable)
        while low < high:
            middle = (low + high) // 2
            if len(render(middle).input_ids) <= cap:
                high = middle
            else:
                low = middle + 1
        return render(high)
