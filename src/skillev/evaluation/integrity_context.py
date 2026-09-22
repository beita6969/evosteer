"""Episode context assembly shared by owner calls and scoped advice requests."""

from __future__ import annotations

from dataclasses import dataclass, field

from .public_context import ArchiveKind, PromptBlock, PublicArchive, PublicPrompt, PublicView


@dataclass(slots=True)
class EpisodeContext:
    root_messages: tuple[dict[str, str], ...]
    root_task: str
    owner_only_system: bool = False
    peer_reference: str = ""
    archive: PublicArchive = field(default_factory=PublicArchive)
    latest_environment_id: str | None = None
    latest_reasoning_id: str | None = None
    _archived_repairs: set[str] = field(default_factory=set, init=False)
    _pending_repair: tuple[str, ...] = field(default=(), init=False)

    def history_page(
        self, kind: ArchiveKind, *, cursor: int = 0, limit: int = 4
    ) -> dict[str, object]:
        """Separate the supplied conversation from this episode's execution log.

        An empty execution log does not mean that the user supplied no context,
        or that a search for external records found nothing. The source dialogue
        stays verbatim in the prompt and is also accessible through this reader.
        """
        archive = self.archive
        if kind is ArchiveKind.CONVERSATION:
            archive = PublicArchive()
            for message in self.root_messages:
                archive.add(kind, 0, message["role"], message["content"])
        page = archive.page(kind, cursor=cursor, limit=limit)
        page["scope"] = (
            "The initial public conversation, already supplied in your prompt."
            if kind is ArchiveKind.CONVERSATION
            else "Execution entries created during this episode only. An empty page says "
            "nothing about external records. The initial public conversation is available "
            "in your prompt and in the conversation archive."
        )
        page["total_entries"] = sum(entry.kind is kind for entry in archive.entries)
        return page

    def remember(self, role: str, content: str, revision: int, kind: ArchiveKind) -> None:
        entry = self.archive.add(kind, revision, role, content)
        if kind is ArchiveKind.ENVIRONMENT:
            self.latest_environment_id = entry.event_id

    def interface_feedback(self, content: str, revision: int) -> None:
        """Replace an unexecuted attempt in active context, not in its archive.

        Repeated rejected drafts are not additional environment observations.
        Keep the latest draft/feedback pair for repair and retain every earlier
        pair verbatim through the existing paged history capability.
        """
        self.resolve_interface_feedback()
        latest = self.archive.entries[-1] if self.archive.entries else None
        draft = (
            (latest.event_id,)
            if latest is not None
            and latest.kind is ArchiveKind.DISCUSSION
            and latest.role == "assistant"
            and latest.revision == revision
            else ()
        )
        # No tool executed: this is a runtime notice to the owner, not a native
        # tool response to a call that never happened.
        feedback = self.archive.add(ArchiveKind.TOOL_RESULT, revision, "user", content)
        self._pending_repair = (*draft, feedback.event_id)

    def remember_reasoning(self, content: str, revision: int) -> None:
        entry = self.archive.add(ArchiveKind.DISCUSSION, revision, "assistant", content)
        self.latest_reasoning_id = entry.event_id

    def resolve_interface_feedback(self) -> None:
        self._archived_repairs.update(self._pending_repair)
        self._pending_repair = ()

    def prompt(self, *, maximum_input_tokens: int | None) -> PublicPrompt:
        prompt = PublicPrompt.required(self.root_messages)
        for entry in self.archive.entries:
            if entry.event_id == self.latest_reasoning_id:
                # A labelled user carrier survives templates that deliberately
                # discard past assistant reasoning. It remains model discussion,
                # never a new environment observation or an evaluator hint.
                prompt = prompt.append(
                    PromptBlock(
                        "user",
                        "Your preceding reasoning, preserved verbatim as model discussion "
                        "(not environment facts):\n" + entry.content,
                        event_id=entry.event_id,
                    )
                )
                continue
            if entry.event_id in self._archived_repairs:
                continue
            if entry.event_id == self.latest_environment_id or (
                self._pending_repair and entry.event_id == self._pending_repair[-1]
            ):
                prompt = prompt.append(
                    PromptBlock(entry.role, entry.content, event_id=entry.event_id)
                )
            else:
                prompt = prompt.remember(entry)
        if self._archived_repairs:
            prompt = prompt.append(
                PromptBlock(
                    "user",
                    f"{len(self._archived_repairs)} earlier unexecuted draft/feedback messages "
                    "are archived and remain available through history. Those drafts caused "
                    "no environment action; they are not additional observations.",
                )
            )
        return PublicPrompt(
            prompt.blocks,
            None if self.latest_reasoning_id else maximum_input_tokens,
            len(self._archived_repairs),
        )

    def peer_prompt(
        self, view: PublicView, *, referenced_ids: tuple[str, ...], maximum_input_tokens: int | None
    ) -> PublicPrompt:
        # Native owner command instructions are intentionally not copied into
        # an advice-only peer. Source dialogue itself remains verbatim.
        messages = tuple(
            item
            for item in self.root_messages
            if not self.owner_only_system or item["role"] != "system"
        )
        prompt = PublicPrompt.required(messages)
        if self.peer_reference:
            prompt = prompt.append(
                PromptBlock(
                    "tool",
                    "Public API reference for the owner; "
                    "this does not grant you tool permissions:\n" + self.peer_reference,
                )
            )
        if view.observation:
            prompt = prompt.append(
                PromptBlock(
                    "tool",
                    f"Current public revision {view.revision}:\n{view.observation}\n"
                    "Current actions available to the owner (not to you):\n"
                    + "\n".join(view.actions),
                )
            )
        for entry in self.archive.resolve(referenced_ids):
            prompt = prompt.append(
                PromptBlock(
                    "tool",
                    f"Referenced public event {entry.event_id}; revision {entry.revision}; "
                    f"kind {entry.kind.value}:\n{entry.content}",
                    event_id=entry.event_id,
                )
            )
        return PublicPrompt(prompt.blocks, maximum_input_tokens)
