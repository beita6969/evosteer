"""Lossless runtime-addressed communication, separate from tool execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class MessageKind(StrEnum):
    REQUEST = "request"
    OPINION = "opinion"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class AgentMessage:
    run_id: str
    arm_id: str
    episode_id: str
    message_id: str
    sender: str
    recipient: str
    kind: MessageKind
    body: str
    state_revision: int
    parent_id: str | None = None
    late: bool = False
    visible_event_ids: tuple[str, ...] = ()


class EpisodeMailbox:
    """An episode mailbox accepts peer advice, never executable environment actions."""

    def __init__(self, run_id: str, arm_id: str, episode_id: str, agents: tuple[str, ...]) -> None:
        self.scope = (run_id, arm_id, episode_id)
        self.agents = agents
        self.messages: dict[str, AgentMessage] = {}

    def deliver(self, message: AgentMessage, *, current_revision: int) -> AgentMessage:
        if (message.run_id, message.arm_id, message.episode_id) != self.scope:
            raise ValueError("message belongs to another evaluation episode")
        if message.sender not in self.agents or message.recipient not in self.agents:
            raise ValueError("message addresses an unknown participant")
        if message.state_revision > current_revision:
            raise ValueError("message references a future environment state")
        if not message.message_id or not isinstance(message.kind, MessageKind):
            raise ValueError("message identity or channel is missing")
        previous = self.messages.get(message.message_id)
        if previous is not None:
            if replace(previous, late=False) != replace(message, late=False):
                raise ValueError("message identity was reused with a different body or route")
            return previous
        if message.parent_id is not None:
            parent = self.messages.get(message.parent_id)
            if parent is None or (parent.sender, parent.recipient) != (
                message.recipient,
                message.sender,
            ):
                raise ValueError("reply is not addressed back to the requesting participant")
        delivered = replace(message, late=message.state_revision < current_revision)
        self.messages[message.message_id] = delivered
        return delivered

    def inbox(self, recipient: str) -> tuple[AgentMessage, ...]:
        return tuple(item for item in self.messages.values() if item.recipient == recipient)
