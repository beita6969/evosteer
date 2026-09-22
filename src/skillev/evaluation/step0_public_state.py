"""Task, environment history and model messages without a benchmark policy.

This is the model-visible state for current Step-0 evaluation. Historical
subgoal/constraint-ledger helpers are not an authority for this state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum


class ExecutionStatus(StrEnum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    ACKNOWLEDGED = "acknowledged"
    UNKNOWN = "unknown"


class TaskStatus(StrEnum):
    ACTIVE = "active"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class PublicInteraction:
    action: str | None
    observation: str
    revision: int = 0
    execution_status: ExecutionStatus = ExecutionStatus.UNKNOWN
    task_status: TaskStatus = TaskStatus.ACTIVE

    def render(self) -> str:
        execution_detail = (
            "The environment returned this observation for the attempted command; "
            "it did not report a separate action-validity flag.\n"
            if self.execution_status is ExecutionStatus.ACKNOWLEDGED
            else ""
        )
        return (
            f"Public event {self.revision}; attempted action: {self.action}\n"
            f"Execution status: {self.execution_status}; task status: {self.task_status}\n"
            + execution_detail
            + f"Environment observation:\n{self.observation}\n"
        )


@dataclass(frozen=True, slots=True)
class ModelDiscussion:
    author: str
    text: str
    public_revision: int

    def render(self) -> str:
        return f"{self.author} at public revision {self.public_revision}:\n{self.text}\n"


@dataclass(frozen=True, slots=True)
class PublicEpisodeState:
    task: str
    history: tuple[PublicInteraction, ...] = ()
    model_notes: str = ""
    history_window_steps: int | None = 4
    history_maximum_characters: int | None = 24000
    discussions: tuple[ModelDiscussion, ...] = ()

    @property
    def revision(self) -> int:
        return len(self.history)

    def observe(
        self,
        action: str | None,
        observation: str,
        *,
        execution_status: ExecutionStatus = ExecutionStatus.UNKNOWN,
        task_status: TaskStatus = TaskStatus.ACTIVE,
    ) -> PublicEpisodeState:
        event = PublicInteraction(
            action, observation, self.revision + 1, execution_status, task_status
        )
        return replace(self, history=(*self.history, event))

    def remember(self, author: str, text: str) -> PublicEpisodeState:
        """Keep visible model communication, never infer environment facts from it."""
        if not text.strip():
            return self
        return replace(
            self,
            discussions=(*self.discussions, ModelDiscussion(author, text, self.revision)),
        )

    def render(
        self, *, maximum_tokens: int | None = None, count_tokens: Callable[[str], int] | None = None
    ) -> str:
        rows = self.history
        discussions = self.discussions
        if self.history_window_steps is not None:
            rows = rows[-self.history_window_steps :]
        # Bound only the history suffix, not the task or the latest observation.
        # Never cut an object ID or qualifier halfway through an observation.
        if self.history_maximum_characters is not None:
            while len(rows) > 1 and sum(len(row.render()) for row in rows) > (
                self.history_maximum_characters
            ):
                rows = rows[1:]

        def compose() -> str:
            omitted = len(self.history) - len(rows)
            # A discussion at revision N precedes the next environment event.
            # Rendering every observation before every old proposal made an
            # unexecuted plan more recent in the prompt than its real feedback.
            events = [(row.revision, 0, row.render()) for row in rows]
            events.extend((message.public_revision, 1, message.render()) for message in discussions)
            events.sort(key=lambda event: (event[0], event[1]))
            return (
                f"Current task (verbatim):\n{self.task}\n"
                f"Earlier environment turns omitted from this context: {omitted}\n"
                + (
                    f"Public archive revisions 1..{omitted}; retrieve with "
                    f'{{"kind":"history","first":1,"last":{omitted}}}.\n'
                    if omitted
                    else ""
                )
                + f"Model working notes (not environment facts):\n{self.model_notes}\n"
                + (
                    "Model conversation entries below are advice and attempted decisions, "
                    "not environment facts.\n"
                    f"Earlier model messages omitted: {len(self.discussions) - len(discussions)}\n"
                    if self.discussions
                    else ""
                )
                + "".join(event[2] for event in events)
            )

        if maximum_tokens is not None:
            if count_tokens is None:
                raise ValueError("token-bounded history requires the actual tokenizer")
            # Keep whole messages, the complete task and the latest observation.
            # The full discussion remains in state and the private model trace.
            while (len(rows) > 1 or discussions) and count_tokens(compose()) > maximum_tokens:
                if discussions and (
                    len(rows) <= 1 or discussions[0].public_revision <= rows[0].revision
                ):
                    discussions = discussions[1:]
                else:
                    rows = rows[1:]
            if count_tokens(compose()) > maximum_tokens:
                raise ValueError(
                    "full task, current observation and notes exceed the public context budget"
                )
        return compose()
