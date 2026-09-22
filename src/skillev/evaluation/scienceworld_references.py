"""Presentation-only state from the simulator's public ambiguity message.

Never resolves objects, consults an object tree, or rewrites/blocks a command.
Original multiline observations (including containment indentation) survive.
"""

import re
from dataclasses import dataclass

from .scienceworld_commands import REFERENCE_PROFILE

_CHOICE = re.compile(r"^(\d+):\s*(.+)$", re.MULTILINE)
_PREFIX = "Ambiguous request: Please enter the number for the action you intended"


@dataclass(frozen=True, slots=True)
class PendingScienceWorldChoice:
    command: str
    public_revision: int
    options: tuple[tuple[str, str], ...]


@dataclass(slots=True)
class ScienceWorldReferences:
    revision: int = 1
    pending: PendingScienceWorldChoice | None = None

    def observe(self, command: str, observation: str, *, terminal: bool) -> None:
        self.revision += 1
        if not terminal and observation.startswith(_PREFIX):
            options = tuple(_CHOICE.findall(observation))
            self.pending = (
                PendingScienceWorldChoice(command, self.revision, options) if options else None
            )
        else:
            # Native resolveAmbiguity consumes/cancels its pending selection even
            # for invalid input. A later number must not retain an old association.
            self.pending = None

    def render(self) -> str:
        heading = f"Public reference protocol ({REFERENCE_PROFILE}), revision {self.revision}."
        if self.pending is None:
            return heading + " No pending numbered choice."
        return (
            heading
            + f"\nPending command: {self.pending.command}\nChoices for this revision only:\n"
            + "\n".join(f"{number}: {text}" for number, text in self.pending.options)
            + "\nUse act(command) with the chosen number as its entire command argument. "
            "The parenthetical locations describe the options, not executable commands. "
            "This choice is consumed by the next command; the simulator, not the framework, "
            "resolves it."
        )
