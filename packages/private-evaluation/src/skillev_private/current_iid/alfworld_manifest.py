"""Single-source ALFWorld final-panel split identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ALFWorldSplit(StrEnum):
    VALID_SEEN = "valid-seen"
    VALID_UNSEEN = "valid-unseen"


@dataclass(frozen=True, slots=True)
class ALFWorldManifestEntry:
    task_id: str
    game_id: str
    split: ALFWorldSplit
    task_type: str
    seed: int
    max_steps: int
    source_revision: str


@dataclass(frozen=True, slots=True)
class ALFWorldPanelManifest:
    entries: tuple[ALFWorldManifestEntry, ...]
    expected_seen: int
    expected_unseen: int

    def validate(self) -> None:
        if len(self.entries) != 128:
            raise ValueError("ALFWorld final panel requires 128 games")
        seen = sum(item.split is ALFWorldSplit.VALID_SEEN for item in self.entries)
        unseen = len(self.entries) - seen
        if (seen, unseen) != (self.expected_seen, self.expected_unseen):
            raise ValueError("ALFWorld manifest split counts differ")
        if (seen, unseen) != (97, 31):
            raise ValueError("Protocol 12 freezes the observed 97/31 panel identity")
        if len({item.game_id for item in self.entries}) != 128:
            raise ValueError("ALFWorld final game IDs must be unique")


__all__ = ["ALFWorldManifestEntry", "ALFWorldPanelManifest", "ALFWorldSplit"]
