"""Stable ALFWorld six-task classification from official game identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ALFWorldTaskType(StrEnum):
    PICK_AND_PLACE = "pick_and_place"
    PICK_TWO_OBJECTS = "pick_two_obj"
    CLEAN_THEN_PLACE = "pick_clean_then_place"
    HEAT_THEN_PLACE = "pick_heat_then_place"
    COOL_THEN_PLACE = "pick_cool_then_place"
    LOOK_AT_OBJECT = "look_at_obj"


PREFIX_TO_TASK_TYPE = tuple((item.value, item) for item in ALFWorldTaskType)


def classify_alfworld_game_id(game_id: str) -> ALFWorldTaskType:
    identity = game_id.strip("/")
    split, separator, remainder = identity.partition("/")
    if separator and split in {"train", "valid_seen", "valid_unseen"}:
        identity = remainder
    for prefix, task_type in PREFIX_TO_TASK_TYPE:
        if identity.startswith(prefix):
            return task_type
    raise ValueError(f"unknown ALFWorld task type: {game_id}")


@dataclass(frozen=True, slots=True)
class ALFWorldManifestCase:
    task_id: str
    game_id: str
    split: str
    task_type: ALFWorldTaskType
    seed: int
    max_steps: int
    environment_id: str
    instruction: str
    deployment: str

    def __post_init__(self) -> None:
        if self.split not in {"valid_seen", "valid_unseen"}:
            raise ValueError("ALFWorld final split is invalid")
        if classify_alfworld_game_id(self.game_id) is not self.task_type:
            raise ValueError("ALFWorld task type differs from game ID")
        if self.seed < 0 or self.max_steps <= 0:
            raise ValueError("ALFWorld seed/horizon is invalid")


def validate_protocol13_alfworld_panel(cases: tuple[ALFWorldManifestCase, ...]) -> None:
    if len(cases) != 128 or len({case.game_id for case in cases}) != 128:
        raise ValueError("ALFWorld Protocol 13 panel must contain 128 unique games")
    counts = (
        sum(case.split == "valid_seen" for case in cases),
        sum(case.split == "valid_unseen" for case in cases),
    )
    if counts != (97, 31):
        raise ValueError(f"ALFWorld panel split differs: seen={counts[0]}, unseen={counts[1]}")
    missing = set(ALFWorldTaskType).difference(case.task_type for case in cases)
    if missing:
        raise ValueError(f"ALFWorld final panel lacks task types: {sorted(missing)}")


__all__ = [
    "ALFWorldManifestCase",
    "ALFWorldTaskType",
    "classify_alfworld_game_id",
    "validate_protocol13_alfworld_panel",
]
