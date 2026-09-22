"""Source-proven, train-only demonstrations for native agent benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from skillev.evaluation.direct_baseline.config import DirectBenchmark
from skillev.evaluation.direct_baseline.interactive_tasks import (
    action_matches_public_surface,
)


class DemoSourceKind(StrEnum):
    OFFICIAL_TRAIN_REPLAY = "official-train-replay"
    PUBLIC_DOCUMENTATION = "public-documentation"


class ALFWorldTaskType(StrEnum):
    PICK_AND_PLACE = "pick_and_place"
    PICK_TWO_OBJECTS = "pick_two_obj"
    CLEAN_THEN_PLACE = "pick_clean_then_place"
    HEAT_THEN_PLACE = "pick_heat_then_place"
    COOL_THEN_PLACE = "pick_cool_then_place"
    LOOK_AT_OBJECT = "look_at_obj"


@dataclass(frozen=True, slots=True)
class WebShopDemoStep:
    observation_before: str
    available_actions_before: tuple[str, ...]
    action: str
    observation_after: str
    available_actions_after: tuple[str, ...]
    reward_after: Decimal
    terminal_after: bool

    def __post_init__(self) -> None:
        if not self.observation_before.strip() or not self.observation_after.strip():
            raise ValueError("WebShop demonstration observation is empty")
        if not self.available_actions_before:
            raise ValueError("WebShop demonstration lacks an action surface")
        if len(self.available_actions_before) != len(set(self.available_actions_before)):
            raise ValueError("WebShop demonstration actions are not unique")
        if not action_matches_public_surface(
            DirectBenchmark.WEB_SHOP,
            self.action,
            self.available_actions_before,
        ):
            raise ValueError("WebShop demonstration action was not available")
        if not Decimal(0) <= self.reward_after <= Decimal(1):
            raise ValueError("WebShop demonstration reward is outside [0,1]")


@dataclass(frozen=True, slots=True)
class WebShopDemonstration:
    example_id: str
    source_kind: DemoSourceKind
    source_revision: str
    source_split: str
    goal_id: str
    session_id: str
    task: str
    steps: tuple[WebShopDemoStep, ...]

    def __post_init__(self) -> None:
        identity = (
            self.example_id,
            self.source_revision,
            self.source_split,
            self.goal_id,
            self.session_id,
            self.task,
        )
        if any(not item.strip() for item in identity):
            raise ValueError("WebShop demonstration identity is incomplete")
        if self.source_split != "train" or not self.steps:
            raise ValueError("WebShop formal demonstrations must be non-empty train replays")


@dataclass(frozen=True, slots=True)
class ALFWorldDemoStep:
    observation_before: str
    admissible_commands_before: tuple[str, ...]
    action: str
    observation_after: str
    admissible_commands_after: tuple[str, ...]
    terminal_after: bool
    won_after: bool | None

    def __post_init__(self) -> None:
        if self.action not in self.admissible_commands_before:
            raise ValueError("ALFWorld demonstration action was not admissible")
        if not self.terminal_after and self.won_after is not None:
            raise ValueError("non-terminal ALFWorld demonstration exposes won")


@dataclass(frozen=True, slots=True)
class ALFWorldDemonstration:
    example_id: str
    source_revision: str
    source_split: str
    game_id: str
    task_type: ALFWorldTaskType
    seed: int
    max_steps: int
    task: str
    steps: tuple[ALFWorldDemoStep, ...]

    def __post_init__(self) -> None:
        if self.source_split != "train" or not self.steps:
            raise ValueError("ALFWorld formal demonstrations must be non-empty train replays")
        if self.max_steps <= 0 or self.seed < 0:
            raise ValueError("ALFWorld demonstration seed/horizon is invalid")


def require_complete_alfworld_demo_coverage(
    examples: tuple[ALFWorldDemonstration, ...],
) -> None:
    observed = {item.task_type for item in examples}
    required = set(ALFWorldTaskType)
    if observed != required:
        raise ValueError(f"ALFWorld demonstration coverage differs: {required - observed}")


__all__ = [
    "ALFWorldDemoStep",
    "ALFWorldDemonstration",
    "ALFWorldTaskType",
    "DemoSourceKind",
    "WebShopDemoStep",
    "WebShopDemonstration",
    "require_complete_alfworld_demo_coverage",
]
