"""Public-only ALFWorld state and task taxonomy helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .interactive_demo_assets import ALFWorldTaskType


def classify_alfworld_game_id(game_id: str) -> ALFWorldTaskType:
    for task_type in ALFWorldTaskType:
        if game_id.startswith(task_type.value):
            return task_type
    raise ValueError(f"unknown ALFWorld task type: {game_id}")


@dataclass(frozen=True, slots=True)
class ALFWorldPublicState:
    observation: str
    admissible_commands: tuple[str, ...]
    remaining_steps: int
    previous_action: str | None
    inventory_observation: str | None

    def __post_init__(self) -> None:
        if not self.observation.strip() or self.remaining_steps <= 0:
            raise ValueError("ALFWorld public state is incomplete")
        if not self.admissible_commands:
            raise ValueError("ALFWorld public state lacks admissible commands")

    def render(self) -> str:
        fields = [
            f"Observation:\n{self.observation}",
            f"Remaining action budget: {self.remaining_steps}",
        ]
        if self.previous_action is not None:
            fields.append(f"Previous accepted action: {self.previous_action}")
        if self.inventory_observation is not None:
            fields.append(f"Last inventory observation:\n{self.inventory_observation}")
        commands = "\n".join(f"- {item}" for item in self.admissible_commands)
        fields.append(f"Admissible commands:\n{commands}")
        return "\n\n".join(fields)


__all__ = ["ALFWorldPublicState", "classify_alfworld_game_id"]
