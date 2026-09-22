"""Build source-proven prompt assets by replaying pinned train environments."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from skillev.evaluation.current_iid.protocol13.interactive_demo_assets import (
    ALFWorldDemonstration,
    ALFWorldDemoStep,
    ALFWorldTaskType,
    DemoSourceKind,
    WebShopDemonstration,
    WebShopDemoStep,
)
from skillev.evaluation.direct_baseline.config import DirectBenchmark
from skillev.evaluation.direct_baseline.interactive_tasks import (
    action_matches_public_surface,
)
from skillev_private.benchmarks.alfworld_official import (
    OfficialALFWorldTask,
    OfficialALFWorldTextEnv,
)
from skillev_private.benchmarks.webshop_official import (
    OfficialWebShopGoal,
    OfficialWebShopTextEnv,
)


def replay_webshop_demo(
    *,
    environment: OfficialWebShopTextEnv,
    goal: OfficialWebShopGoal,
    actions: tuple[str, ...],
    source_revision: str,
) -> WebShopDemonstration:
    observation = environment.reset(goal.session_id)
    if environment.instruction_text is None:
        raise ValueError("WebShop replay lacks a public train instruction")
    available = environment.get_available_actions()
    steps: list[WebShopDemoStep] = []
    for action in actions:
        if not action_matches_public_surface(DirectBenchmark.WEB_SHOP, action, available):
            raise ValueError(f"demo action is unavailable: {goal.goal_id}: {action}")
        result = environment.step(action)
        next_available = environment.get_available_actions()
        steps.append(
            WebShopDemoStep(
                observation,
                available,
                action,
                result.observation_text,
                next_available,
                Decimal(str(result.reward)),
                result.terminal,
            )
        )
        observation = result.observation_text
        available = next_available
        if result.terminal:
            break
    return WebShopDemonstration(
        example_id=f"train/{goal.goal_id}",
        source_kind=DemoSourceKind.OFFICIAL_TRAIN_REPLAY,
        source_revision=source_revision,
        source_split="train",
        goal_id=goal.goal_id,
        session_id=goal.session_id,
        task=environment.instruction_text,
        steps=tuple(steps),
    )


def replay_alfworld_demo(
    *,
    environment: OfficialALFWorldTextEnv,
    task: OfficialALFWorldTask,
    task_type: ALFWorldTaskType,
    actions: tuple[str, ...],
    source_revision: str,
) -> ALFWorldDemonstration:
    state = environment.reset(task.seed)
    observation = state.observation_text
    admissible = state.admissible_commands
    steps: list[ALFWorldDemoStep] = []
    for action in actions:
        if action not in admissible:
            raise ValueError(f"demo action is inadmissible: {task.game_id}: {action}")
        result = environment.step(action)
        steps.append(
            ALFWorldDemoStep(
                observation,
                admissible,
                action,
                result.observation_text,
                result.admissible_commands,
                result.terminal,
                result.success if result.terminal else None,
            )
        )
        observation = result.observation_text
        admissible = result.admissible_commands
        if result.terminal:
            break
    return ALFWorldDemonstration(
        example_id=f"train/{task.game_id}",
        source_revision=source_revision,
        source_split="train",
        game_id=task.game_id,
        task_type=task_type,
        seed=task.seed,
        max_steps=task.max_steps,
        task=state.instruction_text,
        steps=tuple(steps),
    )


def write_private_prompt_asset(path: Path, value: dict[str, object]) -> None:
    """Create one private asset once; do not silently replace replay evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("private prompt asset already exists")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = [
    "replay_alfworld_demo",
    "replay_webshop_demo",
    "write_private_prompt_asset",
]
