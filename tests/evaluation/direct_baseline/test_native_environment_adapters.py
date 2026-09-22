from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from skillev_private.benchmarks.alfworld_official import (
    OfficialALFWorldResetResult,
    OfficialALFWorldStepResult,
)
from skillev_private.benchmarks.scienceworld_official import OfficialScienceWorldStepResult
from skillev_private.benchmarks.webshop_official import OfficialWebShopStepResult
from skillev_private.direct_reference.environments import (
    DirectALFWorldEnvironment,
    DirectScienceWorldEnvironment,
    DirectWebShopEnvironment,
)


@dataclass
class _WebShop:
    instruction_text: str = "buy item"
    actions: tuple[str, ...] = ("search", "click[item]")
    stepped: list[str] = field(default_factory=list)
    closed: bool = False

    def reset(self, session_id: str) -> str:
        return f"session {session_id}"

    def step(self, action: str) -> OfficialWebShopStepResult:
        self.stepped.append(action)
        return OfficialWebShopStepResult("found", 0.4, False)

    def get_available_actions(self) -> tuple[str, ...]:
        return self.actions

    async def close(self) -> None:
        self.closed = True


def test_webshop_delegates_invalid_semantics_to_official_environment() -> None:
    official = _WebShop()
    environment = DirectWebShopEnvironment(official, "s1", "buy item")  # type: ignore[arg-type]
    asyncio.run(environment.reset())
    result = asyncio.run(environment.step("click[other]"))
    assert result.action_valid is None
    assert result.terminal is False
    assert official.stepped == ["click[other]"]
    asyncio.run(environment.close())
    assert official.closed


@dataclass
class _ALFWorld:
    closed: bool = False

    def reset(self, seed: int) -> OfficialALFWorldResetResult:
        assert seed == 42
        return OfficialALFWorldResetResult("room", "put apple away", ("go north",))

    def step(self, action: str) -> OfficialALFWorldStepResult:
        assert action == "go north"
        return OfficialALFWorldStepResult("done", (), True, True)

    async def close(self) -> None:
        self.closed = True


def test_alfworld_uses_exact_admissible_command_and_terminal_success() -> None:
    official = _ALFWorld()
    environment = DirectALFWorldEnvironment(official, 42, "put apple away")  # type: ignore[arg-type]
    asyncio.run(environment.reset())
    result = asyncio.run(environment.step("go north"))
    assert result.action_valid is None
    assert result.success is True
    assert result.reward == 1


@dataclass
class _ScienceWorld:
    closed: bool = False

    def reset(self, *, task_name: str, variation_index: int, seed: int, max_steps: int) -> str:
        assert (task_name, variation_index, seed, max_steps) == ("boil", 2, 42, 100)
        return "lab"

    def step(self, action: str) -> OfficialScienceWorldStepResult:
        assert action == "look around"
        return OfficialScienceWorldStepResult("lab", 45, False)

    async def close(self) -> None:
        self.closed = True


def test_scienceworld_normalizes_official_score_and_preserves_partial_outcome() -> None:
    official = _ScienceWorld()
    environment = DirectScienceWorldEnvironment(official, "boil", 2, 42, 100)  # type: ignore[arg-type]
    asyncio.run(environment.reset())
    result = asyncio.run(environment.step("look around"))
    outcome = asyncio.run(environment.outcome())
    assert result.reward == 0.45
    assert outcome.reward == 0.45
    assert outcome.terminal_reached is False
    assert outcome.terminated_by_horizon is True


def test_scienceworld_clips_negative_native_score_before_aggregation() -> None:
    result = OfficialScienceWorldStepResult("failed", -10, True)
    assert result.score == 0
