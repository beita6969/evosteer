"""ScienceWorld adapter reusing the existing official isolated JVM worker.

The owner submits arbitrary native commands. We do not expose gold paths or
an oracle-filtered action/object list; invalid commands reach the simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from skillev.evaluation.direct_baseline.interactive_tasks import (
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativePublicState,
)
from skillev.evaluation.environment_transition import EnvironmentTransitionReceipt
from skillev.evaluation.scienceworld_commands import (
    COMMAND_PROFILE,
    LEGACY_COMMAND_PROFILE,
    TYPED_COMMAND_PROFILES,
    native_functions,
)
from skillev.evaluation.scienceworld_observation import (
    CLOCK_OBSERVATION_PROFILE,
    public_containment_view,
    public_simulator_clock,
)
from skillev.evaluation.scienceworld_references import ScienceWorldReferences
from skillev_private.benchmarks.official_process import (
    OfficialScienceWorldProcessFactory,
    PinnedOfficialProcess,
)
from skillev_private.benchmarks.scienceworld_official import OfficialScienceWorldTask
from skillev_private.direct_reference.environments import DirectScienceWorldEnvironment


@dataclass(frozen=True, slots=True)
class OODScienceWorldOutcome(NativeEnvironmentOutcome):
    """Private terminal evidence; never part of a model-visible observation."""

    native_final_score: float | None = None
    native_steps: int = 0
    termination_reason: str = "no-native-step"
    simulator_moves: int | None = None
    observation_profile: str = "text-only@1"
    command_profile: str = LEGACY_COMMAND_PROFILE


@dataclass(slots=True)
class OODScienceWorldEnvironment(DirectScienceWorldEnvironment):
    observation_profile: str = "text-only@1"
    last_transition: EnvironmentTransitionReceipt | None = None
    command_profile: str = LEGACY_COMMAND_PROFILE
    references: ScienceWorldReferences = field(default_factory=ScienceWorldReferences)

    def public_observation(self, text: str) -> str:
        state_profiles = {"public-state@1", "public-state@2", CLOCK_OBSERVATION_PROFILE}
        if self.observation_profile not in {"text-only@1", *state_profiles}:
            raise ValueError("unknown ScienceWorld observation profile")
        parts = [text]
        if self.observation_profile in state_profiles:
            names = ("current_look", "inventory", "task_description")
            if any(name not in self._public_state for name in names):
                raise RuntimeError("the selected public-state profile lacks native observations")
            # Verbatim multiline rendering keeps native containment indentation.
            parts.extend(f"{name}:\n{self._public_state[name]}" for name in names)
            if self.observation_profile == "public-state@2":
                view = public_containment_view(
                    self._public_state["current_look"], self._public_state["inventory"]
                )
                if view:
                    parts.append(view)
            if self.observation_profile == CLOCK_OBSERVATION_PROFILE:
                parts.append(public_simulator_clock(self._native_moves, self.max_steps))
        if self.command_profile in {COMMAND_PROFILE, *TYPED_COMMAND_PROFILES}:
            parts.append(self.references.render())
        elif self.command_profile != LEGACY_COMMAND_PROFILE:
            raise ValueError("unknown ScienceWorld command profile")
        return "\n\n".join(parts)

    async def reset(self) -> NativePublicState:
        state = await super(OODScienceWorldEnvironment, self).reset()
        self.references = ScienceWorldReferences()
        return NativePublicState(
            "Task: "
            + self.environment.task_description
            + "\n\n"
            + self.public_observation(state.observation_text),
            native_functions(self.command_profile),
        )

    async def step(self, action: str) -> NativeEnvironmentStep:
        previous = self._raw_score
        previous_moves = self._native_moves
        result = await super(OODScienceWorldEnvironment, self).step(action)
        self.references.observe(action, result.observation, terminal=result.terminal)
        # The official parser explicitly reports this rejection in its public
        # observation. Preserve it as rejection, not an ambiguous execution ACK.
        # Other observations do not establish validity; no oracle action list
        # is queried and the original command, observation and reward survive.
        valid = result.action_valid
        if result.observation.strip() == "No known action matches that input." or (
            result.observation.startswith("Unknown action.  Type 'help' for a list of actions")
        ):
            valid = False
        self.last_transition = EnvironmentTransitionReceipt(
            action,
            result.observation,
            valid,
            previous,
            self._raw_score,
            result.terminal,
            self._native_moves,
            self.observation_profile,
            previous_simulator_moves=previous_moves,
            native_diagnostics=self._private_diagnostics,
        )
        return replace(
            result,
            observation=self.public_observation(result.observation),
            action_valid=valid,
            available_actions=native_functions(self.command_profile),
        )

    async def outcome(self) -> OODScienceWorldOutcome:
        outcome = await super(OODScienceWorldEnvironment, self).outcome()
        reason = (
            "no-native-step"
            if self._raw_score is None
            else "native-negative-score"
            if self._raw_score < 0
            else "score-100"
            if self._raw_score == 100
            else "simulator-horizon"
            if self._terminal
            and self._native_moves is not None
            and self._native_moves > self.max_steps
            else "runner-horizon"
            if self._steps >= self.max_steps
            else "environment-terminal-unspecified"
            if self._terminal
            else "owner-stopped-or-budget-exhausted"
        )
        # A budget stop is not necessarily the configured environment horizon.
        return OODScienceWorldOutcome(
            **{
                **asdict(outcome),
                "terminated_by_horizon": reason in {"runner-horizon", "simulator-horizon"},
            },
            native_final_score=self._raw_score,
            native_steps=self._steps,
            termination_reason=reason,
            simulator_moves=self._native_moves,
            observation_profile=self.observation_profile,
            command_profile=self.command_profile,
        )


def create_scienceworld_environment(
    specification: dict[str, Any],
    *,
    seed: int,
    stderr_path: Path | None,
    maximum_steps: int | None,
    observation_profile: str = "text-only@1",
    command_profile: str = LEGACY_COMMAND_PROFILE,
) -> OODScienceWorldEnvironment:
    case, manifest = specification["case"], specification["manifest"]
    deployment = manifest["deployments"][case["deployment"]]
    raw = manifest["runtimes"][deployment["runtime"]]
    runtime = PinnedOfficialProcess(
        Path(raw["interpreter_path"]),
        Path(raw["source_root"]),
        raw["source_revision"],
        float(raw.get("request_timeout_seconds", 180)),
        worker_stderr_path=stderr_path,
    )
    factory = OfficialScienceWorldProcessFactory(
        runtime,
        Path(deployment["jar_path"]),
        deployment["simplification"],
        seed,
        private_diagnostics=command_profile in {COMMAND_PROFILE, *TYPED_COMMAND_PROFILES},
    )
    payload = case["payload"]
    steps = int(case["max_steps"]) if maximum_steps is None else maximum_steps
    task = OfficialScienceWorldTask(
        case["task_id"],
        payload["environment_id"],
        payload["task_name"],
        int(payload["variation_index"]),
        seed,
        steps,
        payload,
    )
    return OODScienceWorldEnvironment(
        factory.create(task),
        task.task_name,
        task.variation_index,
        seed,
        steps,
        observation_profile=observation_profile,
        command_profile=command_profile,
    )
