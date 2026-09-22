"""Explicit clean and historical Step-0 entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skillev.rollout.evaluation_sglang import EvaluationRolloutGenerator
from skillev.runtime import SkillLibraryState

from .capability_registry import CapabilityRegistry
from .direct_baseline import DirectGenerationRequest, DirectGenerationResult
from .direct_baseline.interactive_tasks import (
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativeInteractiveTask,
    NativePublicState,
)
from .integrity_controller import IntegrityStepZeroClient
from .sealed_candidates import CandidateJournal
from .skill_library_config import FrozenSkillLibrary
from .step0_interactive_agent import ArchitectureEpisodeArtifact
from .step0_receipts import ArchitectureIdentityReceipt
from .step0_types import StepZeroArchitectureConfig, StepZeroDiagnostics, StepZeroTaskBinding

if TYPE_CHECKING:
    from .legacy_step0_architecture import StepZeroArchitectureDirectClient as LegacyClient


class StepZeroArchitectureDirectClient:
    """Run a clean, explicitly selected arm through the integrity controller."""

    def __init__(
        self,
        *,
        generator: EvaluationRolloutGenerator,
        library_state: SkillLibraryState,
        bindings: tuple[StepZeroTaskBinding, ...],
        config: StepZeroArchitectureConfig,
        skill_library: FrozenSkillLibrary | None = None,
        journal: CandidateJournal | None = None,
        run_id: str = "unpublished-development",
    ) -> None:
        if config.arm.legacy:
            raise ValueError(
                "legacy arms require LegacyStepZeroArchitectureDirectClient explicitly"
            )
        self._implementation: IntegrityStepZeroClient = IntegrityStepZeroClient(
            generator=generator,
            library_state=library_state,
            bindings=bindings,
            config=config,
            skill_library=skill_library,
            journal=journal,
            run_id=run_id,
        )

    @property
    def _config(self) -> StepZeroArchitectureConfig:
        return self._implementation._config

    @_config.setter
    def _config(self, value: StepZeroArchitectureConfig) -> None:
        if value.arm != self._implementation._config.arm:
            raise ValueError("an active controller cannot change evaluation arms")
        self._implementation._config = value

    @property
    def architecture_identity(self) -> ArchitectureIdentityReceipt:
        return self._implementation.architecture_identity

    @property
    def diagnostics(self) -> StepZeroDiagnostics:
        return self._implementation.diagnostics

    def register_binding(self, binding: StepZeroTaskBinding) -> None:
        self._implementation.register_binding(binding)

    async def generate(self, request: DirectGenerationRequest) -> DirectGenerationResult:
        return await self._implementation.generate(request)

    async def begin_interactive_episode(
        self, task: NativeInteractiveTask, state: NativePublicState
    ) -> None:
        await self._implementation.begin_interactive_episode(task, state)

    async def observe_interactive_episode(
        self, task_id: str, action: str | None, result: NativeEnvironmentStep | NativePublicState
    ) -> None:
        await self._implementation.observe_interactive_episode(task_id, action, result)

    def current_interactive_memory(self, task_id: str) -> str:
        return self._implementation.current_interactive_memory(task_id)

    async def close_public_episode(self, task_id: str, *, reason: str) -> None:
        await self._implementation.close_public_episode(task_id, reason=reason)

    async def finish_interactive_episode(
        self, task_id: str, outcome: NativeEnvironmentOutcome
    ) -> ArchitectureEpisodeArtifact:
        return await self._implementation.finish_interactive_episode(task_id, outcome)


class LegacyStepZeroArchitectureDirectClient:
    """Run a historical arm only when a caller names legacy reproduction."""

    def __init__(
        self,
        *,
        generator: EvaluationRolloutGenerator,
        library_state: SkillLibraryState,
        bindings: tuple[StepZeroTaskBinding, ...],
        config: StepZeroArchitectureConfig,
        capabilities: CapabilityRegistry | None = None,
        journal: CandidateJournal | None = None,
        run_id: str = "unpublished-development",
    ) -> None:
        if config.arm.legacy:
            from .legacy_step0_architecture import StepZeroArchitectureDirectClient as LegacyClient

            if capabilities is not None or journal is not None:
                raise ValueError(
                    "new integrity capabilities cannot be attached to a historical arm"
                )
            self._implementation: LegacyClient = LegacyClient(
                generator=generator, library_state=library_state, bindings=bindings, config=config
            )
            return
        del run_id
        raise ValueError("legacy reproduction requires a legacy arm")

    @property
    def _config(self) -> StepZeroArchitectureConfig:
        return self._implementation._config

    @_config.setter
    def _config(self, value: StepZeroArchitectureConfig) -> None:
        if value.arm != self._implementation._config.arm:
            raise ValueError("an active controller cannot change evaluation arms")
        self._implementation._config = value

    @property
    def architecture_identity(self) -> ArchitectureIdentityReceipt:
        return self._implementation.architecture_identity

    @property
    def diagnostics(self) -> StepZeroDiagnostics:
        return self._implementation.diagnostics

    def register_binding(self, binding: StepZeroTaskBinding) -> None:
        self._implementation.register_binding(binding)

    async def generate(self, request: DirectGenerationRequest) -> DirectGenerationResult:
        return await self._implementation.generate(request)

    async def begin_interactive_episode(
        self, task: NativeInteractiveTask, state: NativePublicState
    ) -> None:
        await self._implementation.begin_interactive_episode(task, state)

    async def observe_interactive_episode(
        self, task_id: str, action: str | None, result: NativeEnvironmentStep | NativePublicState
    ) -> None:
        await self._implementation.observe_interactive_episode(task_id, action, result)

    def current_interactive_memory(self, task_id: str) -> str:
        return self._implementation.current_interactive_memory(task_id)

    async def finish_interactive_episode(
        self, task_id: str, outcome: NativeEnvironmentOutcome
    ) -> ArchitectureEpisodeArtifact:
        return await self._implementation.finish_interactive_episode(task_id, outcome)
