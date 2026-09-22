"""Canonical private composition root for Protocol 10 training methods.

This module deliberately knows nothing about the historical Protocol 9 catalog
or its semantic-selection machinery. It seals the common 4,608-episode mix
once, then dispatches exactly one of the three Protocol 10 method builders with
the same runtime dependencies and ordered tasks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from skillev.application import FormalRuntimeDependencies
from skillev.experiments import FormalMethodV10, ProtocolV10FormalExperimentSpec
from skillev.rollout import RolloutTask, UnskilledRolloutSessionBundle
from skillev.runtime import OrderedTaskCursorState
from skillev.training import RolloutWorkflowBinding, TaskProvider
from skillev_private.benchmarks import (
    ProtocolV10PopulationCatalog,
    ProtocolV10PopulationSessionRegistry,
    ProtocolV10TrainingMix,
    ProtocolV10TrainingSelection,
    materialize_protocol_v10_training_mix,
)

ApplicationT = TypeVar("ApplicationT")


@dataclass(slots=True)
class ProtocolV10OrderedTaskProvider(TaskProvider):
    """Cursor over the sealed SkillFlow-style 4,608-episode task sequence."""

    tasks: tuple[RolloutTask, ...]
    curriculum_id: str
    cursor: int = 0

    def __post_init__(self) -> None:
        if not self.tasks or len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Protocol 10 task provider requires unique ordered tasks")
        if not self.curriculum_id.strip() or not 0 <= self.cursor <= len(self.tasks):
            raise ValueError("Protocol 10 task provider cursor is invalid")

    def next_task(self) -> RolloutTask:
        if self.cursor >= len(self.tasks):
            raise RuntimeError("Protocol 10 training curriculum is exhausted")
        task = self.tasks[self.cursor]
        self.cursor += 1
        return task

    @property
    def runtime_state(self) -> OrderedTaskCursorState:
        return OrderedTaskCursorState(self.curriculum_id, self.cursor)


@dataclass(frozen=True, slots=True)
class ProtocolV10TaskProviderFactory:
    tasks: tuple[RolloutTask, ...]
    curriculum_id: str

    def fresh(self) -> ProtocolV10OrderedTaskProvider:
        return ProtocolV10OrderedTaskProvider(self.tasks, self.curriculum_id)

    def from_exact_state(self, state: OrderedTaskCursorState) -> TaskProvider:
        if state.curriculum_id != self.curriculum_id:
            raise ValueError("Protocol 10 resume cursor belongs to another curriculum")
        return ProtocolV10OrderedTaskProvider(self.tasks, self.curriculum_id, state.cursor)


@dataclass(frozen=True, slots=True)
class ProtocolV10BaseSessionFactory:
    """Expose trusted V10 sessions before method-specific skill retrieval."""

    training_mix: ProtocolV10TrainingMix

    def create(self, task: RolloutTask) -> UnskilledRolloutSessionBundle:
        bundle = self.training_mix.session_factory.create(task)
        if bundle.retrieved_skills:
            raise ValueError("Protocol 10 private session injected method-visible skills")
        return UnskilledRolloutSessionBundle(
            environment=bundle.environment,
            evaluator=bundle.evaluator,
            cleanup=bundle.cleanup,
        )


@dataclass(frozen=True, slots=True)
class ProtocolV10ApplicationInputs:
    """Method-independent scientific and execution inputs for one arm."""

    method: FormalMethodV10
    training_mix: ProtocolV10TrainingMix
    runtime: FormalRuntimeDependencies
    workflow_binding: RolloutWorkflowBinding
    task_provider_factory: ProtocolV10TaskProviderFactory = field(init=False)
    base_session_factory: ProtocolV10BaseSessionFactory = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.method, FormalMethodV10):
            raise TypeError("Protocol 10 application method is not closed")
        if not isinstance(self.training_mix, ProtocolV10TrainingMix):
            raise TypeError("Protocol 10 application requires the sealed training mix")
        if not isinstance(self.runtime, FormalRuntimeDependencies):
            raise TypeError("Protocol 10 application requires formal runtime dependencies")
        if self.workflow_binding != self.runtime.workflow_resources.binding:
            raise ValueError("Protocol 10 workflow binding differs from runtime resources")
        object.__setattr__(
            self,
            "task_provider_factory",
            ProtocolV10TaskProviderFactory(
                self.training_mix.tasks,
                (f"{self.training_mix.selection.format}:{self.training_mix.selection.algorithm}"),
            ),
        )
        object.__setattr__(
            self,
            "base_session_factory",
            ProtocolV10BaseSessionFactory(self.training_mix),
        )


ProtocolV10ApplicationBuilder = Callable[[ProtocolV10ApplicationInputs], ApplicationT]


@dataclass(frozen=True, slots=True)
class ProtocolV10MethodBuilders(Generic[ApplicationT]):
    """Three non-interchangeable executable builders; no feature flags."""

    skillflow_baseline: ProtocolV10ApplicationBuilder[ApplicationT]
    bayesian_improve_full: ProtocolV10ApplicationBuilder[ApplicationT]
    bayesian_improve_no_calibration: ProtocolV10ApplicationBuilder[ApplicationT]

    def __post_init__(self) -> None:
        builders = (
            self.skillflow_baseline,
            self.bayesian_improve_full,
            self.bayesian_improve_no_calibration,
        )
        if any(not callable(builder) for builder in builders):
            raise TypeError("Protocol 10 requires all three executable method builders")
        if len({id(builder) for builder in builders}) != len(builders):
            raise ValueError("Protocol 10 methods cannot share one ambiguous builder")

    def for_method(
        self,
        method: FormalMethodV10,
    ) -> ProtocolV10ApplicationBuilder[ApplicationT]:
        match method:
            case FormalMethodV10.SKILLFLOW_BASELINE:
                return self.skillflow_baseline
            case FormalMethodV10.BAYESIAN_IMPROVE_FULL:
                return self.bayesian_improve_full
            case FormalMethodV10.BAYESIAN_IMPROVE_NO_CALIBRATION:
                return self.bayesian_improve_no_calibration


@dataclass(frozen=True, slots=True)
class BuiltProtocolV10Application(Generic[ApplicationT]):
    method: FormalMethodV10
    application: ApplicationT
    training_mix: ProtocolV10TrainingMix


@dataclass(frozen=True, slots=True)
class ProtocolV10AttemptBuilder(Generic[ApplicationT]):
    """Build one method only after every Protocol 10 gate is satisfied."""

    experiment: ProtocolV10FormalExperimentSpec
    catalog: ProtocolV10PopulationCatalog
    selection: ProtocolV10TrainingSelection
    sessions: ProtocolV10PopulationSessionRegistry
    runtime: FormalRuntimeDependencies
    workflow_binding: RolloutWorkflowBinding
    methods: ProtocolV10MethodBuilders[ApplicationT]

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, ProtocolV10FormalExperimentSpec):
            raise TypeError("Protocol 10 builder requires the frozen formal experiment")
        if not isinstance(self.catalog, ProtocolV10PopulationCatalog):
            raise TypeError("Protocol 10 builder requires the private population catalog")
        if not isinstance(self.selection, ProtocolV10TrainingSelection):
            raise TypeError("Protocol 10 builder requires the frozen training selection")
        if not isinstance(self.sessions, ProtocolV10PopulationSessionRegistry):
            raise TypeError("Protocol 10 builder requires population session routes")
        if not isinstance(self.runtime, FormalRuntimeDependencies):
            raise TypeError("Protocol 10 builder requires formal runtime dependencies")
        if not isinstance(self.workflow_binding, RolloutWorkflowBinding):
            raise TypeError("Protocol 10 builder requires a workflow binding")
        if not isinstance(self.methods, ProtocolV10MethodBuilders):
            raise TypeError("Protocol 10 builder requires closed method builders")
        if self.catalog.protocol.seed != self.experiment.seed:
            raise ValueError("Protocol 10 experiment and population seed differ")
        if self.experiment.total_episodes != len(self.selection.episodes):
            raise ValueError("Protocol 10 selection does not fill the frozen experiment")
        if self.experiment.batch_size * self.experiment.total_steps != len(self.selection.episodes):
            raise ValueError("Protocol 10 batch partition does not consume the selection")
        if self.workflow_binding != self.runtime.workflow_resources.binding:
            raise ValueError("Protocol 10 workflow binding differs from runtime resources")

    def build(self, method: FormalMethodV10) -> BuiltProtocolV10Application[ApplicationT]:
        self.experiment.require_execution_ready(self.catalog.protocol)
        return self._build(method)

    def _build(self, method: FormalMethodV10) -> BuiltProtocolV10Application[ApplicationT]:
        self.experiment.binding(method)
        training_mix = materialize_protocol_v10_training_mix(
            self.catalog,
            self.selection,
            self.sessions,
        )
        inputs = ProtocolV10ApplicationInputs(
            method=method,
            training_mix=training_mix,
            runtime=self.runtime,
            workflow_binding=self.workflow_binding,
        )
        application = self.methods.for_method(method)(inputs)
        return BuiltProtocolV10Application(method, application, training_mix)


@dataclass(frozen=True, slots=True)
class ProtocolV10IntegrationSmokeBuilder(Generic[ApplicationT]):
    """Use production data/runtime composition without opening either formal gate."""

    formal: ProtocolV10AttemptBuilder[ApplicationT]

    def build(self, method: FormalMethodV10) -> BuiltProtocolV10Application[ApplicationT]:
        return self.formal._build(method)


__all__ = [
    "BuiltProtocolV10Application",
    "ProtocolV10ApplicationBuilder",
    "ProtocolV10ApplicationInputs",
    "ProtocolV10AttemptBuilder",
    "ProtocolV10BaseSessionFactory",
    "ProtocolV10IntegrationSmokeBuilder",
    "ProtocolV10MethodBuilders",
    "ProtocolV10OrderedTaskProvider",
    "ProtocolV10TaskProviderFactory",
]
