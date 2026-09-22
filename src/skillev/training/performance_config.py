"""Execution-only configuration shared by formal and integration runtimes."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import cast

import yaml

from skillev.contracts import JsonValue
from skillev.policy.scoring_execution import TeacherForcingConfig

from .rollout_workflow import RolloutWorkflowBinding


@dataclass(frozen=True, slots=True)
class TrainingPerformanceConfig:
    resident_trajectories: int = 16
    actor_requests: int = 8
    environment_calls: int = 8
    terminal_evaluations: int = 8
    process_graders: int = 4
    transport_threads: int = 16
    coordinator_participates: bool = True
    max_running_requests: int = 12
    max_loras_per_batch: int = 2
    deterministic_gradients: bool = True
    disable_radix_cache: bool = True
    pipeline_mode: str = "within-step"
    request_scheduling_policy: str = "fifo"
    rollout_scheduling_policy: str = "stable-fifo-work-conserving"
    provisional_edge_gradients: bool = False
    provisional_queue_bytes: int = 128 * 1024 * 1024
    provisional_plan_bytes: int = 1024 * 1024 * 1024
    provisional_device_bytes: int = 0
    z_feature_cache_entries: int = 256
    gradient_buffer_bytes: int = 512 * 1024 * 1024
    teacher_forcing: TeacherForcingConfig = field(default_factory=TeacherForcingConfig)
    fla_profile: dict[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        for entry in fields(self):
            value = getattr(self, entry.name)
            if entry.name not in {
                "teacher_forcing",
                "fla_profile",
                "coordinator_participates",
                "pipeline_mode",
                "rollout_scheduling_policy",
                "request_scheduling_policy",
                "provisional_edge_gradients",
                "provisional_device_bytes",
                "disable_radix_cache",
                "deterministic_gradients",
            }:
                if type(value) is not int or value < 1:
                    raise ValueError("execution capacities must be positive integers")
        if type(self.provisional_device_bytes) is not int or self.provisional_device_bytes < 0:
            raise ValueError("device gradient cache capacity must be non-negative")
        if type(self.deterministic_gradients) is not bool:
            raise TypeError("gradient determinism switch must be boolean")
        if type(self.disable_radix_cache) is not bool:
            raise TypeError("prefix cache switch must be boolean")
        if type(self.coordinator_participates) is not bool:
            raise TypeError("coordinator participation must be boolean")
        if self.max_loras_per_batch < 2:
            raise ValueError("base and current LoRA require two serving slots")
        if self.pipeline_mode not in {"sealed-batch", "within-step"}:
            raise ValueError("unsupported gradient execution mode")
        if type(self.provisional_edge_gradients) is not bool:
            raise TypeError("provisional edge gradients switch must be boolean")
        self.workflow()  # Validate the declared execution policy in its sole owner.

    def configure_process(self) -> None:
        """Fix the observed SDPA backward noise before initializing CUDA owners."""
        import torch

        if self.deterministic_gradients:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(self.deterministic_gradients)

    def workflow(self) -> RolloutWorkflowBinding:
        return RolloutWorkflowBinding(
            max_resident_trajectories=self.resident_trajectories,
            max_inflight_model_requests=self.actor_requests,
            max_inflight_environment_calls=self.environment_calls,
            max_inflight_terminal_evaluations=self.terminal_evaluations,
            max_inflight_process_graders=self.process_graders,
            transport_worker_threads=self.transport_threads,
            scheduling_policy=self.rollout_scheduling_policy,
            request_scheduling_policy=self.request_scheduling_policy,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], asdict(self))

    @classmethod
    def from_value(cls, value: object) -> TrainingPerformanceConfig:
        names = {f.name for f in fields(cls)}
        optional = {
            "gradient_buffer_bytes",
            "teacher_forcing",
            "fla_profile",
            "rollout_scheduling_policy",
            "request_scheduling_policy",
            "provisional_edge_gradients",
            "provisional_queue_bytes",
            "provisional_plan_bytes",
            "provisional_device_bytes",
        }
        if not isinstance(value, dict) or not names - optional <= set(value) <= names:
            raise ValueError("execution profile has incompatible fields")
        values = dict(value)
        if "teacher_forcing" in values:
            if not isinstance(values["teacher_forcing"], dict):
                raise TypeError("teacher-forcing profile must be an object")
            values["teacher_forcing"] = TeacherForcingConfig(**values["teacher_forcing"])
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> TrainingPerformanceConfig:
        return cls.from_value(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
