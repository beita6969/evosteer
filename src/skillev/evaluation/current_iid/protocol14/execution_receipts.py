"""Private start/complete runtime receipts for Protocol 14 execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from skillev.evaluation.direct_baseline.config import DirectDecodingProfile

from .contracts import ExecutionContractV4

START_FORMAT = "skillev-protocol14-execution-start@1"
COMPLETE_FORMAT = "skillev-protocol14-execution-complete@1"


@dataclass(frozen=True, slots=True)
class RuntimeGenerationProfile:
    profile_id: str
    sampling_mode: str
    enable_thinking: bool
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    stop: tuple[str, ...]

    @classmethod
    def from_direct_profile(cls, profile: DirectDecodingProfile) -> RuntimeGenerationProfile:
        return cls(
            profile.profile_id,
            profile.sampling_mode,
            profile.enable_thinking,
            profile.temperature,
            profile.top_p,
            profile.top_k,
            profile.min_p,
            profile.presence_penalty,
            profile.repetition_penalty,
            profile.max_new_tokens,
            profile.seed,
            profile.stop,
        )

    def validate_against(self, execution: ExecutionContractV4) -> None:
        decoding = execution.decoding
        observed = (
            self.profile_id,
            self.sampling_mode,
            self.enable_thinking,
            self.temperature,
            self.top_p,
            self.top_k,
            self.min_p,
            self.presence_penalty,
            self.repetition_penalty,
            self.max_new_tokens,
            self.seed,
            self.stop,
        )
        expected = (
            execution.decoding_profile,
            decoding.sampling_mode,
            execution.thinking_mode.value == "enabled",
            decoding.temperature,
            decoding.top_p,
            decoding.top_k,
            decoding.min_p,
            decoding.presence_penalty,
            decoding.repetition_penalty,
            decoding.max_new_tokens,
            decoding.seed,
            decoding.stop,
        )
        if observed != expected:
            raise ValueError("runtime generation profile differs from Protocol 14")

    def to_mapping(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "sampling_mode": self.sampling_mode,
            "enable_thinking": self.enable_thinking,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "stop": list(self.stop),
        }


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentIdentity:
    environment_profile: str
    runtime_source_revision: str
    deployment_id: str
    horizon_policy: str
    max_steps: int
    history_window_steps: int | None
    history_maximum_characters: int
    history_observation_characters: int | None
    include_reasoning_in_history: bool
    invalid_candidate_policy: str
    invalid_environment_action_policy: str
    prompt_source_revision: str

    def validate_against(self, execution: ExecutionContractV4) -> None:
        extension = execution.interactive
        if extension is None or execution.environment_profile is None:
            raise ValueError("runtime environment supplied for a non-interactive execution")
        observed = (
            self.environment_profile,
            self.horizon_policy,
            self.max_steps,
            self.history_window_steps,
            self.history_maximum_characters,
            self.history_observation_characters,
            self.include_reasoning_in_history,
            self.invalid_candidate_policy,
            self.invalid_environment_action_policy,
            self.prompt_source_revision,
        )
        expected = (
            execution.environment_profile,
            extension.horizon_policy,
            extension.max_steps,
            extension.history_window_steps,
            extension.history_maximum_characters,
            extension.history_observation_characters,
            extension.include_reasoning_in_history,
            extension.invalid_candidate_policy,
            extension.invalid_environment_action_policy,
            extension.prompt_source_revision,
        )
        if observed != expected:
            raise ValueError("runtime environment differs from Protocol 14")
        if not self.runtime_source_revision.strip() or not self.deployment_id.strip():
            raise ValueError("runtime environment provenance is incomplete")

    def to_mapping(self) -> dict[str, object]:
        return {
            "environment_profile": self.environment_profile,
            "runtime_source_revision": self.runtime_source_revision,
            "deployment_id": self.deployment_id,
            "horizon_policy": self.horizon_policy,
            "max_steps": self.max_steps,
            "history_window_steps": self.history_window_steps,
            "history_maximum_characters": self.history_maximum_characters,
            "history_observation_characters": self.history_observation_characters,
            "include_reasoning_in_history": self.include_reasoning_in_history,
            "invalid_candidate_policy": self.invalid_candidate_policy,
            "invalid_environment_action_policy": self.invalid_environment_action_policy,
            "prompt_source_revision": self.prompt_source_revision,
        }


@dataclass(frozen=True, slots=True)
class Protocol14ExecutionReceipt:
    format_version: str
    status: str
    attempt_id: str
    attempt_role: str
    execution: dict[str, object]
    generation_profile: RuntimeGenerationProfile
    environment: RuntimeEnvironmentIdentity | None
    model_route: str
    runtime_model_revision: str
    response_model_ids: tuple[str, ...]
    context_length: int
    adapter_active: bool
    panel_manifest_id: str
    planned_task_ids: tuple[str, ...]
    generated_task_ids: tuple[str, ...]
    terminal_task_ids: tuple[str, ...]
    generation_code_revision: str
    scoring_code_revision: str
    evaluator_version: str
    started_at: str
    completed_at: str | None

    def validate_against(
        self, *, execution: ExecutionContractV4, expected_task_ids: tuple[str, ...]
    ) -> None:
        if self.format_version not in {START_FORMAT, COMPLETE_FORMAT}:
            raise ValueError("invalid Protocol 14 execution receipt format")
        if self.status not in {"running", "complete"}:
            raise ValueError("invalid Protocol 14 execution status")
        if self.attempt_role not in {"reference", "candidate", "canary"}:
            raise ValueError("invalid Protocol 14 attempt role")
        if self.execution != execution.to_mapping():
            raise ValueError("runtime execution mapping differs from Protocol 14")
        self.generation_profile.validate_against(execution)
        if (self.environment is None) != (execution.interactive is None):
            raise ValueError("runtime environment presence differs from execution")
        if self.environment is not None:
            self.environment.validate_against(execution)
        if (
            self.model_route != execution.actor_route
            or self.runtime_model_revision != execution.actor_model_revision
            or self.context_length != execution.context_length
            or self.panel_manifest_id != execution.panel_manifest_id
        ):
            raise ValueError("runtime model or panel identity differs from Protocol 14")
        if self.adapter_active:
            raise ValueError("Protocol 14 backbone receipt cannot activate an adapter")
        if self.planned_task_ids != expected_task_ids:
            raise ValueError("runtime planned task order differs from the manifest")
        if len(expected_task_ids) != execution.expected_count:
            raise ValueError("runtime manifest count differs from Protocol 14")
        expected_set = set(expected_task_ids)
        if not set(self.generated_task_ids).issubset(expected_set):
            raise ValueError("generated task IDs escape the panel")
        if not set(self.terminal_task_ids).issubset(set(self.generated_task_ids)):
            raise ValueError("terminal task IDs escape generated tasks")
        required = (
            self.attempt_id,
            self.model_route,
            self.runtime_model_revision,
            self.generation_code_revision,
            self.scoring_code_revision,
            self.evaluator_version,
            self.started_at,
        )
        if any(not value.strip() for value in required):
            raise ValueError("runtime execution provenance is incomplete")
        if self.status == "running":
            if self.format_version != START_FORMAT or self.completed_at is not None:
                raise ValueError("running receipt completion fields differ")
        elif self.format_version != COMPLETE_FORMAT or self.completed_at is None:
            raise ValueError("complete receipt completion fields differ")

    def to_mapping(self) -> dict[str, object]:
        return {
            "format": self.format_version,
            "status": self.status,
            "attempt_id": self.attempt_id,
            "attempt_role": self.attempt_role,
            "execution": self.execution,
            "generation_profile": self.generation_profile.to_mapping(),
            "environment": self.environment.to_mapping() if self.environment else None,
            "model_route": self.model_route,
            "runtime_model_revision": self.runtime_model_revision,
            "response_model_ids": list(self.response_model_ids),
            "context_length": self.context_length,
            "adapter_active": self.adapter_active,
            "panel_manifest_id": self.panel_manifest_id,
            "planned_task_ids": list(self.planned_task_ids),
            "generated_task_ids": list(self.generated_task_ids),
            "terminal_task_ids": list(self.terminal_task_ids),
            "generation_code_revision": self.generation_code_revision,
            "scoring_code_revision": self.scoring_code_revision,
            "evaluator_version": self.evaluator_version,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


def write_execution_receipt(path: Path, receipt: Protocol14ExecutionReceipt) -> None:
    path.write_text(
        json.dumps(receipt.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__ = [
    "COMPLETE_FORMAT",
    "START_FORMAT",
    "Protocol14ExecutionReceipt",
    "RuntimeEnvironmentIdentity",
    "RuntimeGenerationProfile",
    "write_execution_receipt",
]
