"""Runtime-authored Protocol 13 execution evidence.

The receipt proves explicit identities and task-order conservation.  It does
not compute file hashes and never contains answers or benchmark inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from skillev.evaluation.direct_baseline.config import DirectDecodingProfile

from .contracts import ExecutionContractV3

START_FORMAT = "skillev-protocol13-execution-start@1"
COMPLETE_FORMAT = "skillev-protocol13-execution-complete@1"


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _optional_text(value: object, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a mapping")
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    result = tuple(_text(item, label) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique values")
    return result


@dataclass(frozen=True, slots=True)
class RuntimeGenerationProfile:
    profile_id: str
    sampling_mode: str
    enable_thinking: bool
    temperature: str
    top_p: str
    top_k: int
    min_p: str
    presence_penalty: str
    repetition_penalty: str
    max_new_tokens: int
    seed: int
    stop: tuple[str, ...]

    @classmethod
    def from_direct_profile(cls, profile: DirectDecodingProfile) -> RuntimeGenerationProfile:
        return cls(
            profile_id=profile.profile_id,
            sampling_mode=profile.sampling_mode,
            enable_thinking=profile.enable_thinking,
            temperature=str(profile.temperature),
            top_p=str(profile.top_p),
            top_k=profile.top_k,
            min_p=str(profile.min_p),
            presence_penalty=str(profile.presence_penalty),
            repetition_penalty=str(profile.repetition_penalty),
            max_new_tokens=profile.max_new_tokens,
            seed=profile.seed,
            stop=profile.stop,
        )

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

    @classmethod
    def from_mapping(cls, value: object) -> RuntimeGenerationProfile:
        row = _mapping(value, "generation profile")
        expected = {
            "profile_id",
            "sampling_mode",
            "enable_thinking",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
            "max_new_tokens",
            "seed",
            "stop",
        }
        if set(row) != expected:
            raise ValueError("generation profile fields differ")
        return cls(
            profile_id=_text(row["profile_id"], "profile ID"),
            sampling_mode=_text(row["sampling_mode"], "sampling mode"),
            enable_thinking=_boolean(row["enable_thinking"], "enable thinking"),
            temperature=_text(row["temperature"], "temperature"),
            top_p=_text(row["top_p"], "top p"),
            top_k=_integer(row["top_k"], "top k"),
            min_p=_text(row["min_p"], "min p"),
            presence_penalty=_text(row["presence_penalty"], "presence penalty"),
            repetition_penalty=_text(row["repetition_penalty"], "repetition penalty"),
            max_new_tokens=_integer(row["max_new_tokens"], "maximum new tokens"),
            seed=_integer(row["seed"], "seed"),
            stop=_strings(row["stop"], "stop sequences"),
        )


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentIdentity:
    environment_profile: str
    runtime_source_revision: str
    deployment_id: str
    horizon_policy: str
    max_steps_cap: int
    required_max_steps: int | None
    history_window_steps: int | None
    history_maximum_characters: int | None
    include_reasoning_in_history: bool
    invalid_candidate_policy: str
    invalid_environment_action_policy: str
    prompt_asset_id: str
    prompt_asset_source_revision: str
    manifest_identity: dict[str, object]

    def to_mapping(self) -> dict[str, object]:
        return {
            "environment_profile": self.environment_profile,
            "runtime_source_revision": self.runtime_source_revision,
            "deployment_id": self.deployment_id,
            "horizon_policy": self.horizon_policy,
            "max_steps_cap": self.max_steps_cap,
            "required_max_steps": self.required_max_steps,
            "history_window_steps": self.history_window_steps,
            "history_maximum_characters": self.history_maximum_characters,
            "include_reasoning_in_history": self.include_reasoning_in_history,
            "invalid_candidate_policy": self.invalid_candidate_policy,
            "invalid_environment_action_policy": self.invalid_environment_action_policy,
            "prompt_asset_id": self.prompt_asset_id,
            "prompt_asset_source_revision": self.prompt_asset_source_revision,
            "manifest_identity": self.manifest_identity,
        }

    @classmethod
    def from_mapping(cls, value: object) -> RuntimeEnvironmentIdentity:
        row = _mapping(value, "environment identity")
        expected = {
            "environment_profile",
            "runtime_source_revision",
            "deployment_id",
            "horizon_policy",
            "max_steps_cap",
            "required_max_steps",
            "history_window_steps",
            "history_maximum_characters",
            "include_reasoning_in_history",
            "invalid_candidate_policy",
            "invalid_environment_action_policy",
            "prompt_asset_id",
            "prompt_asset_source_revision",
            "manifest_identity",
        }
        if set(row) != expected:
            raise ValueError("environment identity fields differ")
        window = row["history_window_steps"]
        if window is not None:
            window = _integer(window, "history window steps")
        required_steps = row["required_max_steps"]
        if required_steps is not None:
            required_steps = _integer(required_steps, "required maximum steps")
        history_characters = row["history_maximum_characters"]
        if history_characters is not None:
            history_characters = _integer(history_characters, "history maximum characters")
        return cls(
            environment_profile=_text(row["environment_profile"], "environment profile"),
            runtime_source_revision=_text(
                row["runtime_source_revision"], "runtime source revision"
            ),
            deployment_id=_text(row["deployment_id"], "deployment ID"),
            horizon_policy=_text(row["horizon_policy"], "horizon policy"),
            max_steps_cap=_integer(row["max_steps_cap"], "maximum step cap"),
            required_max_steps=required_steps,
            history_window_steps=window,
            history_maximum_characters=history_characters,
            include_reasoning_in_history=_boolean(
                row["include_reasoning_in_history"], "include reasoning in history"
            ),
            invalid_candidate_policy=_text(
                row["invalid_candidate_policy"], "invalid candidate policy"
            ),
            invalid_environment_action_policy=_text(
                row["invalid_environment_action_policy"],
                "invalid environment action policy",
            ),
            prompt_asset_id=_text(row["prompt_asset_id"], "prompt asset ID"),
            prompt_asset_source_revision=_text(
                row["prompt_asset_source_revision"], "prompt asset source revision"
            ),
            manifest_identity=_mapping(row["manifest_identity"], "environment manifest identity"),
        )


@dataclass(frozen=True, slots=True)
class Protocol13ExecutionReceipt:
    format_version: str
    status: str
    attempt_id: str
    execution: dict[str, object]
    generation_profile: RuntimeGenerationProfile
    environment: RuntimeEnvironmentIdentity | None
    model_route: str
    service_instance_ids: tuple[str, ...]
    response_model_ids: tuple[str, ...]
    context_length: int
    adapter_active: bool
    panel_manifest_id: str
    planned_task_ids: tuple[str, ...]
    generated_task_ids: tuple[str, ...]
    scored_task_ids: tuple[str, ...]
    generation_code_revision: str
    scoring_code_revision: str
    evaluator_version: str
    evaluator_contract: dict[str, object] | None
    started_at: str
    completed_at: str | None

    def __post_init__(self) -> None:
        if self.status not in {"running", "complete"}:
            raise ValueError("execution status must be running or complete")
        expected_format = COMPLETE_FORMAT if self.status == "complete" else START_FORMAT
        if self.format_version != expected_format:
            raise ValueError("execution receipt format differs from status")
        if self.status == "complete" and self.completed_at is None:
            raise ValueError("complete execution lacks completion time")
        if self.status == "running" and self.completed_at is not None:
            raise ValueError("running execution cannot have completion time")
        if self.context_length <= 0:
            raise ValueError("context length must be positive")
        if not self.service_instance_ids:
            raise ValueError("execution receipt lacks model service instances")
        if len(self.service_instance_ids) != len(set(self.service_instance_ids)):
            raise ValueError("execution receipt service instances are duplicated")
        required = (
            self.attempt_id,
            self.model_route,
            self.panel_manifest_id,
            self.generation_code_revision,
            self.scoring_code_revision,
            self.evaluator_version,
            self.started_at,
        )
        if any(not item.strip() for item in required):
            raise ValueError("execution receipt identity is incomplete")
        if len(self.planned_task_ids) != len(set(self.planned_task_ids)):
            raise ValueError("planned task IDs are duplicated")

    def validate_against(
        self,
        *,
        execution: ExecutionContractV3,
        expected_task_ids: tuple[str, ...],
        expected_generation_profile: DirectDecodingProfile,
        expected_evaluator_contract: dict[str, object] | None = None,
        expected_environment_manifest: dict[str, object] | None = None,
    ) -> None:
        if self.status != "complete" or self.format_version != COMPLETE_FORMAT:
            raise ValueError("publication requires a complete execution")
        if self.execution != execution.to_mapping():
            raise ValueError("runtime execution differs from condition")
        if self.panel_manifest_id != execution.panel_manifest_id:
            raise ValueError("runtime panel manifest differs")
        if self.planned_task_ids != expected_task_ids:
            raise ValueError("planned task order differs")
        if self.generated_task_ids != expected_task_ids:
            raise ValueError("generation coverage differs")
        if self.scored_task_ids != expected_task_ids:
            raise ValueError("scoring coverage differs")
        if self.model_route != execution.actor_route:
            raise ValueError("runtime model route differs")
        if self.response_model_ids != (execution.actor_route,):
            raise ValueError("response model identity differs")
        if self.context_length != execution.context_length:
            raise ValueError("context length differs")
        if self.adapter_active:
            raise ValueError("backbone-only execution used an adapter")
        expected_profile = RuntimeGenerationProfile.from_direct_profile(expected_generation_profile)
        if self.generation_profile != expected_profile:
            raise ValueError("runtime generation parameters differ")
        if expected_profile.profile_id != execution.decoding_profile:
            raise ValueError("frozen generation profile differs from execution")
        if execution.evaluator_profile is None:
            if self.evaluator_version != execution.scorer_profile:
                raise ValueError("runtime scorer identity differs")
            if self.evaluator_contract is not None:
                raise ValueError("non-code execution unexpectedly has an evaluator contract")
        else:
            if self.evaluator_version != execution.evaluator_profile:
                raise ValueError("runtime evaluator identity differs")
            if self.evaluator_contract is None:
                raise ValueError("code execution lacks its evaluator contract")
            if self.evaluator_contract.get("profile_id") != execution.evaluator_profile:
                raise ValueError("runtime evaluator profile differs")
        if (
            expected_evaluator_contract is not None
            and self.evaluator_contract != expected_evaluator_contract
        ):
            raise ValueError("runtime evaluator contract differs")
        if execution.environment_profile is None:
            if self.environment is not None:
                raise ValueError("static execution unexpectedly used an environment")
        elif self.environment is None:
            raise ValueError("interactive execution lacks environment identity")
        elif self.environment.environment_profile != execution.environment_profile:
            raise ValueError("environment profile differs")
        elif execution.interactive is None:
            raise ValueError("interactive execution lacks its condition extension")
        else:
            expected_environment = {
                "horizon_policy": execution.interactive.horizon_policy,
                "max_steps_cap": execution.interactive.max_steps_cap,
                "required_max_steps": execution.interactive.required_max_steps,
                "history_window_steps": execution.interactive.history_window_steps,
                "history_maximum_characters": (execution.interactive.history_maximum_characters),
                "include_reasoning_in_history": (
                    execution.interactive.include_reasoning_in_history
                ),
                "invalid_candidate_policy": execution.interactive.invalid_candidate_policy,
                "invalid_environment_action_policy": (
                    execution.interactive.invalid_environment_action_policy
                ),
                "prompt_asset_id": execution.interactive.prompt_asset_id,
                "prompt_asset_source_revision": (
                    execution.interactive.prompt_asset_source_revision
                ),
            }
            observed_environment = self.environment.to_mapping()
            for name, expected in expected_environment.items():
                if observed_environment[name] != expected:
                    raise ValueError(f"runtime environment {name} differs")
            if self.environment.manifest_identity != expected_environment_manifest:
                raise ValueError("runtime environment manifest identity differs")

    def to_mapping(self) -> dict[str, object]:
        return {
            "format": self.format_version,
            "status": self.status,
            "attempt_id": self.attempt_id,
            "execution": self.execution,
            "generation_profile": self.generation_profile.to_mapping(),
            "environment": self.environment.to_mapping() if self.environment else None,
            "model_route": self.model_route,
            "service_instance_ids": list(self.service_instance_ids),
            "response_model_ids": list(self.response_model_ids),
            "context_length": self.context_length,
            "adapter_active": self.adapter_active,
            "panel_manifest_id": self.panel_manifest_id,
            "planned_task_ids": list(self.planned_task_ids),
            "generated_task_ids": list(self.generated_task_ids),
            "scored_task_ids": list(self.scored_task_ids),
            "generation_code_revision": self.generation_code_revision,
            "scoring_code_revision": self.scoring_code_revision,
            "evaluator_version": self.evaluator_version,
            "evaluator_contract": self.evaluator_contract,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


_RECEIPT_FIELDS = {
    "format",
    "status",
    "attempt_id",
    "execution",
    "generation_profile",
    "environment",
    "model_route",
    "service_instance_ids",
    "response_model_ids",
    "context_length",
    "adapter_active",
    "panel_manifest_id",
    "planned_task_ids",
    "generated_task_ids",
    "scored_task_ids",
    "generation_code_revision",
    "scoring_code_revision",
    "evaluator_version",
    "evaluator_contract",
    "started_at",
    "completed_at",
}


def load_execution_receipt(path: Path) -> Protocol13ExecutionReceipt:
    row = _mapping(json.loads(path.read_text(encoding="utf-8")), "execution receipt")
    if set(row) != _RECEIPT_FIELDS:
        raise ValueError("execution receipt fields differ")
    environment = row["environment"]
    evaluator_contract = row["evaluator_contract"]
    return Protocol13ExecutionReceipt(
        format_version=_text(row["format"], "format"),
        status=_text(row["status"], "status"),
        attempt_id=_text(row["attempt_id"], "attempt ID"),
        execution=_mapping(row["execution"], "execution"),
        generation_profile=RuntimeGenerationProfile.from_mapping(row["generation_profile"]),
        environment=(
            None if environment is None else RuntimeEnvironmentIdentity.from_mapping(environment)
        ),
        model_route=_text(row["model_route"], "model route"),
        service_instance_ids=_strings(row["service_instance_ids"], "service instance IDs"),
        response_model_ids=_strings(row["response_model_ids"], "response model IDs"),
        context_length=_integer(row["context_length"], "context length"),
        adapter_active=_boolean(row["adapter_active"], "adapter active"),
        panel_manifest_id=_text(row["panel_manifest_id"], "panel manifest ID"),
        planned_task_ids=_strings(row["planned_task_ids"], "planned task IDs"),
        generated_task_ids=_strings(row["generated_task_ids"], "generated task IDs"),
        scored_task_ids=_strings(row["scored_task_ids"], "scored task IDs"),
        generation_code_revision=_text(row["generation_code_revision"], "generation code revision"),
        scoring_code_revision=_text(row["scoring_code_revision"], "scoring code revision"),
        evaluator_version=_text(row["evaluator_version"], "evaluator version"),
        evaluator_contract=(
            None
            if evaluator_contract is None
            else _mapping(evaluator_contract, "evaluator contract")
        ),
        started_at=_text(row["started_at"], "started at"),
        completed_at=_optional_text(row["completed_at"], "completed at"),
    )


def write_execution_receipt(path: Path, receipt: Protocol13ExecutionReceipt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = receipt.to_mapping()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise FileExistsError("execution receipt already differs")
        return
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "COMPLETE_FORMAT",
    "START_FORMAT",
    "Protocol13ExecutionReceipt",
    "RuntimeEnvironmentIdentity",
    "RuntimeGenerationProfile",
    "load_execution_receipt",
    "write_execution_receipt",
]
