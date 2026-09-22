"""Read-only binding of an explicitly chosen checkpoint to preloaded SGLang routes.

No checkpoint discovery, best/latest selection, tensor loading, service mutation,
or answer access. Paths and expanded checkpoint metadata stay in the coordinator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from skillev.evaluation.step0_integrity import InferenceArm
from skillev.evaluation.step0_types import ArchitectureInferenceState
from skillev.rollout.evaluation_sglang import EvaluationPolicyDescriptor


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    return value


def _identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(char in value for char in ("/", "\\", "\n"))
    ):
        raise ValueError("policy and adapter names must be labels, not private paths")
    return value


def _checkpoint(directory: Path) -> tuple[Path, dict[str, Any]]:
    policy_directory = (
        directory if (directory / "policy_state.json").is_file() else directory / "policy"
    )
    state = _object(policy_directory / "policy_state.json")
    if state.get("format") != "skillev-policy-checkpoint@4":
        raise ValueError("unsupported policy checkpoint format")
    step = state.get("optimizer_step")
    if type(step) is not int or step < 1:
        raise ValueError("trained evaluation needs a nonzero checkpoint update count")
    for field in ("backbone_id", "forward_version"):
        if not isinstance(state.get(field), str) or not state[field].strip():
            raise ValueError("checkpoint is missing its policy identity")
    if state["forward_version"] == "adapter-free":
        raise ValueError("trained checkpoint cannot have an adapter-free forward version")
    runtime_directory = directory if policy_directory != directory else directory.parent
    if policy_directory != directory or (runtime_directory / "COMPLETE").is_file():
        # This is the training writer's atomic completion marker, not a new approval gate.
        if not (runtime_directory / "COMPLETE").is_file():
            raise ValueError("training checkpoint has not finished writing")
        runtime = _object(runtime_directory / "runtime_state.json")
        if runtime.get("optimizer_step") != step:
            raise ValueError("training and forward checkpoint steps disagree")
    forward_directory = policy_directory / "forward_adapter"
    adapter = _object(forward_directory / "adapter_config.json")
    if adapter.get("peft_type") != "LORA" or not adapter.get("base_model_name_or_path"):
        raise ValueError("checkpoint must contain its forward PEFT LoRA configuration")
    weights = forward_directory / "adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise ValueError("forward adapter weights are missing or empty")
    return forward_directory, {
        "optimizer_steps": step,
        "forward_version": state["forward_version"],
        "backbone_id": state["backbone_id"],
        "adapter_config": adapter,
        "runtime_checkpoint_directory": str(runtime_directory)
        if (runtime_directory / "COMPLETE").is_file()
        else None,
    }


@dataclass(frozen=True, slots=True)
class TrainedPolicyBinding:
    policy: EvaluationPolicyDescriptor
    adapter_name: str
    checkpoint_directory: Path
    forward_directory: Path
    served_adapter_paths: tuple[str, ...]
    metadata: dict[str, Any]

    @classmethod
    def read(
        cls,
        policy_id: str,
        settings: dict[str, Any],
        *,
        observed: list[dict[str, Any]],
        tokenizer_id: str,
    ) -> TrainedPolicyBinding:
        policy_id = _identifier(policy_id)
        adapter_name = _identifier(settings["adapter_name"])
        directory = Path(settings["checkpoint_directory"]).expanduser()
        if not directory.is_absolute():
            raise ValueError("checkpoint_directory must be an explicit absolute directory")
        directory = directory.resolve(strict=True)
        forward_directory, metadata = _checkpoint(directory)
        raw_paths = settings.get("served_adapter_paths", [str(forward_directory)] * len(observed))
        if (
            not isinstance(raw_paths, list)
            or len(raw_paths) != len(observed)
            or any(
                not isinstance(path, str) or not PurePosixPath(path).is_absolute()
                for path in raw_paths
            )
        ):
            raise ValueError("served_adapter_paths must give one absolute forward path per replica")
        binding = cls(
            EvaluationPolicyDescriptor(
                policy_id, "Qwen3.5-9B", tokenizer_id, metadata["forward_version"]
            ),
            adapter_name,
            directory,
            forward_directory,
            tuple(raw_paths),
            metadata,
        )
        binding.validate(observed)
        return binding

    def require_arm(self, arm: InferenceArm) -> None:
        if (
            arm.policy_id != self.policy.snapshot_id
            or arm.optimizer_steps != self.metadata["optimizer_steps"]
        ):
            raise ValueError("arm identity/update count differs from its explicit checkpoint")

    @property
    def inference_state(self) -> ArchitectureInferenceState:
        # Backward/Z/posterior/operator state is not executed by this forward-only evaluator.
        return ArchitectureInferenceState(
            method_id="skillev-bayesian-improve-forward-evaluation@1",
            optimizer_steps=self.metadata["optimizer_steps"],
            forward_adapter_active=True,
            action_policy_authority="evaluated-forward-policy-single-owner-final",
        )

    def controls(self) -> dict[str, object]:
        return {
            **self.metadata,
            "checkpoint_directory": str(self.checkpoint_directory),
            "forward_directory": str(self.forward_directory),
            "served_adapter_paths": list(self.served_adapter_paths),
        }

    def validate(self, observed: list[dict[str, Any]]) -> None:
        directory, metadata = _checkpoint(self.checkpoint_directory)
        if directory != self.forward_directory or metadata != self.metadata:
            raise ValueError("the explicitly selected checkpoint changed during evaluation")
        if len(observed) != len(self.served_adapter_paths):
            raise ValueError("the configured adapter replica population changed")
        for replica, expected_path in zip(observed, self.served_adapter_paths, strict=True):
            arguments = replica["server_info"].get("server_args", replica["server_info"])
            if arguments.get("enable_lora") is not True:
                raise ValueError("the actual inference service has not enabled LoRA")
            if (
                self.metadata["adapter_config"]["base_model_name_or_path"]
                != replica["model_info"]["model_path"]
            ):
                raise ValueError(
                    "forward adapter and actual service use different base model paths"
                )
            cards = replica.get("models", {}).get("data", [])
            matched = [card for card in cards if card.get("id") == self.adapter_name]
            if (
                len(matched) != 1
                or matched[0].get("root") != expected_path
                or not matched[0].get("parent")
                or matched[0]["parent"] != arguments.get("served_model_name")
            ):
                raise ValueError("the named service route is not the selected forward adapter")
