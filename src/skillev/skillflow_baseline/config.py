"""Frozen upstream semantics and permitted Protocol 10 adaptations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from skillev.contracts import JsonValue, normalize_json

SKILLFLOW_UPSTREAM_REVISION: Final = "74be52bb6bd9f0e9e68dacb72636b75649197983"
SKILLFLOW_PARITY_CONTRACT: Final = "skillflow-upstream-parity@1"


@dataclass(frozen=True, slots=True)
class ExactSkillFlowProtocolConfig:
    """Validated projection of upstream controls plus Protocol 10 shape."""

    values: dict[str, JsonValue]
    upstream_revision: str = SKILLFLOW_UPSTREAM_REVISION
    parity_contract: str = SKILLFLOW_PARITY_CONTRACT

    def __post_init__(self) -> None:
        normalized = normalize_json(self.values)
        if not isinstance(normalized, dict):
            raise TypeError("SkillFlow configuration projection must be an object")
        object.__setattr__(self, "values", normalized)
        if self.upstream_revision != SKILLFLOW_UPSTREAM_REVISION:
            raise ValueError("SkillFlow upstream revision differs from the parity oracle")
        if self.parity_contract != SKILLFLOW_PARITY_CONTRACT:
            raise ValueError("SkillFlow parity contract differs")
        required = {
            "backward_lora_rank": 16,
            "batch_size": 16,
            "beta": 1.0,
            "epsilon_min": 0.1,
            "kl_coeff": 0.01,
            "lora_alpha": 128,
            "lora_rank": 64,
            "max_grad_norm": 3.0,
            "max_steps": 288,
            "n_trajectories_per_question": 1,
            "plateau_window_size": 10,
            "reward_mode": "outcome_only",
            "skill_mode": "policy_action",
            "ttb_edge_normalization": "per-token",
            "ttb_length_normalization": "steps",
        }
        differing = {
            name: (normalized.get(name), expected)
            for name, expected in required.items()
            if normalized.get(name) != expected
        }
        if differing:
            raise ValueError(f"SkillFlow configuration changes frozen semantics: {differing}")
        targets = normalized.get("lora_target_modules")
        if targets != ["q_proj", "k_proj", "v_proj", "o_proj"]:
            raise ValueError("SkillFlow theta LoRA targets differ from upstream")

    def runtime_values(
        self,
        *,
        base_model_path: str,
        output_directory: str,
        supervisor_api_base: str,
        supervisor_model: str,
    ) -> dict[str, object]:
        values: dict[str, object] = dict(self.values)
        values.update(
            {
                "base_model": base_model_path,
                "device": "cuda:0",
                "extra_device": None,
                "formal_checkpoint": True,
                "formal_runtime": True,
                "output_dir": output_directory,
                "parity_contract": self.parity_contract,
                "rollout_workers": 16,
                "supervisor_api_base": supervisor_api_base,
                "supervisor_mode": "external",
                "supervisor_model": supervisor_model,
                "tracking_mode": "disabled",
                "upstream_revision": self.upstream_revision,
            }
        )
        return values


__all__ = [
    "SKILLFLOW_PARITY_CONTRACT",
    "SKILLFLOW_UPSTREAM_REVISION",
    "ExactSkillFlowProtocolConfig",
]
